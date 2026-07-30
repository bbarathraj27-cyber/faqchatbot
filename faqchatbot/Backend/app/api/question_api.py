import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.config import settings
from app.models.chat_model import ChatLog
from app.models.feedback_model import Feedback
from app.services.cache_service import (
    get_cached_answer,
    set_cached_answer,
    make_signature,
    extend_cached_answer_ttl,
    purge_cached_answer,
)
from app.services.embedding_service import embed_text
from app.services.pinecone_service import query_similar_chunks, query_by_topics
from app.services.neo4j_service import get_related_topics
from app.services.llm_service import (
    generate_answer,
    correct_and_extract_topics,
    AnswerGenerationError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/question", tags=["question"])


def _log_chat(db: Session, question: str, answer: str, sources: list[str]) -> int | None:
    """Write a ChatLog row without letting a DB hiccup break the response.

    Previously db.add()/db.commit() ran unguarded in ask_question, so any
    transient DB issue (e.g. a free-tier Postgres cold-start/idle timeout)
    surfaced as an unhandled 500 to the frontend ("Something went wrong
    getting an answer"), even though retrieval + the LLM answer had already
    succeeded. Now we log best-effort and still return the answer.
    """
    try:
        log = ChatLog(
            question=question,
            answer=answer,
            source_urls=",".join(sources),
        )
        db.add(log)
        db.commit()
        db.refresh(log)
        return log.id
    except Exception as e:
        logger.warning(f"Could not write chat log (continuing without it): {e}")
        try:
            db.rollback()
        except Exception:
            pass
        return None

def _retrieve(corrected_question: str, question_topics: list[str]) -> list[dict]:
    """
    Runs RAG (Pinecone vector search) + KAG (Neo4j topic expansion).
    question_topics comes from the merged spelling+topic-extraction LLM
    call in ask_question, instead of a separate extract_topics() call here.
    """
    query_embedding = embed_text(corrected_question)
    vector_matches = query_similar_chunks(query_embedding, top_k=5)

    related_topics: list[str] = []
    if question_topics:
        with ThreadPoolExecutor(max_workers=len(question_topics)) as executor:
            for topics in executor.map(lambda t: get_related_topics(t, limit=3), question_topics):
                related_topics.extend(topics)
    related_topics = list(set(related_topics) - set(question_topics))

    graph_matches = query_by_topics(related_topics, top_k=3) if related_topics else []

    combined: dict[str, dict] = {}
    for m in vector_matches + graph_matches:
        key = m.get("id") or m["text"]
        combined.setdefault(key, m)
    return list(combined.values())


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    chat_log_id: int | None = None
    answer: str
    sources: list[str]
    from_cache: bool


class FeedbackRequest(BaseModel):
    chat_log_id: int
    is_helpful: bool
    comment: str | None = None


@router.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest, db: Session = Depends(get_db)):
    corrected_question, question_topics = correct_and_extract_topics(request.question)

    # --- Always retrieve first (RAG + KAG) ---
    all_matches = _retrieve(corrected_question, question_topics)
    chunk_ids = [m.get("id") or m["text"] for m in all_matches]
    fresh_signature = make_signature(chunk_ids) if chunk_ids else None

    # --- CAG: only serve from cache if the retrieved content is unchanged ---
    cached = get_cached_answer(corrected_question)
    if cached and fresh_signature and cached.get("signature") == fresh_signature:
        chat_log_id = _log_chat(db, corrected_question, cached["answer"], cached.get("sources", []))
        return QuestionResponse(
            answer=cached["answer"],
            sources=cached.get("sources", []),
            from_cache=True,
            chat_log_id=chat_log_id,
        )
    if cached and fresh_signature and cached.get("signature") != fresh_signature:
        logger.info(f"Fresh content detected for '{corrected_question}', regenerating instead of using cache")

    try:
        answer = generate_answer(corrected_question, all_matches)
    except AnswerGenerationError:
        fallback = "I'm having trouble reaching the answer service right now. Please try again in a moment."
        chat_log_id = _log_chat(db, corrected_question, fallback, [])
        return QuestionResponse(answer=fallback, sources=[], from_cache=False, chat_log_id=chat_log_id)

    sources = list({m["source_url"] for m in all_matches if m["source_url"]})
    result = {"answer": answer, "sources": sources}

    _NO_ANSWER_PHRASES = (
        "i don't know",
        "i do not know",
        "i don't have that information",
        "i don't have enough information",
        "not available in the",
        "based on the available documentation",
        "trouble reaching the answer service",
    )
    is_no_answer = any(p in answer.lower() for p in _NO_ANSWER_PHRASES)

    if all_matches and fresh_signature and not is_no_answer:
        set_cached_answer(corrected_question, result, fresh_signature)
    elif is_no_answer:
        logger.info(f"LLM gave a 'no answer' response for '{corrected_question}', skipping cache write")
    else:
        logger.info(f"No context found for '{corrected_question}', skipping cache write")

    chat_log_id = _log_chat(db, corrected_question, answer, sources)

    return QuestionResponse(**result, from_cache=False, chat_log_id=chat_log_id)


@router.post("/feedback")
def submit_feedback(request: FeedbackRequest, db: Session = Depends(get_db)):
    chat_log = db.query(ChatLog).filter(ChatLog.id == request.chat_log_id).first()
    if not chat_log:
        raise HTTPException(status_code=404, detail="chat_log_id not found")

    feedback = Feedback(
        chat_log_id=request.chat_log_id,
        is_helpful=request.is_helpful,
        comment=request.comment,
    )
    db.add(feedback)
    db.commit()

    # Make the vote actually do something to the cache, instead of just
    # sitting unused in the feedback table:
    #   👍 -> this cached answer is verified good, keep it around longer
    #   👎 -> purge it now so the next person gets a freshly regenerated
    #         answer instead of the same disliked one
    if request.is_helpful:
        extend_cached_answer_ttl(chat_log.question, settings.CACHE_TTL_SECONDS * 2)
    else:
        purge_cached_answer(chat_log.question)

    return {"status": "ok"}