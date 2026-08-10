import logging
import re
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

def _vector_search(corrected_question: str) -> list[dict]:
    """RAG leg: embed the question, then search Pinecone by vector similarity."""
    query_embedding = embed_text(corrected_question)
    return query_similar_chunks(query_embedding, top_k=12)


def _graph_search(question_topics: list[str]) -> list[dict]:
    """KAG leg: expand the question's topics via Neo4j, then search Pinecone by topic match."""
    related_topics: list[str] = []
    if question_topics:
        with ThreadPoolExecutor(max_workers=min(len(question_topics), 4)) as executor:
            for topics in executor.map(lambda t: get_related_topics(t, limit=5), question_topics):
                related_topics.extend(topics)

    related_topics = [t for t in dict.fromkeys(related_topics) if t not in set(question_topics)]
    if not related_topics and question_topics:
        related_topics = [t for t in dict.fromkeys(question_topics) if t]

    return query_by_topics(related_topics, top_k=8) if related_topics else []


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def _lexical_overlap_score(question_text: str, question_topics: list[str], match: dict) -> float:
    if not question_text and not question_topics:
        return 0.0

    question_tokens = _tokenize(question_text)
    if not question_tokens and not question_topics:
        return 0.0

    chunk_text = " ".join([
        match.get("text", ""),
        match.get("question", ""),
        " ".join(match.get("topics", []) or []),
    ])
    chunk_tokens = _tokenize(chunk_text)

    overlap = len(question_tokens & chunk_tokens)
    overlap_score = min(1.0, overlap / max(1, min(6, len(question_tokens))))

    topic_overlap = 0
    if question_topics and match.get("topics"):
        topic_overlap = len(set(question_topics) & set(match.get("topics", [])))
        topic_overlap = min(1.0, topic_overlap / max(1, min(3, len(question_topics))))

    return overlap_score * 0.7 + topic_overlap * 0.3


def rank_hybrid_matches(
    vector_matches: list[dict],
    graph_matches: list[dict],
    top_k: int = 8,
    question_text: str = "",
    question_topics: list[str] | None = None,
) -> list[dict]:
    """Fuse vector and topic-based matches into a single ranked list.

    The key improvement is to reward candidates that appear in both legs and to
    preserve the more informative score from either leg while preventing a single
    source from dominating the final ordering.
    """
    scored: dict[str, dict] = {}

    for match in vector_matches:
        key = match.get("id") or match["text"]
        scored[key] = {
            **match,
            "_vector_score": float(match.get("score", 0.0) or 0.0),
            "_graph_score": 0.0,
            "_support": 0,
        }

    for match in graph_matches:
        key = match.get("id") or match["text"]
        if key not in scored:
            scored[key] = {
                **match,
                "_vector_score": 0.0,
                "_graph_score": float(match.get("score", 0.0) or 0.0),
                "_support": 0,
            }
        else:
            scored[key]["_support"] += 1
            scored[key]["_graph_score"] = max(scored[key]["_graph_score"], float(match.get("score", 0.0) or 0.0))

    for entry in scored.values():
        entry["_support"] = int(entry.get("_support", 0))
        entry["_lexical_score"] = _lexical_overlap_score(
            question_text,
            question_topics or [],
            entry,
        )
        entry["_hybrid_score"] = (
            (entry.get("_vector_score", 0.0) * 0.45)
            + (entry.get("_graph_score", 0.0) * 0.30)
            + (0.20 if entry.get("_support", 0) > 0 else 0.0)
            + (0.10 if entry.get("_graph_score", 0.0) > 0.0 and entry.get("_vector_score", 0.0) > 0.0 else 0.0)
            + (entry.get("_lexical_score", 0.0) * 0.70)
            + (0.15 if entry.get("_lexical_score", 0.0) >= 0.75 else 0.0)
        )

    ranked = sorted(scored.values(), key=lambda item: item.get("_hybrid_score", 0.0), reverse=True)
    return ranked[:top_k]


def _retrieve(corrected_question: str, question_topics: list[str]) -> list[dict]:
    """
    Runs RAG (Pinecone vector search) + KAG (Neo4j topic expansion).
    The two legs run concurrently since they don't depend on each other -
    only on corrected_question/question_topics, both already available.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        vector_future = executor.submit(_vector_search, corrected_question)
        graph_future = executor.submit(_graph_search, question_topics)
        vector_matches = vector_future.result()
        graph_matches = graph_future.result()

    return rank_hybrid_matches(
        vector_matches,
        graph_matches,
        top_k=6,
        question_text=corrected_question,
        question_topics=question_topics,
    )

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

    RELEVANCE_THRESHOLD = 0.35
    sources = list({
        m["source_url"]
        for m in all_matches
        if m["source_url"] and m.get("_hybrid_score", 0.0) >= RELEVANCE_THRESHOLD
    })

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

    if is_no_answer:
        sources = []

    result = {"answer": answer, "sources": sources}

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