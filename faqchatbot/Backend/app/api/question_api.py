import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.chat_model import ChatLog
from app.models.feedback_model import Feedback
from app.services.cache_service import get_cached_answer, set_cached_answer, make_signature
from app.services.embedding_service import embed_text
from app.services.pinecone_service import query_similar_chunks, query_by_topics
from app.services.neo4j_service import extract_topics, get_related_topics
from app.services.llm_service import (
    generate_answer,
    correct_question_spelling,
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


def _retrieve(corrected_question: str) -> list[dict]:
    """
    Runs RAG (Pinecone vector search) + KAG (Neo4j topic expansion) and
    returns the deduped, combined match list. This is intentionally cheap —
    one embedding call + one Pinecone query + graph lookups, no LLM call —
    so it's safe to run on every request, even ones that end up being
    served from cache. Running this first is what lets us detect whether
    the underlying content has changed since an answer was cached.
    """
    query_embedding = embed_text(corrected_question)
    vector_matches = query_similar_chunks(query_embedding, top_k=5)

    question_topics = extract_topics(corrected_question)
    related_topics: list[str] = []
    for topic in question_topics:
        related_topics.extend(get_related_topics(topic, limit=3))
    related_topics = list(set(related_topics) - set(question_topics))

    graph_matches = query_by_topics(related_topics, top_k=3) if related_topics else []

    # Dedupe by chunk id (falls back to text) rather than raw text alone, so
    # two distinct chunks that happen to share identical text don't collide.
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
    corrected_question = correct_question_spelling(request.question)

    # --- Always retrieve first (RAG + KAG) ---
    # This is the freshness check: it's cheap (no LLM call), so we run it on
    # every request rather than blindly trusting a 24h cache entry. If new
    # content was ingested, or old chunks were replaced/deleted, the set of
    # matching chunk ids will differ from what was cached, and we know to
    # regenerate instead of serving a stale answer.
    all_matches = _retrieve(corrected_question)
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
        # The LLM call itself failed (network/API error) — this is NOT a real
        # answer, so it must never be written to the cache. Previously this
        # case produced a fake "answer" string that got cached for 24h,
        # which is exactly why the same question kept coming back with a
        # stale "I don't have that information" cached response.
        fallback = "I'm having trouble reaching the answer service right now. Please try again in a moment."
        chat_log_id = _log_chat(db, corrected_question, fallback, [])
        return QuestionResponse(answer=fallback, sources=[], from_cache=False, chat_log_id=chat_log_id)

    sources = list({m["source_url"] for m in all_matches if m["source_url"]})
    result = {"answer": answer, "sources": sources}

    # Only cache answers that were actually grounded in retrieved context.
    # If RAG/KAG retrieval came up empty (e.g. a transient Pinecone/Neo4j/
    # embedding failure, or content genuinely not ingested yet), the LLM's
    # honest "I don't have that information" would otherwise get locked into
    # Redis for CACHE_TTL_SECONDS and served to every future identical
    # question — which is the bug behind the "cached answer" screenshots.
    if all_matches and fresh_signature:
        set_cached_answer(corrected_question, result, fresh_signature)
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

    return {"status": "ok"}