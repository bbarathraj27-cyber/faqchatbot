"""
RAG + KAG + CAG retrieval benchmark for faqchatbot.

Drop this into Backend/api_tests/ (or Backend/ root) and run it with the
venv activated, from the Backend/ directory:

    source .venv/bin/activate
    python api_tests/benchmark_retrieval.py --sample-size 40 --top-k 5

WHAT IT MEASURES
----------------
1. RAG+KAG retrieval accuracy (the live path used by question_api._retrieve):
   - Hit Rate@K   : % of queries where the source chunk appears in top-K
   - MRR          : Mean Reciprocal Rank of the source chunk (0 if missed)
   - Precision@K  : (only meaningful with human-labeled data; here we report
                     the synthetic-only version, see NOTE below)
   - Latency      : ms per retrieval call

2. CAG (Redis cache) correctness + speedup:
   - Confirms a repeated question hits cache and signature check passes
   - Reports latency of cold call vs cached call

METHODOLOGY (synthetic self-retrieval test)
--------------------------------------------
We don't have hand-labeled questions, so we generate a synthetic "ground
truth" the fast way: for a random sample of chunks already stored in
Pinecone, ask an LLM to write the question that chunk would answer. Then we
run that question back through your real retrieval path and check whether
the original chunk comes back in the top-K results.

NOTE: This measures "can retrieval find the chunk a question was literally
derived from" — a lower bound / sanity check, not a full accuracy audit.
For a more rigorous number, add a small hand-labeled set (see
HAND_LABELED_EXAMPLES below) alongside the synthetic set.
"""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

# Make `app` importable when running from Backend/ or Backend/api_tests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.orm import Session  # noqa: E402

from app.database import SessionLocal  # noqa: E402
from app.models.source_model import Source  # noqa: E402
from app.services.pinecone_service import get_chunks_by_source  # noqa: E402
from app.services.groq_key_pool import chat_completion  # noqa: E402
from app.services.cache_service import get_cached_answer, set_cached_answer, make_signature  # noqa: E402
from app.config import settings  # noqa: E402

# The same retrieval function question_api.py uses live (RAG + KAG combined).
from app.api.question_api import _retrieve  # noqa: E402
from app.services.llm_service import correct_and_extract_topics  # noqa: E402


# ---------------------------------------------------------------------------
# Optional: add real user-style questions with known correct source_url here.
# This is the more trustworthy signal — fill in a few as you gather them.
# ---------------------------------------------------------------------------
HAND_LABELED_EXAMPLES: list[dict] = [
    {"question": "How do I install TensorFlow with pip?", "expected_source_url": "https://www.tensorflow.org/install"},
    {"question": "What is tf.data used for?", "expected_source_url": "https://www.tensorflow.org/guide/data"},
    {"question": "How do I enable GPU support?", "expected_source_url": "https://www.tensorflow.org/install/gpu"},
]


def generate_question_for_chunk(chunk_text: str) -> str | None:
    """Ask the LLM to write a natural question this chunk would answer."""
    messages = [
        {
            "role": "system",
            "content": (
                "You write a single, natural user question that the given "
                "passage directly and fully answers. Respond with ONLY the "
                "question text, nothing else."
            ),
        },
        {"role": "user", "content": chunk_text[:1500]},
    ]
    try:
        question = chat_completion(messages, model=settings.LLM_MODEL, temperature=0.3)
        question = question.strip().strip('"')
        return question if question else None
    except Exception as e:
        print(f"  [warn] question generation failed: {e}")
        return None


def sample_chunks(db: Session, sample_size: int) -> list[dict]:
    """Pull real ingested chunks (with real chunk IDs) to use as ground truth."""
    sources = db.query(Source).filter(Source.status == "success").all()
    if not sources:
        print("No successfully ingested sources found in the database.")
        return []

    random.shuffle(sources)
    chunks: list[dict] = []
    for source in sources:
        if len(chunks) >= sample_size:
            break
        source_chunks = get_chunks_by_source(source.url)
        if not source_chunks:
            continue
        # Attach source_url onto each chunk dict; get_chunks_by_source doesn't include it
        for c in source_chunks:
            c["source_url"] = source.url
        random.shuffle(source_chunks)
        take = min(len(source_chunks), max(1, sample_size // 10))
        chunks.extend(source_chunks[:take])

    random.shuffle(chunks)
    return chunks[:sample_size]


def summarize_ranking_results(ranks: list[int | None], latencies_ms: list[float], top_k: int) -> dict:
    hits = [r for r in ranks if r is not None]
    tested = len(ranks)
    hit_rate = (len(hits) / tested) if tested else 0.0
    mrr = (sum(1 / r for r in hits) / tested) if tested else 0.0
    avg_latency = statistics.mean(latencies_ms) if latencies_ms else 0.0
    p95_latency = (
        statistics.quantiles(latencies_ms, n=20)[18] if len(latencies_ms) >= 20 else max(latencies_ms, default=0.0)
    )
    return {
        "tested": tested,
        "hits": len(hits),
        "hit_rate": hit_rate,
        "mrr": mrr,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "top_k": top_k,
    }


def run_rag_kag_benchmark(sample_size: int, top_k: int) -> dict:
    print(f"\n{'=' * 60}\nRAG + KAG RETRIEVAL BENCHMARK (synthetic, sample={sample_size}, top_k={top_k})\n{'=' * 60}")
    db = SessionLocal()
    try:
        chunks = sample_chunks(db, sample_size)
    finally:
        db.close()

    if not chunks:
        print("No chunks available to benchmark. Ingest some sources first.")
        return

    ranks: list[int | None] = []  # 1-indexed rank of correct chunk, None if missed
    latencies_ms: list[float] = []
    tested = 0

    for i, chunk in enumerate(chunks, 1):
        chunk_text = chunk.get("text", "")
        chunk_id = chunk.get("id")
        if not chunk_text or not chunk_id:
            continue

        question = generate_question_for_chunk(chunk_text)
        if not question:
            continue

        tested += 1
        print(f"[{i}/{len(chunks)}] Q: {question[:80]}")

        start = time.perf_counter()
        try:
            corrected_question, question_topics = correct_and_extract_topics(question)
            matches = _retrieve(corrected_question, question_topics)
        except Exception as e:
            print(f"  [error] retrieval failed: {e}")
            matches = []
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

        returned_ids = [m.get("id") for m in matches[:top_k]]
        if chunk_id in returned_ids:
            rank = returned_ids.index(chunk_id) + 1
            ranks.append(rank)
            print(f"  HIT at rank {rank}  ({elapsed_ms:.0f}ms)")
        else:
            ranks.append(None)
            print(f"  MISS  ({elapsed_ms:.0f}ms, {len(matches)} results returned)")

    if tested == 0:
        print("No valid test cases were generated.")
        return

    summary = summarize_ranking_results(ranks, latencies_ms, top_k)
    hit_rate_pct = summary["hit_rate"] * 100

    print(f"\n{'-' * 60}")
    print(f"Tested queries        : {summary['tested']}")
    print(f"Hit Rate@{top_k}           : {hit_rate_pct:.1f}%  ({summary['hits']}/{summary['tested']})")
    print(f"MRR                    : {summary['mrr']:.3f}")
    print(f"Avg retrieval latency  : {summary['avg_latency_ms']:.0f} ms")
    print(f"P95 retrieval latency  : {summary['p95_latency_ms']:.0f} ms")
    print(f"{'-' * 60}")

    if hand_labeled_results := run_hand_labeled(top_k):
        print("\nHand-labeled set results:")
        print(hand_labeled_results)

    return summary


def run_hand_labeled(top_k: int) -> str | None:
    if not HAND_LABELED_EXAMPLES:
        return None
    hits = 0
    for ex in HAND_LABELED_EXAMPLES:
        corrected_question, question_topics = correct_and_extract_topics(ex["question"])
        matches = _retrieve(corrected_question, question_topics)
        urls = [m.get("source_url") for m in matches[:top_k]]
        if ex["expected_source_url"] in urls:
            hits += 1
    total = len(HAND_LABELED_EXAMPLES)
    return f"{hits}/{total} correct ({hits / total * 100:.1f}%)"


def run_cag_benchmark() -> dict:
    print(f"\n{'=' * 60}\nCAG (Redis cache) CORRECTNESS + SPEED TEST\n{'=' * 60}")

    test_question = "What is TensorFlow used for?"
    fake_chunk_ids = ["benchmark-test-chunk-1", "benchmark-test-chunk-2"]
    signature = make_signature(fake_chunk_ids)

    # Write
    start = time.perf_counter()
    set_cached_answer(test_question, {"answer": "Test answer.", "sources": ["https://example.com"]}, signature)
    write_ms = (time.perf_counter() - start) * 1000

    # Cold miss simulation (different question)
    start = time.perf_counter()
    miss = get_cached_answer("a question that was never asked before, benchmark probe")
    miss_ms = (time.perf_counter() - start) * 1000

    # Read (hit)
    start = time.perf_counter()
    cached = get_cached_answer(test_question)
    hit_ms = (time.perf_counter() - start) * 1000

    print(f"Cache write latency     : {write_ms:.1f} ms")
    print(f"Cache miss latency      : {miss_ms:.1f} ms  (expect: fast, returns None -> {miss is None})")
    print(f"Cache hit latency       : {hit_ms:.1f} ms  (expect: fast, returns data -> {cached is not None})")
    if cached:
        signature_ok = cached.get("signature") == signature
        print(f"Signature integrity     : {'OK' if signature_ok else 'MISMATCH (bug!)'}")
    else:
        print("Cache hit FAILED — Redis may be disconnected. Check REDIS_HOST/PASSWORD in .env.")

    return {
        "write_latency_ms": write_ms,
        "miss_latency_ms": miss_ms,
        "hit_latency_ms": hit_ms,
        "cache_hit": cached is not None,
        "signature_ok": bool(cached and cached.get("signature") == signature),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark RAG+KAG retrieval accuracy and CAG cache behavior.")
    parser.add_argument("--sample-size", type=int, default=30, help="Number of synthetic test questions to generate")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K to evaluate hit rate against")
    parser.add_argument("--skip-cag", action="store_true", help="Skip the Redis cache test")
    args = parser.parse_args()

    rag_summary = run_rag_kag_benchmark(args.sample_size, args.top_k)
    if not args.skip_cag:
        cag_summary = run_cag_benchmark()
    else:
        cag_summary = None

    report = {
        "workflow": "RAG+KAG+CAG",
        "rag_kag": rag_summary,
        "cag": cag_summary,
    }
    print(f"\n{'=' * 60}\nFINAL BENCHMARK REPORT\n{'=' * 60}")
    print(json.dumps(report, indent=2))