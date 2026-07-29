"""
Benchmark suite for the FAQ chatbot.

Covers:
  1. Answer quality  - keyword-based pass/fail against known-good questions,
                        plus an out-of-scope question that SHOULD say "I don't know".
  2. Latency          - cold (first call, LLM-generated) vs cached (second call).
  3. Load test        - N concurrent requests, throughput + error rate.

Run from the Backend/ folder with the server already running (./start.sh),
via:  python3 benchmark.py
"""

import asyncio
import time
import statistics

import httpx

BASE_URL = "http://localhost:8000"
ASK_URL = f"{BASE_URL}/question/ask"
TOPICS_URL = f"{BASE_URL}/sources/topics"

# Questions with an expected keyword that MUST appear in a correct answer.
# Extend this list with more of your real, DB-backed topics as you verify them.
QUALITY_CASES = [
    {"question": "How do I install TensorFlow with pip?", "expect_keyword": "pip"},
    {"question": "What is tf.data used for?", "expect_keyword": "data"},
    {"question": "How do I enable GPU support?", "expect_keyword": "GPU"},
]

# This one has no real basis in tensorflow.org docs - a correct system
# should say it doesn't know, NOT hallucinate an answer.
OUT_OF_SCOPE_CASE = {
    "question": "What is the capital of France?",
    "must_not_contain": None,  # we just check it doesn't confidently answer
}

CONCURRENCY_LEVELS = [1, 5, 10]


async def ask(client: httpx.AsyncClient, question: str) -> dict:
    start = time.perf_counter()
    try:
        resp = await client.post(ASK_URL, json={"question": question}, timeout=60)
        elapsed = time.perf_counter() - start
        if resp.status_code != 200:
            return {"ok": False, "elapsed": elapsed, "error": f"HTTP {resp.status_code}"}
        data = resp.json()
        return {
            "ok": True,
            "elapsed": elapsed,
            "answer": data.get("answer", ""),
            "from_cache": data.get("from_cache", False),
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {"ok": False, "elapsed": elapsed, "error": str(e)}


async def run_quality_and_latency():
    print("\n=== 1. ANSWER QUALITY + LATENCY (cold vs cached) ===\n")
    async with httpx.AsyncClient() as client:
        for case in QUALITY_CASES:
            q = case["question"]
            keyword = case["expect_keyword"]

            cold = await ask(client, q)
            cached = await ask(client, q)  # same question -> should hit cache

            passed = cold["ok"] and keyword.lower() in cold.get("answer", "").lower()
            status = "PASS" if passed else "FAIL"

            print(f"[{status}] \"{q}\"")
            print(f"    cold:   {cold['elapsed']:.2f}s  from_cache={cold.get('from_cache')}")
            print(f"    2nd:    {cached['elapsed']:.2f}s  from_cache={cached.get('from_cache')}")
            if not passed:
                print(f"    answer: {cold.get('answer', cold.get('error'))[:200]}")
            print()

        # Out-of-scope check
        oos = await ask(client, OUT_OF_SCOPE_CASE["question"])
        answer_lower = oos.get("answer", "").lower()
        looks_like_refusal = any(
            phrase in answer_lower
            for phrase in ["don't know", "not available", "no relevant", "cannot find"]
        )
        status = "PASS" if looks_like_refusal else "REVIEW"
        print(f"[{status}] Out-of-scope question: \"{OUT_OF_SCOPE_CASE['question']}\"")
        print(f"    answer: {oos.get('answer', oos.get('error'))[:200]}\n")


async def run_load_test(concurrency: int, question: str):
    async with httpx.AsyncClient() as client:
        start = time.perf_counter()
        results = await asyncio.gather(*[ask(client, question) for _ in range(concurrency)])
        total_time = time.perf_counter() - start

    successes = [r for r in results if r["ok"]]
    failures = [r for r in results if not r["ok"]]
    latencies = [r["elapsed"] for r in successes]

    print(f"--- Concurrency: {concurrency} ---")
    print(f"  total wall time:   {total_time:.2f}s")
    print(f"  successes/fails:   {len(successes)}/{len(failures)}")
    if latencies:
        print(f"  avg latency:       {statistics.mean(latencies):.2f}s")
        print(f"  p95 latency:       {sorted(latencies)[int(len(latencies) * 0.95) - 1]:.2f}s")
        print(f"  throughput:        {len(successes) / total_time:.2f} req/s")
    if failures:
        print(f"  sample error:      {failures[0].get('error')}")
    print()


async def main():
    await run_quality_and_latency()

    print("=== 2. LOAD TEST ===\n")
    test_question = QUALITY_CASES[0]["question"]
    for level in CONCURRENCY_LEVELS:
        await run_load_test(level, test_question)

    print("=== 3. RETRIEVAL MODE NOTE ===")
    print("KAG (Neo4j) is currently disabled per your last session, so this")
    print("run only reflects RAG+CAG performance. Re-run once Neo4j credentials")
    print("are restored to get a fair RAG vs RAG+CAG vs RAG+CAG+KAG comparison.\n")


if __name__ == "__main__":
    asyncio.run(main())