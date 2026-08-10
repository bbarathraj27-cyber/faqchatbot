import pytest

from api_tests.benchmark_retrieval import summarize_ranking_results
from app.api.question_api import rank_hybrid_matches


def test_summarize_ranking_results_aggregates_metrics():
    ranks = [1, 3, None]
    latencies_ms = [100.0, 200.0, 300.0]

    summary = summarize_ranking_results(ranks, latencies_ms, top_k=3)

    assert summary["tested"] == 3
    assert summary["hits"] == 2
    assert summary["hit_rate"] == pytest.approx(2 / 3)
    assert summary["mrr"] == pytest.approx((1.0 + 1.0 / 3.0) / 3.0)
    assert summary["avg_latency_ms"] == pytest.approx(200.0)
    assert summary["p95_latency_ms"] == pytest.approx(300.0)


def test_rank_hybrid_matches_prefers_overlap_across_sources():
    vector_matches = [
        {"id": "a", "text": "alpha", "score": 0.95},
        {"id": "b", "text": "beta", "score": 0.90},
    ]
    topic_matches = [
        {"id": "b", "text": "beta", "score": 0.91},
        {"id": "c", "text": "gamma", "score": 0.89},
    ]

    ranked = rank_hybrid_matches(vector_matches, topic_matches, top_k=3)

    assert [m["id"] for m in ranked[:2]] == ["b", "a"]
    assert ranked[0]["id"] == "b"


def test_rank_hybrid_matches_prioritizes_lexical_and_topic_signal():
    vector_matches = [
        {"id": "a", "text": "space mission overview", "score": 0.95},
    ]
    topic_matches = [
        {"id": "b", "text": "apples and bananas are common fruits", "score": 0.01, "topics": ["fruit"]},
    ]

    ranked = rank_hybrid_matches(
        vector_matches,
        topic_matches,
        top_k=3,
        question_text="apples bananas",
        question_topics=["fruit"],
    )

    assert ranked[0]["id"] == "b"
