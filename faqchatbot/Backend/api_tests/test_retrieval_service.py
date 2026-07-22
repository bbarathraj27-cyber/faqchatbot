import time

from app.services.retrieval_service import run_with_timeout


def test_run_with_timeout_returns_default_when_callable_exceeds_timeout():
    def slow_callable():
        time.sleep(0.2)
        return "done"

    result = run_with_timeout(slow_callable, timeout_seconds=0.05, default="fallback")

    assert result == "fallback"


def test_run_with_timeout_returns_callable_result_when_fast():
    def fast_callable():
        return "done"

    result = run_with_timeout(fast_callable, timeout_seconds=0.2, default="fallback")

    assert result == "done"
