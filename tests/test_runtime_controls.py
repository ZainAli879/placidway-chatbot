from datetime import datetime, timedelta, timezone
import threading
import time

from app.runtime.controls import (
    ControlledGenerator, GenerationGate, GenerationUnavailable,
    SlidingWindowLimiter, trim_messages,
)


def test_sliding_window_allows_ten_then_reports_retry_time():
    limiter = SlidingWindowLimiter(limit=10, window_seconds=60)
    for second in range(10):
        assert limiter.allow(float(second))[0]

    allowed, retry_after = limiter.allow(10.0)
    assert not allowed
    assert retry_after == 50.0
    assert limiter.allow(60.0)[0]


def test_generation_gate_limits_concurrency_without_incrementing_denied_requests():
    gate = GenerationGate(concurrency=2, daily_limit=10)
    now = datetime(2026, 10, 28, tzinfo=timezone.utc)
    first = gate.try_acquire(now)
    second = gate.try_acquire(now)

    with first as allowed_first, second as allowed_second:
        assert allowed_first and allowed_second
        with gate.try_acquire(now) as allowed_third:
            assert not allowed_third
        assert gate.calls_today == 2

    with gate.try_acquire(now) as allowed_after_release:
        assert allowed_after_release
    assert gate.calls_today == 3


def test_generation_gate_resets_daily_and_enforces_ceiling():
    gate = GenerationGate(concurrency=1, daily_limit=1)
    day = datetime(2026, 10, 28, tzinfo=timezone.utc)
    with gate.try_acquire(day) as allowed:
        assert allowed
    with gate.try_acquire(day + timedelta(hours=1)) as allowed:
        assert not allowed
    with gate.try_acquire(day + timedelta(days=1)) as allowed:
        assert allowed


def test_generation_gate_is_thread_safe():
    gate = GenerationGate(concurrency=2, daily_limit=20)
    barrier = threading.Barrier(5)
    release = threading.Event()
    allowed = []

    def attempt():
        barrier.wait()
        with gate.try_acquire(datetime(2026, 10, 28, tzinfo=timezone.utc)) as value:
            allowed.append(value)
            if value:
                release.wait(timeout=2)

    threads = [threading.Thread(target=attempt) for _ in range(4)]
    for thread in threads:
        thread.start()
    barrier.wait()
    # Release the two successful workers without waiting for denied workers.
    while sum(allowed) < 2:
        time.sleep(0.001)
    release.set()
    for thread in threads:
        thread.join()
    assert sum(allowed) == 2


def test_trim_messages_retains_newest_twenty_without_mutating_input():
    messages = [{"role": "user", "content": str(i)} for i in range(25)]
    trimmed = trim_messages(messages)
    assert len(trimmed) == 20
    assert trimmed[0]["content"] == "5"
    assert len(messages) == 25


def test_controlled_generator_does_not_call_delegate_when_daily_limit_is_exhausted():
    class Delegate:
        calls = 0
        def generate(self, question, evidence, package_key=None):
            self.calls += 1
            self.package_key = package_key
            return "answer"

    delegate = Delegate()
    gate = GenerationGate(concurrency=1, daily_limit=1)
    controlled = ControlledGenerator(delegate, gate, clock=lambda: datetime(2026, 10, 28, tzinfo=timezone.utc))
    assert controlled.generate("q", [], package_key="anal") == "answer"
    assert delegate.package_key == "anal"
    try:
        controlled.generate("q", [])
    except GenerationUnavailable:
        pass
    else:
        raise AssertionError("expected GenerationUnavailable")
    assert delegate.calls == 1
