"""Lightweight in-process controls suitable for a free single-instance app."""
from __future__ import annotations

from collections import deque
from datetime import datetime
import threading
from typing import Callable, Sequence


class SlidingWindowLimiter:
    def __init__(self, *, limit: int, window_seconds: int):
        if limit < 1 or window_seconds < 1:
            raise ValueError("limit and window_seconds must be positive")
        self.limit = limit
        self.window_seconds = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def allow(self, now: float) -> tuple[bool, float]:
        with self._lock:
            cutoff = now - self.window_seconds
            while self._timestamps and self._timestamps[0] <= cutoff:
                self._timestamps.popleft()
            if len(self._timestamps) >= self.limit:
                return False, max(0.0, self._timestamps[0] + self.window_seconds - now)
            self._timestamps.append(now)
            return True, 0.0


class GenerationUnavailable(RuntimeError):
    """The configured daily generation allowance has been exhausted."""


class DailyLimitedGenerator:
    """Enforce the configured daily cost guard without limiting concurrency."""

    def __init__(
        self,
        delegate,
        daily_limit: int,
        clock: Callable[[], datetime] | None = None,
    ):
        if daily_limit < 1:
            raise ValueError("daily_limit must be positive")
        self.delegate = delegate
        self._daily_limit = daily_limit
        self._clock = clock or (lambda: datetime.now().astimezone())
        self._day = None
        self._calls_today = 0
        self._counter_lock = threading.Lock()

    def generate(self, question, evidence, package_key=None):
        day = self._clock().astimezone().date()
        with self._counter_lock:
            if day != self._day:
                self._day = day
                self._calls_today = 0
            if self._calls_today >= self._daily_limit:
                raise GenerationUnavailable("Daily generation allowance reached")
            self._calls_today += 1
        return self.delegate.generate(question, evidence, package_key=package_key)


def trim_messages(messages: Sequence[dict], limit: int = 20) -> list[dict]:
    if limit < 1:
        raise ValueError("limit must be positive")
    return [dict(message) for message in messages[-limit:]]
