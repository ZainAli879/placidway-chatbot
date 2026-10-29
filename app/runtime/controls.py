"""Lightweight in-process controls suitable for a free single-instance app."""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from datetime import datetime
import threading
from typing import Callable, Iterator, Sequence


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


class GenerationGate:
    def __init__(self, *, concurrency: int, daily_limit: int):
        if concurrency < 1 or daily_limit < 1:
            raise ValueError("concurrency and daily_limit must be positive")
        self._semaphore = threading.BoundedSemaphore(concurrency)
        self._daily_limit = daily_limit
        self._day = None
        self._calls_today = 0
        self._lock = threading.Lock()

    @property
    def calls_today(self) -> int:
        with self._lock:
            return self._calls_today

    @contextmanager
    def try_acquire(self, now: datetime) -> Iterator[bool]:
        acquired = self._semaphore.acquire(blocking=False)
        if not acquired:
            yield False
            return
        allowed = False
        try:
            day = now.astimezone().date()
            with self._lock:
                if day != self._day:
                    self._day = day
                    self._calls_today = 0
                if self._calls_today < self._daily_limit:
                    self._calls_today += 1
                    allowed = True
            yield allowed
        finally:
            self._semaphore.release()


class GenerationUnavailable(RuntimeError):
    """Generation was denied by an in-process concurrency or daily limit."""


class ControlledGenerator:
    def __init__(
        self,
        delegate,
        gate: GenerationGate,
        *,
        clock: Callable[[], datetime] | None = None,
    ):
        self.delegate = delegate
        self.gate = gate
        self.clock = clock or (lambda: datetime.now().astimezone())

    def generate(self, question, evidence, package_key=None):
        with self.gate.try_acquire(self.clock()) as allowed:
            if not allowed:
                raise GenerationUnavailable("Generation capacity is currently unavailable")
            return self.delegate.generate(question, evidence, package_key=package_key)


def trim_messages(messages: Sequence[dict], limit: int = 20) -> list[dict]:
    if limit < 1:
        raise ValueError("limit must be positive")
    return [dict(message) for message in messages[-limit:]]
