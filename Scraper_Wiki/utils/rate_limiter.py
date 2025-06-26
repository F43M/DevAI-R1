from __future__ import annotations

import asyncio
import time
from typing import Any, Optional


class RateLimiter:
    """Leaky bucket rate limiter with exponential backoff."""

    def __init__(
        self,
        min_delay: float,
        max_delay: float | None = None,
        *,
        metrics: Optional[Any] = None,
    ) -> None:
        if max_delay is None:
            max_delay = min_delay
        self.base_min = min_delay
        self.base_max = max_delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.metrics = metrics
        self.consecutive_failures = 0
        self.next_time = time.monotonic() + self.min_delay

    def _record_wait(self, delay: float) -> None:
        if delay > 0 and self.metrics is not None:
            try:
                self.metrics.rate_limited_total.inc()
            except Exception:
                pass

    def wait(self) -> None:
        now = time.monotonic()
        if now < self.next_time:
            delay = self.next_time - now
            self._record_wait(delay)
            time.sleep(delay)
            now = time.monotonic()
        self.next_time = max(now, self.next_time) + self.min_delay

    async def async_wait(self) -> None:
        now = time.monotonic()
        if now < self.next_time:
            delay = self.next_time - now
            self._record_wait(delay)
            await asyncio.sleep(delay)
            now = time.monotonic()
        self.next_time = max(now, self.next_time) + self.min_delay

    def record_error(self) -> None:
        self.consecutive_failures += 1
        self.min_delay = min(self.max_delay, self.min_delay * 2)
        self.next_time = time.monotonic() + self.min_delay

    def record_success(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.consecutive_failures = 0
        self.min_delay = self.base_min
        self.max_delay = self.base_max
        self.next_time = time.monotonic() + self.min_delay
