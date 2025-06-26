"""Crawling utilities and rate limiting."""

from __future__ import annotations

from typing import Dict, Iterable, List

import requests


class RateLimiter:
    """Simple exponential backoff rate limiter."""

    def __init__(self, min_delay: float, max_delay: float | None = None):
        if max_delay is None:
            max_delay = min_delay
        self.base_min = min_delay
        self.base_max = max_delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.consecutive_failures = 0

    def _sample_delay(self) -> float:
        import random

        return random.uniform(self.min_delay, self.max_delay)

    def wait(self) -> None:
        import time

        time.sleep(self._sample_delay())

    def record_error(self) -> None:
        self.consecutive_failures += 1
        self.min_delay = min(self.max_delay, self.min_delay * 2)
        self.max_delay = min(self.max_delay, self.max_delay * 2)

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.min_delay = self.base_min
        self.max_delay = self.base_max


class DynamicRateLimiter:
    """Manage per-host rate limiters with exponential backoff."""

    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0) -> None:
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.limiters: Dict[str, RateLimiter] = {}

    def get(self, host: str) -> RateLimiter:
        if host not in self.limiters:
            self.limiters[host] = RateLimiter(self.base_delay, self.max_delay)
        return self.limiters[host]


rate_limiter = DynamicRateLimiter()


class BrightDataProxy:
    """Simple wrapper for Bright Data proxy credentials."""

    def __init__(self, user: str, password: str, host: str, port: int = 22225) -> None:
        self.user = user
        self.password = password
        self.host = host
        self.port = port

    def as_dict(self) -> Dict[str, str]:
        url = f"http://{self.user}:{self.password}@{self.host}:{self.port}"
        return {"http": url, "https": url}


class TwoCaptchaSolver:
    """Minimal 2Captcha API client."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def solve_recaptcha(self, site_key: str, url: str) -> str:
        """Return captcha solution for ``site_key`` on ``url``."""

        resp = requests.post(
            "http://2captcha.com/in.php",
            data={
                "key": self.api_key,
                "method": "userrecaptcha",
                "googlekey": site_key,
                "pageurl": url,
                "json": 1,
            },
            timeout=10,
        )
        resp.raise_for_status()
        captcha_id = resp.json().get("request")
        for _ in range(20):
            rate_limiter.wait()
            res = requests.get(
                "http://2captcha.com/res.php",
                params={
                    "key": self.api_key,
                    "action": "get",
                    "id": captcha_id,
                    "json": 1,
                },
                timeout=10,
            )
            res.raise_for_status()
            data = res.json()
            if data.get("status") == 1:
                return data["request"]
            if data.get("request") != "CAPCHA_NOT_READY":
                raise RuntimeError(data.get("request"))
        raise RuntimeError("Captcha solving timed out")


def fallback_request(
    url: str, proxies: Iterable[str] | None, user_agents: Iterable[str] | None
) -> str:
    """Try multiple proxy/UA combinations until the request succeeds."""

    proxy_cycle = list(proxies or [None])
    ua_cycle = list(user_agents or [None])
    attempts = max(len(proxy_cycle), 1) * max(len(ua_cycle), 1)
    idx = 0
    last_exc: Exception | None = None
    for _ in range(attempts):
        proxy = proxy_cycle[idx % len(proxy_cycle)]
        ua = ua_cycle[idx % len(ua_cycle)]
        idx += 1
        headers = {"User-Agent": ua} if ua else {}
        proxy_dict = {"http": proxy, "https": proxy} if proxy else None
        try:
            resp = requests.get(url, headers=headers, proxies=proxy_dict, timeout=10)
            if resp.status_code in {403, 429}:
                continue
            resp.raise_for_status()
            return resp.text
        except Exception as exc:  # pragma: no cover - network errors
            last_exc = exc
    if last_exc:
        raise last_exc
    raise RuntimeError("All attempts failed")


__all__ = [
    "rate_limiter",
    "DynamicRateLimiter",
    "BrightDataProxy",
    "TwoCaptchaSolver",
    "fallback_request",
]
