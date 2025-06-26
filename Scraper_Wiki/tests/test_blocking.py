import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import crawling
import requests


def test_fallback_request_switches_proxy_and_user_agent(monkeypatch):
    calls = []

    class Resp:
        def __init__(self, status):
            self.status_code = status
            self.text = "ok"

        def raise_for_status(self):
            if self.status_code != 200:
                raise requests.HTTPError()

    def fake_get(url, headers=None, proxies=None, timeout=10):
        calls.append((headers.get("User-Agent"), proxies.get("http") if proxies else None))
        if len(calls) == 1:
            return Resp(403)
        return Resp(200)

    monkeypatch.setattr(requests, "get", fake_get)

    proxies = ["http://p1", "http://p2"]
    uas = ["ua1", "ua2"]

    text = crawling.fallback_request("http://x", proxies, uas)

    assert text == "ok"
    assert calls == [("ua1", "http://p1"), ("ua2", "http://p2")]
