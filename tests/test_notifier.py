import asyncio
from devai.notifier import Notifier
from devai.config import config

class DummySMTP:
    def __init__(self, host):
        self.host = host
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        pass
    def send_message(self, msg):
        DummySMTP.sent.append(msg)

DummySMTP.sent = []

class DummySession:
    def __init__(self, *a, **k):
        pass
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc, tb):
        pass
    async def post(self, url, json):
        DummySession.posts.append((url, json))
        class Resp:
            async def json(self):
                return {}
            async def text(self):
                return ""
        return Resp()

DummySession.posts = []

def test_send_email_and_slack(monkeypatch):
    DummySMTP.sent = []
    DummySession.posts = []
    monkeypatch.setattr(config, "NOTIFY_EMAIL", "user@example.com", raising=False)
    monkeypatch.setattr(config, "NOTIFY_SLACK", "http://hook", raising=False)
    monkeypatch.setattr("smtplib.SMTP", lambda host: DummySMTP(host))
    monkeypatch.setattr("aiohttp.ClientSession", DummySession)

    notifier = Notifier()
    notifier.send("Alerta", "Teste")

    assert DummySMTP.sent
    assert DummySMTP.sent[0]["Subject"] == "Alerta"
    assert DummySession.posts == [("http://hook", {"text": "*Alerta*\nTeste"})]
