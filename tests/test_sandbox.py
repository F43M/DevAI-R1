import subprocess
import pytest
from devai import sandbox


def test_run_executes(monkeypatch):
    sb = sandbox.Sandbox("img")

    captured = {}

    class DummyResult:
        stdout = "ok"

    def fake_run(cmd, capture_output, text, timeout):
        captured["cmd"] = cmd
        captured["timeout"] = timeout
        return DummyResult()

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)
    out = sb.run(["echo", "hi"], timeout=5)
    assert out == "ok"
    assert captured["cmd"] == ["docker", "run", "--rm", "img", "echo", "hi"]
    assert captured["timeout"] == 5


def test_run_timeout(monkeypatch):
    sb = sandbox.Sandbox()

    def fake_run(cmd, capture_output, text, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)
    with pytest.raises(TimeoutError):
        sb.run(["sleep", "2"], timeout=1)
