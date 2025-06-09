import subprocess
import pytest
import subprocess
import pytest
from devai import sandbox


def test_run_executes(monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which", lambda x: "/usr/bin/docker")
    sb = sandbox.Sandbox("img", cpus="2", memory="128m")

    captured = {}

    class DummyProc:
        def communicate(self, timeout=None):
            captured["timeout"] = timeout
            return ("ok", "")

    def fake_popen(cmd, stdout, stderr, text):
        captured["cmd"] = cmd
        return DummyProc()

    monkeypatch.setattr(sandbox.subprocess, "Popen", fake_popen)
    out = sb.run(["echo", "hi"], timeout=5)
    assert out == "ok"
    assert captured["cmd"] == [
        "docker",
        "run",
        "--rm",
        "--cpus",
        "2",
        "--memory",
        "128m",
        "img",
        "echo",
        "hi",
    ]
    assert captured["timeout"] == 5


def test_run_timeout(monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which", lambda x: "/usr/bin/docker")
    sb = sandbox.Sandbox()

    class DummyProc:
        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)
        def kill(self):
            captured.append("killed")

    captured = []

    def fake_popen(cmd, stdout, stderr, text):
        return DummyProc()

    monkeypatch.setattr(sandbox.subprocess, "Popen", fake_popen)
    with pytest.raises(TimeoutError):
        sb.run(["sleep", "2"], timeout=1)
    assert "killed" in captured


def test_shutdown_terminates_processes(monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which", lambda x: "/usr/bin/docker")
    sb = sandbox.Sandbox()

    class DummyProc:
        def kill(self):
            captured.append(True)

    captured = []
    sb._processes.append(DummyProc())
    sb.shutdown()
    assert captured == [True]
    assert sb._processes == []


def test_docker_missing(monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which", lambda x: None)
    sb = sandbox.Sandbox()
    assert not sb.enabled
