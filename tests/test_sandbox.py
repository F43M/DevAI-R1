import subprocess
import asyncio
import sys
import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="Sandbox não suportado no Windows"
)
from devai import sandbox
from devai import tasks as tasks_module
from devai import test_runner


def test_run_executes(monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which", lambda x: "/usr/bin/docker")
    monkeypatch.setattr(sandbox.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sandbox.os, "getcwd", lambda: "/tmp/project")
    monkeypatch.setattr(sandbox.config, "SANDBOX_NETWORK", "none")
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
    out = sb.run_command(["echo", "hi"], timeout=5)
    assert out == "ok"
    assert captured["cmd"] == [
        "docker",
        "run",
        "--rm",
        "-v",
        "/tmp/project:/app",
        "--workdir",
        "/app",
        "--network",
        "none",
        "--cpus",
        "2",
        "--memory",
        "128m",
        "img",
        "echo",
        "hi",
    ]
    assert captured["timeout"] == 5


def test_network_option(monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which", lambda x: "/usr/bin/docker")
    monkeypatch.setattr(sandbox.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sandbox.os, "getcwd", lambda: "/tmp/project")
    monkeypatch.setattr(sandbox.config, "SANDBOX_NETWORK", "host")
    sb = sandbox.Sandbox()

    captured = {}

    class DummyProc:
        def communicate(self, timeout=None):
            return ("ok", "")

    def fake_popen(cmd, stdout, stderr, text):
        captured["cmd"] = cmd
        return DummyProc()

    monkeypatch.setattr(sandbox.subprocess, "Popen", fake_popen)
    sb.run_command(["echo"], timeout=3)
    assert "--network" in captured["cmd"]
    idx = captured["cmd"].index("--network") + 1
    assert captured["cmd"][idx] == "host"


def test_allowed_hosts_creates_network(monkeypatch):
    def fake_which(cmd):
        if cmd == "docker":
            return "/usr/bin/docker"
        if cmd == "iptables":
            return "/usr/sbin/iptables"
        return None

    monkeypatch.setattr(sandbox.shutil, "which", fake_which)
    monkeypatch.setattr(sandbox.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sandbox.os, "getcwd", lambda: "/tmp/project")
    monkeypatch.setattr(sandbox.config, "SANDBOX_NETWORK", "bridge")
    monkeypatch.setattr(sandbox.config, "SANDBOX_ALLOWED_HOSTS", ["example.com"])
    created = []
    removed = []
    ipt_cmds = []

    class DummyProc:
        def communicate(self, timeout=None):
            return ("ok", "")

    def fake_run(cmd, check=False):
        if cmd[:3] == ["docker", "network", "create"]:
            created.append(cmd[3])
        elif cmd[:3] == ["docker", "network", "rm"]:
            removed.append(cmd[3])
        elif "iptables" in cmd[0]:
            ipt_cmds.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    def fake_popen(cmd, stdout, stderr, text):
        return DummyProc()

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)
    monkeypatch.setattr(sandbox.subprocess, "Popen", fake_popen)
    sb = sandbox.Sandbox()
    sb.run_command(["echo"], timeout=3)
    assert created and created[0] in removed
    assert ipt_cmds


def test_run_timeout(monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which", lambda x: "/usr/bin/docker")
    monkeypatch.setattr(sandbox.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sandbox.os, "getcwd", lambda: "/tmp/project")
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
        sb.run_command(["sleep", "2"], timeout=1)
    assert "killed" in captured


def test_shutdown_terminates_processes(monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which", lambda x: "/usr/bin/docker")
    monkeypatch.setattr(sandbox.platform, "system", lambda: "Linux")
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


def test_docker_missing_windows_warning(monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which", lambda x: None)
    monkeypatch.setattr(sandbox.platform, "system", lambda: "Windows")
    messages = []

    def fake_warn(msg, **kw):
        messages.append(msg)

    monkeypatch.setattr(sandbox.logger, "warning", fake_warn)
    sb = sandbox.Sandbox()
    assert not sb.enabled
    assert any("Docker Desktop" in m for m in messages)


def test_run_in_sandbox_delegates(monkeypatch):
    called = {}

    def fake_run(self, cmd, timeout=30):
        called["cmd"] = cmd
        called["timeout"] = timeout
        return "ok"

    monkeypatch.setattr(sandbox.Sandbox, "run", fake_run)
    out = sandbox.run_in_sandbox(["echo", "hi"], timeout=5)
    assert out == "ok"
    assert called["cmd"] == ["echo", "hi"]
    assert called["timeout"] == 5


def test_static_analysis_uses_sandbox(monkeypatch):
    class DummyAnalyzer:
        def __init__(self):
            self.code_root = "."
            self.code_chunks = {}
            self.code_graph = tasks_module.nx.DiGraph()
            self.learned_rules = {}

    analyzer = DummyAnalyzer()
    mem = type("M", (), {"save": lambda self, *a, **k: None})()
    tm = tasks_module.TaskManager("missing.yaml", analyzer, mem)

    captured = {}

    def fake_run(cmd, timeout=30):
        captured["cmd"] = cmd
        return "output"

    monkeypatch.setattr(tasks_module, "run_in_sandbox", fake_run)

    async def run():
        return await tm._perform_static_analysis_task(tm.tasks["static_analysis"])

    res = asyncio.run(run())
    assert res == ["output"]
    assert captured["cmd"] == ["flake8", str(analyzer.code_root)]


def test_run_pytest_isolation(monkeypatch, tmp_path):
    monkeypatch.setattr(test_runner.config, "TESTS_USE_ISOLATION", True)
    monkeypatch.setattr(test_runner.config, "LOG_DIR", str(tmp_path))

    captured = {}

    def fake_run(cmd, timeout=30):
        captured["cmd"] = cmd
        return "1 passed"

    monkeypatch.setattr(test_runner, "run_in_sandbox", fake_run)
    ok, out = test_runner.run_pytest(tmp_path)
    assert ok
    assert "1 passed" in out
    assert captured["cmd"] == ["pytest", "-q"]
