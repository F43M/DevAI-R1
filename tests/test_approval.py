import types
import pytest
import config_utils
import asyncio
from devai import approval
from devai.config import Config, config
from devai.approval import requires_approval
import devai.command_router as command_router
import devai.decision_log as decision_log
from datetime import datetime, timedelta


def _write_cfg(tmp_path, text):
    p = tmp_path / "config.yaml"
    p.write_text(text)
    return str(p)


def test_config_approval(tmp_path, monkeypatch):
    path = _write_cfg(tmp_path, "APPROVAL_MODE: auto_edit\n")
    fake_yaml = types.SimpleNamespace(
        safe_load=lambda f: {"APPROVAL_MODE": "auto_edit"}
    )
    monkeypatch.setattr(config_utils, "yaml", fake_yaml)
    cfg = Config(path)
    assert cfg.APPROVAL_MODE == "auto_edit"


def test_config_invalid_approval(tmp_path, monkeypatch):
    path = _write_cfg(tmp_path, "APPROVAL_MODE: bad\n")
    fake_yaml = types.SimpleNamespace(safe_load=lambda f: {"APPROVAL_MODE": "bad"})
    monkeypatch.setattr(config_utils, "yaml", fake_yaml)
    with pytest.raises(ValueError):
        Config(path)


def test_requires_approval(monkeypatch):
    monkeypatch.setattr(config, "APPROVAL_MODE", "full_auto")
    assert not requires_approval("patch")
    assert not requires_approval("shell_safe")
    monkeypatch.setattr(config, "APPROVAL_MODE", "auto_edit")
    assert requires_approval("shell")
    assert not requires_approval("patch")
    assert not requires_approval("shell_safe")
    monkeypatch.setattr(config, "APPROVAL_MODE", "suggest")
    assert requires_approval("patch")
    assert requires_approval("shell")
    assert not requires_approval("shell_safe")
    assert not requires_approval("other")


def test_requires_approval_remember(monkeypatch, tmp_path):
    log_path = tmp_path / "decision_log.yaml"
    import json, types
    monkeypatch.setattr(decision_log, "yaml", types.SimpleNamespace(safe_load=json.loads))
    log_path.write_text(
        json.dumps([
            {
                "id": "001",
                "tipo": "edit",
                "modulo": "x.txt",
                "motivo": "editar",
                "modelo_ia": "cli",
                "hash_resultado": "x",
                "timestamp": "2024-01-01",
                "remember": True,
                "expires_at": "9999-01-01T00:00:00",
            }
        ])
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "APPROVAL_MODE", "suggest")
    assert not requires_approval("edit", "x.txt")
    assert requires_approval("edit", "y.txt")


def test_auto_approval_rules(monkeypatch):
    monkeypatch.setattr(
        config,
        "AUTO_APPROVAL_RULES",
        [{"action": "edit", "path": "docs/**", "approve": True}],
    )
    monkeypatch.setattr(config, "APPROVAL_MODE", "suggest")
    assert not requires_approval("edit", "docs/file.md")
    assert requires_approval("edit", "src/file.py")


def test_auto_approval_rules_force(monkeypatch):
    monkeypatch.setattr(
        config,
        "AUTO_APPROVAL_RULES",
        [{"action": "edit", "path": "docs/**", "approve": False}],
    )
    monkeypatch.setattr(config, "APPROVAL_MODE", "full_auto")
    assert requires_approval("edit", "docs/file.md")
    assert not requires_approval("edit", "src/file.py")


def test_auto_approval_shell_safe(monkeypatch):
    monkeypatch.setattr(
        config,
        "AUTO_APPROVAL_RULES",
        [{"action": "shell_safe", "path": "scripts/**", "approve": False}],
    )
    monkeypatch.setattr(config, "APPROVAL_MODE", "suggest")
    assert requires_approval("shell_safe", "scripts/run.sh")
    assert not requires_approval("shell_safe", "other/run.sh")


def test_shell_safe_constants():
    assert "shell_safe" in approval.SAFE_ACTIONS
    assert "shell" in approval.SHELL_ACTIONS
    assert "shell_safe" in approval.SHELL_ACTIONS


def test_temporary_auto_approval(monkeypatch):
    monkeypatch.setattr(config, "APPROVAL_MODE", "suggest")
    command_router.approval.auto_approve_remaining = 0
    asyncio.run(command_router.handle_aprovar_proxima(None, None, "2", plain=True, feedback_db=None))
    assert command_router.approval.auto_approve_remaining == 2
    assert not requires_approval("patch")
    assert not requires_approval("patch")
    assert command_router.approval.auto_approve_remaining == 0
    assert requires_approval("patch")


def test_request_approval_notifies(monkeypatch):
    sent = []

    class DummyNotifier:
        def __init__(self):
            self.enabled = True

        def send(self, subj, body, details=None):
            if details:
                body = f"{body}\n{details}"
            sent.append(body)

    monkeypatch.setattr("devai.approval.Notifier", DummyNotifier)
    monkeypatch.setattr(config, "API_PORT", 1234, raising=False)

    async def run():
        fut = asyncio.create_task(approval.request_approval("Permitir?"))
        req = await approval.wait_for_request()
        approval.resolve_request(True)
        result = await fut
        return req, result

    req, result = asyncio.run(run())
    assert result is True
    assert req["token"] in sent[0]


def test_temporary_auto_approval_time(monkeypatch):
    monkeypatch.setattr(config, "APPROVAL_MODE", "suggest")
    command_router.approval.auto_approve_until = None

    base = datetime(2024, 1, 1, 12, 0, 0)
    current = base

    class DummyDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return current

    monkeypatch.setattr(approval, "datetime", DummyDateTime)
    monkeypatch.setattr(command_router, "datetime", DummyDateTime)

    asyncio.run(
        command_router.handle_aprovar_durante(None, None, "10", plain=True, feedback_db=None)
    )
    assert approval.auto_approve_until == base + timedelta(seconds=10)
    assert not requires_approval("patch")

    current = base + timedelta(seconds=11)
    assert requires_approval("patch")


def test_patch_threshold_forces_confirmation(monkeypatch, tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("a\nb\nc\n")
    diff = "@@ -1,3 +1,5 @@\n a\n-b\n-c\n+d\n+e\n+f\n+g"
    monkeypatch.setattr(command_router.config, "APPROVAL_DIFF_THRESHOLD", 2)
    monkeypatch.setattr(command_router.config, "APPROVAL_MODE", "auto_edit")
    called = []

    def fake_req(action, path=None):
        called.append(action)
        return True

    monkeypatch.setattr(command_router, "requires_approval", fake_req)
    command_router._apply_patch_to_file(path, diff)
    assert called
