import types
import pytest
import config_utils
import asyncio
from devai import approval
from devai.config import Config, config
from devai.approval import requires_approval
import devai.command_router as command_router
import devai.decision_log as decision_log


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

        def send(self, subj, body):
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
