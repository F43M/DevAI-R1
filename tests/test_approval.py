import types
import pytest
import config_utils
from devai.config import Config, config
from devai.approval import requires_approval
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
    monkeypatch.setattr(config, "APPROVAL_MODE", "auto_edit")
    assert requires_approval("shell")
    assert not requires_approval("patch")
    monkeypatch.setattr(config, "APPROVAL_MODE", "suggest")
    assert requires_approval("patch")
    assert requires_approval("shell")
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
            }
        ])
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "APPROVAL_MODE", "suggest")
    assert not requires_approval("edit", "x.txt")
    assert requires_approval("edit", "y.txt")
