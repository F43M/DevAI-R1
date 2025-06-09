import types
import pytest
import config_utils
from devai.config import Config, config
from devai.approval import requires_approval


def _write_cfg(tmp_path, text):
    p = tmp_path / "config.yaml"
    p.write_text(text)
    return str(p)


def test_config_approval(tmp_path, monkeypatch):
    path = _write_cfg(tmp_path, "APPROVAL_MODE: manual\n")
    fake_yaml = types.SimpleNamespace(safe_load=lambda f: {"APPROVAL_MODE": "manual"})
    monkeypatch.setattr(config_utils, "yaml", fake_yaml)
    cfg = Config(path)
    assert cfg.APPROVAL_MODE == "manual"


def test_config_invalid_approval(tmp_path, monkeypatch):
    path = _write_cfg(tmp_path, "APPROVAL_MODE: bad\n")
    fake_yaml = types.SimpleNamespace(safe_load=lambda f: {"APPROVAL_MODE": "bad"})
    monkeypatch.setattr(config_utils, "yaml", fake_yaml)
    with pytest.raises(ValueError):
        Config(path)


def test_requires_approval(monkeypatch):
    monkeypatch.setattr(config, "APPROVAL_MODE", "auto")
    assert not requires_approval("patch")
    monkeypatch.setattr(config, "APPROVAL_MODE", "manual")
    assert requires_approval("anything")
    monkeypatch.setattr(config, "APPROVAL_MODE", "suggest")
    assert requires_approval("patch")
    assert not requires_approval("other")
