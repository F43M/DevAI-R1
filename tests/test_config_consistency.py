from pathlib import Path
import logging

import types
import config_utils
from devai.config import Config


def _write_cfg(tmp_path: Path, text: str) -> str:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(text)
    return str(cfg_file)


def test_model_name_from_models(tmp_path, monkeypatch):
    path = _write_cfg(tmp_path, "MODELS:\n  default:\n    name: mymodel\n")
    fake_yaml = types.SimpleNamespace(safe_load=lambda f: {"MODELS": {"default": {"name": "mymodel"}}})
    monkeypatch.setattr(config_utils, "yaml", fake_yaml)
    cfg = Config(path)
    assert cfg.model_name == "mymodel"


def test_model_name_fallback(tmp_path, monkeypatch):
    path = _write_cfg(tmp_path, "MODEL_NAME: only\n")
    fake_yaml = types.SimpleNamespace(safe_load=lambda f: {"MODEL_NAME": "only"})
    monkeypatch.setattr(config_utils, "yaml", fake_yaml)
    cfg = Config(path)
    assert cfg.model_name == "only"


def test_warning_on_deprecated(caplog, tmp_path, monkeypatch):
    path = _write_cfg(
        tmp_path,
        "MODEL_NAME: old\nMODELS:\n  default:\n    name: new\n",
    )
    fake_yaml = types.SimpleNamespace(
        safe_load=lambda f: {"MODEL_NAME": "old", "MODELS": {"default": {"name": "new"}}}
    )
    monkeypatch.setattr(config_utils, "yaml", fake_yaml)
    with caplog.at_level(logging.WARNING):
        cfg = Config(path)
    assert any("obsoleto" in r.message for r in caplog.records)
    assert any("divergentes" in r.message for r in caplog.records)
    assert cfg.model_name == "new"

