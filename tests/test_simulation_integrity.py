from pathlib import Path
import pytest

import devai.shadow_mode as shadow_mode
from devai.shadow_mode import simulate_update, log_simulation
from devai.config import config


def test_decline_change(tmp_path, monkeypatch):
    code_root = tmp_path / "app"
    code_root.mkdir()
    monkeypatch.setattr(config, "CODE_ROOT", str(code_root))
    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path / "logs"))

    f = code_root / "m.py"
    f.write_text("x = 1\n")
    testf = code_root / "test_m.py"
    testf.write_text("import m\n\ndef test_x():\n    assert m.x == 1\n")

    monkeypatch.setattr(shadow_mode, "run_tests_in_temp", lambda d: (False, ""))
    diff, tests_ok, _, sim_id, patch_file = simulate_update(
        str(f), "x = 2\n", cleanup_cb=lambda d: None
    )
    assert not tests_ok
    evaluation = {"analysis": "negado"}
    log_simulation(sim_id, str(f), tests_ok, evaluation["analysis"], "shadow_failed")
    assert f.read_text() == "x = 1\n"
    log = Path(config.LOG_DIR) / "simulation_history.md"
    assert log.exists()
    text = log.read_text()
    assert "shadow_failed" in text


def test_accept_change(tmp_path, monkeypatch):
    code_root = tmp_path / "app"
    code_root.mkdir()
    monkeypatch.setattr(config, "CODE_ROOT", str(code_root))
    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path / "logs2"))
    f = code_root / "n.py"
    f.write_text("print('old')\n")

    monkeypatch.setattr(shadow_mode, "run_tests_in_temp", lambda d: (True, ""))
    diff, tests_ok, _, sim_id, patch_file = simulate_update(
        str(f), "print('new')\n", cleanup_cb=lambda d: None
    )
    evaluation = {"analysis": "ok"}
    if tests_ok:
        f.write_text("print('new')\n")
        action = "shadow_approved"
    else:
        action = "shadow_failed"
    log_simulation(sim_id, str(f), tests_ok, evaluation["analysis"], action)
    assert f.read_text() == "print('new')\n"
    log = Path(config.LOG_DIR) / "simulation_history.md"
    assert log.exists()
    text = log.read_text()
    assert action in text

