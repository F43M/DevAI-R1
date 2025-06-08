import time
from pathlib import Path
import devai.shadow_mode as sm


def test_run_test_isolated_success(tmp_path, monkeypatch):
    monkeypatch.setattr(sm.config, "CODE_ROOT", str(tmp_path))
    monkeypatch.setattr(sm.config, "TESTS_USE_ISOLATION", False)
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    ok, out = sm.run_test_isolated(tmp_path)
    assert ok
    assert "1 passed" in out


def test_run_test_isolated_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(sm.config, "CODE_ROOT", str(tmp_path))
    monkeypatch.setattr(sm.config, "TESTS_USE_ISOLATION", False)
    (tmp_path / "test_sleep.py").write_text("import time\n\ndef test_sleep():\n    time.sleep(5)\n")
    ok, out = sm.run_test_isolated(tmp_path, timeout=1)
    assert not ok
    assert "Tempo excedido" in out

