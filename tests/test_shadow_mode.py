from pathlib import Path
import os
import pytest
from devai.shadow_mode import simulate_update
from devai.config import config


def test_simulate_update_no_apply(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CODE_ROOT", str(tmp_path))
    file = Path(config.CODE_ROOT) / "a.py"
    file.write_text("print('old')\n")
    diff, ok, out, sim_id, patch_file = simulate_update(
        str(file), "print('new')\n", cleanup_cb=lambda d: None
    )
    assert "-print('old')" in diff
    assert "+print('new')" in diff
    assert file.read_text() == "print('old')\n"
    assert Path(patch_file).exists()


def test_simulate_update_prevents_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CODE_ROOT", str(tmp_path))
    file = Path(config.CODE_ROOT) / "b.py"
    file.write_text("print('old')\n")
    monkeypatch.setattr(os.path, "samefile", lambda a, b: True)
    with pytest.raises(AssertionError):
        simulate_update(str(file), "print('x')\n")


