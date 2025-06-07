from pathlib import Path
from devai.shadow_mode import simulate_update
from devai.config import config


def test_simulate_update_no_apply(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CODE_ROOT", str(tmp_path))
    file = Path(config.CODE_ROOT) / "a.py"
    file.write_text("print('old')\n")
    diff, temp_root, sim_id = simulate_update(str(file), "print('new')\n")
    assert "-print('old')" in diff
    assert "+print('new')" in diff
    assert file.read_text() == "print('old')\n"
    assert Path(temp_root).exists()


