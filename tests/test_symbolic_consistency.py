from pathlib import Path
from devai.file_history import FileHistory
from devai.memory import MemoryManager


def test_symbolic_consistency(tmp_path):
    hist = FileHistory(str(tmp_path / "h.json"))
    mem = MemoryManager(str(tmp_path / "mem.sqlite"), "dummy", model=None, index=None)
    hist.record("a.py", "edit", old=["x"], new=["y"])
    mem.save({
        "type": "refatoracao",
        "memory_type": "refatoracao aprovada",
        "content": "Refatoracao aplicada em a.py",
        "metadata": {"arquivo": "a.py"},
    })
    assert hist.history("a.py")
    assert mem.search("Refatoracao aplicada", memory_type="refatoracao aprovada")
    log = Path(tmp_path / "logs")
    log.mkdir()
    log_file = log / "simulation_history.md"
    log_file.write_text("## Simulacao\nA\nA\nA\nA\nshadow_failed\n")
    text = log_file.read_text()
    assert "shadow_failed" in text
