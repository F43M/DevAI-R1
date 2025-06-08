import asyncio
from pathlib import Path
import types

import pytest

from devai.shadow_mode import simulate_update, evaluate_change_with_ia, log_simulation
from devai.update_manager import UpdateManager
from devai.file_history import FileHistory
from devai.memory import MemoryManager
from devai.analyzer import CodeAnalyzer
from devai.config import config


class DummyAI:
    async def safe_api_call(self, prompt, max_tokens, context="", memory=None):
        return "sugestao aprovada"

    async def close(self):
        pass


def test_simulated_update_and_application(tmp_path, monkeypatch):
    code_root = tmp_path / "app"
    code_root.mkdir()
    monkeypatch.setattr(config, "CODE_ROOT", str(code_root))
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(config, "LOG_DIR", str(log_dir))

    file_path = code_root / "mod.py"
    file_path.write_text("x = 1\n")
    test_file = code_root / "test_mod.py"
    test_file.write_text("import mod\n\ndef test_x():\n    assert mod.x == 2\n")

    diff, temp_root, sim_id = simulate_update(str(file_path), "x = 2\n")

    # avoid real subprocess and API calls
    monkeypatch.setattr("devai.shadow_mode.run_tests_in_temp", lambda d: (True, ""))
    monkeypatch.setattr("devai.shadow_mode.AIModel", lambda: DummyAI())

    history = FileHistory(str(tmp_path / "hist.json"))
    mem = MemoryManager(str(tmp_path / "mem.sqlite"), "dummy", model=None, index=None)
    analyzer = CodeAnalyzer(str(code_root), mem, history)

    async def run_flow():
        evaluation = await evaluate_change_with_ia(diff)
        updater = UpdateManager()
        monkeypatch.setattr(updater, "run_tests", lambda capture_output=False: (True, "") if capture_output else True)
        success, _ = updater.safe_apply(file_path, lambda p: p.write_text("x = 2\n"), capture_output=True)
        if success:
            history.record(str(file_path), "edit", old=["x = 1"], new=["x = 2"])
            mem.save(
                {
                    "type": "refatoracao",
                    "memory_type": "refatoracao aprovada",
                    "content": f"Refatoracao aplicada em {file_path}",
                    "metadata": {"arquivo": str(file_path), "contexto": "dry_run"},
                }
            )
            log_simulation(sim_id, str(file_path), True, evaluation["analysis"], "shadow_approved")
        return success

    result = asyncio.run(run_flow())
    assert result
    assert file_path.read_text() == "x = 2\n"
    log = Path(config.LOG_DIR) / "simulation_history.md"
    assert log.exists()
    assert mem.search("Refatoracao aplicada", memory_type="refatoracao aprovada")
