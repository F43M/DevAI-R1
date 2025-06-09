import asyncio
from pathlib import Path

import pytest

from devai import rlhf
from devai.memory import MemoryManager


class DummyMemory(MemoryManager):
    def __init__(self, path: str, *a, **k):
        super().__init__(path, "dummy", model=None, index=None)


def _create_memory(tmp_path: Path) -> DummyMemory:
    mem = DummyMemory(str(tmp_path / "mem.sqlite"))
    mem.save(
        {
            "type": "dialog",
            "memory_type": "dialog_summary",
            "content": "answer",
            "metadata": {"prompt": "question"},
            "feedback_score": 2,
        }
    )
    return mem


def test_collect_examples_returns_content(tmp_path):
    mem = _create_memory(tmp_path)
    tuner = rlhf.RLFineTuner(mem)
    data = tuner.collect_examples()
    assert data and data[0]["prompt"] == "question"


def test_collect_log_examples(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("User: hi\nAssistant: hello\n")
    mem = _create_memory(tmp_path)
    tuner = rlhf.RLFineTuner(mem)
    data = tuner._collect_from_logs(str(tmp_path))
    assert {"prompt": "hi", "response": "hello", "score": 1} in data


def test_fine_tune_creates_output(tmp_path):
    mem = _create_memory(tmp_path)
    tuner = rlhf.RLFineTuner(mem)
    out = tmp_path / "model"
    result = asyncio.run(tuner.fine_tune("base", str(out)))
    assert out.exists()
    assert "status" in result


def test_cli_main_runs(tmp_path, monkeypatch, capsys):
    _create_memory(tmp_path)
    out = tmp_path / "model"
    import devai.memory as memory_module
    monkeypatch.setattr(memory_module, "MemoryManager", DummyMemory)
    monkeypatch.setattr(rlhf.config, "MEMORY_DB", str(tmp_path / "mem.sqlite"))
    rlhf.main(["base", str(out)])
    out_text = capsys.readouterr().out
    assert out.exists()
    assert "status" in out_text
