import asyncio
from pathlib import Path

import pytest

from devai import rlhf
from devai.memory import MemoryManager


class DummyMemory(MemoryManager):
    def __init__(self, path: str):
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


def test_fine_tune_creates_output(tmp_path):
    mem = _create_memory(tmp_path)
    tuner = rlhf.RLFineTuner(mem)
    out = tmp_path / "model"
    result = asyncio.run(tuner.fine_tune("base", str(out)))
    assert out.exists()
    assert "status" in result
