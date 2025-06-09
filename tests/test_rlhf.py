import asyncio
import json
from pathlib import Path

import pytest

from devai import rlhf, core
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


def test_collect_examples_returns_content(tmp_path, monkeypatch):
    mem = _create_memory(tmp_path)
    # duplicate entry should be removed
    mem.save(
        {
            "type": "dialog",
            "memory_type": "dialog_summary",
            "content": "answer",
            "metadata": {"prompt": "question"},
            "feedback_score": 1,
        }
    )
    monkeypatch.setattr(rlhf.config, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(rlhf.config, "RLHF_OUTPUT_DIR", str(tmp_path / "out"))
    tuner = rlhf.RLFineTuner(mem)
    data = tuner.collect_examples()
    assert len(data) == 1
    assert (tmp_path / "rlhf_dataset.json").exists()
    ds_dir = tmp_path / "out" / "datasets"
    assert len(list(ds_dir.glob("*.sha256"))) == 1


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


def test_train_from_memory_empty(tmp_path, monkeypatch):
    import devai.memory as memory_module
    monkeypatch.setattr(memory_module, "MemoryManager", DummyMemory)
    monkeypatch.setattr(rlhf.config, "MEMORY_DB", str(tmp_path / "mem.sqlite"))

    result = asyncio.run(rlhf.train_from_memory("base", str(tmp_path / "out")))
    assert result["status"] == "no_data"


def test_run_scheduled_rlhf(tmp_path, monkeypatch):
    mem = _create_memory(tmp_path)
    monkeypatch.setattr(core.config, "RLHF_THRESHOLD", 1)
    monkeypatch.setattr(core.config, "RLHF_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(core.config, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(core.config, "MODELS", {"default": {"name": "base"}})

    async def fake_train(base, out):
        Path(out).mkdir(parents=True, exist_ok=True)
        return {"status": "ok"}

    monkeypatch.setattr(core.rlhf, "train_from_memory", fake_train)

    result = asyncio.run(core.run_scheduled_rlhf(mem))
    assert result["status"] == "ok"
    assert "hash" in result
    assert result["num_examples"] == 1
    cur = mem.conn.cursor()
    cur.execute("SELECT metadata FROM memory WHERE type = 'rlhf_training'")
    meta = json.loads(cur.fetchone()[0])
    assert meta["hash"] == result["hash"]


def test_run_scheduled_rlhf_skips_if_same(tmp_path, monkeypatch):
    mem = _create_memory(tmp_path)
    monkeypatch.setattr(core.config, "RLHF_THRESHOLD", 1)
    monkeypatch.setattr(core.config, "RLHF_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(core.config, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(core.config, "MODELS", {"default": {"name": "base"}})

    calls: list[str] = []

    async def fake_train(base, out):
        calls.append(out)
        Path(out).mkdir(parents=True, exist_ok=True)
        return {"status": "ok"}

    monkeypatch.setattr(core.rlhf, "train_from_memory", fake_train)

    result1 = asyncio.run(core.run_scheduled_rlhf(mem))
    assert result1["status"] == "ok"

    result2 = asyncio.run(core.run_scheduled_rlhf(mem))
    assert result2["status"] == "skipped"
    assert len(calls) == 1
    ds_dir = tmp_path / "out" / "datasets"
    assert len(list(ds_dir.glob("*.sha256"))) == 1
