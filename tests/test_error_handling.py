import asyncio
import pytest
from devai.config import config
from devai.error_handler import (
    with_retry_async,
    friendly_message,
    log_error,
    error_memory,
    persist_errors,
    load_persisted_errors,
)


class _TimeoutCounter:
    def __init__(self):
        self.count = 0

    async def __call__(self):
        self.count += 1
        if self.count < 3:
            raise asyncio.TimeoutError()
        return "ok"


def test_retry_timeout():
    c = _TimeoutCounter()
    result = asyncio.run(with_retry_async(c, max_attempts=3, base_delay=0))
    assert result == "ok"
    assert c.count == 3
    assert friendly_message(asyncio.TimeoutError()).startswith("⏱️")


def test_retry_connection_error():
    attempts = 0

    async def failing():
        nonlocal attempts
        attempts += 1
        raise ConnectionError("fail")

    with pytest.raises(Exception):
        asyncio.run(with_retry_async(failing, max_attempts=2, base_delay=0))
    assert attempts == 2
    assert "📡" in friendly_message(ConnectionError())


def test_log_error_records():
    error_memory.clear()
    log_error("unit_test", ValueError("boom"))
    assert error_memory
    assert error_memory[-1]["função"] == "unit_test"


def test_friendly_unknown():
    msg = friendly_message(RuntimeError("x"))
    assert "Algo deu errado" in msg


def test_persist_and_load_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "ERROR_LOG_PATH", str(tmp_path / "errors_log.jsonl"))
    error_memory.clear()
    log_error("unit_test", ValueError("boom"))
    asyncio.run(persist_errors())
    assert (tmp_path / "errors_log.jsonl").exists()
    error_memory.clear()
    load_persisted_errors()
    assert error_memory
    assert error_memory[-1]["função"] == "unit_test"
