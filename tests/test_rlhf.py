import asyncio
import pytest
from devai import rlhf


class DummyMemory:
    pass


def test_collect_examples_returns_list():
    tuner = rlhf.RLFineTuner(DummyMemory())
    assert tuner.collect_examples() == []


def test_fine_tune_not_implemented():
    tuner = rlhf.RLFineTuner(DummyMemory())
    with pytest.raises(NotImplementedError):
        asyncio.run(tuner.fine_tune("base", "out"))
