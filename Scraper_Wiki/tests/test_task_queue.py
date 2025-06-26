import importlib
from task_queue import InMemoryQueue, consume, publish, _BACKEND


def test_manual_ack(monkeypatch):
    q = InMemoryQueue()
    monkeypatch.setattr("task_queue.get_backend", lambda: q)
    publish("q", {"a": 1})
    it = consume("q", manual_ack=True)
    msg, ack = next(it)
    assert msg == {"a": 1}
    # not calling ack should keep internal unfinished count
    assert q._get_queue("q").unfinished_tasks == 1
    ack()
    assert q._get_queue("q").unfinished_tasks == 0
