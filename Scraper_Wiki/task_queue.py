import os
import json
from queue import Queue as _Queue

try:
    import pika
except Exception:  # pragma: no cover - optional dependency
    pika = None

try:
    import redis
except Exception:  # pragma: no cover - optional dependency
    redis = None


class BaseQueue:
    """Abstract queue backend."""

    def publish(self, queue: str, message: dict):
        raise NotImplementedError

    def consume(self, queue: str, manual_ack: bool = False):
        raise NotImplementedError

    def clear(self, queue: str):
        """Remove all pending messages from ``queue``."""
        raise NotImplementedError


class InMemoryQueue(BaseQueue):
    def __init__(self):
        self.queues = {}

    def _get_queue(self, name: str):
        return self.queues.setdefault(name, _Queue())

    def publish(self, queue: str, message: dict):
        self._get_queue(queue).put(json.dumps(message))

    def consume(self, queue: str, manual_ack: bool = False):
        q = self._get_queue(queue)
        while True:
            msg = q.get()
            data = json.loads(msg)
            if manual_ack:
                yield data, q.task_done
            else:
                yield data
                q.task_done()

    def clear(self, queue: str):
        self._get_queue(queue).queue.clear()


class RabbitMQQueue(BaseQueue):
    def __init__(self, url: str):
        if pika is None:
            raise RuntimeError("pika not installed")
        self.connection = pika.BlockingConnection(pika.URLParameters(url))
        self.channel = self.connection.channel()

    def publish(self, queue: str, message: dict):
        self.channel.queue_declare(queue=queue, durable=True)
        body = json.dumps(message).encode()
        self.channel.basic_publish(exchange="", routing_key=queue, body=body)

    def consume(self, queue: str, manual_ack: bool = False):
        self.channel.queue_declare(queue=queue, durable=True)
        for method, _, body in self.channel.consume(queue, inactivity_timeout=1):
            if body:
                data = json.loads(body.decode())
                if manual_ack:
                    yield data, lambda tag=method.delivery_tag: self.channel.basic_ack(
                        tag
                    )
                else:
                    self.channel.basic_ack(method.delivery_tag)
                    yield data

    def clear(self, queue: str):
        self.channel.queue_purge(queue)


class RedisQueue(BaseQueue):
    """Queue backend using Redis lists."""

    def __init__(self, url: str):
        if redis is None:
            raise RuntimeError("redis not installed")
        self.client = redis.from_url(url)

    def publish(self, queue: str, message: dict):
        self.client.rpush(queue, json.dumps(message))

    def consume(self, queue: str, manual_ack: bool = False):
        while True:
            if manual_ack:
                item = self.client.brpoplpush(queue, f"{queue}:processing", timeout=1)
                if item:
                    data = json.loads(item)
                    yield data, lambda val=item: self.client.lrem(
                        f"{queue}:processing", 1, val
                    )
            else:
                item = self.client.blpop(queue, timeout=1)
                if item:
                    yield json.loads(item[1])

    def clear(self, queue: str):
        self.client.delete(queue)


_BACKEND = None


def get_backend() -> BaseQueue:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    url = os.environ.get("QUEUE_URL")
    if url:
        if url.startswith("redis") and redis:
            _BACKEND = RedisQueue(url)
        elif pika:
            _BACKEND = RabbitMQQueue(url)
        else:
            raise RuntimeError("No supported queue backend available")
    else:
        _BACKEND = InMemoryQueue()
    return _BACKEND


def publish(queue_name: str, message: dict):
    get_backend().publish(queue_name, message)


def consume(queue_name: str, manual_ack: bool = False):
    yield from get_backend().consume(queue_name, manual_ack=manual_ack)


def clear(queue_name: str) -> None:
    """Clear all messages from ``queue_name``."""
    get_backend().clear(queue_name)
