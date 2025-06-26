import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Stub tensorflow to avoid heavy dependency
class DummyWriter:
    def __init__(self, path):
        self.f = open(path, "wb")

    def write(self, data):
        self.f.write(data)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.f.close()


sys.modules["tensorflow"] = SimpleNamespace(
    io=SimpleNamespace(TFRecordWriter=DummyWriter)
)

from integrations.storage import LocalStorage


def test_save_tfrecord(tmp_path):
    storage = LocalStorage(str(tmp_path))
    data = [{"a": 1}, {"b": 2}]
    storage.save_dataset(data, fmt="tfrecord", version="1.2.3")
    tf_file = tmp_path / "wikipedia_qa.tfrecord"
    assert tf_file.exists()
    assert b'{"a": 1}' in tf_file.read_bytes()
    assert (tmp_path / "dataset_version.txt").read_text() == "1.2.3"


def test_s3_save_tfrecord(monkeypatch):
    class DummyClient:
        def __init__(self):
            self.kwargs = None
            self.kwargs_version = None

        def put_object(self, Bucket, Key, Body):
            if Key.endswith("dataset_version.txt"):
                self.kwargs_version = {"Bucket": Bucket, "Key": Key, "Body": Body}
            else:
                self.kwargs = {"Bucket": Bucket, "Key": Key, "Body": Body}

    dummy = DummyClient()
    monkeypatch.setitem(
        sys.modules, "boto3", SimpleNamespace(client=lambda *a, **k: dummy)
    )
    import importlib

    storage_mod = importlib.reload(importlib.import_module("integrations.storage"))
    s3 = storage_mod.S3Storage("bucket", prefix="pre", client=dummy)
    data = [{"x": 1}]
    s3.save_dataset(data, fmt="tfrecord", version="2.0.0")
    assert dummy.kwargs["Key"] == "pre/wikipedia_qa.tfrecord"
    assert b'{"x": 1}' in dummy.kwargs["Body"]
    assert dummy.kwargs_version == {
        "Bucket": "bucket",
        "Key": "pre/dataset_version.txt",
        "Body": b"2.0.0",
    }


def test_s3_compression(monkeypatch):
    class DummyClient:
        def __init__(self):
            self.kwargs = None

        def put_object(self, Bucket, Key, Body):
            self.kwargs = {"Bucket": Bucket, "Key": Key, "Body": Body}

    dummy = DummyClient()
    monkeypatch.setitem(
        sys.modules, "boto3", SimpleNamespace(client=lambda *a, **k: dummy)
    )
    import importlib

    storage_mod = importlib.reload(importlib.import_module("integrations.storage"))
    s3 = storage_mod.S3Storage("b", prefix="p", client=dummy)
    data = [{"z": 2}]
    s3.save_dataset(data, fmt="json", compression="gzip")
    assert dummy.kwargs["Key"].endswith("wikipedia_qa.json.gz")
    import gzip, json as js

    assert js.loads(gzip.decompress(dummy.kwargs["Body"]).decode()) == data
