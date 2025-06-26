import importlib
import sys
import os
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _reload(monkeypatch, modules=None):
    if modules:
        for name, mod in modules.items():
            monkeypatch.setitem(sys.modules, name, mod)
    return importlib.reload(importlib.import_module("integrations.storage"))


def test_get_backend_postgres(monkeypatch, tmp_path):
    class DummyCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def execute(self, *args, **kwargs):
            pass

    class DummyConn:
        def cursor(self):
            return DummyCursor()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    pg_mod = SimpleNamespace(connect=lambda dsn: DummyConn())
    storage_mod = _reload(monkeypatch, {"psycopg2": pg_mod})
    backend = storage_mod.get_backend("postgres", str(tmp_path))
    assert isinstance(backend, storage_mod.PostgreSQLStorage)


def test_get_backend_mongodb(monkeypatch, tmp_path):
    class DummyCollection:
        def insert_many(self, data):
            self.data = data

    class DummyDB:
        def __getitem__(self, name):
            return DummyCollection()

    class DummyClient:
        def __init__(self, uri):
            self.uri = uri

        def __getitem__(self, name):
            return DummyDB()

    pymod = SimpleNamespace(MongoClient=DummyClient)
    storage_mod = _reload(monkeypatch, {"pymongo": pymod})
    backend = storage_mod.get_backend("mongodb", str(tmp_path))
    assert isinstance(backend, storage_mod.MongoDBStorage)


def test_get_backend_s3(monkeypatch, tmp_path):
    class DummyClient:
        def put_object(self, **kw):
            self.kw = kw

    boto = SimpleNamespace(client=lambda *a, **k: DummyClient())
    storage_mod = _reload(monkeypatch, {"boto3": boto})
    backend = storage_mod.get_backend("s3", str(tmp_path))
    assert isinstance(backend, storage_mod.S3Storage)


def test_get_backend_local(monkeypatch, tmp_path):
    storage_mod = _reload(monkeypatch, {})
    backend = storage_mod.get_backend("local", str(tmp_path))
    assert isinstance(backend, storage_mod.LocalStorage)


def test_offload_to_s3(monkeypatch, tmp_path):
    uploaded = {}

    class DummyClient:
        def put_object(self, **kw):
            uploaded.update(kw)

    boto = SimpleNamespace(client=lambda *a, **k: DummyClient())

    class DummyCollection:
        def insert_many(self, data):
            self.data = data

    class DummyDB:
        def __getitem__(self, name):
            return DummyCollection()

    class DummyClientDB:
        def __init__(self, uri):
            self.uri = uri

        def __getitem__(self, name):
            return DummyDB()

    pymod = SimpleNamespace(MongoClient=DummyClientDB)
    monkeypatch.setenv("S3_UPLOAD_BUCKET", "bucket")
    monkeypatch.setenv("UPLOAD_THRESHOLD_MB", "0")
    storage_mod = _reload(monkeypatch, {"boto3": boto, "pymongo": pymod})
    backend = storage_mod.get_backend("mongodb", str(tmp_path))
    backend.save_dataset([{"a": 1}])
    assert uploaded["Bucket"] == "bucket"


def test_get_backend_neo4j(monkeypatch, tmp_path):
    storage_mod = _reload(monkeypatch, {})
    backend = storage_mod.get_backend("neo4j", str(tmp_path))
    assert isinstance(backend, storage_mod.GraphStorage)


def test_get_backend_iceberg(monkeypatch, tmp_path):
    storage_mod = _reload(monkeypatch, {})
    backend = storage_mod.get_backend("iceberg", str(tmp_path))
    assert isinstance(backend, storage_mod.DatalakeStorage)


def test_get_backend_milvus(monkeypatch, tmp_path):
    storage_mod = _reload(monkeypatch, {})
    backend = storage_mod.get_backend("milvus", str(tmp_path))
    assert isinstance(backend, storage_mod.MilvusVectorStore)


def test_get_backend_weaviate(monkeypatch, tmp_path):
    storage_mod = _reload(monkeypatch, {})
    backend = storage_mod.get_backend("weaviate", str(tmp_path))
    assert isinstance(backend, storage_mod.WeaviateVectorStore)
