import sys
from types import ModuleType, SimpleNamespace
from integrations import datalake


def test_write_parquet(tmp_path, monkeypatch):
    dummy_pa = ModuleType("pyarrow")
    dummy_pa.Table = SimpleNamespace(from_pylist=lambda d: d)
    dummy_ds = ModuleType("pyarrow.dataset")
    dummy_ds.write_dataset = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "pyarrow", dummy_pa)
    monkeypatch.setitem(sys.modules, "pyarrow.dataset", dummy_ds)

    data = [
        {"lang": "en", "domain": "wiki", "text": "a"},
        {"lang": "en", "domain": "wiki", "text": "b"},
    ]
    datalake.write_parquet(data, str(tmp_path))
