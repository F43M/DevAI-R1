import importlib
import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType

sys.modules.setdefault("spacy", SimpleNamespace(load=lambda *a, **k: None))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _reload(monkeypatch, modules=None):
    if modules:
        for name, mod in modules.items():
            monkeypatch.setitem(sys.modules, name, mod)
    if "training" in sys.modules and not hasattr(sys.modules["training"], "__path__"):
        sys.modules.pop("training")
    sys.modules.pop("training.formats", None)
    return importlib.import_module("training.formats")


def test_save_arrow_dataset(tmp_path, monkeypatch):
    written = {}

    def dummy_write_dataset(
        table,
        base_dir,
        format="parquet",
        partitioning=None,
        existing_data_behavior=None,
    ):
        written["dir"] = base_dir

    pa_mod = ModuleType("pyarrow")
    pa_mod.Table = SimpleNamespace(from_pylist=lambda recs: "table")
    ds_mod = ModuleType("pyarrow.dataset")
    ds_mod.write_dataset = dummy_write_dataset
    pa_mod.dataset = ds_mod
    formats = _reload(monkeypatch, {"pyarrow": pa_mod, "pyarrow.dataset": ds_mod})

    formats.save_arrow_dataset([{"a": 1}], tmp_path)
    assert written["dir"] == str(tmp_path)


def test_save_delta_table(tmp_path, monkeypatch):
    written = {}

    def dummy_write_deltalake(path, table, mode="error"):
        written["path"] = path

    pa_mod = ModuleType("pyarrow")
    pa_mod.Table = SimpleNamespace(from_pylist=lambda recs: "table")
    delta_mod = ModuleType("deltalake")
    delta_mod.write_deltalake = dummy_write_deltalake
    formats = _reload(monkeypatch, {"pyarrow": pa_mod, "deltalake": delta_mod})

    formats.save_delta_table([{"x": 1}], tmp_path)
    assert written["path"] == str(tmp_path)
