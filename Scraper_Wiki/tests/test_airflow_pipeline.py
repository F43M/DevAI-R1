import importlib.util
import sys
from types import ModuleType
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

# Load the module without executing ``training.__init__``
training_mod = ModuleType("training")
training_mod.__path__ = [str(ROOT / "training")]
sys.modules["training"] = training_mod
spec = importlib.util.spec_from_file_location(
    "training.airflow_pipeline", ROOT / "training" / "airflow_pipeline.py"
)
pipe = importlib.util.module_from_spec(spec)
sys.modules["training.airflow_pipeline"] = pipe
spec.loader.exec_module(pipe)


def test_create_dag():
    if pipe.DAG is None:
        with pytest.raises(RuntimeError):
            pipe.create_dag()
    else:
        dag = pipe.create_dag()
        assert {
            "scrape_data",
            "postprocess_dataset",
            "publish_dataset",
        }.issubset(dag.task_dict)


def test_publish_dataset_fine_tune(tmp_path, monkeypatch):
    """Fine-tune step is triggered when FINE_TUNE_MODEL is set."""
    processed = tmp_path / "processed.json"
    processed.write_text("[]", encoding="utf-8")

    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    logged = {}
    dummy_mlflow = ModuleType("mlflow")
    dummy_mlflow.start_run = lambda *a, **k: DummyRun()
    dummy_mlflow.log_param = lambda k, v: logged.setdefault(k, v)
    dummy_mlflow.log_metric = lambda *a, **k: None
    sys.modules["mlflow"] = dummy_mlflow

    dummy_utils = ModuleType("training.pretrained_utils")

    def fake_fine(path, model_name="bert-base-uncased"):
        logged["fine"] = str(path)
        return "ver1234"

    dummy_utils.fine_tune_model = fake_fine
    sys.modules["training.pretrained_utils"] = dummy_utils

    builder = ModuleType("builder")
    builder_obj = type(
        "B", (), {"dataset": [], "save_dataset": lambda self, format: None}
    )()
    monkeypatch.setattr(pipe, "DatasetBuilder", lambda: builder_obj)

    ti = type("TI", (), {"xcom_pull": lambda self, task_ids: str(processed)})()
    monkeypatch.setenv("FINE_TUNE_MODEL", "1")
    pipe.publish_dataset(ti)

    assert logged.get("fine") == str(processed)
