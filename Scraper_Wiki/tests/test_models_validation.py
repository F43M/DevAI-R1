import importlib.util
from pathlib import Path
import pytest
from pydantic import ValidationError

MODELS_PATH = (
    Path(__file__).resolve().parents[1] / "scraper_wiki" / "models" / "__init__.py"
)
spec = importlib.util.spec_from_file_location("scraper_wiki.models", MODELS_PATH)
models = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(models)
DatasetRecord = models.DatasetRecord
QARecord = models.QARecord


def test_datasetrecord_valid():
    rec = DatasetRecord.parse_obj(
        {
            "id": "1",
            "language": "en",
            "category": "c",
            "content": "text",
            "created_at": "now",
        }
    )
    assert rec.id == "1"


def test_datasetrecord_missing_field():
    with pytest.raises(ValidationError):
        DatasetRecord.parse_obj({"language": "en", "content": "x", "created_at": "now"})


def test_datasetrecord_wrong_type():
    with pytest.raises(ValidationError):
        DatasetRecord.parse_obj(
            {
                "id": "1",
                "language": "en",
                "category": "c",
                "content": "x",
                "created_at": "now",
                "content_embedding": "bad",
            }
        )


def test_run_plugin_validation():
    class DummyPlugin:
        def fetch_items(self, lang, category, since=None):
            return [{}]

        def parse_item(self, item):
            return {
                "id": 1,
                "language": "en",
                "category": "c",
                "content": "c",
                "created_at": "now",
            }

    def run_plugin(plugin):
        for item in plugin.fetch_items("en", "c"):
            result = plugin.parse_item(item)
            DatasetRecord.parse_obj(result)

    with pytest.raises(ValidationError):
        run_plugin(DummyPlugin())


def test_qarecord():
    qa = QARecord(question="q", answer="a")
    assert qa.question == "q" and qa.answer == "a"
