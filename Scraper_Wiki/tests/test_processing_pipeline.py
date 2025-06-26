import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pipeline = importlib.import_module("processing.pipeline")


def test_pipeline_processes_records(monkeypatch):
    data = [
        {"content": "def foo(x):\n    # comment\n    if x:\n        eval('x')"},
        {"content": "def foo(x):\n    # comment\n    if x:\n        eval('x')"},
    ]
    for rec in data:
        rec["content_embedding"] = [0.1, 0.2]

    monkeypatch.setattr(pipeline, "lint_code", lambda code: ["LINT"])
    monkeypatch.setattr(pipeline, "scan_vulnerabilities", lambda code: ["VULN"])

    processed = pipeline.get_pipeline()(data)
    assert len(processed) == 1
    rec = processed[0]
    assert rec["lint"] == ["LINT"]
    assert rec["vulnerabilities"] == ["VULN"]
    assert "tokens" in rec
    assert "complexities" in rec.get("metadata", {})
