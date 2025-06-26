import json
from pathlib import Path
from typer.testing import CliRunner

import cli
from processing import aggregator


def test_merge_datasets_deduplicates(tmp_path):
    rec = {
        "id": "1",
        "language": "en",
        "content": "print('hi')",
        "created_at": "now",
    }
    file1 = tmp_path / "a.json"
    file2 = tmp_path / "b.json"
    file1.write_text(json.dumps([rec]), encoding="utf-8")
    file2.write_text(json.dumps([rec.copy()]), encoding="utf-8")

    merged = aggregator.merge_datasets([str(file1), str(file2)])
    assert len(merged) == 1


def test_cli_merge_datasets(tmp_path):
    rec = {"id": "1", "language": "en", "content": "print('hi')", "created_at": "now"}
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    f1.write_text(json.dumps([rec]), encoding="utf-8")
    f2.write_text(json.dumps([rec.copy()]), encoding="utf-8")
    out = tmp_path / "out.json"

    runner = CliRunner()
    result = runner.invoke(
        cli.app, ["merge-datasets", str(f1), str(f2), "--output", str(out)]
    )
    assert result.exit_code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data) == 1
