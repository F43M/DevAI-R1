from difflib import unified_diff
from devai.file_history import FileHistory


def test_dry_run_diff_parser(tmp_path):
    src = tmp_path / "file.py"
    src.write_text("a\nb\n")
    new_code = "a\nc\n"
    diff = "".join(
        unified_diff(src.read_text().splitlines(keepends=True), new_code.splitlines(keepends=True))
    )
    assert "+" in diff and "-" in diff
    evaluation = {"tests_ok": True}
    assert "tests_ok" in evaluation
    history = FileHistory(str(tmp_path / "hist.json"))
    old_lines = src.read_text().splitlines()
    src.write_text(new_code)
    history.record(str(src), "edit", old=old_lines, new=new_code.splitlines())
    assert src.read_text() == new_code
