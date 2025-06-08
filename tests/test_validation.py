import asyncio
from pathlib import Path
import pytest

from devai.analyzer import CodeAnalyzer


class DummyMemory:
    def save(self, entry, update_feedback=False):
        pass


def make_analyzer(tmp_path: Path) -> CodeAnalyzer:
    root = tmp_path / "app"
    root.mkdir()
    return CodeAnalyzer(str(root), DummyMemory())


def test_file_outside_root(tmp_path: Path):
    analyzer = make_analyzer(tmp_path)
    with pytest.raises(ValueError):
        asyncio.run(analyzer.create_file("../hack.txt", "x"))


def test_negative_line(tmp_path: Path):
    analyzer = make_analyzer(tmp_path)
    file = Path(analyzer.code_root) / "f.py"
    file.write_text("a\nb\n")
    with pytest.raises(ValueError):
        asyncio.run(analyzer.edit_line("f.py", -1, "x"))


def test_empty_content(tmp_path: Path):
    analyzer = make_analyzer(tmp_path)
    file = Path(analyzer.code_root) / "g.py"
    file.write_text("a\n")
    with pytest.raises(ValueError):
        asyncio.run(analyzer.edit_line("g.py", 1, ""))


def test_valid_edit(tmp_path: Path):
    analyzer = make_analyzer(tmp_path)
    file = Path(analyzer.code_root) / "h.py"
    file.write_text("old\n")
    result = asyncio.run(analyzer.edit_line("h.py", 1, "new"))
    assert result
    assert file.read_text().strip() == "new"
