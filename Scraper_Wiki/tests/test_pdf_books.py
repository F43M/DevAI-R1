import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def create_pdf(path: Path, text: str) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    pdf.cell(40, 10, text)
    pdf.output(str(path))


def test_pdf_books_single(monkeypatch, tmp_path):
    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    pdf_path = tmp_path / "sample.pdf"
    create_pdf(pdf_path, "Hello World")

    mod = importlib.import_module("plugins.pdf_books")
    plugin = mod.Plugin()
    items = plugin.fetch_items("en", str(pdf_path))
    assert items == [{"path": str(pdf_path), "lang": "en", "category": str(pdf_path)}]
    parsed = plugin.parse_item(items[0])
    assert "Hello World" in parsed["content"]
    assert parsed["language"] == "en"
    for field in [
        "raw_code",
        "context",
        "problems",
        "fixed_version",
        "lessons",
        "origin_metrics",
        "challenge",
    ]:
        assert field in parsed


def test_pdf_books_directory(monkeypatch, tmp_path):
    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    dir_path = tmp_path / "pdfs"
    dir_path.mkdir()
    pdf1 = dir_path / "a.pdf"
    pdf2 = dir_path / "b.pdf"
    create_pdf(pdf1, "A")
    create_pdf(pdf2, "B")

    mod = importlib.import_module("plugins.pdf_books")
    plugin = mod.Plugin()
    items = plugin.fetch_items("en", str(dir_path))
    paths = [str(pdf1), str(pdf2)]
    assert [it["path"] for it in items] == paths
    parsed = plugin.parse_item(items[0])
    assert "A" in parsed["content"]
    for field in [
        "raw_code",
        "context",
        "problems",
        "fixed_version",
        "lessons",
        "origin_metrics",
        "challenge",
    ]:
        assert field in parsed
