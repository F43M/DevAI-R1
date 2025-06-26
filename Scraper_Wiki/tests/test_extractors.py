import sys
from pathlib import Path

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scraper_wiki as sw


def test_extract_infobox():
    html = "<table class='infobox'><tr><th>Name</th><td>Python</td></tr></table>"
    result = sw.extract_infobox(html)
    assert result == {"Name": "Python"}


def test_extract_infobox_malformed():
    html = "<table class='infobox'><tr><th>Name</th><td>Python<tr><th>Released</th><td>1991"  # missing closing tags
    result = sw.extract_infobox(html)
    assert "Name" in result


def test_extract_tables():
    html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
    tables = sw.extract_tables(html)
    assert tables == [[['A', 'B'], ['1', '2']]]


def test_extract_tables_malformed():
    html = "<table><tr><th>A<th>B<tr><td>1<td>2"  # unclosed tags
    tables = sw.extract_tables(html)
    assert tables != []
