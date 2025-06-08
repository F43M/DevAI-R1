import io
import pytest
from devai import yaml_fallback


def test_safe_load_simple_values():
    data = yaml_fallback.safe_load(io.StringIO("a: 1\nb: two\nc: 3.5"))
    assert data == {"a": 1, "b": "two", "c": 3.5}


def test_safe_load_lines_without_colon_are_ignored():
    data = yaml_fallback.safe_load("a: 1\ninvalid\nb: 2")
    assert data == {"a": 1, "b": 2}


def test_safe_dump_and_load_roundtrip_dict():
    original = {"x": 10, "y": "z"}
    dumped = yaml_fallback.safe_dump(original)
    loaded = yaml_fallback.safe_load(dumped)
    assert loaded == original


def test_safe_dump_and_load_roundtrip_list_of_dicts():
    original = [{"a": 1}, {"b": 2}]
    dumped = yaml_fallback.safe_dump(original)
    assert dumped.strip().splitlines() == ["- a: 1", "- b: 2"]
    loaded = yaml_fallback.safe_load(dumped)
    assert loaded == {"- a": 1, "- b": 2}
