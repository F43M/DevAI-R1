from devai.symbolic_memory_tagger import tag_memory_entry, previous_hashes


def test_tag_memory_entry():
    previous_hashes.clear()
    tags1 = tag_memory_entry({"name": "f", "hash": "1"})
    tags2 = tag_memory_entry({"name": "f", "hash": "2"})
    assert "@nova_funcao" in tags1
    assert "@refatorado" in tags2


def test_complexity_and_docstring_tags(monkeypatch):
    previous_hashes.clear()
    from devai.config import config
    monkeypatch.setattr(config, "COMPLEXITY_TAG_THRESHOLD", 5)
    tags = tag_memory_entry({
        "name": "g",
        "hash": "1",
        "complexity": 6,
        "docstring": "deprecated: old"
    })
    assert "@complexo" in tags
    assert "@descontinuado" in tags
