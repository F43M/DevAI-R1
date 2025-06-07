from devai.symbolic_memory_tagger import tag_memory_entry, previous_hashes


def test_tag_memory_entry():
    previous_hashes.clear()
    tags1 = tag_memory_entry({"name": "f", "hash": "1"})
    tags2 = tag_memory_entry({"name": "f", "hash": "2"})
    assert "@nova_funcao" in tags1
    assert "@refatorado" in tags2
