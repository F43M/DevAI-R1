from devai.file_history import FileHistory


def test_history_record(tmp_path):
    log = tmp_path / "hist.json"
    hist = FileHistory(str(log))
    hist.record("a.txt", "create", new=["x"])
    hist.record("a.txt", "edit", old=["x"], new=["y"])
    items = hist.history("a.txt")
    assert len(items) == 2
    assert items[0]["type"] == "create"
    assert items[1]["old"] == ["x"]
