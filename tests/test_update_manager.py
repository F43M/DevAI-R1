from devai.update_manager import UpdateManager


def test_safe_apply_success(tmp_path, monkeypatch):
    file = tmp_path / "f.txt"
    file.write_text("old")
    mgr = UpdateManager()
    monkeypatch.setattr(mgr, "run_tests", lambda: True)
    result = mgr.safe_apply(file, lambda p: p.write_text("new"))
    assert result
    assert file.read_text() == "new"
    assert not file.with_suffix(".txt.bak").exists()


def test_safe_apply_failure(tmp_path, monkeypatch):
    file = tmp_path / "f.txt"
    file.write_text("old")
    mgr = UpdateManager()
    monkeypatch.setattr(mgr, "run_tests", lambda: False)
    result = mgr.safe_apply(file, lambda p: p.write_text("new"))
    assert not result
    assert file.read_text() == "old"
    # o backup eh movido de volta ao restaurar
    assert not file.with_suffix(".txt.bak").exists()
