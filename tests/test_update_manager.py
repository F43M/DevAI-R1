from devai.update_manager import UpdateManager
import devai.update_manager as upd


class DummyModel:
    async def generate(self, prompt, max_length=0):
        return "retry"

    async def close(self):
        pass


def test_safe_apply_success(tmp_path, monkeypatch):
    file = tmp_path / "f.txt"
    file.write_text("old")
    mgr = UpdateManager()
    monkeypatch.setattr(
        mgr,
        "run_tests",
        lambda capture_output=False: (True, "") if capture_output else True,
    )
    result = mgr.safe_apply(file, lambda p: p.write_text("new"))
    assert result
    assert file.read_text() == "new"
    assert not file.with_suffix(".txt.bak").exists()


def test_safe_apply_failure(tmp_path, monkeypatch):
    file = tmp_path / "f.txt"
    file.write_text("old")
    mgr = UpdateManager()
    monkeypatch.setattr(
        mgr,
        "run_tests",
        lambda capture_output=False: (False, "erro") if capture_output else False,
    )
    monkeypatch.setattr(upd, "AIModel", lambda: DummyModel())
    monkeypatch.setattr(upd, "registrar_licao_negativa", lambda a, b: None)
    monkeypatch.setattr(upd, "registrar_feedback_negativo", lambda a, b: None)
    result = mgr.safe_apply(file, lambda p: p.write_text("new"))
    assert not result
    assert file.read_text() == "old"
    # o backup eh movido de volta ao restaurar
    assert not file.with_suffix(".txt.bak").exists()
