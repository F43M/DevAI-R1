import json
import asyncio
from pathlib import Path
from devai import feedback


def test_registrar_preferencia(tmp_path, monkeypatch):
    prefs_file = tmp_path / "prefs.json"
    prefs_file.write_text(json.dumps({"preferencias": []}))
    monkeypatch.setattr(feedback, "PREFS_FILE", prefs_file)
    feedback.registrar_preferencia("use aspas simples")
    feedback.registrar_preferencia("use aspas simples")
    feedback.registrar_preferencia("evite loops")
    assert feedback.listar_preferencias() == [
        "use aspas simples",
        "evite loops",
    ]
