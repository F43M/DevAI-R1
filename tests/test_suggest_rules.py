import json
import types
import devai.decision_log as dl


def test_suggest_rules(monkeypatch, tmp_path):
    log_file = tmp_path / "decision_log.yaml"
    data = [
        {"tipo": "edit", "modulo": "docs/a.txt", "hash_resultado": dl.OK_HASH},
        {"tipo": "edit", "modulo": "docs/a.txt", "hash_resultado": dl.OK_HASH},
        {"tipo": "edit", "modulo": "docs/a.txt", "hash_resultado": dl.OK_HASH},
        {"tipo": "create", "modulo": "src/b.py", "hash_resultado": dl.OK_HASH},
    ]
    log_file.write_text(json.dumps(data))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dl, "yaml", types.SimpleNamespace(safe_load=json.loads))
    rules = dl.suggest_rules(3)
    assert {"action": "edit", "path": "docs/a.txt", "approve": True} in rules
    assert all(r["action"] != "create" for r in rules)
