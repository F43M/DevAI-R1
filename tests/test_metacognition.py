import asyncio
import json
from datetime import datetime
import yaml

import devai.metacognition as metacog


def test_global_reflection_log(tmp_path, monkeypatch):
    now = datetime.now().isoformat()
    history = [
        {"timestamp": now, "tipo": "ok", "modulo": "a.py", "decision_score": 1},
        {"timestamp": now, "tipo": "erro", "modulo": "a.py", "decision_score": -1},
    ]
    hist_file = tmp_path / "decisions.yaml"
    hist_file.write_text(yaml.safe_dump(history))

    score_map = tmp_path / "score.json"
    self_log = tmp_path / "self.md"
    global_log = tmp_path / "global.md"

    monkeypatch.setattr(metacog, "SCORE_MAP", score_map)
    monkeypatch.setattr(metacog, "SELF_LOG", self_log)
    monkeypatch.setattr(metacog, "GLOBAL_LOG", global_log)

    loop = metacog.MetacognitionLoop(str(hist_file))
    asyncio.run(loop._analyze())
    content = global_log.read_text()
    assert "# Reflexão gerada" in content
    assert "- Contexto: a.py" in content

    asyncio.run(loop._analyze())
    content2 = global_log.read_text()
    assert content2.count("# Reflexão gerada") == 2
    data = json.loads(score_map.read_text())
    assert data.get("a.py") == 0
