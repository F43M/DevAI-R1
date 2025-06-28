import types
import json
import asyncio
from datetime import datetime

from devai.core import CodeMemoryAI
from devai.conversation_handler import ConversationHandler
from devai.config import config


def _setup_ai(log_dir):
    ai = object.__new__(CodeMemoryAI)
    ai.memory = types.SimpleNamespace(
        search=lambda *a, **k: [],
        recent_entries=lambda *a, **k: [],
        indexed_ids=[],
    )
    ai.analyzer = types.SimpleNamespace(
        code_chunks={},
        learned_rules=[],
        last_analysis_time=datetime.now(),
        scan_progress=0.0,
        get_code_graph=lambda: {},
        read_lines=lambda *a, **k: [],
        list_dir=lambda *a, **k: [],
        edit_line=lambda *a, **k: True,
        create_file=lambda *a, **k: True,
        delete_file=lambda *a, **k: True,
        create_directory=lambda *a, **k: True,
        delete_directory=lambda *a, **k: True,
    )
    ai.history = types.SimpleNamespace(history=lambda *a, **k: [])
    ai.tasks = types.SimpleNamespace(last_actions=lambda: [])
    ai.log_monitor = types.SimpleNamespace()
    ai.conv_handler = ConversationHandler(memory=ai.memory)
    ai.reason_stack = []
    ai.double_check = False
    ai.app = types.SimpleNamespace()
    routes = {}

    def fake_get(path):
        def decorator(fn):
            routes[path] = fn
            return fn

        return decorator

    ai.app.get = ai.app.post = fake_get
    ai.app.mount = lambda *a, **k: None
    config.LOG_DIR = str(log_dir)
    CodeMemoryAI._setup_api_routes(ai)
    return routes


def test_stats_endpoint(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "scraper_progress.json").write_text(json.dumps({"p": 1}))

    routes = _setup_ai(log_dir)
    result = asyncio.run(routes["/stats"]())
    assert result["p"] == 1
