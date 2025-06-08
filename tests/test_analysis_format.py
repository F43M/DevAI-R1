import asyncio
from datetime import datetime
from devai.core import CodeMemoryAI
from devai.conversation_handler import ConversationHandler
import types

class DummyGraph:
    def __init__(self):
        self._adj = {'f': {}}
    def add_node(self, n):
        self._adj.setdefault(n, {})
    def out_degree(self, n):
        return len(self._adj.get(n, {}))

class DummyAnalyzer:
    def __init__(self):
        self.code_root = '.'
        self.code_graph = DummyGraph()
        self.code_chunks = {
            'f': {
                'name': 'f',
                'file': 'core.py',
                'code': 'def f(): pass',
                'complexity': 2,
                'dependencies': [],
                'docstring': ''
            }
        }
        self.learned_rules = {}
        self.last_analysis_time = datetime.now()

    async def deep_scan_app(self):
        pass

    async def summary_by_module(self):
        return {'core.py': {'complex_functions': 0, 'todos': 0, 'score': '✅ Estável'}}

class DummyMemory:
    def save(self, *a, **k):
        pass

async def run_deep():
    ai = object.__new__(CodeMemoryAI)
    ai.memory = DummyMemory()
    ai.analyzer = DummyAnalyzer()
    ai.ai_model = None
    ai.learning_engine = None
    ai.tasks = None
    ai.log_monitor = None
    ai.complexity_tracker = None
    ai.conv_handler = ConversationHandler(memory=ai.memory)
    ai.conversation_history = []
    ai.double_check = False

    record = {}
    app = types.SimpleNamespace()
    def fake_get(path):
        def decorator(fn):
            record[path] = fn
            return fn
        return decorator
    app.get = fake_get
    app.post = fake_get
    app.mount = lambda *a, **k: None
    ai.app = app

    CodeMemoryAI._setup_api_routes(ai)
    deep_fn = record['/deep_analysis']
    return await deep_fn(token='')

result = asyncio.run(run_deep())

def test_symbols_present():
    assert '🚩' in result['report']
    assert '⚠️' in result['report']
    assert '💡' in result['report']

def test_module_listed():
    assert 'core.py' in result['modules']
    assert 'core.py' in result['report']

def test_formatting():
    assert '🧠 Análise do Projeto' in result['report']
    assert '\n' in result['report']
