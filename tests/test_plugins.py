import asyncio
from pathlib import Path

from devai.tasks import TaskManager
from devai.plugin_manager import PluginManager


class DummyAnalyzer:
    def __init__(self, root):
        self.code_root = root
        self.code_chunks = {}
        self.code_graph = {}
        self.learned_rules = {}


class DummyMemory:
    def save(self, entry, update_feedback=False):
        pass


def test_plugin_task(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    plugin_file = plugin_dir / "plug.py"
    plugin_file.write_text(
        """\
from pathlib import Path

PLUGIN_INFO = {'name': 'Dummy', 'version': '1.0', 'description': 'd'}

def register(tm):
    async def _perform_dummy_task(self, task, *args):
        return ['ok']
    tm.tasks['dummy'] = {'name': 'Dummy', 'type': 'dummy'}
    setattr(tm, '_perform_dummy_task', _perform_dummy_task.__get__(tm))

def unregister(tm):
    tm.tasks.pop('dummy', None)
    if hasattr(tm, '_perform_dummy_task'):
        delattr(tm, '_perform_dummy_task')
"""
    )
    analyzer = DummyAnalyzer(".")
    mem = DummyMemory()
    tm = TaskManager("missing.yaml", analyzer, mem)
    pm = PluginManager(tm)
    pm.load_plugins(str(plugin_dir))

    async def run():
        return await tm.run_task('dummy')

    res = asyncio.run(run())
    assert res == ['ok']


def test_enable_disable_plugin(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "plug.py").write_text(
        """\
PLUGIN_INFO = {'name': 'Dummy', 'version': '1.0', 'description': 'd'}

def register(tm):
    async def _perform_dummy_task(self, task, *args):
        return ['ok']
    tm.tasks['dummy'] = {'name': 'Dummy', 'type': 'dummy'}
    setattr(tm, '_perform_dummy_task', _perform_dummy_task.__get__(tm))

def unregister(tm):
    tm.tasks.pop('dummy', None)
    if hasattr(tm, '_perform_dummy_task'):
        delattr(tm, '_perform_dummy_task')
"""
    )
    analyzer = DummyAnalyzer(".")
    mem = DummyMemory()
    tm = TaskManager("missing.yaml", analyzer, mem)
    pm = PluginManager(tm, db_file=str(tmp_path / "db.sqlite"))
    pm.load_plugins(str(plugin_dir))
    assert 'dummy' in tm.tasks
    pm.disable_plugin('Dummy')
    assert 'dummy' not in tm.tasks
    pm.enable_plugin('Dummy')
    assert 'dummy' in tm.tasks
