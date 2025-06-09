import asyncio
from pathlib import Path

from devai.analyzer import CodeAnalyzer
from devai.memory import MemoryManager


def test_deep_scan_app_no_nameerror(tmp_path, monkeypatch):
    code_root = tmp_path / "app"
    code_root.mkdir()
    (code_root / "m.py").write_text("def foo():\n    pass\n")

    mem = MemoryManager(str(tmp_path / "mem.sqlite"), "dummy", model=None, index=None)
    analyzer = CodeAnalyzer(str(code_root), mem)
    monkeypatch.chdir(tmp_path)

    def edges_with_data(self, data=False):
        return [(u, v, {}) for u, vs in self._adj.items() for v in vs]

    # ensure stub graph handles `data` argument
    analyzer.code_graph.edges = edges_with_data.__get__(analyzer.code_graph, type(analyzer.code_graph))

    async def run():
        await analyzer.deep_scan_app()

    asyncio.run(run())
