import importlib.util
import pathlib
import sqlite3
from typing import Dict, Set, Any, List

from .config import logger


class PluginManager:
    """Load and manage optional plugins stored in the plugins folder."""

    def __init__(self, task_manager, db_file: str = "plugins.sqlite") -> None:
        self.task_manager = task_manager
        self.plugins: Dict[str, Any] = {}
        self.plugin_tasks: Dict[str, Set[str]] = {}
        self.plugin_paths: Dict[str, pathlib.Path] = {}
        self.conn = sqlite3.connect(db_file)
        self._init_db()
        self.load_plugins()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS plugins (name TEXT PRIMARY KEY, version TEXT, active INTEGER)"
        )
        self.conn.commit()

    def list_plugins(self) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT name, version, active FROM plugins")
        return [
            {"name": n, "version": v, "active": bool(a)} for n, v, a in cur.fetchall()
        ]

    def _register_module(self, name: str, module: Any) -> None:
        before = set(self.task_manager.tasks.keys())
        if hasattr(module, "register"):
            module.register(self.task_manager)
        added = set(self.task_manager.tasks.keys()) - before
        self.plugins[name] = module
        self.plugin_tasks[name] = added

    def load_plugins(self, path: str = "plugins") -> None:
        plugin_dir = pathlib.Path(path)
        if not plugin_dir.exists():
            return
        for file in plugin_dir.glob("*.py"):
            spec = importlib.util.spec_from_file_location(file.stem, file)
            if not spec or not spec.loader:
                continue
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as e:  # pragma: no cover - plugin errors should not crash
                logger.error("Erro ao carregar plugin", plugin=str(file), error=str(e))
                continue
            info = getattr(module, "PLUGIN_INFO", {"name": file.stem, "version": "0.0"})
            name = info.get("name", file.stem)
            version = info.get("version", "0.0")
            self.plugin_paths[name] = file
            cur = self.conn.cursor()
            cur.execute("SELECT active FROM plugins WHERE name=?", (name,))
            row = cur.fetchone()
            if row is None:
                cur.execute(
                    "INSERT INTO plugins (name, version, active) VALUES (?, ?, 1)",
                    (name, version),
                )
                self.conn.commit()
                active = True
            else:
                active = bool(row[0])
                cur.execute("UPDATE plugins SET version=? WHERE name=?", (version, name))
                self.conn.commit()
            if active:
                self._register_module(name, module)

    def enable_plugin(self, name: str) -> bool:
        if name in self.plugins:
            self.conn.execute("UPDATE plugins SET active=1 WHERE name=?", (name,))
            self.conn.commit()
            return True
        path = self.plugin_paths.get(name)
        if not path:
            return False
        spec = importlib.util.spec_from_file_location(name, path)
        if not spec or not spec.loader:
            return False
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:  # pragma: no cover - plugin errors should not crash
            logger.error("Erro ao ativar plugin", plugin=name, error=str(e))
            return False
        self._register_module(name, module)
        self.conn.execute("UPDATE plugins SET active=1 WHERE name=?", (name,))
        self.conn.commit()
        return True

    def disable_plugin(self, name: str) -> bool:
        module = self.plugins.pop(name, None)
        tasks = self.plugin_tasks.pop(name, set())
        for t in tasks:
            self.task_manager.tasks.pop(t, None)
        if module and hasattr(module, "unregister"):
            try:
                module.unregister(self.task_manager)
            except Exception:
                pass
        self.conn.execute("UPDATE plugins SET active=0 WHERE name=?", (name,))
        self.conn.commit()
        return module is not None
