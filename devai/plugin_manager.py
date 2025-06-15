"""Plugin management utilities for optional extensions."""

import importlib.util
import pathlib
import sqlite3
from typing import Any, Dict, List, Set

from .config import logger


class PluginManager:
    """Load and manage optional plugins stored in the plugins folder."""

    def __init__(self, task_manager, db_file: str = "plugins.sqlite") -> None:
        self.task_manager = task_manager
        self.plugins: Dict[str, Any] = {}
        self.plugin_tasks: Dict[str, Set[str]] = {}
        self.plugin_paths: Dict[str, pathlib.Path] = {}
        self.conn = sqlite3.connect(db_file, check_same_thread=False)
        self._init_db()
        self.load_plugins()

    def _init_db(self) -> None:
        """Initialize the plugin database table."""
        cur = self.conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS plugins ("
            "name TEXT PRIMARY KEY, version TEXT, active INTEGER)"
        )
        self.conn.commit()

    def list_plugins(self) -> List[Dict[str, Any]]:
        """Return metadata for all installed plugins."""
        cur = self.conn.cursor()
        cur.execute("SELECT name, version, active FROM plugins")
        return [
            {"name": n, "version": v, "active": bool(a)} for n, v, a in cur.fetchall()
        ]

    def _register_module(self, name: str, module: Any) -> None:
        """Register plugin tasks and metadata."""
        before = set(self.task_manager.tasks.keys())
        if hasattr(module, "register"):
            module.register(self.task_manager)
        added = set(self.task_manager.tasks.keys()) - before
        self.plugins[name] = module
        self.plugin_tasks[name] = added

    def load_plugins(self, path: str = "plugins") -> None:
        """Discover and load plugins from the given folder."""
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
            except Exception as e:
                # pragma: no cover - plugin errors should not crash
                logger.error("Erro ao carregar plugin", plugin=str(file), error=str(e))
                continue
            info = getattr(
                module,
                "PLUGIN_INFO",
                {"name": file.stem, "version": "0.0"},
            )
            name = info.get("name", file.stem)
            version = info.get("version", "0.0")
            self.plugin_paths[name] = file
            cur = self.conn.cursor()
            cur.execute("SELECT active FROM plugins WHERE name=?", (name,))
            row = cur.fetchone()
            if row is None:
                cur.execute(
                    "INSERT INTO plugins (name, version, active) VALUES " "(?, ?, 1)",
                    (name, version),
                )
                self.conn.commit()
                active = True
            else:
                active = bool(row[0])
                cur.execute(
                    "UPDATE plugins SET version=? WHERE name=?",
                    (version, name),
                )
                self.conn.commit()
            if active:
                self._register_module(name, module)

    def enable_plugin(self, name: str) -> bool:
        """Dynamically enable a plugin by name."""
        if name in self.plugins:
            self.conn.execute(
                "UPDATE plugins SET active=1 WHERE name=?",
                (name,),
            )
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
        except Exception as e:
            # pragma: no cover - plugin errors should not crash
            logger.error("Erro ao ativar plugin", plugin=name, error=str(e))
            return False
        self._register_module(name, module)
        self.conn.execute(
            "UPDATE plugins SET active=1 WHERE name=?",
            (name,),
        )
        self.conn.commit()
        return True

    def disable_plugin(self, name: str) -> bool:
        """Disable a previously loaded plugin."""
        module = self.plugins.pop(name, None)
        tasks = self.plugin_tasks.pop(name, set())
        for t in tasks:
            self.task_manager.tasks.pop(t, None)
        if module and hasattr(module, "unregister"):
            try:
                module.unregister(self.task_manager)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to unregister plugin", plugin=name, error=str(exc))
        self.conn.execute(
            "UPDATE plugins SET active=0 WHERE name=?",
            (name,),
        )
        self.conn.commit()
        return module is not None
