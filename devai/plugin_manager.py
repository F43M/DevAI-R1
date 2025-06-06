class PluginManager:
    """Load and register optional plugins from the plugins folder."""
    def __init__(self, task_manager):
        self.task_manager = task_manager
        self.plugins = []
        self.load_plugins()

    def load_plugins(self, path="plugins"):
        import importlib.util
        import pathlib

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
                if hasattr(module, "register"):
                    module.register(self.task_manager)
                    self.plugins.append(module)
            except Exception as e:  # pragma: no cover - plugin errors should not crash
                from .config import logger
                logger.error("Erro ao carregar plugin", plugin=str(file), error=str(e))
