import importlib
import os
from .config import logger


def check_dependencies() -> None:
    """Warn if simplified stubs are being used instead of real libraries."""
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    libs = ["aiohttp", "fastapi", "uvicorn", "networkx"]
    for lib in libs:
        try:
            mod = importlib.import_module(lib)
            path = getattr(mod, "__file__", "")
            if path and os.path.abspath(path).startswith(os.path.join(project_dir, lib)):
                logger.warning(
                    "Dependência %s está usando versão simplificada; instale as bibliotecas reais para integração completa",
                    lib,
                )
        except Exception as e:  # pragma: no cover - optional dependency
            logger.error("Falha ao importar %s", lib, error=str(e))
