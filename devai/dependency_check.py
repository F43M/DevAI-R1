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
                    f"Dependência {lib} está usando versão simplificada; instale as bibliotecas reais para integração completa"
                )
        except Exception as e:  # pragma: no cover - optional dependency
            logger.error(
                f"Falha ao importar {lib}", error=str(e)
            )
