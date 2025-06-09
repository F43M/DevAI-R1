import importlib
from .config import logger


def check_dependencies() -> None:
    """Warn only when core dependencies are missing."""
    libs = ["aiohttp", "fastapi", "uvicorn", "networkx"]
    for lib in libs:
        try:
            importlib.import_module(lib)
        except Exception as e:  # pragma: no cover - optional dependency
            logger.warning(
                f"Falha ao importar {lib}; alguns recursos podem ficar indisponíveis",
                error=str(e),
            )
