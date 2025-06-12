import importlib.util
from .config import logger


def check_dependencies() -> None:
    """Report missing required libraries."""
    libs = ["aiohttp", "fastapi", "uvicorn", "networkx", "pydantic", "yaml"]
    missing = [lib for lib in libs if importlib.util.find_spec(lib) is None]
    if missing:  # pragma: no cover - reporting only
        logger.error("Dependências ausentes: %s", ", ".join(missing))
