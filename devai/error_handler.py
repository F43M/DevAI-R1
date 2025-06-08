import asyncio
import random
import subprocess
from datetime import datetime
from typing import Callable, Awaitable, TypeVar
from pathlib import Path

from .config import logger
import json
import aiofiles

# simple in-memory record of recent errors
error_memory = []  # persist across runs

def load_persisted_errors() -> None:
    """Load errors from previous runs into memory."""
    path = Path("errors_log.jsonl")
    if not path.exists():
        return
    try:
        for line in path.read_text().splitlines():
            try:
                data = json.loads(line)
                ts = data.get("timestamp")
                if ts:
                    timestamp = datetime.fromisoformat(ts)
                else:
                    timestamp = datetime.now()
                func = data.get("função") or data.get("funcao", "")
                error_memory.append(
                    {
                        "timestamp": timestamp,
                        "tipo": data.get("tipo", ""),
                        "mensagem": data.get("mensagem", ""),
                        "função": func,
                    }
                )
                if len(error_memory) > 100:
                    error_memory.pop(0)
            except Exception as e:
                logger.warning("Erro ao carregar linha do log", error=str(e))
    except Exception as e:
        logger.warning("Erro ao ler log persistido", error=str(e))

load_persisted_errors()

T = TypeVar("T")

async def with_retry_async(func: Callable[[], Awaitable[T]], max_attempts: int = 3, base_delay: float = 1.0) -> T:
    """Execute async function with retries using exponential backoff."""
    for attempt in range(1, max_attempts + 1):
        try:
            return await func()
        except (asyncio.TimeoutError, ConnectionError) as e:
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.4)
            logger.warning(
                f"⚠️ Tentativa {attempt} falhou: {e} — Retentando em {delay:.2f}s"
            )
            await asyncio.sleep(delay)
    raise Exception("🚫 Todas as tentativas falharam. A IA continua indisponível.")

def log_error(func_name: str, e: Exception) -> None:
    """Log error and store basic info for future analysis."""
    logger.error(f"[ERRO SIMBÓLICO] {type(e).__name__} em {func_name}: {e}")
    error_memory.append(
        {
            "timestamp": datetime.now(),
            "tipo": type(e).__name__,
            "mensagem": str(e),
            "função": func_name,
        }
    )
    if len(error_memory) > 100:
        error_memory.pop(0)


async def persist_errors() -> None:
    """Persist the in-memory error log to disk."""
    async with aiofiles.open("errors_log.jsonl", "a") as f:
        for err in error_memory:
            data = {
                "timestamp": err["timestamp"].isoformat(),
                "tipo": err["tipo"],
                "mensagem": err["mensagem"],
                "funcao": err["função"],
            }
            await f.write(json.dumps(data) + "\n")
    error_memory.clear()

def friendly_message(e: Exception) -> str:
    """Map technical errors to friendly messages for the user."""
    if isinstance(e, asyncio.TimeoutError):
        return "⏱️ A IA demorou para responder. Pode estar ocupada."
    if isinstance(e, subprocess.TimeoutExpired):
        return "🕒 Testes cancelados por excederem o tempo máximo permitido"
    if isinstance(e, ConnectionError):
        return "📡 Não foi possível conectar à IA. Verifique sua rede ou aguarde reconexão automática."
    if isinstance(e, MemoryError):
        return "💥 Testes encerrados por falta de memória"
    if getattr(e, "status", 0) >= 500:
        return "🧱 A IA retornou um erro interno. Isso pode ser temporário."
    return "⚠️ Algo deu errado. Consulte os logs ou tente novamente."
