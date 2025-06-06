import os
import logging
import logging.handlers
from dataclasses import dataclass, field, fields, MISSING
from typing import Dict, Any
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None
try:
    import structlog
except Exception:  # pragma: no cover - optional
    structlog = None
from config_utils import load_config

@dataclass(init=False)
class Config:
    """Application configuration loaded from YAML with simple validation."""

    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    MODEL_NAME: str = "deepseek/deepseek-r1-0528:free"
    MODELS: Dict[str, Any] = field(default_factory=dict)
    CODE_ROOT: str = "./app"
    MEMORY_DB: str = "memory.sqlite"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    TASK_DEFINITIONS: str = "tasks.yaml"
    LOG_DIR: str = "./logs"
    FILE_HISTORY: str = "file_history.json"
    API_SECRET: str = os.getenv("API_SECRET", "")
    API_PORT: int = 8000
    LEARNING_LOOP_INTERVAL: int = 300
    MAX_CONTEXT_LENGTH: int = 160000
    OPENROUTER_URL: str = "https://openrouter.ai/api/v1/chat/completions"
    INDEX_FILE: str = "faiss.index"
    INDEX_IDS_FILE: str = "faiss_ids.json"
    NOTIFY_EMAIL: str = os.getenv("NOTIFY_EMAIL", "")
    LOCAL_MODEL: str = os.getenv("LOCAL_MODEL", "")
    COMPLEXITY_HISTORY: str = "complexity_history.json"
    LOG_AGGREGATOR_URL: str = os.getenv("LOG_AGGREGATOR_URL", "")

    def __init__(self, path: str = "config.yaml") -> None:
        defaults: Dict[str, Any] = {}
        for f in fields(self.__class__):
            if f.default is not MISSING:
                defaults[f.name] = f.default
            elif f.default_factory is not MISSING:  # type: ignore
                defaults[f.name] = f.default_factory()  # type: ignore
        cfg = load_config(path, defaults)
        for key, value in cfg.items():
            setattr(self, key, value)
        self._validate()

    def _validate(self) -> None:
        if not isinstance(self.API_PORT, int):
            raise ValueError("API_PORT must be integer")
        if not isinstance(self.LEARNING_LOOP_INTERVAL, int):
            raise ValueError("LEARNING_LOOP_INTERVAL must be integer")


class Metrics:
    """Simple metrics collector."""

    def __init__(self):
        self.api_calls = 0
        self.total_response_time = 0.0
        self.errors = 0
        self.max_cpu_percent = 0.0
        self.max_memory_percent = 0.0

    def record_call(self, duration: float):
        self.api_calls += 1
        self.total_response_time += duration
        self.update_resources()

    def record_error(self):
        self.errors += 1

    def update_resources(self) -> None:
        if not psutil:
            return
        try:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
            self.max_cpu_percent = max(self.max_cpu_percent, cpu)
            self.max_memory_percent = max(self.max_memory_percent, mem)
        except Exception:
            pass

    def summary(self) -> Dict[str, Any]:
        avg = self.total_response_time / self.api_calls if self.api_calls else 0
        data = {
            "api_calls": self.api_calls,
            "avg_response_time": avg,
            "errors": self.errors,
            "max_cpu_percent": self.max_cpu_percent,
            "max_memory_percent": self.max_memory_percent,
        }
        if psutil:
            try:
                data.update(
                    {
                        "cpu_percent": psutil.cpu_percent(interval=None),
                        "memory_percent": psutil.virtual_memory().percent,
                    }
                )
            except Exception:
                pass
        return data


def configure_logging(log_dir: str, aggregator: str = ""):
    os.makedirs(log_dir, exist_ok=True)
    if structlog:
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.stdlib.add_log_level,
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        logger_obj = structlog.get_logger()
    else:
        logging.basicConfig(level=logging.INFO)
        base_logger = logging.getLogger("devai")
        class SimpleLogger:
            def __init__(self, l):
                self._l = l
            def info(self, msg, **kw):
                self._l.info(msg + (" " + " ".join(f"{k}={v}" for k,v in kw.items()) if kw else ""))
            def warning(self, msg, **kw):
                self._l.warning(msg + (" " + " ".join(f"{k}={v}" for k,v in kw.items()) if kw else ""))
            def error(self, msg, **kw):
                self._l.error(msg + (" " + " ".join(f"{k}={v}" for k,v in kw.items()) if kw else ""))
        logger_obj = SimpleLogger(base_logger)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, "ai_core.log"), maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    if aggregator:
        http_handler = logging.handlers.HTTPHandler(aggregator, "/", method="POST")
        root_logger.addHandler(http_handler)
    root_logger.setLevel(logging.INFO)
    return logger_obj


config = Config()
logger = configure_logging(config.LOG_DIR, config.LOG_AGGREGATOR_URL)
metrics = Metrics()
