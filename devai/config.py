import os
import logging
import logging.handlers
from typing import Dict, Any
try:
    import structlog
except Exception:  # pragma: no cover - optional
    structlog = None
from config_utils import load_config

class Config:
    """Load configuration from YAML merging with defaults."""
    def __init__(self, path: str = "config.yaml"):
        defaults = {
            "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", ""),
            "MODEL_NAME": "deepseek/deepseek-r1-0528:free",
            "CODE_ROOT": "./app",
            "MEMORY_DB": "memory.sqlite",
            "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
            "TASK_DEFINITIONS": "tasks.yaml",
            "LOG_DIR": "./logs",
            "FILE_HISTORY": "file_history.json",
            "API_TOKEN": os.getenv("API_TOKEN", ""),
            "API_PORT": 8000,
            "LEARNING_LOOP_INTERVAL": 300,
            "MAX_CONTEXT_LENGTH": 160000,
            "OPENROUTER_URL": "https://openrouter.ai/api/v1/chat/completions",
            "INDEX_FILE": "faiss.index",
            "INDEX_IDS_FILE": "faiss_ids.json",
        }
        cfg = load_config(path, defaults)
        for key, value in cfg.items():
            setattr(self, key, value)


class Metrics:
    """Simple metrics collector."""

    def __init__(self):
        self.api_calls = 0
        self.total_response_time = 0.0
        self.errors = 0

    def record_call(self, duration: float):
        self.api_calls += 1
        self.total_response_time += duration

    def record_error(self):
        self.errors += 1

    def summary(self) -> Dict[str, Any]:
        avg = self.total_response_time / self.api_calls if self.api_calls else 0
        return {"api_calls": self.api_calls, "avg_response_time": avg, "errors": self.errors}


def configure_logging(log_dir: str):
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
    root_logger.setLevel(logging.INFO)
    return logger_obj


config = Config()
logger = configure_logging(config.LOG_DIR)
metrics = Metrics()
