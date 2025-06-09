import os
import logging
import logging.handlers
from dataclasses import dataclass, field, fields, MISSING
from typing import Dict, Any

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass
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
    MODEL_NAME: str = "deepseek/deepseek-r1-0528:free"  # DEPRECATED
    MODELS: Dict[str, Any] = field(default_factory=dict)
    CODE_ROOT: str = "./app"
    MEMORY_DB: str = "memory.sqlite"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    TASK_DEFINITIONS: str = "tasks.yaml"
    LOG_DIR: str = "./logs"
    LOG_MONITOR_INTERVAL: int = 60
    FILE_HISTORY: str = "file_history.json"
    API_SECRET: str = os.getenv("API_SECRET", "")
    API_PORT: int = 8000
    LEARNING_LOOP_INTERVAL: int = 300
    MAX_CONTEXT_LENGTH: int = 160000
    OPENROUTER_URL: str = "https://openrouter.ai/api/v1/chat/completions"
    MODEL_TIMEOUT: int = 60
    INDEX_FILE: str = "faiss.index"
    INDEX_IDS_FILE: str = "faiss_ids.json"
    NOTIFY_EMAIL: str = os.getenv("NOTIFY_EMAIL", "")
    LOCAL_MODEL: str = os.getenv("LOCAL_MODEL", "")
    MAX_SESSION_TOKENS: int = 1000
    COMPLEXITY_HISTORY: str = "complexity_history.json"
    LOG_AGGREGATOR_URL: str = os.getenv("LOG_AGGREGATOR_URL", "")
    DOUBLE_CHECK: bool = False
    SHOW_REASONING_BY_DEFAULT: bool = False
    SHOW_CONTEXT_BUTTON: bool = False
    ENABLE_AI_SUMMARY: bool = False
    START_MODE: str = "fast"  # options: fast, full, custom
    START_TASKS: list[str] = field(default_factory=list)
    RESCAN_INTERVAL_MINUTES: int = 15  # intervalo mínimo para novas varreduras
    TESTS_USE_ISOLATION: bool = True
    TEST_CPU_LIMIT: int = 1  # limite de segundos de CPU por processo de teste
    TEST_MEMORY_LIMIT_MB: int = 512  # memória máxima em MB para testes
    LEARNING_RATE_LIMIT: int = 5
    AUTO_MONITOR_FAILURES: int = 3
    AUTO_MONITOR_FILES: int = 5
    AUTO_MONITOR_HOURS: int = 72
    RLHF_THRESHOLD: int = 10
    RLHF_OUTPUT_DIR: str = "./logs/rlhf_results"

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
        if "MODEL_NAME" in cfg:
            logging.getLogger("devai").warning(
                "\u26a0\ufe0f 'MODEL_NAME' est\xe1 obsoleto. Use 'MODELS.default.name'."
            )
            models_name = cfg.get("MODELS", {}).get("default", {}).get("name")
            if models_name and models_name != cfg["MODEL_NAME"]:
                logging.getLogger("devai").warning(
                    "Valores divergentes para 'MODEL_NAME' e 'MODELS.default.name'"
                )
        self._validate()

    def _validate(self) -> None:
        if not isinstance(self.API_PORT, int):
            raise ValueError("API_PORT must be integer")
        if not isinstance(self.LEARNING_LOOP_INTERVAL, int):
            raise ValueError("LEARNING_LOOP_INTERVAL must be integer")
        if not isinstance(self.LOG_MONITOR_INTERVAL, int):
            raise ValueError("LOG_MONITOR_INTERVAL must be integer")
        if self.START_MODE not in {"fast", "full", "custom"}:
            raise ValueError("START_MODE must be 'fast', 'full' or 'custom'")
        if not isinstance(self.START_TASKS, list):
            raise ValueError("START_TASKS must be a list")
        allowed = {"scan", "watch", "monitor"}
        for t in self.START_TASKS:
            if t not in allowed:
                raise ValueError("Invalid task in START_TASKS")
        if not isinstance(self.RESCAN_INTERVAL_MINUTES, int):
            raise ValueError("RESCAN_INTERVAL_MINUTES must be integer")
        if not isinstance(self.TEST_CPU_LIMIT, int):
            raise ValueError("TEST_CPU_LIMIT must be integer")
        if not isinstance(self.TEST_MEMORY_LIMIT_MB, int):
            raise ValueError("TEST_MEMORY_LIMIT_MB must be integer")
        if not isinstance(self.LEARNING_RATE_LIMIT, int):
            raise ValueError("LEARNING_RATE_LIMIT must be integer")
        if not isinstance(self.TESTS_USE_ISOLATION, bool):
            raise ValueError("TESTS_USE_ISOLATION must be boolean")
        if not isinstance(self.ENABLE_AI_SUMMARY, bool):
            raise ValueError("ENABLE_AI_SUMMARY must be boolean")
        if not isinstance(self.AUTO_MONITOR_FAILURES, int):
            raise ValueError("AUTO_MONITOR_FAILURES must be integer")
        if not isinstance(self.AUTO_MONITOR_FILES, int):
            raise ValueError("AUTO_MONITOR_FILES must be integer")
        if not isinstance(self.AUTO_MONITOR_HOURS, int):
            raise ValueError("AUTO_MONITOR_HOURS must be integer")
        if not isinstance(self.RLHF_THRESHOLD, int):
            raise ValueError("RLHF_THRESHOLD must be integer")
        if not isinstance(self.RLHF_OUTPUT_DIR, str):
            raise ValueError("RLHF_OUTPUT_DIR must be string")

    @property
    def model_name(self) -> str:
        """Return the active model name from MODELS.default."""
        name = self.MODELS.get("default", {}).get("name")
        if not name and hasattr(self, "MODEL_NAME"):
            name = self.MODEL_NAME
        if hasattr(self, "MODEL_NAME") and name != self.MODEL_NAME:
            logging.getLogger("devai").warning(
                "\u26a0\ufe0f 'MODEL_NAME' est\xe1 obsoleto. Use 'MODELS.default.name'."
            )
        return name


class Metrics:
    """Simple metrics collector."""

    def __init__(self):
        self.api_calls = 0
        self.total_response_time = 0.0
        self.errors = 0
        self.max_cpu_percent = 0.0
        self.max_memory_percent = 0.0
        self.model_usage: Dict[str, int] = {}
        self.incomplete_responses = 0

    def record_call(self, duration: float):
        self.api_calls += 1
        self.total_response_time += duration
        self.update_resources()

    def record_error(self):
        self.errors += 1

    def record_model_usage(self, name: str) -> None:
        self.model_usage[name] = self.model_usage.get(name, 0) + 1

    def record_incomplete(self) -> None:
        self.incomplete_responses += 1

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
            "model_usage": dict(self.model_usage),
            "incomplete_responses": self.incomplete_responses,
            "error_percent": (self.errors / self.api_calls * 100) if self.api_calls else 0,
            "incomplete_percent": (self.incomplete_responses / self.api_calls * 100) if self.api_calls else 0,
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
            def critical(self, msg, **kw):
                self._l.critical(msg + (" " + " ".join(f"{k}={v}" for k,v in kw.items()) if kw else ""))
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
api_key_missing = not bool(config.OPENROUTER_API_KEY)
logger = configure_logging(config.LOG_DIR, config.LOG_AGGREGATOR_URL)
if api_key_missing:
    logger.critical(
        "❌ OPENROUTER_API_KEY não configurada. A IA não funcionará corretamente."
    )
metrics = Metrics()
