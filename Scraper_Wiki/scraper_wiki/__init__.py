# 🚀 F43M Wikipedia Scraper Ultra Pro Max - GodMode++
# CEO: Fabio | Engenharia de Nível Industrial
"""Core scraping utilities and dataset builder."""

import wikipediaapi

# Some versions of the wikipedia-api package do not expose
# ``WikipediaException``.  This attribute is referenced throughout the
# codebase when specifying retry logic for API calls.  To maintain
# compatibility with those versions we provide a simple fallback to the
# base ``Exception`` type when the attribute is missing.
if not hasattr(wikipediaapi, "WikipediaException"):

    class WikipediaException(Exception):
        """Fallback exception used when ``wikipediaapi`` lacks one."""

    wikipediaapi.WikipediaException = WikipediaException
import os
import re
import time
import json
import csv
import random
import logging

try:
    import structlog
except ImportError:  # pragma: no cover - optional dependency
    structlog = None
import asyncio
import inspect
from tqdm import tqdm
from unidecode import unidecode
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed,
    Future,
)

try:  # optional heavy dependency
    from datasets import Dataset, concatenate_datasets
except ImportError:  # pragma: no cover - missing deps
    Dataset = object

    def concatenate_datasets(*args, **kwargs):
        return None


from pathlib import Path
import hashlib
import pickle
import zlib
import uuid
from bs4 import BeautifulSoup
import requests
import aiohttp
from training.formats import save_qa_dataset, save_text_corpus
from training.postprocessing import filter_by_complexity
from urllib.parse import urlparse, urljoin
import html2text
import ast
from typing import List, Dict, Tuple, Optional, Set, Protocol
from datetime import datetime, timedelta
import multiprocessing
import signal
from scraper_wiki.state import load_last_scraped, save_last_scraped
from integrations.alerts import send_alert
from integrations.dvc_utils import track_path
import backoff
import tenacity
import numpy as np
import spacy

try:  # optional heavy dependency
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - missing deps
    SentenceTransformer = object
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import sqlite3
from integrations import storage
import dq
import metrics
from integrations import storage_sqlite
from integrations.binary_storage import BinaryStorage
from utils.text import clean_text, extract_entities, translate_text
from utils.relation import extract_relations, extract_relations_regex
from utils.quality import (
    classify_github_repo,
    classify_stackoverflow_answer,
    balance_quality,
    generate_challenge_prompt,
    evaluate_code_quality,
)
from utils.cleaner import clean_wiki_text, split_sentences
from utils.code import (
    normalize_indentation,
    remove_comments,
    detect_programming_language,
    docstring_to_google,
    parse_function_signature,
    parse_with_language,
    extract_context_from_code,
)
from utils.ast_tools import get_functions_complexity
from utils.code_sniffer import scan as scan_code
from utils.contextualizer import search_discussions
from utils.rate_limiter import RateLimiter
from processing.pipeline import get_pipeline
from enrichment.generator import (
    generate_diagram,
    generate_explanations,
    link_theory,
    generate_synthetic_qa,
)
from utils.sonarqube import analyze_code
from utils.gap_analysis import identify_gaps


HANDLED_EXCEPTIONS = (
    requests.RequestException,
    aiohttp.ClientError,
    asyncio.TimeoutError,
    json.JSONDecodeError,
    pickle.PickleError,
    zlib.error,
    OSError,
    sqlite3.Error,
    ValueError,
)

# Precompiled regex patterns for text cleaning
_CLEAN_COMBINED_RE = re.compile(
    r"\[\d+\]|\{\|.*?\|\}|\b(?:ver também|see also|véase também|voir aussi)\b.*",
    flags=re.IGNORECASE | re.DOTALL,
)
_SECTION_RE = re.compile(
    r"==\s*(?:referências|references|referencias|bibliografia|bibliography|bibliografía|ligações externas|external links|enlaces externos)\s*==.*",
    flags=re.IGNORECASE | re.DOTALL,
)


# ============================
# 🔧 Configurações Avançadas
# ============================
class Config:
    # Idiomas suportados (prioridade ordenada)
    LANGUAGES = ["pt", "en", "es", "fr", "de", "it", "ja", "zh"]

    # Diretórios de saída
    OUTPUT_DIR = "datasets_wikipedia_pro"
    RAW_DIR = "datasets/raw"
    CACHE_DIR = ".wiki_cache"
    LOG_DIR = "logs"
    ASSETS_DIR = os.environ.get("ASSETS_DIR", "assets")

    # Categorias avançadas com pesos
    CATEGORIES = {
        "Programação": 1.0,
        "Algoritmos": 0.9,
        "Linguagens de programação": 1.2,
        "Estruturas de dados": 0.95,
        "Engenharia de software": 1.1,
        "Ciência da computação": 0.85,
        "Desenvolvimento web": 1.15,
        "Banco de dados": 0.9,
        "Inteligência artificial": 1.3,
        "Segurança da informação": 0.95,
        "Machine Learning": 1.25,
        "Redes neurais": 1.1,
        "Visão computacional": 0.9,
        "Processamento de linguagem natural": 1.2,
        "Sistemas distribuídos": 0.85,
        "Computação quântica": 0.8,
        "Blockchain": 0.9,
        "Internet das Coisas": 0.85,
        "Realidade virtual": 0.8,
        "DevOps": 1.0,
    }

    # Parâmetros avançados
    MAX_DEPTH = 3  # Profundidade máxima de navegação em categorias
    MAX_THREADS = int(os.environ.get("MAX_THREADS", multiprocessing.cpu_count() * 2))
    MAX_PROCESSES = int(os.environ.get("MAX_PROCESSES", multiprocessing.cpu_count()))
    MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "5"))
    WORKER_CONCURRENCY = int(os.environ.get("WORKER_CONCURRENCY", "5"))
    RETRIES = 5
    TIMEOUT = 10
    RATE_LIMIT_DELAY = float(os.environ.get("RATE_LIMIT_DELAY", 0.5))
    PLUGIN_RATE_LIMITS: Dict[str, float] = {}
    _raw_plugin_limits = os.environ.get("PLUGIN_RATE_LIMITS")
    if _raw_plugin_limits:
        for part in _raw_plugin_limits.split(","):
            if "=" in part:
                name, val = part.split("=", 1)
                try:
                    PLUGIN_RATE_LIMITS[name.strip()] = float(val)
                except ValueError:
                    pass
    PLUGIN_RATE_LIMITS.setdefault("default", RATE_LIMIT_DELAY)
    MAX_PAGES_PER_CATEGORY = 1000
    MIN_TEXT_LENGTH = 150  # mínimo de caracteres para considerar uma página
    MAX_TEXT_LENGTH = 10000  # máximo de caracteres a extrair por página
    PAGEVIEW_DAYS = int(os.environ.get("PAGEVIEW_DAYS", "30"))
    MIN_PAGEVIEWS = int(os.environ.get("MIN_PAGEVIEWS", "0"))
    REMOVE_STOPWORDS = os.environ.get("REMOVE_STOPWORDS", "0") == "1"
    MIN_CODE_QUALITY = float(os.environ.get("MIN_CODE_QUALITY", "1.0"))
    MAX_LINT_ERRORS = int(os.environ.get("MAX_LINT_ERRORS", "10"))

    # Modelos de NLP
    NLP_MODELS = {
        "en": "en_core_web_lg",
        "pt": "pt_core_news_lg",
        "es": "es_core_news_lg",
        "fr": "fr_core_news_lg",
        "de": "de_core_news_lg",
        "it": "it_core_news_lg",
    }

    # Configuração de embeddings
    EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    # GPU
    _use_gpu_flag = os.environ.get("USE_GPU", "auto").lower()
    if _use_gpu_flag == "auto":
        try:
            import torch  # type: ignore

            USE_GPU: bool = torch.cuda.is_available()
        except ImportError:  # pragma: no cover - optional dependency
            USE_GPU = False
    else:
        USE_GPU = _use_gpu_flag in {"1", "true", "yes"}

    # Configuração de sumarização
    SUMMARY_SENTENCES = 3

    # Configuração de clustering
    CLUSTER_K = 10

    # Proxies e headers
    PROXIES = []  # Lista de proxies rotativos pode ser adicionada
    PROXY_PROVIDER_URL = os.environ.get("PROXY_PROVIDER_URL")
    PROXY_PROVIDER_AUTH = os.environ.get("PROXY_PROVIDER_AUTH")
    CUSTOM_USER_AGENT = os.environ.get("CUSTOM_USER_AGENT")
    USE_UNDETECTED_CHROMEDRIVER = (
        os.environ.get("USE_UNDETECTED_CHROMEDRIVER", "0") == "1"
    )
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    ]

    # Backend de cache: 'file' (padrão), 'redis' ou 'sqlite'
    CACHE_BACKEND = "file"
    REDIS_URL = "redis://localhost:6379/0"
    SQLITE_PATH = os.path.join(CACHE_DIR, "cache.sqlite")
    CACHE_TTL: Optional[int] = int(os.environ.get("CACHE_TTL", "0")) or None
    STORAGE_BACKEND = os.environ.get("STORAGE_BACKEND", "local")
    """Backend used for dataset storage.

    Choices are ``local``, ``s3``, ``mongodb``, ``postgres``, ``iceberg``,
    ``delta``, ``datalake``, ``neo4j``, ``milvus`` and ``weaviate``.
    """
    COMPRESSION = os.environ.get("COMPRESSION", "none")
    SCRAPER_BACKEND = os.environ.get("SCRAPER_BACKEND", "selenium")
    LOG_SERVICE_URL = os.environ.get("LOG_SERVICE_URL")
    LOG_SERVICE_TYPE = os.environ.get("LOG_SERVICE_TYPE", "loki")
    ALERT_WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL")

    # API endpoints and keys for optional plugins
    STACKOVERFLOW_API_KEY = os.environ.get("STACKOVERFLOW_API_KEY")
    STACKOVERFLOW_API_ENDPOINT = os.environ.get(
        "STACKOVERFLOW_API_ENDPOINT", "https://api.stackexchange.com/2.3"
    )

    # Generic Stack Exchange configuration
    STACKEXCHANGE_API_KEY = os.environ.get(
        "STACKEXCHANGE_API_KEY", STACKOVERFLOW_API_KEY
    )
    STACKEXCHANGE_API_ENDPOINT = os.environ.get(
        "STACKEXCHANGE_API_ENDPOINT", STACKOVERFLOW_API_ENDPOINT
    )
    STACKEXCHANGE_SITE = os.environ.get("STACKEXCHANGE_SITE", "stackoverflow")
    STACKEXCHANGE_MIN_SCORE = int(os.environ.get("STACKEXCHANGE_MIN_SCORE", "0"))
    WIKIDATA_API_ENDPOINT = os.environ.get(
        "WIKIDATA_API_ENDPOINT", "https://www.wikidata.org/w/api.php"
    )

    @classmethod
    def get_random_user_agent(cls):
        if cls.CUSTOM_USER_AGENT:
            return cls.CUSTOM_USER_AGENT
        return random.choice(cls.USER_AGENTS)


_proxy_index = 0

# Storage for downloaded images and other binary assets
binary_storage = BinaryStorage(Config.ASSETS_DIR)


def _fetch_premium_proxy() -> str | None:
    """Retrieve a proxy from a premium provider if configured."""
    if not Config.PROXY_PROVIDER_URL:
        return None
    headers = {}
    if Config.PROXY_PROVIDER_AUTH:
        headers["Authorization"] = Config.PROXY_PROVIDER_AUTH
    try:
        resp = requests.get(Config.PROXY_PROVIDER_URL, headers=headers, timeout=5)
        resp.raise_for_status()
        return resp.text.strip()
    except requests.RequestException as exc:  # pragma: no cover - network issues
        logger.warning(
            "Failed to fetch premium proxy",
            extra={"error_type": type(exc).__name__, "error_message": str(exc)},
        )
        return None


def get_next_proxy() -> str | None:
    """Return the next proxy cycling through ``Config.PROXIES`` or provider."""
    global _proxy_index
    if Config.PROXY_PROVIDER_URL:
        proxy = _fetch_premium_proxy()
        if proxy:
            return proxy
    if not Config.PROXIES:
        return None
    proxy = Config.PROXIES[_proxy_index % len(Config.PROXIES)]
    _proxy_index += 1
    return proxy


BASE_URLS = {
    "en": "https://en.wikipedia.org",
    "pt": "https://pt.wikipedia.org",
}


def get_base_url(lang: str) -> str:
    """Return base Wikipedia URL for a given language."""
    return BASE_URLS.get(lang, f"https://{lang}.wikipedia.org")


# ============================
# 🗂 Categoria Normalização
# ============================

# Mapas de alias para nomes canônicos de categorias. As chaves devem estar
# normalizadas (sem acentos e em minúsculas) para facilitar a busca.
CATEGORY_ALIASES = {
    "programacao": "Programação",
}


def normalize_category(name: str) -> Optional[str]:
    """Retorna o nome canônico de uma categoria.

    A comparação ignora acentos e diferenças de maiúsculas/minúsculas.
    Se a categoria ou um de seus aliases for encontrado, devolve o nome
    oficial; caso contrário, ``None``.
    """

    normalized = unidecode(name).lower()

    for canonical in Config.CATEGORIES:
        if unidecode(canonical).lower() == normalized:
            return canonical

    return CATEGORY_ALIASES.get(normalized)


# ============================
# 📊 Logging Avançado
# ============================
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: grey + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class JsonFormatter(logging.Formatter):
    """Simple JSON formatter with standard fields."""

    def format(self, record: logging.LogRecord) -> str:
        rec = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "file": record.filename,
            "line": record.lineno,
        }
        return json.dumps(rec, ensure_ascii=False)


class LokiHandler(logging.Handler):
    """Simple handler that sends logs to a Loki HTTP endpoint."""

    def __init__(self, url: str):
        super().__init__()
        self.url = url.rstrip("/")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if isinstance(record.msg, dict):
                data = record.msg
            else:
                data = {"message": record.getMessage()}
            data.update(
                {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "level": record.levelname,
                    "name": record.name,
                    "file": record.filename,
                    "line": record.lineno,
                }
            )
            line = json.dumps(data, ensure_ascii=False)
            payload = {
                "streams": [
                    {
                        "labels": '{job="scraper"}',
                        "entries": [
                            {
                                "ts": data["timestamp"],
                                "line": line,
                            }
                        ],
                    }
                ]
            }
            requests.post(self.url, json=payload, timeout=1)
        except requests.RequestException as exc:
            logger.warning(
                "Failed to send log to Loki",
                extra={"error_type": type(exc).__name__, "error_message": str(exc)},
            )


class ElasticsearchHandler(logging.Handler):
    """Send logs to an Elasticsearch index."""

    def __init__(self, url: str, index: str = "scraper"):
        super().__init__()
        self.url = url.rstrip("/")
        self.index = index

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if isinstance(record.msg, dict):
                doc = record.msg
            else:
                doc = {"message": record.getMessage()}
            doc.update(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": record.levelname,
                    "name": record.name,
                    "file": record.filename,
                    "line": record.lineno,
                }
            )
            requests.post(f"{self.url}/{self.index}/_doc", json=doc, timeout=1)
        except requests.RequestException as exc:
            logger.warning(
                "Failed to send log to Elasticsearch",
                extra={"error_type": type(exc).__name__, "error_message": str(exc)},
            )


def setup_logger(name, log_file, level: int = logging.INFO, fmt: str = "text"):
    """Configure and return a logger with console and file handlers."""
    os.makedirs(Config.LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = JsonFormatter() if fmt == "json" else CustomFormatter()

    handler = logging.FileHandler(os.path.join(Config.LOG_DIR, log_file))
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if Config.LOG_SERVICE_URL:
        if Config.LOG_SERVICE_TYPE == "elastic":
            remote = ElasticsearchHandler(Config.LOG_SERVICE_URL)
        else:
            remote = LokiHandler(Config.LOG_SERVICE_URL)
        remote.setFormatter(formatter)
        logger.addHandler(remote)

    if structlog:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(level),
            processors=[
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
            ],
        )
        wrapped = structlog.wrap_logger(
            logger,
            processors=[
                (
                    structlog.processors.JSONRenderer()
                    if fmt == "json"
                    else structlog.processors.KeyValueRenderer()
                )
            ],
        )
        return wrapped

    return logger


logger = setup_logger("wiki_scraper", "scraper.log")


def log_error(message: str, exc: Exception) -> None:
    """Log ``message`` with error details in structured form."""
    logger.error(
        message, extra={"error_type": type(exc).__name__, "error_message": str(exc)}
    )


def log_failed_url(url: str) -> None:
    """Append a failing URL to ``failed_urls.log`` within :data:`Config.LOG_DIR`."""
    try:
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        with open(os.path.join(Config.LOG_DIR, "failed_urls.log"), "a") as fh:
            fh.write(f"{url}\n")
    except OSError as exc:
        logger.warning(
            "Failed to write failed URL log",
            extra={"error_type": type(exc).__name__, "error_message": str(exc)},
        )


# ============================
# 🧠 Cache Inteligente
# ============================


class CacheBackend(Protocol):
    def get(self, key: str): ...

    def set(self, key: str, data, ttl: Optional[int] = None): ...

    def stats(self) -> dict: ...


class FileCache(CacheBackend):
    def __init__(self):
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _get_cache_path(self, key: str) -> str:
        hash_key = hashlib.md5(key.encode("utf-8")).hexdigest()
        return os.path.join(Config.CACHE_DIR, f"{hash_key}.pkl.gz")

    @backoff.on_exception(backoff.expo, HANDLED_EXCEPTIONS, max_tries=3)
    def get(self, key: str):
        cache_file = self._get_cache_path(key)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    compressed_data = f.read()
                data = pickle.loads(zlib.decompress(compressed_data))
                self.hits += 1
                return data
            except HANDLED_EXCEPTIONS as e:
                log_error(f"Erro ao ler cache {key}", e)
                os.remove(cache_file)

        self.misses += 1
        return None

    @backoff.on_exception(backoff.expo, HANDLED_EXCEPTIONS, max_tries=3)
    def set(self, key: str, data, ttl: Optional[int] = None):
        cache_file = self._get_cache_path(key)
        try:
            compressed_data = zlib.compress(pickle.dumps(data))
            temp_file = cache_file + ".tmp"
            with open(temp_file, "wb") as f:
                f.write(compressed_data)
            os.replace(temp_file, cache_file)
        except HANDLED_EXCEPTIONS as e:
            log_error(f"Erro ao salvar cache {key}", e)
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def stats(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (
                self.hits / (self.hits + self.misses)
                if (self.hits + self.misses) > 0
                else 0
            ),
        }


class RedisCache(CacheBackend):
    def __init__(self, url: str):
        try:
            import redis  # type: ignore
        except ImportError as exc:  # pragma: no cover - import error
            raise ImportError("redis package required for RedisCache") from exc

        self.client = redis.Redis.from_url(url)
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        val = self.client.get(key)
        if val is not None:
            try:
                data = pickle.loads(zlib.decompress(val))
            except (pickle.UnpicklingError, zlib.error, OSError) as exc:
                self.client.delete(key)
                self.misses += 1
                log_error(f"Erro ao ler cache {key}", exc)
                return None
            self.hits += 1
            return data
        self.misses += 1
        return None

    def set(self, key: str, data, ttl: Optional[int] = None):
        val = zlib.compress(pickle.dumps(data))
        if ttl:
            self.client.setex(key, ttl, val)
        else:
            self.client.set(key, val)

    def stats(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (
                self.hits / (self.hits + self.misses)
                if (self.hits + self.misses) > 0
                else 0
            ),
        }


class SQLiteCache(CacheBackend):
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value BLOB, expires_at INTEGER)"
        )
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        cur = self.conn.execute(
            "SELECT value, expires_at FROM cache WHERE key=?", (key,)
        )
        row = cur.fetchone()
        if row:
            value, exp = row
            if exp is not None and exp < int(time.time()):
                self.conn.execute("DELETE FROM cache WHERE key=?", (key,))
                self.conn.commit()
                self.misses += 1
                return None
            try:
                data = pickle.loads(zlib.decompress(value))
            except (pickle.UnpicklingError, zlib.error, sqlite3.Error) as exc:
                self.conn.execute("DELETE FROM cache WHERE key=?", (key,))
                self.conn.commit()
                self.misses += 1
                log_error(f"Erro ao ler cache {key}", exc)
                return None
            self.hits += 1
            return data
        self.misses += 1
        return None

    def set(self, key: str, data, ttl: Optional[int] = None):
        val = zlib.compress(pickle.dumps(data))
        exp = int(time.time()) + ttl if ttl else None
        self.conn.execute(
            "REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
            (key, val, exp),
        )
        self.conn.commit()

    def stats(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (
                self.hits / (self.hits + self.misses)
                if (self.hits + self.misses) > 0
                else 0
            ),
        }


def init_cache() -> CacheBackend:
    if Config.CACHE_BACKEND == "redis":
        return RedisCache(Config.REDIS_URL)
    if Config.CACHE_BACKEND == "sqlite":
        return SQLiteCache(Config.SQLITE_PATH)
    return FileCache()


cache: CacheBackend = init_cache()


def clear_cache() -> None:
    """Remove arquivos ou registros expirados do cache."""
    ttl = Config.CACHE_TTL
    if Config.CACHE_BACKEND == "sqlite":
        if not os.path.exists(Config.SQLITE_PATH):
            return
        conn = sqlite3.connect(Config.SQLITE_PATH)
        conn.execute(
            "DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at < ?",
            (int(time.time()),),
        )
        conn.commit()
        conn.close()
    elif Config.CACHE_BACKEND == "file":
        if ttl is None:
            return
        cutoff = time.time() - ttl
        path = Path(Config.CACHE_DIR)
        if path.exists():
            for p in path.glob("*.pkl.gz"):
                if p.stat().st_mtime < cutoff:
                    p.unlink()
    else:
        # Redis remove chaves expiradas automaticamente
        pass


# Global rate limiter for all network operations
rate_limiter = RateLimiter(Config.RATE_LIMIT_DELAY)
# Limit the number of concurrent HTTP requests
fetch_semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_REQUESTS)


class AsyncRunner:
    """Maintain a shared process pool for async operations."""

    def __init__(self) -> None:
        self._proc_pool: ProcessPoolExecutor | None = None

    def get_process_pool(self) -> ProcessPoolExecutor:
        if self._proc_pool is None:
            self._proc_pool = ProcessPoolExecutor(max_workers=Config.MAX_PROCESSES)
        return self._proc_pool

    def shutdown(self) -> None:
        if self._proc_pool:
            self._proc_pool.shutdown()
            self._proc_pool = None


async_runner = AsyncRunner()


# ============================
# 🔍 Funções Avançadas de NLP
# ============================
class NLPProcessor:
    _instances = {}

    @classmethod
    def get_instance(cls, lang: str):
        if lang not in cls._instances:
            if lang in Config.NLP_MODELS:
                try:
                    if Config.USE_GPU:
                        try:
                            spacy.require_gpu()
                        except Exception as e:  # pragma: no cover - optional
                            logger.warning(f"GPU não disponível para spaCy: {e}")
                    cls._instances[lang] = spacy.load(Config.NLP_MODELS[lang])
                    logger.info(f"Carregado modelo NLP para {lang}")
                except OSError:
                    logger.warning(
                        f"Modelo NLP para {lang} não encontrado, tentando modelo pequeno"
                    )
                    try:
                        cls._instances[lang] = spacy.load(
                            Config.NLP_MODELS[lang].replace("_lg", "_sm")
                        )
                    except OSError:
                        logger.warning(
                            f"Modelos lg e sm para {lang} indisponíveis, usando 'en_core_web_sm'"
                        )
                        cls._instances[lang] = spacy.load("en_core_web_sm")
            else:
                logger.warning(f"Modelo NLP para {lang} não configurado, usando inglês")
                cls._instances[lang] = spacy.load(Config.NLP_MODELS["en"])
        return cls._instances[lang]

    @classmethod
    def get_embedding_model(cls):
        if not hasattr(cls, "_embedding_model"):
            cls._embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
            if Config.USE_GPU:
                try:
                    cls._embedding_model = cls._embedding_model.to("cuda")
                except Exception as e:  # pragma: no cover - optional
                    logger.warning(f"Não foi possível mover modelo para GPU: {e}")
        return cls._embedding_model


def extract_keywords(text: str, lang: str = "en", n: int = 10) -> List[str]:
    try:
        nlp = NLPProcessor.get_instance(lang)
        doc = nlp(text)

        # Filtra substantivos e nomes próprios
        keywords = [
            chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3
        ]

        # Remove duplicatas e conta frequência
        freq = {}
        for word in keywords:
            word_lower = word.lower()
            if word_lower in freq:
                freq[word_lower] += 1
            else:
                freq[word_lower] = 1

        # Ordena por frequência e pega os top N
        sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [kw[0] for kw in sorted_keywords[:n]]

    except Exception as e:
        logger.error(f"Erro ao extrair keywords: {e}")
        return []


def summarize_text(text: str, lang: str = "en") -> str:
    try:
        parser = PlaintextParser.from_string(text, Tokenizer(lang))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, Config.SUMMARY_SENTENCES)
        return " ".join([str(sentence) for sentence in summary])
    except Exception as e:
        logger.error(f"Erro ao sumarizar texto: {e}")
        return (
            text[: Config.MIN_TEXT_LENGTH]
            if len(text) > Config.MIN_TEXT_LENGTH
            else text
        )


def cluster_texts(texts: List[str], k: int = Config.CLUSTER_K) -> np.ndarray:
    try:
        model = NLPProcessor.get_embedding_model()
        embeddings = model.encode(texts, show_progress_bar=False)

        # Normaliza os embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms

        # Clusteriza usando K-means
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(normalized_embeddings)

        return clusters
    except Exception as e:
        logger.error(f"Erro no clustering: {e}")
        return np.zeros(len(texts))


# ============================
# 🧹 Limpeza Avançada de Texto
# ============================
def advanced_clean_text(
    text: str,
    lang: str = "en",
    remove_stopwords: bool = False,
    split: bool = False,
    stem: bool = False,
) -> str | list[str]:
    """Clean wiki text with optional stopword removal, stemming and sentence splitting."""
    try:
        text = clean_wiki_text(text)

        # Remove caracteres especiais, mantendo acentos quando relevante
        if lang in ["en", "de"]:
            text = unidecode(text)

        # Remove Wikipedia-specific markup in a single pass
        text = _CLEAN_COMBINED_RE.sub("", text)

        # Remove specific sections
        text = _SECTION_RE.sub("", text)

        if remove_stopwords or stem:
            removed = False
            try:
                nlp = NLPProcessor.get_instance(lang)
                doc = nlp(text)
                tokens = []
                for t in doc:
                    if remove_stopwords and getattr(t, "is_stop", False):
                        continue
                    tokens.append(t.lemma_ if stem else t.text)
                text = " ".join(tokens)
                removed = True
            except Exception as e:
                logger.error(f"Erro ao processar texto com spaCy: {e}")
            if not removed:
                try:
                    import nltk
                    from nltk.corpus import stopwords
                    from nltk.stem import SnowballStemmer, PorterStemmer

                    stop_words = set()
                    if remove_stopwords:
                        stop_words = set(stopwords.words(lang))
                    if stem:
                        try:
                            stemmer = SnowballStemmer(lang)
                        except Exception:
                            stemmer = PorterStemmer()
                    tokens = []
                    for w in text.split():
                        if remove_stopwords and w.lower() in stop_words:
                            continue
                        if stem:
                            tokens.append(stemmer.stem(w))
                        else:
                            tokens.append(w)
                    text = " ".join(tokens)
                except Exception as e:
                    logger.error(f"Erro ao processar texto com NLTK: {e}")

        text = text.strip()
        if split:
            return split_sentences(text, lang)
        return text
    except Exception as e:
        logger.error(f"Erro na limpeza de texto: {e}")
        return text


def extract_main_content(page_html: str) -> str:
    try:
        soup = BeautifulSoup(page_html, "html.parser")

        # Remove elementos indesejados
        for element in soup.find_all(
            [
                "table",
                "div.infobox",
                "span.reference",
                "ol.references",
                "div.navbox",
                "div.hatnote",
                "div.thumb",
                "div.notice",
            ]
        ):
            element.decompose()

        # Extrai conteúdo principal
        content_div = soup.find("div", {"id": "mw-content-text"})
        if content_div:
            return str(content_div)
        return page_html
    except Exception as e:
        logger.error(f"Erro ao extrair conteúdo principal: {e}")
        return page_html


def extract_links(html: str, base_url: str) -> List[str]:
    """Return absolute links from ``html`` that contain ``/wiki/`` in the href."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/wiki/" in href:
                links.append(urljoin(base_url, href))
        return links
    except Exception as e:
        logger.error(f"Erro ao extrair links: {e}")
        return []


def extract_infobox(html: str) -> Dict[str, str]:
    """Return a dictionary with key/value pairs from the first infobox table."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        box = soup.find("table", class_=lambda c: c and "infobox" in c)
        if not box:
            return {}
        data: Dict[str, str] = {}
        for row in box.find_all("tr"):
            header = row.find("th")
            value = row.find("td")
            if header and value:
                key = header.get_text(strip=True)
                val = value.get_text(strip=True)
                data[key] = val
        return data
    except Exception as e:
        logger.error(f"Erro ao extrair infobox: {e}")
        return {}


def extract_tables(html: str) -> List[List[List[str]]]:
    """Return all HTML tables as lists of rows and cells."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        tables: List[List[List[str]]] = []
        for table in soup.find_all("table"):
            rows: List[List[str]] = []
            for tr in table.find_all("tr"):
                cells = [c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
                if cells:
                    rows.append(cells)
            if rows:
                tables.append(rows)
        return tables
    except Exception as e:
        logger.error(f"Erro ao extrair tabelas: {e}")
        return []


def extract_images(html: str) -> List[Dict[str, str]]:
    """Return image URLs and captions from thumbnail/figure elements."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        results: List[Dict[str, str]] = []

        for div in soup.find_all("div", class_=lambda c: c and "thumb" in c.split()):
            img = div.find("img")
            if not img or not img.get("src"):
                continue
            url = img["src"]
            if url.startswith("//"):
                url = "https:" + url
            caption_tag = div.find(class_="thumbcaption")
            caption = caption_tag.get_text(strip=True) if caption_tag else ""
            results.append({"image_url": url, "caption": caption})

        for fig in soup.find_all("figure"):
            img = fig.find("img")
            if not img or not img.get("src"):
                continue
            url = img["src"]
            if url.startswith("//"):
                url = "https:" + url
            caption_tag = fig.find("figcaption")
            caption = caption_tag.get_text(strip=True) if caption_tag else ""
            results.append({"image_url": url, "caption": caption})

        return results
    except Exception as e:
        logger.error(f"Erro ao extrair imagens: {e}")
        return []


def extract_videos(html: str) -> List[str]:
    """Return video URLs from ``html`` content."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        videos: List[str] = []
        for vid in soup.find_all("video"):
            src = vid.get("src") or ""
            if not src:
                source = vid.find("source")
                if source and source.get("src"):
                    src = source["src"]
            if src:
                if src.startswith("//"):
                    src = "https:" + src
                videos.append(src)
        for iframe in soup.find_all("iframe"):
            src = iframe.get("src") or ""
            if any(domain in src for domain in ["youtube.com", "vimeo.com"]):
                if src.startswith("//"):
                    src = "https:" + src
                videos.append(src)
        return videos
    except Exception as e:
        logger.error(f"Erro ao extrair videos: {e}")
        return []


def dump_raw_page(
    title: str,
    lang: str,
    category: str,
    html: str,
    text: str,
    output_dir: str = Config.RAW_DIR,
) -> None:
    """Persist raw page ``html`` and ``text`` to ``output_dir``."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", title)[:50]
        path = Path(output_dir) / f"{lang}_{safe}.json"
        data = {
            "title": title,
            "lang": lang,
            "category": category,
            "html": html,
            "text": text,
        }
        tmp = str(path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as exc:  # pragma: no cover - filesystem errors
        logger.error(f"Erro ao salvar RAW para {title}: {exc}")


def is_captcha_page(html: str) -> bool:
    """Return ``True`` if the page appears to be a CAPTCHA challenge."""
    lowered = html.lower()
    patterns = ["captcha", "g-recaptcha", "cloudflare"]
    return any(p in lowered for p in patterns)


class CaptchaDetected(Exception):
    """Raised when a CAPTCHA page is detected."""


class TooManyRequests(CaptchaDetected):
    """Raised when a HTTP 429 status is encountered."""


async def fetch_with_retry(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    retries: int = Config.RETRIES,
    proxy: str | None = None,
) -> tuple[int, str]:
    """Fetch a URL using ``aiohttp`` with tenacity-based retries."""

    if proxy is None:
        proxy = get_next_proxy()
    if headers is None:
        headers = {"User-Agent": Config.get_random_user_agent()}

    async def _attempt() -> tuple[int, str]:
        timeout = aiohttp.ClientTimeout(total=Config.TIMEOUT)
        acquired_dummy = False
        if (
            hasattr(fetch_semaphore, "active")
            and hasattr(fetch_semaphore, "limit")
            and not hasattr(fetch_semaphore, "acquire")
        ):
            while fetch_semaphore.active >= fetch_semaphore.limit:
                await asyncio.sleep(0)
            await fetch_semaphore.__aenter__()
            acquired_dummy = True
        else:
            await fetch_semaphore.acquire()
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    url, params=params, headers=headers, proxy=proxy
                ) as resp:
                    if resp.status == 429:
                        metrics.scrape_block.inc()
                        metrics.requests_429_total.inc()
                        rate_limiter.record_error()
                        raise TooManyRequests()
                    if resp.status == 403:
                        metrics.scrape_block.inc()
                        rate_limiter.record_error()
                        raise CaptchaDetected()
                    if 500 <= resp.status < 600:
                        raise aiohttp.ClientResponseError(
                            resp.request_info,
                            resp.history,
                            status=resp.status,
                            message=resp.reason,
                            headers=resp.headers,
                        )
                    resp.raise_for_status()
                    text = await resp.text()
                    if is_captcha_page(text):
                        rate_limiter.record_error()
                        raise CaptchaDetected()
                    rate_limiter.record_success()
                    return resp.status, text
        finally:
            if acquired_dummy:
                await fetch_semaphore.__aexit__(None, None, None)
            else:
                fetch_semaphore.release()

    def _wait_strategy(retry_state: tenacity.RetryCallState) -> float:
        exc = retry_state.outcome.exception()
        if isinstance(exc, TooManyRequests):
            return tenacity.wait_random_exponential(multiplier=5, max=10)(retry_state)
        return tenacity.wait_random_exponential(max=10)(retry_state)

    retryer = tenacity.AsyncRetrying(
        stop=tenacity.stop_after_attempt(retries),
        wait=_wait_strategy,
        retry=tenacity.retry_if_exception_type(
            (aiohttp.ClientError, asyncio.TimeoutError, CaptchaDetected)
        ),
        reraise=True,
    )
    exc: Exception | None = None
    async for attempt in retryer:
        with attempt:
            try:
                return await _attempt()
            except CaptchaDetected:
                proxy = get_next_proxy()
                headers["User-Agent"] = Config.get_random_user_agent()
                exc = CaptchaDetected()
                raise
            except Exception as e:
                proxy = get_next_proxy()
                headers["User-Agent"] = Config.get_random_user_agent()
                exc = e
                raise

    log_failed_url(url)
    metrics.requests_failed_total.inc()
    raise exc  # type: ignore[misc]


def fetch_html_content(title: str, lang: str) -> str:
    """Retrieve the raw HTML for a Wikipedia page using the REST API."""

    return asyncio.run(fetch_html_content_async(title, lang))


async def fetch_html_content_async(title: str, lang: str) -> str:
    """Asynchronous version of :func:`fetch_html_content`."""
    cache_key = f"html_{lang}_{title}"
    cached = cache.get(cache_key)

    url = f"{get_base_url(lang)}/api/rest_v1/page/html/{title}"
    headers = {"User-Agent": Config.get_random_user_agent()}

    try:
        await rate_limiter.async_wait()
        _, html = await fetch_with_retry(url, headers=headers)
        metrics.scrape_success.inc()
        cache.set(cache_key, html, ttl=Config.CACHE_TTL)
        return html
    except Exception as e:
        metrics.scrape_error.inc()
        logger.error(f"Erro ao buscar HTML para {title}: {e}")
        if cached is not None:
            return cached
        return ""


async def fetch_all(urls: List[str]) -> List[str]:
    """Fetch multiple URLs concurrently using ``fetch_with_retry``."""
    results = await asyncio.gather(*(fetch_with_retry(u) for u in urls))
    return [text for _, text in results]


def fetch_pageviews(title: str, lang: str) -> int:
    """Return the total pageviews for ``title`` in ``lang`` over ``Config.PAGEVIEW_DAYS`` days."""

    return asyncio.run(fetch_pageviews_async(title, lang))


async def fetch_pageviews_async(title: str, lang: str) -> int:
    """Asynchronous helper for :func:`fetch_pageviews`."""

    cache_key = f"pageviews_{lang}_{title}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    end = (datetime.utcnow() - timedelta(days=1)).strftime("%Y%m%d")
    start = (datetime.utcnow() - timedelta(days=Config.PAGEVIEW_DAYS)).strftime(
        "%Y%m%d"
    )
    quoted = requests.utils.quote(title, safe="")
    url = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"{lang}.wikipedia/all-access/user/{quoted}/daily/{start}/{end}"
    )
    await rate_limiter.async_wait()
    try:
        status, text = await fetch_with_retry(
            url, headers={"User-Agent": Config.get_random_user_agent()}
        )
    except Exception:
        cache.set(cache_key, 0, ttl=Config.CACHE_TTL)
        return 0

    if status != 200:
        cache.set(cache_key, 0, ttl=Config.CACHE_TTL)
        return 0

    try:
        data = json.loads(text)
        views = sum(int(it.get("views", 0)) for it in data.get("items", []))
    except Exception:
        views = 0

    cache.set(cache_key, views, ttl=Config.CACHE_TTL)
    return views


def search_category(keyword: str, lang: str) -> Optional[str]:
    """Search for a similar category name using the Wikipedia API."""
    cache_key = f"search_category_{lang}_{keyword}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    url = f"{get_base_url(lang)}/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srnamespace": 14,
        "srsearch": keyword,
        "format": "json",
    }

    async def _async_search() -> Optional[str]:
        await rate_limiter.async_wait()
        status, text = await fetch_with_retry(
            url, params=params, headers={"User-Agent": Config.get_random_user_agent()}
        )
        if status == 429:
            rate_limiter.record_error()
        data = json.loads(text)
        results = data.get("query", {}).get("search", [])
        if results:
            title = results[0].get("title", "")
            title = title.replace("Category:", "").replace("Categoria:", "")
            cache.set(cache_key, title, ttl=Config.CACHE_TTL)
            return title
        return None

    try:
        return asyncio.run(_async_search()) or cache.get(cache_key)
    except Exception as e:  # pragma: no cover - network issues
        rate_limiter.record_error()
        logger.error(f"Erro ao buscar categorias para {keyword}: {e}")
        return cache.get(cache_key)


def get_revision_history(title: str, lang: str, limit: int) -> List[dict]:
    """Return revision metadata for a page using the Wikipedia API."""

    cache_key = f"revisions_{lang}_{title}_{limit}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    url = f"{get_base_url(lang)}/w/api.php"
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": title,
        "rvlimit": limit,
        "rvprop": "timestamp|user",
        "format": "json",
    }

    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.RequestException,
        max_tries=Config.RETRIES,
    )
    def _request() -> List[dict]:
        rate_limiter.wait()
        resp = requests.get(
            url,
            params=params,
            headers={"User-Agent": Config.get_random_user_agent()},
            timeout=Config.TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        revs: List[dict] = []
        for page in pages.values():
            for rev in page.get("revisions", []):
                revs.append(
                    {
                        "timestamp": rev.get("timestamp"),
                        "user": rev.get("user"),
                    }
                )
        return revs

    try:
        revisions = _request()
        cache.set(cache_key, revisions, ttl=Config.CACHE_TTL)
        return revisions
    except Exception as e:  # pragma: no cover - network issues
        logger.error(f"Erro ao buscar revisões de {title}: {e}")
        fallback = cache.get(cache_key)
        if fallback is not None:
            return fallback
        return []


# ============================
# 🔗 Coletor Avançado com Retry
# ============================
class WikipediaAdvanced:
    def __init__(self, lang: str):
        self.lang = lang
        self.wiki = wikipediaapi.Wikipedia(
            language=lang,
            extract_format=wikipediaapi.ExtractFormat.HTML,
            user_agent=Config.get_random_user_agent(),
        )
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": Config.get_random_user_agent()})

    def _prepare_session(self):
        """Refresh User-Agent and proxy settings before a request."""
        self.session.headers["User-Agent"] = Config.get_random_user_agent()
        if Config.PROXIES:
            self.session.proxies = get_next_proxy()
        else:
            self.session.proxies = {}

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, wikipediaapi.WikipediaException),
        max_tries=Config.RETRIES,
    )
    def fetch_page(self, page_title: str) -> Optional[wikipediaapi.WikipediaPage]:
        cache_key = f"page_{self.lang}_{page_title}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        self._prepare_session()
        rate_limiter.wait()

        try:
            page = self.wiki.page(page_title)
            if page.exists():
                # Melhora a qualidade do conteúdo
                page._fullurl = self.wiki.api.article_url(page_title)
                rate_limiter.wait()
                page._html = fetch_html_content(page_title, self.lang)
                page._html = extract_main_content(page._html)

                cache.set(cache_key, page, ttl=Config.CACHE_TTL)
                return page
        except Exception as e:
            rate_limiter.record_error()
            logger.error(f"Erro ao buscar página {page_title}: {e}")
            log_failed_url(self.wiki.api.article_url(page_title))
            raise

        return None

    async def fetch_page_async(
        self, page_title: str
    ) -> Optional[wikipediaapi.WikipediaPage]:
        """Asynchronous version of :meth:`fetch_page` using ``aiohttp``."""
        cache_key = f"page_{self.lang}_{page_title}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        await rate_limiter.async_wait()

        try:
            page = await asyncio.to_thread(self.wiki.page, page_title)
            if page.exists():
                page._fullurl = self.wiki.api.article_url(page_title)
                await rate_limiter.async_wait()
                page._html = await fetch_html_content_async(page_title, self.lang)
                page._html = extract_main_content(page._html)
                cache.set(cache_key, page, ttl=Config.CACHE_TTL)
                return page
        except Exception as e:
            rate_limiter.record_error()
            logger.error(f"Erro ao buscar página {page_title}: {e}")
            log_failed_url(self.wiki.api.article_url(page_title))
            raise

        return None

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, wikipediaapi.WikipediaException),
        max_tries=Config.RETRIES,
    )
    def fetch_category(
        self, category_name: str
    ) -> Optional[wikipediaapi.WikipediaPage]:
        category_title = (
            f"Category:{category_name}"
            if self.lang != "pt"
            else f"Categoria:{category_name}"
        )
        self._prepare_session()
        rate_limiter.wait()
        return self.fetch_page(category_title)

    def get_links_from_category_page(self, category_name: str) -> List[str]:
        """Return wiki links from the HTML of a category page."""
        try:
            title = (
                f"Category:{category_name}"
                if self.lang != "pt"
                else f"Categoria:{category_name}"
            )
            html = fetch_html_content(title, self.lang)
            base_url = get_base_url(self.lang)
            return extract_links(html, base_url)
        except Exception as e:
            logger.error(f"Erro ao obter links da categoria {category_name}: {e}")
            return []

    def should_queue(self, title: str) -> bool:
        """Return ``True`` if the page should be queued for processing."""

        try:
            page = self.fetch_page(title)
        except Exception:
            return False

        if not page or not page.exists():
            return False

        categories = getattr(page, "categories", {})
        for cat in categories:
            lowered = cat.lower()
            if "disambiguation" in lowered or "desambigua" in lowered:
                return False

        if len(clean_text(page.text)) < Config.MIN_TEXT_LENGTH:
            return False

        views = fetch_pageviews(title, self.lang)
        return views >= Config.MIN_PAGEVIEWS

    def get_related_pages(self, page_title: str, limit: int = 10) -> List[dict]:
        cache_key = f"related_{self.lang}_{page_title}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        self._prepare_session()

        @backoff.on_exception(
            backoff.expo,
            (requests.exceptions.RequestException, wikipediaapi.WikipediaException),
            max_tries=Config.RETRIES,
        )
        def _request():
            url = f"{get_base_url(self.lang)}/w/api.php"
            params = {
                "action": "query",
                "titles": page_title,
                "prop": "links",
                "pllimit": limit,
                "format": "json",
            }
            rate_limiter.wait()
            response = self.session.get(url, params=params, timeout=Config.TIMEOUT)
            response.raise_for_status()
            return response.json()

        try:
            data = _request()

            pages = data.get("query", {}).get("pages", {})
            links = []

            for page in pages.values():
                for link in page.get("links", []):
                    if "ns" in link and link["ns"] == 0:  # Only main namespace
                        links.append({"title": link["title"], "lang": self.lang})

            cache.set(cache_key, links, ttl=Config.CACHE_TTL)
            return links
        except Exception as e:
            logger.error(f"Erro ao buscar páginas relacionadas para {page_title}: {e}")
            log_failed_url(f"{get_base_url(self.lang)}/w/api.php")
            fallback = cache.get(cache_key)
            if fallback is not None:
                return fallback
            return []

    def get_revision_history(self, page_title: str, limit: int = 5) -> List[dict]:
        """Wrapper around :func:`get_revision_history` for this language."""

        return get_revision_history(page_title, self.lang, limit)

    def get_category_members(
        self, category_name: str, depth: int = 0, visited: Optional[Set[str]] = None
    ) -> List[dict]:
        if visited is None:
            visited = set()

        category = self.fetch_category(category_name)
        if not category or not category.exists():
            alt = search_category(category_name, self.lang)
            if alt:
                category_name = alt
                category = self.fetch_category(category_name)

        if not category or not category.exists():
            logger.warning(f"Categoria não encontrada: {category_name}")
            return []

        members = []
        for member in category.categorymembers.values():
            if member.title not in visited:
                visited.add(member.title)

                if (
                    member.ns == wikipediaapi.Namespace.CATEGORY
                    and depth < Config.MAX_DEPTH
                ):
                    sub_members = self.get_category_members(
                        member.title.replace("Category:", "").replace("Categoria:", ""),
                        depth + 1,
                        visited,
                    )
                    members.extend(sub_members)
                elif member.ns == wikipediaapi.Namespace.MAIN:
                    if self.should_queue(member.title):
                        members.append(
                            {
                                "title": member.title,
                                "url": member.fullurl,
                                "lang": self.lang,
                                "category": category_name,
                                "depth": depth,
                            }
                        )

                if len(members) >= Config.MAX_PAGES_PER_CATEGORY:
                    break

        return members

    async def get_category_members_async(
        self,
        category_name: str,
        depth: int = 0,
        visited: Optional[Set[str]] = None,
        max_concurrency: Optional[int] = None,
    ) -> List[dict]:
        """Asynchronously fetch category members recursively."""
        if visited is None:
            visited = set()

        max_concurrency = max_concurrency or Config.MAX_CONCURRENT_REQUESTS
        sem = asyncio.Semaphore(max_concurrency)

        category = await asyncio.to_thread(self.fetch_category, category_name)
        if not category or not category.exists():
            alt = search_category(category_name, self.lang)
            if alt:
                category_name = alt
                category = await asyncio.to_thread(self.fetch_category, category_name)

        if not category or not category.exists():
            logger.warning(f"Categoria não encontrada: {category_name}")
            return []

        members: List[dict] = []

        async def handle_member(member):
            if member.title in visited:
                return []
            visited.add(member.title)

            if (
                member.ns == wikipediaapi.Namespace.CATEGORY
                and depth < Config.MAX_DEPTH
            ):
                async with sem:
                    return await self.get_category_members_async(
                        member.title.replace("Category:", "").replace("Categoria:", ""),
                        depth + 1,
                        visited,
                        max_concurrency,
                    )
            elif member.ns == wikipediaapi.Namespace.MAIN:
                if self.should_queue(member.title):
                    return [
                        {
                            "title": member.title,
                            "url": member.fullurl,
                            "lang": self.lang,
                            "category": category_name,
                            "depth": depth,
                        }
                    ]
                return []
            return []

        tasks = [handle_member(m) for m in category.categorymembers.values()]
        for res in await asyncio.gather(*tasks):
            members.extend(res)
            if len(members) >= Config.MAX_PAGES_PER_CATEGORY:
                break

        return members

    def crawl_links(
        self,
        start_page: str,
        depth: int = 1,
        visited: Optional[Set[str]] = None,
    ) -> List[dict]:
        """Breadth-first crawl of ``/wiki/`` links starting from ``start_page``.

        Parameters
        ----------
        start_page: str
            Title of the initial page to crawl.
        depth: int
            Maximum link depth to follow.
        visited: Optional[Set[str]]
            Optional set of already visited titles to avoid repeats.

        Returns
        -------
        List[dict]
            List of pages in the same format as :meth:`get_category_members`.
        """
        if visited is None:
            visited = set()

        queue: List[Tuple[str, int]] = [(start_page, 0)]
        results: List[dict] = []
        base_url = get_base_url(self.lang)

        while queue and len(results) < Config.MAX_PAGES_PER_CATEGORY:
            current, cur_depth = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            try:
                html = fetch_html_content(current, self.lang)
                links = extract_links(html, base_url)
            except Exception as e:
                logger.error(f"Erro ao rastrear links de {current}: {e}")
                continue

            for link in links:
                if len(results) >= Config.MAX_PAGES_PER_CATEGORY:
                    break
                parsed = urlparse(link)
                if "/wiki/" not in parsed.path:
                    continue
                title = parsed.path.split("/wiki/")[-1]
                title = requests.utils.unquote(title).replace("_", " ")
                if title in visited:
                    continue
                if self.should_queue(title):
                    results.append(
                        {
                            "title": title,
                            "url": link,
                            "lang": self.lang,
                            "category": start_page,
                            "depth": cur_depth,
                        }
                    )
                    if cur_depth < depth:
                        queue.append((title, cur_depth + 1))

        return results

    async def crawl_links_async(
        self,
        start_page: str,
        depth: int = 1,
        visited: Optional[Set[str]] = None,
    ) -> List[dict]:
        """Asynchronous version of :meth:`crawl_links`."""

        if visited is None:
            visited = set()

        queue: List[Tuple[str, int]] = [(start_page, 0)]
        results: List[dict] = []
        base_url = get_base_url(self.lang)

        while queue and len(results) < Config.MAX_PAGES_PER_CATEGORY:
            current, cur_depth = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            try:
                html = await fetch_html_content_async(current, self.lang)
                links = extract_links(html, base_url)
            except Exception as e:
                logger.error(f"Erro ao rastrear links de {current}: {e}")
                continue

            for link in links:
                if len(results) >= Config.MAX_PAGES_PER_CATEGORY:
                    break
                parsed = urlparse(link)
                if "/wiki/" not in parsed.path:
                    continue
                title = parsed.path.split("/wiki/")[-1]
                title = requests.utils.unquote(title).replace("_", " ")
                if title in visited:
                    continue
                if self.should_queue(title):
                    results.append(
                        {
                            "title": title,
                            "url": link,
                            "lang": self.lang,
                            "category": start_page,
                            "depth": cur_depth,
                        }
                    )
                    if cur_depth < depth:
                        queue.append((title, cur_depth + 1))

        return results


# ============================
# 🏗️ Builder de Dataset Profissional
# ============================


def cpu_process_page(
    title: str,
    content: str,
    lang: str,
    category: str,
    images: List[Dict[str, str]] | None = None,
    videos: List[str] | None = None,
    revisions: List[dict] | None = None,
    rev_limit: int = 5,
    translate_to: str | None = None,
) -> dict:
    """Executes CPU intensive operations for a page."""
    builder = DatasetBuilder(
        include_revisions=revisions, rev_limit=rev_limit, translate_to=translate_to
    )
    summary = summarize_text(content, lang)
    record = builder.generate_qa_pairs(
        title=title,
        content=content,
        summary=summary,
        lang=lang,
        category=category,
    )
    record["entities"] = extract_entities(content)
    if images is not None:
        record["images"] = images
        paths: List[str] = []
        for img in images:
            url = img.get("image_url")
            if not url:
                continue
            try:
                paths.append(binary_storage.save(url))
            except Exception:
                continue
        record["image_paths"] = paths
    if videos:
        record["videos"] = videos
        record["video_urls"] = videos
    if revisions is not None:
        record.setdefault("metadata", {})["revisions"] = revisions
    return record


class DatasetBuilder:
    def __init__(
        self,
        include_revisions: bool = False,
        rev_limit: int = 5,
        min_complexity: int | None = None,
        max_complexity: int | None = None,
        thread_executor: ThreadPoolExecutor | None = None,
        process_executor: ProcessPoolExecutor | None = None,
        translate_to: str | None = None,
        synthetic_pairs_per_gap: int = 0,
        **_,
    ):
        """Initialize the builder and optionally reuse executors.

        Args:
            include_revisions: Whether to include revision history.
            rev_limit: Maximum number of revisions to fetch.
            min_complexity: Minimum allowed code complexity.
            max_complexity: Maximum allowed code complexity.
            thread_executor: Existing ``ThreadPoolExecutor`` instance for I/O
                tasks. When ``None`` a new executor is created on demand.
            process_executor: Existing ``ProcessPoolExecutor`` instance for CPU
                intensive tasks.
        """

        self.embedding_model = NLPProcessor.get_embedding_model()
        self.dataset = []
        self.qa_pairs = []
        self.pending_pages: List[dict] = []
        self.duplicates_removed = 0
        self.invalid_records = 0
        self.include_revisions = include_revisions
        self.rev_limit = rev_limit
        self.min_complexity = min_complexity
        self.max_complexity = max_complexity
        self.last_scraped = load_last_scraped()
        self.translate_to = translate_to
        self.synthetic_pairs_per_gap = synthetic_pairs_per_gap

        self.thread_executor = thread_executor
        self.process_executor = process_executor
        self._own_thread_executor = thread_executor is None
        self._own_process_executor = process_executor is None

        # Load checkpoints if available
        pages_path = os.path.join(Config.LOG_DIR, "checkpoint_pages.json")
        data_path = os.path.join(Config.LOG_DIR, "checkpoint_data.json")
        if os.path.exists(data_path):
            try:
                with open(data_path, "r", encoding="utf-8") as f:
                    self.dataset = json.load(f)
            except Exception as e:  # pragma: no cover - corrupted file
                logger.error(f"Erro ao carregar {data_path}: {e}")
            else:
                try:
                    os.remove(data_path)
                except OSError:
                    pass
        if os.path.exists(pages_path):
            try:
                with open(pages_path, "r", encoding="utf-8") as f:
                    self.pending_pages = json.load(f)
            except Exception as e:  # pragma: no cover - corrupted file
                logger.error(f"Erro ao carregar {pages_path}: {e}")
            else:
                try:
                    os.remove(pages_path)
                except OSError:
                    pass
        if self.dataset or self.pending_pages:
            self._update_progress()

    def _update_progress(self):
        """Update progress information in logs/progress.json"""
        try:
            os.makedirs(Config.LOG_DIR, exist_ok=True)
            progress_file = os.path.join(Config.LOG_DIR, "progress.json")
            scraper_file = os.path.join(Config.LOG_DIR, "scraper_progress.json")
            temp_file = progress_file + ".tmp"
            temp_file_new = scraper_file + ".tmp"

            clusters = sorted(
                {item.get("cluster") for item in self.dataset if "cluster" in item}
            )
            topics = sorted(
                {item.get("topic") for item in self.dataset if item.get("topic")}
            )
            languages = sorted(
                {item.get("language") for item in self.dataset if item.get("language")}
            )
            categories = sorted(
                {item.get("category") for item in self.dataset if item.get("category")}
            )

            progress = {
                "pages_processed": len(self.dataset),
                "clusters": clusters,
                "topics": topics,
                "languages": languages,
                "categories": categories,
                "duplicates_removed": self.duplicates_removed,
                "invalid_records": self.invalid_records,
            }

            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
            os.replace(temp_file, progress_file)

            with open(temp_file_new, "w", encoding="utf-8") as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
            os.replace(temp_file_new, scraper_file)

            data_path = os.path.join(Config.LOG_DIR, "checkpoint_data.json")
            pages_path = os.path.join(Config.LOG_DIR, "checkpoint_pages.json")
            with open(data_path + ".tmp", "w", encoding="utf-8") as df:
                json.dump(self.dataset, df, ensure_ascii=False)
            os.replace(data_path + ".tmp", data_path)
            with open(pages_path + ".tmp", "w", encoding="utf-8") as pf:
                json.dump(self.pending_pages, pf, ensure_ascii=False)
            os.replace(pages_path + ".tmp", pages_path)
        except Exception as e:
            logger.error(f"Erro ao atualizar progresso: {e}")

    def process_page(
        self, page_info: dict, proc_executor: Optional[ProcessPoolExecutor] = None
    ) -> Optional[object]:
        start_time = time.perf_counter()
        try:
            wiki = WikipediaAdvanced(page_info["lang"])
            key = f"{page_info['lang']}:{page_info['title']}"
            last_ts = self.last_scraped.get(key)
            revs = []
            if last_ts:
                revs = wiki.get_revision_history(page_info["title"], 1)
                if revs:
                    try:
                        last_dt = datetime.fromisoformat(last_ts)
                        rev_dt = datetime.fromisoformat(
                            revs[0]["timestamp"].replace("Z", "+00:00")
                        )
                        if rev_dt <= last_dt:
                            return None
                    except Exception:
                        pass

            page = wiki.fetch_page(page_info["title"])

            if not page or not page.exists():
                return None

            # Extrai e limpa o texto
            raw_text = clean_text(page.text)
            dump_raw_page(
                page_info["title"],
                page_info["lang"],
                page_info.get("category", ""),
                getattr(page, "_html", ""),
                raw_text,
            )
            clean_content = advanced_clean_text(
                raw_text,
                page_info["lang"],
                remove_stopwords=Config.REMOVE_STOPWORDS,
            )

            if len(clean_content) < Config.MIN_TEXT_LENGTH:
                return None

            if proc_executor:
                images = extract_images(getattr(page, "_html", ""))
                videos = extract_videos(getattr(page, "_html", ""))
                revisions = None
                if self.include_revisions:
                    revisions = wiki.get_revision_history(
                        page_info["title"], self.rev_limit
                    )
                return proc_executor.submit(
                    cpu_process_page,
                    page_info["title"],
                    clean_content,
                    page_info["lang"],
                    page_info.get("category", ""),
                    images,
                    videos,
                    revisions,
                    self.rev_limit,
                    self.translate_to,
                )

            # Sumariza o conteúdo
            summary = summarize_text(clean_content, page_info["lang"])

            # Gera QA pairs avançadas
            qa_data = self.generate_qa_pairs(
                title=page_info["title"],
                content=clean_content,
                summary=summary,
                lang=page_info["lang"],
                category=page_info.get("category", ""),
            )
            try:
                from plugins import load_plugin

                wikidata = load_plugin("wikidata")
                items = wikidata.fetch_items(page_info["lang"], page_info["title"])
                if items:
                    parsed = wikidata.parse_item(items[0])
                    qid = parsed.get("wikidata_id")
                    if qid:
                        qa_data.setdefault("metadata", {})["entity_ids"] = [qid]
            except Exception:
                pass
            qa_data["entities"] = extract_entities(clean_content)
            images = extract_images(getattr(page, "_html", ""))
            videos = extract_videos(getattr(page, "_html", ""))
            qa_data["images"] = images
            qa_data["image_paths"] = [
                binary_storage.save(img["image_url"])
                for img in images
                if img.get("image_url")
            ]
            if videos:
                qa_data["videos"] = videos
                qa_data["video_urls"] = videos
            if self.include_revisions:
                qa_data.setdefault("metadata", {})["revisions"] = (
                    wiki.get_revision_history(page_info["title"], self.rev_limit)
                )
            metrics.scrape_success.inc()
            metrics.pages_scraped_total.inc()
            ts = None
            if revs:
                ts = revs[0].get("timestamp")
            if not ts:
                ts = datetime.utcnow().isoformat()
            self.last_scraped[key] = ts
            return qa_data
        except Exception as e:
            metrics.scrape_error.inc()
            logger.error(f"Erro ao processar página {page_info.get('title', '')}: {e}")
            return None
        finally:
            metrics.page_processing_seconds.observe(time.perf_counter() - start_time)

    async def process_page_async(
        self, page_info: dict, proc_executor: Optional[ProcessPoolExecutor] = None
    ) -> Optional[object]:
        """Asynchronous wrapper for :meth:`process_page`."""
        return await asyncio.to_thread(self.process_page, page_info, proc_executor)

    def generate_qa_pairs(
        self,
        title: str,
        content: str,
        summary: str,
        lang: str,
        category: str,
        extra_metadata: dict | None = None,
    ) -> dict:
        raw_code = content
        code_lang = detect_programming_language(content)
        complexities = {}
        docstring = ""
        signature = ""
        problems: list[str] = []
        fixed_version = content
        parse_ok = True
        context_from_code = ""
        if code_lang != "unknown":
            parse_ok = parse_with_language(content, code_lang)
            context_from_code = extract_context_from_code(content, code_lang)
            if not parse_ok:
                return {}
            signature = parse_function_signature(content)
            try:
                tree = ast.parse(content)
                func = next(
                    (
                        n
                        for n in tree.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ),
                    None,
                )
                if func:
                    ds = ast.get_docstring(func)
                    if ds:
                        docstring = docstring_to_google(ds)
            except Exception:
                pass
            content = normalize_indentation(remove_comments(content, code_lang))
            complexities = get_functions_complexity(content, code_lang)
            problems, fixed_version = scan_code(content)
            if self.min_complexity is not None and complexities:
                if max(complexities.values()) < self.min_complexity:
                    return {}
            if self.max_complexity is not None and complexities:
                if max(complexities.values()) > self.max_complexity:
                    return {}
        else:
            problems, fixed_version = scan_code(content)

        quality_metrics = evaluate_code_quality(raw_code, code_lang)
        if (
            quality_metrics["score"] < Config.MIN_CODE_QUALITY
            or quality_metrics["lint_errors"] > Config.MAX_LINT_ERRORS
            or any("eval" in p or "exec" in p for p in problems)
        ):
            return {}

        # Extrai keywords para gerar perguntas variadas
        keywords = extract_keywords(content, lang)

        tags = self._extract_tags(keywords, extra_metadata)

        trans_content = content
        trans_summary = summary
        if self.translate_to:
            trans_content = translate_text(content, self.translate_to)
            trans_summary = translate_text(summary, self.translate_to)

        # Gera múltiplas perguntas baseadas no conteúdo
        questions = self._generate_questions(title, content, lang, keywords)

        # Gera respostas completas
        answers = self._generate_answers(content, summary, lang)

        # Relações semânticas básicas
        relations = extract_relations(content, lang)
        relations_regex = extract_relations_regex(content)
        relations.extend(r for r in relations_regex if r not in relations)

        # Cria embeddings para busca semântica
        embeddings = self.embedding_model.encode(
            [trans_content, trans_summary], show_progress_bar=False
        )
        content_embedding, summary_embedding = embeddings

        if self.translate_to:
            content = trans_content
            summary = trans_summary
            lang = self.translate_to

        # Classificação avançada de tópicos
        topic, subtopic = self._classify_topic(title, content, lang)

        rec_id = hashlib.md5(f"{title}_{lang}".encode("utf-8")).hexdigest()
        diagram_path = ""
        if code_lang != "unknown":
            try:
                diagram_path = generate_diagram(
                    content, os.path.join(Config.LOG_DIR, "diagrams"), rec_id
                )
            except Exception:
                diagram_path = ""
        explanations = generate_explanations(content, lang)
        theory_links = link_theory(content)

        record = {
            "id": rec_id,
            "title": title,
            "language": lang,
            "category": category,
            "topic": topic,
            "subtopic": subtopic,
            "keywords": keywords,
            "tags": tags,
            "content": content,
            "raw_code": raw_code if code_lang != "unknown" else "",
            "summary": summary,
            "context": context_from_code or summary,
            "problems": problems,
            "fixed_version": fixed_version,
            "lessons": extra_metadata.get("lessons", "") if extra_metadata else "",
            "origin_metrics": (
                extra_metadata.get("origin_metrics", {}) if extra_metadata else {}
            ),
            "challenge": (extra_metadata.get("challenge") if extra_metadata else None)
            or generate_challenge_prompt(problems),
            "content_embedding": content_embedding.tolist(),
            "summary_embedding": summary_embedding.tolist(),
            "quality_score": (
                extra_metadata.get("quality_score", 0.0) if extra_metadata else 0.0
            )
            + quality_metrics["score"],
            "quality": None,
            "reason": "",
            "quality_level": None,
            "quality_reason": "",
            "questions": questions,
            "answers": answers,
            "relations": relations,
            "tests": extra_metadata.get("tests", []) if extra_metadata else [],
            "docstring": docstring,
            "diagram_path": diagram_path,
            "theory_links": theory_links,
            "explanations": explanations,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": {
                "length": len(content),
                "source": "wikipedia",
                "source_url": f"{get_base_url(lang)}/wiki/{title.replace(' ', '_')}",
            },
        }

        record.setdefault("origin_metrics", {})["code_quality"] = {
            "cyclomatic_complexity": quality_metrics["complexity"],
            "lint_errors": quality_metrics["lint_errors"],
        }

        metrics_sq = analyze_code(content)
        if metrics_sq:
            record.setdefault("origin_metrics", {})["sonarqube"] = metrics_sq

        try:
            from provenance.tracker import record_provenance

            prov = record_provenance(record["metadata"]["source_url"], content)
            record["metadata"].update(prov)
        except Exception:
            pass

        try:
            from provenance.compliance import check_license

            lic = check_license(record["metadata"]["source_url"])
            record["metadata"]["license"] = lic
        except Exception:
            record["metadata"]["license"] = "unknown"

        # Determine quality classification when possible
        if extra_metadata:
            if any(
                k in extra_metadata for k in ["stars", "stargazers_count", "repository"]
            ):
                q, r = classify_github_repo(extra_metadata)
                record["quality"], record["reason"] = q, r
                record["quality_level"], record["quality_reason"] = q, r
            elif "score" in extra_metadata:
                q, r = classify_stackoverflow_answer(extra_metadata)
                record["quality"], record["reason"] = q, r
                record["quality_level"], record["quality_reason"] = q, r
        if record["quality"] is None:
            score = record.get("quality_score", 0.0)
            if score >= 5:
                q = "high"
            elif score >= 2:
                q = "medium"
            else:
                q = "low"
            record["quality"] = q
            record["quality_level"] = q
            record["reason"] = "quality score"
            record["quality_reason"] = "quality score"
        if code_lang != "unknown":
            record["metadata"]["code_language"] = code_lang
            if complexities:
                record["metadata"]["complexities"] = complexities
            if docstring:
                record["docstring"] = docstring
            if signature:
                record["signature"] = signature
        if extra_metadata:
            if "wikidata_id" in extra_metadata:
                record["wikidata_id"] = extra_metadata["wikidata_id"]
            if "image_url" in extra_metadata:
                record["image_url"] = extra_metadata["image_url"]
            if "discussion_links" in extra_metadata:
                record["discussion_links"] = extra_metadata["discussion_links"]

        if "discussion_links" not in record:
            try:
                links = search_discussions(record["id"])
            except Exception:
                links = []
            record["discussion_links"] = links
        try:
            storage_sqlite.save_to_db(
                {"id": record["id"], "metadata": record.get("metadata", {})}
            )
        except Exception as e:
            logger.error(f"Erro ao salvar no SQLite: {e}")
        return record

    def _generate_questions(
        self, title: str, content: str, lang: str, keywords: List[str]
    ) -> List[dict]:
        """Generate diverse questions for a given page."""

        questions = []

        # Pergunta básica sobre o título
        base_question = {
            "text": self._translate_question(f"What is {title}?", lang),
            "type": "definition",
            "difficulty": "easy",
        }
        questions.append(base_question)

        # Perguntas baseadas em keywords
        for keyword in keywords[:5]:  # Limita a 5 perguntas por keyword
            question_types = [
                (f"How does {keyword} relate to {title}?", "relation", "medium"),
                (f"What is the role of {keyword} in {title}?", "role", "medium"),
                (
                    f"Can you explain {keyword} in the context of {title}?",
                    "explanation",
                    "hard",
                ),
            ]

            for q_text, q_type, q_diff in question_types:
                questions.append(
                    {
                        "text": self._translate_question(q_text, lang),
                        "type": q_type,
                        "difficulty": q_diff,
                    }
                )

        # Perguntas baseadas em conteúdo (usando NLP)
        try:
            nlp = NLPProcessor.get_instance(lang)
            doc = nlp(content)

            # Perguntas sobre entidades nomeadas
            for ent in doc.ents[:5]:  # Limita a 5 entidades
                if ent.label_ in ["PERSON", "ORG", "LOC", "PRODUCT"]:
                    questions.append(
                        {
                            "text": self._translate_question(
                                f"What is the significance of {ent.text} in {title}?",
                                lang,
                            ),
                            "type": "significance",
                            "difficulty": "medium",
                        }
                    )

            # Perguntas sobre verbos/ações
            for sent in doc.sents[:3]:  # Analisa as primeiras 3 sentenças
                root = [token for token in sent if token.dep_ == "ROOT"]
                if root:
                    verb = root[0].lemma_
                    questions.append(
                        {
                            "text": self._translate_question(
                                f"How does {title} {verb}?", lang
                            ),
                            "type": "process",
                            "difficulty": "hard",
                        }
                    )
        except Exception as e:
            logger.error(f"Erro ao gerar perguntas NLP para {title}: {e}")

        return questions

    def _generate_answers(self, content: str, summary: str, lang: str) -> List[dict]:
        answers = []

        # Resposta resumida
        answers.append({"text": summary, "type": "summary", "length": "short"})

        # Resposta completa
        answers.append(
            {
                "text": content[: Config.MAX_TEXT_LENGTH],  # Limita o tamanho
                "type": "detailed",
                "length": "long",
            }
        )

        # Respostas específicas por parágrafo
        paragraphs = [p for p in content.split("\n") if len(p.strip()) > 0]
        for para in paragraphs[:3]:  # Limita a 3 parágrafos
            answers.append({"text": para, "type": "paragraph", "length": "medium"})

        return answers

    def _translate_question(self, question: str, lang: str) -> str:
        """Translate common question fragments to ``lang``."""

        translations = {
            "pt": {
                "What is": "O que é",
                "How does": "Como",
                "relate to": "se relaciona com",
                "What is the role of": "Qual é o papel de",
                "Can you explain": "Você pode explicar",
                "in the context of": "no contexto de",
                "What is the significance of": "Qual é a importância de",
            },
            "es": {
                "What is": "Qué es",
                "How does": "Cómo",
                "relate to": "se relaciona con",
                "What is the role of": "Cuál es el papel de",
                "Can you explain": "Puedes explicar",
                "in the context of": "en el contexto de",
                "What is the significance of": "Cuál es la importancia de",
            },
            "fr": {
                "What is": "Qu'est-ce que",
                "How does": "Comment",
                "relate to": "se rapporte à",
                "What is the role of": "Quel est le rôle de",
                "Can you explain": "Pouvez-vous expliquer",
                "in the context of": "dans le contexte de",
                "What is the significance of": "Quelle est l'importance de",
            },
        }

        if lang in translations:
            for eng, trans in translations[lang].items():
                question = question.replace(eng, trans)

        return question

    def _classify_topic(self, title: str, content: str, lang: str) -> Tuple[str, str]:
        title_lower = title.lower()
        content_lower = content.lower()

        # Mapeamento de tópicos principais e subtópicos
        topics = {
            "ai": {
                "keywords": [
                    "inteligência artificial",
                    "machine learning",
                    "ai",
                    "deep learning",
                    "redes neurais",
                ],
                "subtopics": {
                    "nlp": ["processamento de linguagem natural", "pln", "nlu", "nlp"],
                    "vision": [
                        "visão computacional",
                        "computer vision",
                        "reconhecimento de imagem",
                    ],
                    "robotics": ["robótica", "robots", "autonomous systems"],
                },
            },
            "web": {
                "keywords": [
                    "desenvolvimento web",
                    "frontend",
                    "backend",
                    "full stack",
                    "javascript",
                    "html",
                    "css",
                ],
                "subtopics": {
                    "frontend": [
                        "frontend",
                        "interface",
                        "ui",
                        "ux",
                        "react",
                        "vue",
                        "angular",
                    ],
                    "backend": [
                        "backend",
                        "servidor",
                        "api",
                        "rest",
                        "graphql",
                        "node.js",
                    ],
                    "fullstack": ["full stack", "full-stack", "mern", "mean"],
                },
            },
            "data": {
                "keywords": [
                    "banco de dados",
                    "big data",
                    "data science",
                    "data analytics",
                    "sql",
                    "nosql",
                ],
                "subtopics": {
                    "sql": ["sql", "relacional", "mysql", "postgresql", "oracle"],
                    "nosql": [
                        "nosql",
                        "mongodb",
                        "cassandra",
                        "redis",
                        "elasticsearch",
                    ],
                    "bigdata": ["big data", "hadoop", "spark", "data lake"],
                },
            },
        }

        # Tenta encontrar o tópico principal
        main_topic = "engenharia de software"
        subtopic = "geral"

        for topic, data in topics.items():
            if any(kw in title_lower or kw in content_lower for kw in data["keywords"]):
                main_topic = topic

                # Tenta encontrar subtópico
                for sub, sub_kws in data["subtopics"].items():
                    if any(
                        skw in title_lower or skw in content_lower for skw in sub_kws
                    ):
                        subtopic = sub
                        break
                break

        return (main_topic, subtopic)

    def _extract_tags(
        self, keywords: List[str], extra_metadata: dict | None
    ) -> List[dict]:
        """Return tags with optional explanation links."""
        if extra_metadata and extra_metadata.get("tags"):
            tag_links = extra_metadata.get("tag_links", {})
            return [
                {"tag": t, "link": tag_links.get(t)}
                for t in extra_metadata.get("tags", [])
            ]
        return [{"tag": kw, "link": None} for kw in keywords[:5]]

    def build_from_pages(
        self,
        pages: List[dict],
        progress_desc: str = "Processando páginas",
        use_queue: bool = False,
        client=None,
        pipeline_name: str | None = None,
    ) -> List[dict]:
        """Process pages locally or enfileira tarefas para processamento."""
        if self.pending_pages:
            pages = self.pending_pages
        else:
            self.pending_pages = pages.copy()

        if client:
            futures = [client.submit(self.process_page, page) for page in pages]
            processed = 0
            total = len(futures)
            for fut in tqdm(futures, total=total, desc=progress_desc):
                result = fut.result()
                processed += 1
                if processed % 10 == 0 or processed == total:
                    logger.info(f"Distributed progress: {processed}/{total}")
                page = pages[processed - 1]
                if page in self.pending_pages:
                    self.pending_pages.remove(page)
                if result:
                    self.dataset.append(result)
                self._update_progress()
            if pipeline_name:
                pipe = get_pipeline(pipeline_name)
                self.dataset = pipe(self.dataset)
            return self.dataset

        if not use_queue:
            cpu_futures = []
            if self.thread_executor is None:
                self.thread_executor = ThreadPoolExecutor(
                    max_workers=Config.MAX_THREADS
                )
            if self.process_executor is None:
                self.process_executor = ProcessPoolExecutor(
                    max_workers=Config.MAX_PROCESSES
                )

            page_futures = {
                self.thread_executor.submit(
                    self.process_page, page, self.process_executor
                ): page
                for page in pages
            }

            processed = 0
            total = len(page_futures)
            for future in tqdm(
                as_completed(page_futures), total=total, desc=progress_desc
            ):
                cpu_future = future.result()
                page = page_futures[future]
                processed += 1
                if page in self.pending_pages:
                    self.pending_pages.remove(page)
                if processed % 10 == 0 or processed == total:
                    logger.info(f"Thread progress: {processed}/{total}")
                if cpu_future:
                    cpu_futures.append(cpu_future)
                self._update_progress()

            processed_cpu = 0
            total_cpu = len(cpu_futures)
            for future in tqdm(
                as_completed(cpu_futures),
                total=total_cpu,
                desc="Processando conteúdo",
            ):
                result = future.result()
                processed_cpu += 1
                if processed_cpu % 10 == 0 or processed_cpu == total_cpu:
                    logger.info(f"Process progress: {processed_cpu}/{total_cpu}")
                if result:
                    self.dataset.append(result)
                self._update_progress()

            if pipeline_name:
                pipe = get_pipeline(pipeline_name)
                self.dataset = pipe(self.dataset)
            return self.dataset

        # Queue based processing
        from task_queue import publish, consume

        for page in pages:
            publish("scrape_tasks", page)

        processed = 0
        total = len(pages)
        for result in consume("scrape_results"):
            if result is None:
                continue
            processed += 1
            for p in list(self.pending_pages):
                if p.get("title") == result.get("title") and p.get(
                    "lang", p.get("language")
                ) == result.get("language"):
                    self.pending_pages.remove(p)
                    break
            self.dataset.append(result)
            self._update_progress()
            if processed % 10 == 0 or processed == total:
                logger.info(f"Queue progress: {processed}/{total}")
            if processed == total:
                break

        if pipeline_name:
            pipe = get_pipeline(pipeline_name)
            self.dataset = pipe(self.dataset)
        return self.dataset

    async def build_from_pages_async(
        self,
        pages: List[dict],
        progress_desc: str = "Processando páginas",
        pipeline_name: str | None = None,
    ) -> List[dict]:
        """Asynchronous version of :meth:`build_from_pages`."""
        if self.pending_pages:
            pages = self.pending_pages
        else:
            self.pending_pages = pages.copy()

        cpu_futures: list[Future] = []
        pr_executor = async_runner.get_process_pool()
        sem = asyncio.Semaphore(Config.MAX_CONCURRENT_REQUESTS)

        processed = 0
        total = len(pages)

        async def handle(page: dict) -> None:
            nonlocal processed
            async with sem:
                cpu_future = await self.process_page_async(page, pr_executor)
            processed += 1
            if page in self.pending_pages:
                self.pending_pages.remove(page)
            if processed % 10 == 0 or processed == total:
                logger.info(f"Async progress: {processed}/{total}")
            if cpu_future:
                cpu_futures.append(cpu_future)
            self._update_progress()

        await asyncio.gather(*(handle(p) for p in pages))

        processed_cpu = 0
        total_cpu = len(cpu_futures)
        for future in tqdm(
            as_completed(cpu_futures), total=total_cpu, desc="Processando conteúdo"
        ):
            result = future.result()
            processed_cpu += 1
            if processed_cpu % 10 == 0 or processed_cpu == total_cpu:
                logger.info(f"Process progress: {processed_cpu}/{total_cpu}")
            if result:
                self.dataset.append(result)
            self._update_progress()

        if pipeline_name:
            pipe = get_pipeline(pipeline_name)
            self.dataset = pipe(self.dataset)
        return self.dataset

    def add_io_pairs(self, pairs: List[dict]) -> None:
        """Append input/output ``pairs`` to :attr:`qa_pairs`."""

        self.qa_pairs.extend(pairs)

    def enhance_with_clustering(self):
        if not self.dataset:
            return

        # Clusteriza baseado nos embeddings de conteúdo
        texts = [item["content"] for item in self.dataset]
        clusters = cluster_texts(texts)

        # Adiciona clusters ao dataset
        for i, item in enumerate(self.dataset):
            item["cluster"] = int(clusters[i])

        # Opcionalmente filtra registros pelo nível de complexidade
        self.dataset = filter_by_complexity(
            self.dataset, self.min_complexity, self.max_complexity
        )

    def augment_with_synthetic_data(self, min_ratio: float = 0.1) -> None:
        """Augment dataset generating synthetic question/answer pairs."""

        if self.synthetic_pairs_per_gap <= 0 or not self.dataset:
            return

        gaps = identify_gaps(self.dataset, min_ratio=min_ratio)
        languages = gaps.get("languages", []) or ["en"]
        topics = gaps.get("topics", []) or ["general"]

        for lang in languages:
            for topic in topics:
                pairs = generate_synthetic_qa(
                    topic, lang, n=self.synthetic_pairs_per_gap
                )
                for q in pairs:
                    self.dataset.append(
                        {
                            "id": uuid.uuid4().hex,
                            "language": lang,
                            "category": "synthetic",
                            "topic": topic,
                            "content": "",
                            "summary": "",
                            "questions": [q["question"]],
                            "answers": [q["answer"]],
                            "relations": [],
                            "created_at": datetime.utcnow().isoformat(),
                            "metadata": {"synthetic": True},
                        }
                    )

    def save_dataset(
        self,
        format: str = "all",
        output_dir: str = Config.OUTPUT_DIR,
        *,
        incremental: bool = False,
    ):
        os.makedirs(output_dir, exist_ok=True)

        # Optionally generate synthetic question/answer pairs
        self.augment_with_synthetic_data()

        if not self.dataset:
            logger.warning("Nenhum dado para salvar")
            return

        # Balance records by quality before further processing
        try:
            self.dataset = balance_quality(self.dataset)
        except Exception:
            pass

        for rec in self.dataset:
            if "content" in rec:
                rec["content"] = dq.remove_pii(dq.strip_credentials(rec["content"]))
            if "summary" in rec:
                rec["summary"] = dq.remove_pii(rec["summary"])

        # Remove near-duplicates based on Simhash
        try:
            self.dataset, rem_sim = dq.deduplicate_by_simhash(self.dataset)
            self.duplicates_removed += rem_sim
        except Exception as e:  # pragma: no cover - library issues
            logger.error(f"Erro na deduplicação Simhash: {e}")

        try:
            plag = dq.detect_code_plagiarism(self.dataset)
            if plag:
                logger.warning(f"Registros plagiados removidos: {len(plag)}")
                self.dataset = [r for r in self.dataset if r not in plag]
                self.invalid_records += len(plag)
        except Exception as e:  # pragma: no cover - library issues
            logger.error(f"Erro na verificação de plágio: {e}")

        # Validação dos registros antes de salvar
        validated_data = []
        for item in self.dataset:
            valid = True

            # Checa se embeddings contêm apenas números finitos
            if not np.all(np.isfinite(item.get("content_embedding", []))):
                logger.warning(
                    f"content_embedding inválido para {item.get('id', 'desconhecido')}"
                )
                valid = False

            if not np.all(np.isfinite(item.get("summary_embedding", []))):
                logger.warning(
                    f"summary_embedding inválido para {item.get('id', 'desconhecido')}"
                )
                valid = False

            # Checa presença de perguntas e respostas
            if not item.get("questions"):
                logger.warning(
                    f"Registro {item.get('id', 'desconhecido')} sem perguntas"
                )
                valid = False

            if not item.get("answers"):
                logger.warning(
                    f"Registro {item.get('id', 'desconhecido')} sem respostas"
                )
                valid = False

            if "relations" not in item:
                logger.warning(
                    f"Registro {item.get('id', 'desconhecido')} sem relações"
                )
                valid = False

            # Verifica tamanho do resumo
            summary_text = item.get("summary", "")
            if len(summary_text) < Config.MIN_TEXT_LENGTH:
                logger.warning(
                    f"Resumo muito curto para {item.get('id', 'desconhecido')}"
                )
                valid = False

            if valid:
                validated_data.append(item)

        # Skip records already saved when incremental mode is on
        if incremental:
            try:
                from provenance import tracker as prov

                filtered = []
                for rec in validated_data:
                    h = prov.compute_record_hash(rec)
                    if prov.dataset_hash_exists(h):
                        self.duplicates_removed += 1
                        continue
                    prov.record_dataset_hash(rec)
                    rec.setdefault("metadata", {})["record_hash"] = h
                    filtered.append(rec)
                validated_data = filtered
            except Exception as e:  # pragma: no cover - db errors
                logger.error(f"Erro verificação incremental: {e}")

        # Quality metrics
        try:
            metrics.dataset_completeness.set(
                len(validated_data) / len(self.dataset) if self.dataset else 0.0
            )
            topics = {i.get("topic") for i in validated_data if i.get("topic")}
            metrics.dataset_topic_diversity.set(
                len(topics) / len(validated_data) if validated_data else 0.0
            )
            langs = {i.get("language") for i in validated_data if i.get("language")}
            cats = {i.get("category") for i in validated_data if i.get("category")}
            metrics.dataset_language_coverage.set(
                len(langs) / len(Config.LANGUAGES) if Config.LANGUAGES else 0.0
            )
            metrics.dataset_domain_coverage.set(
                len(cats) / len(Config.CATEGORIES) if Config.CATEGORIES else 0.0
            )
            bias = 1.0 if len(langs) < len(Config.LANGUAGES) / 2 else 0.0
            metrics.dataset_bias_detected.set(bias)
        except Exception:  # pragma: no cover - metrics failures
            pass

        if not validated_data:
            logger.warning("Nenhum registro válido para salvar")
            return

        # Ordena por idioma e tópico
        sorted_data = sorted(validated_data, key=lambda x: (x["language"], x["topic"]))

        backend = storage.get_backend(Config.STORAGE_BACKEND, output_dir)

        def _bump(ver: str) -> str:
            try:
                major, minor, patch = [int(x) for x in ver.split(".")]
            except Exception:
                return "1.0.0"
            patch += 1
            return f"{major}.{minor}.{patch}"

        info_path = os.path.join(output_dir, "dataset_info.json")
        old_info = {}
        if os.path.exists(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    old_info = json.load(f)
            except Exception:
                old_info = {}
        old_version = old_info.get("version", "0.0.0")
        old_size = old_info.get("size", 0)
        old_ids = set(old_info.get("record_ids", []))
        new_version = _bump(old_version)

        backend.save_dataset(
            sorted_data,
            format,
            version=new_version,
            compression=Config.COMPRESSION,
        )
        logger.info(f"Dataset salvo usando backend {Config.STORAGE_BACKEND}")
        track_path(output_dir)
        track_path(Config.RAW_DIR)
        save_last_scraped(self.last_scraped)

        if format == "qa":
            save_qa_dataset(sorted_data, Path(output_dir) / "qa_pairs.json")
        if format == "text":
            save_text_corpus(sorted_data, Path(output_dir) / "text_corpus")

        if format in ["all", "hf", "tfrecord"]:
            try:
                hf_dataset = Dataset.from_list(sorted_data)
                hf_dataset.save_to_disk(os.path.join(output_dir, "huggingface"))
                logger.info(
                    f"Dataset salvo para HuggingFace: {os.path.join(output_dir, 'huggingface')}"
                )
            except Exception as e:
                logger.error(f"Erro ao salvar dataset HuggingFace: {e}")

        # Audit diff between current and previous dataset
        current_ids = [item.get("id") for item in sorted_data if item.get("id")]
        diff_path = os.path.join(output_dir, f"diff_{new_version}.json")
        try:
            with open(diff_path, "w", encoding="utf-8") as df:
                json.dump(
                    {
                        "added": sorted(set(current_ids) - old_ids),
                        "removed": sorted(old_ids - set(current_ids)),
                    },
                    df,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as e:  # pragma: no cover - unexpected I/O errors
            logger.error(f"Erro ao salvar diff file: {e}")

        # Write metadata about the dataset
        try:
            sources = {
                item.get("metadata", {}).get("source", "unknown")
                for item in sorted_data
            }
            licenses = {
                item.get("metadata", {}).get("license", "unknown")
                for item in sorted_data
            }
            info = {
                "source": list(sources)[0] if len(sources) == 1 else sorted(sources),
                "collection_date": datetime.utcnow().date().isoformat(),
                "license": (
                    list(licenses)[0] if len(licenses) == 1 else sorted(licenses)
                ),
                "version": new_version,
                "size": len(sorted_data),
                "record_ids": current_ids,
            }
            with open(
                os.path.join(output_dir, "dataset_info.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(info, f, ensure_ascii=False, indent=2)

            diff_ratio = (
                abs(len(sorted_data) - old_size) / max(old_size, 1) if old_info else 1.0
            )
            if diff_ratio > 0.1:
                with open(
                    os.path.join(output_dir, "CHANGELOG.txt"), "a", encoding="utf-8"
                ) as cf:
                    cf.write(
                        f"{datetime.utcnow().isoformat()} - v{new_version} size changed {old_size}->{len(sorted_data)} diff: diff_{new_version}.json\n"
                    )
            if diff_ratio > 0.5:
                send_alert(
                    f"Dataset size changed significantly: {old_size} -> {len(sorted_data)}"
                )
        except Exception as e:  # pragma: no cover - unexpected I/O errors
            logger.error(f"Erro ao salvar dataset_info.json: {e}")
            send_alert(f"Failed to save dataset info: {e}")

    def cleanup(self) -> None:
        """Shut down executors owned by this instance."""
        if self._own_thread_executor and self.thread_executor:
            self.thread_executor.shutdown()
            self.thread_executor = None
        if self._own_process_executor and self.process_executor:
            self.process_executor.shutdown()
            self.process_executor = None
        async_runner.shutdown()


# ============================
# 🚦 Pipeline de Execução Principal
# ============================
def main(
    langs: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    fmt: str = "all",
    rate_limit_delay: Optional[float] = None,
    *,
    start_pages: Optional[List[str]] = None,
    depth: int = 1,
    revisions: bool = False,
    rev_limit: int = 5,
    translate_to: str | None = None,
    client=None,
    incremental: bool = False,
):
    """Gera o dataset utilizando os parâmetros fornecidos."""
    start_time = time.perf_counter()
    metrics.start_metrics_server(int(os.environ.get("METRICS_PORT", "8001")))
    metrics.start_system_metrics_loop()
    logger.info("🚀 Iniciando Wikipedia Scraper Ultra Pro Max - GodMode++")

    if rate_limit_delay is not None:
        Config.RATE_LIMIT_DELAY = rate_limit_delay
        rate_limiter.base_min = rate_limit_delay
        rate_limiter.base_max = rate_limit_delay
        rate_limiter.reset()

    languages = langs or Config.LANGUAGES
    cats = Config.CATEGORIES
    if categories:
        normalized = [normalize_category(c) or c for c in categories]
        cats = {c: Config.CATEGORIES.get(c, 1.0) for c in normalized}

    builder = DatasetBuilder(
        include_revisions=revisions, rev_limit=rev_limit, translate_to=translate_to
    )

    all_pages: List[dict] = []
    for lang in languages:
        logger.info(f"🌐 Processando idioma: {lang.upper()}")
        wiki = WikipediaAdvanced(lang)

        for category, weight in cats.items():
            logger.info(f"🔍 Buscando na categoria: {category} (peso: {weight})")

            pages = wiki.get_category_members(category)
            logger.info(f"📄 Páginas encontradas em {category}: {len(pages)}")

            for page in pages:
                page["weight"] = weight

            all_pages.extend(pages)
            time.sleep(Config.RATE_LIMIT_DELAY * 2)

        if start_pages:
            for sp in start_pages:
                logger.info(
                    f"🕸️ Rastreiando links a partir de: {sp} (profundidade: {depth})"
                )
                pages = wiki.crawl_links(sp, depth)
                logger.info(f"📄 Páginas coletadas de {sp}: {len(pages)}")
                all_pages.extend(pages)
                time.sleep(Config.RATE_LIMIT_DELAY * 2)

    logger.info(f"📚 Total de páginas coletadas: {len(all_pages)}")

    sig = inspect.signature(builder.build_from_pages)
    if "pipeline_name" in sig.parameters:
        builder.build_from_pages(
            all_pages,
            "Construindo dataset",
            client=client,
            pipeline_name="default",
        )
    else:
        builder.build_from_pages(all_pages, "Construindo dataset", client=client)

    logger.info("🧠 Aplicando técnicas avançadas de NLP...")
    builder.enhance_with_clustering()

    # Carrega dados extras de plugins para completar registros
    extra_data: List[dict] = []
    try:
        from plugins import load_plugin

        for plugin_name in [
            "stackoverflow",
            "wikidata",
            "infobox_parser",
            "table_parser",
        ]:
            try:
                plugin = load_plugin(plugin_name)
            except Exception as e:
                logger.error(f"Erro ao carregar plugin {plugin_name}: {e}")
                continue
            for lang in languages:
                for category in cats:
                    try:
                        items = plugin.fetch_items(lang, category)
                        for item in items:
                            parsed = plugin.parse_item(item)
                            if parsed:
                                extra_data.append(parsed)
                    except Exception as e:  # pragma: no cover - network errors
                        logger.error(f"Erro plugin {plugin_name}: {e}")
    except Exception:
        pass

    # Deduplicação e validação de qualidade
    if hasattr(builder, "dataset"):
        data, rem_hash = dq.deduplicate_by_hash(getattr(builder, "dataset", []))
        data, rem_emb = dq.deduplicate_by_embedding(data)
        data = dq.complete_missing_fields(data, extra_data)
        data, invalid = dq.validate_semantics(data)

        builder.duplicates_removed = rem_hash + rem_emb
        builder.invalid_records = invalid
        builder.dataset = data
        if hasattr(builder, "_update_progress"):
            builder._update_progress()

    logger.info("💾 Salvando dataset completo...")
    builder.save_dataset(format=fmt, incremental=incremental)

    logger.info("✅ Dataset finalizado com sucesso!")
    logger.info(f"📊 Estatísticas de cache: {cache.stats()}")
    metrics.scrape_session_seconds.observe(time.perf_counter() - start_time)


async def main_async(
    langs: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    fmt: str = "all",
    rate_limit_delay: Optional[float] = None,
    *,
    start_pages: Optional[List[str]] = None,
    depth: int = 1,
    revisions: bool = False,
    rev_limit: int = 5,
    translate_to: str | None = None,
    incremental: bool = False,
) -> None:
    """Asynchronous version of :func:`main`."""
    start_time = time.perf_counter()
    metrics.start_metrics_server(int(os.environ.get("METRICS_PORT", "8001")))
    metrics.start_system_metrics_loop()
    logger.info("🚀 Iniciando Wikipedia Scraper Ultra Pro Max - GodMode++ (async)")

    if rate_limit_delay is not None:
        Config.RATE_LIMIT_DELAY = rate_limit_delay
        rate_limiter.base_min = rate_limit_delay
        rate_limiter.base_max = rate_limit_delay
        rate_limiter.reset()

    languages = langs or Config.LANGUAGES
    cats = Config.CATEGORIES
    if categories:
        normalized = [normalize_category(c) or c for c in categories]
        cats = {c: Config.CATEGORIES.get(c, 1.0) for c in normalized}

    builder = DatasetBuilder(
        include_revisions=revisions, rev_limit=rev_limit, translate_to=translate_to
    )

    all_pages: List[dict] = []
    for lang in languages:
        logger.info(f"🌐 Processando idioma: {lang.upper()}")
        wiki = WikipediaAdvanced(lang)

        for category, weight in cats.items():
            logger.info(f"🔍 Buscando na categoria: {category} (peso: {weight})")

            pages = await wiki.get_category_members_async(category)
            logger.info(f"📄 Páginas encontradas em {category}: {len(pages)}")

            for page in pages:
                page["weight"] = weight

            all_pages.extend(pages)
            await asyncio.sleep(Config.RATE_LIMIT_DELAY * 2)

        if start_pages:
            for sp in start_pages:
                logger.info(
                    f"🕸️ Rastreiando links a partir de: {sp} (profundidade: {depth})"
                )
                pages = await wiki.crawl_links_async(sp, depth)
                logger.info(f"📄 Páginas coletadas de {sp}: {len(pages)}")
                all_pages.extend(pages)
                await asyncio.sleep(Config.RATE_LIMIT_DELAY * 2)

    logger.info(f"📚 Total de páginas coletadas: {len(all_pages)}")

    sig = inspect.signature(builder.build_from_pages_async)
    if "pipeline_name" in sig.parameters:
        await builder.build_from_pages_async(
            all_pages, "Construindo dataset", pipeline_name="default"
        )
    else:
        await builder.build_from_pages_async(all_pages, "Construindo dataset")

    logger.info("🧠 Aplicando técnicas avançadas de NLP...")
    builder.enhance_with_clustering()

    extra_data: List[dict] = []
    try:
        from plugins import load_plugin

        for plugin_name in [
            "stackoverflow",
            "wikidata",
            "infobox_parser",
            "table_parser",
        ]:
            try:
                plugin = load_plugin(plugin_name)
            except Exception as e:
                logger.error(f"Erro ao carregar plugin {plugin_name}: {e}")
                continue
            for lang in languages:
                for category in cats:
                    try:
                        items = plugin.fetch_items(lang, category)
                        for item in items:
                            parsed = plugin.parse_item(item)
                            if parsed:
                                extra_data.append(parsed)
                    except Exception as e:  # pragma: no cover - network errors
                        logger.error(f"Erro plugin {plugin_name}: {e}")
    except Exception:
        pass

    if hasattr(builder, "dataset"):
        data, rem_hash = dq.deduplicate_by_hash(getattr(builder, "dataset", []))
        data, rem_emb = dq.deduplicate_by_embedding(data)
        data = dq.complete_missing_fields(data, extra_data)
        data, invalid = dq.validate_semantics(data)

        builder.duplicates_removed = rem_hash + rem_emb
        builder.invalid_records = invalid
        builder.dataset = data
        if hasattr(builder, "_update_progress"):
            builder._update_progress()

    logger.info("💾 Salvando dataset completo...")
    builder.save_dataset(format=fmt, incremental=incremental)

    logger.info("✅ Dataset finalizado com sucesso!")
    logger.info(f"📊 Estatísticas de cache: {cache.stats()}")
    metrics.scrape_session_seconds.observe(time.perf_counter() - start_time)


if __name__ == "__main__":
    main()
