"""Prometheus metrics used to monitor scraping performance."""

from prometheus_client import Counter, Histogram, Gauge, start_http_server
import psutil
import threading
import time

scrape_success = Counter(
    "scrape_success_total", "Total de páginas raspadas com sucesso"
)

scrape_error = Counter("scrape_error_total", "Total de erros ao raspar páginas")

scrape_block = Counter("scrape_block_total", "Total de bloqueios durante a raspagem")

# Total de registros de QA gerados com sucesso
pages_scraped_total = Counter("pages_scraped_total", "Pages successfully scraped")

# Falhas ao executar requisições HTTP
requests_failed_total = Counter("requests_failed_total", "HTTP request failures")

# HTTP 429 responses received
requests_429_total = Counter(
    "requests_429_total", "HTTP 429 responses from target servers"
)

# Total de tentativas extras ao fazer requisições
request_retries_total = Counter("request_retries_total", "HTTP request retries")

# Number of times requests had to wait due to rate limiting
rate_limited_total = Counter("rate_limited_total", "Number of rate limited events")

# Histogram of processing time per page
page_processing_seconds = Histogram(
    "page_processing_seconds", "Time spent processing a page in seconds"
)

# Duração completa de uma sessão de scraping
scrape_session_seconds = Histogram(
    "scrape_session_seconds", "Time spent in a full scraping session in seconds"
)

# Quality metrics
dataset_completeness = Gauge(
    "dataset_completeness_ratio",
    "Ratio of valid records to total records",
)

dataset_topic_diversity = Gauge(
    "dataset_topic_diversity",
    "Unique topics divided by total valid records",
)

# Cobertura de idiomas e categorias/domínios
dataset_language_coverage = Gauge(
    "dataset_language_coverage",
    "Ratio of scraped languages to configured languages",
)

dataset_domain_coverage = Gauge(
    "dataset_domain_coverage",
    "Ratio of scraped categories to configured categories",
)

# Flag de viés detectado no dataset
dataset_bias_detected = Gauge(
    "dataset_bias_detected",
    "1 when dataset exhibits potential bias, 0 otherwise",
)

# System metrics
cpu_usage_percent = Gauge(
    "cpu_usage_percent",
    "Current system CPU utilization percentage",
)

memory_usage_percent = Gauge(
    "memory_usage_percent",
    "Current system memory utilization percentage",
)

# Timestamp da última atualização dos parsers
parser_update_timestamp = Gauge(
    "scraper_parser_update_timestamp",
    "Unix timestamp of the last parser update",
)


def start_metrics_server(port: int = 8001) -> None:
    """Inicia o servidor de métricas para Prometheus."""
    start_http_server(port)


def start_system_metrics_loop(interval: int = 5) -> None:
    """Atualiza métricas de CPU e memória em segundo plano."""

    def _update() -> None:
        while True:
            cpu_usage_percent.set(psutil.cpu_percent())
            memory_usage_percent.set(psutil.virtual_memory().percent)
            time.sleep(interval)

    thread = threading.Thread(target=_update, daemon=True)
    thread.start()
