import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import typer

import scraper_wiki
import dashboard
from search import indexer

app = typer.Typer(help="Scraper Wiki command line interface")


@app.callback(invoke_without_command=False)
def main(
    ctx: typer.Context,
    cache_backend: str = typer.Option(
        None, "--cache-backend", help="Backend de cache", show_default=False
    ),
    cache_ttl: int = typer.Option(
        None,
        "--cache-ttl",
        help="Tempo de vida do cache em segundos",
        show_default=False,
    ),
    log_level: str = typer.Option(
        None,
        "--log-level",
        help="Nível de log (DEBUG, INFO, WARNING...)",
        show_default=False,
    ),
    log_format: str = typer.Option(
        "text", "--log-format", help="Formato do log (text ou json)"
    ),
    max_threads: int = typer.Option(
        None, "--max-threads", help="Número máximo de threads", show_default=False
    ),
    max_processes: int = typer.Option(
        None, "--max-processes", help="Número máximo de processos", show_default=False
    ),
    storage_backend: str = typer.Option(
        None, "--storage-backend", help="Backend de armazenamento", show_default=False
    ),
    compress: str = typer.Option(
        None,
        "--compress",
        help="Compressão (none, gzip, zstd)",
        show_default=False,
    ),
    scraper_backend: str = typer.Option(
        None,
        "--scraper-backend",
        help="Backend do AutoLearnerScraper (selenium ou playwright)",
        show_default=False,
    ),
):
    if cache_backend is not None:
        scraper_wiki.Config.CACHE_BACKEND = cache_backend
        scraper_wiki.cache = scraper_wiki.init_cache()
    if cache_ttl is not None:
        scraper_wiki.Config.CACHE_TTL = cache_ttl

    if log_level is not None or log_format != "text":
        level = (
            getattr(logging, log_level.upper(), logging.INFO)
            if log_level
            else logging.INFO
        )
        scraper_wiki.setup_logger(
            "wiki_scraper", "scraper.log", level=level, fmt=log_format
        )
    if max_threads is not None:
        scraper_wiki.Config.MAX_THREADS = max_threads
    if max_processes is not None:
        scraper_wiki.Config.MAX_PROCESSES = max_processes
    if storage_backend is not None:
        scraper_wiki.Config.STORAGE_BACKEND = storage_backend
    if compress is not None:
        scraper_wiki.Config.COMPRESSION = compress
    if scraper_backend is not None:
        scraper_wiki.Config.SCRAPER_BACKEND = scraper_backend


QUEUE_FILE = Path("jobs_queue.jsonl")


@app.command()
def scrape(
    lang: Optional[List[str]] = typer.Option(
        None, "--lang", help="Idioma a processar", show_default=False
    ),
    category: Optional[List[str]] = typer.Option(
        None, "--category", help="Categoria específica", show_default=False
    ),
    fmt: str = typer.Option(
        "all",
        "--format",
        help="Formato de saída (json, jsonl, csv, parquet, tfrecord, qa, text)",
    ),
    start_page: list[str] = typer.Option(
        None,
        "--start-page",
        help="Página inicial para rastrear links (pode ser repetido)",
        show_default=False,
    ),
    depth: int = typer.Option(
        1,
        "--depth",
        help="Profundidade de navegação para páginas iniciais",
    ),
    rate_limit_delay: float = typer.Option(
        None, "--rate-limit-delay", help="Delay entre requisições", show_default=False
    ),
    revisions: bool = typer.Option(
        False, "--revisions", help="Inclui histórico de revisões", is_flag=True
    ),
    rev_limit: int = typer.Option(5, "--rev-limit", help="Número máximo de revisões"),
    translate_to: str = typer.Option(
        None,
        "--translate",
        help="Traduz conteúdo para o idioma fornecido",
        show_default=False,
    ),
    async_mode: bool = typer.Option(
        False, "--async", help="Usa scraping assíncrono", is_flag=True
    ),
    plugin: str = typer.Option(
        "wikipedia",
        "--plugin",
        help="Plugin de scraping (wikipedia, infobox_parser, table_parser)",
    ),
    distributed: bool = typer.Option(
        False, "--distributed", help="Usa cluster distribuído", is_flag=True
    ),
    train: bool = typer.Option(
        False, "--train", help="Executa conversões para treinamento"
    ),
    incremental: bool = typer.Option(
        False, "--incremental", help="Busca apenas novos itens", is_flag=True
    ),
):
    """Executa o scraper imediatamente."""
    lang = lang or None
    category = category or None
    cats = (
        [scraper_wiki.normalize_category(c) or c for c in category]
        if category
        else None
    )
    client = None
    if distributed:
        from cluster import get_client

        client = get_client()

    if plugin == "wikipedia":
        if async_mode:
            import asyncio

            asyncio.run(
                scraper_wiki.main_async(
                    lang,
                    cats,
                    fmt,
                    rate_limit_delay,
                    start_pages=start_page,
                    depth=depth,
                    revisions=revisions,
                    rev_limit=rev_limit,
                    translate_to=translate_to,
                    incremental=incremental,
                )
            )
        else:
            scraper_wiki.main(
                lang,
                cats,
                fmt,
                rate_limit_delay,
                start_pages=start_page,
                depth=depth,
                revisions=revisions,
                rev_limit=rev_limit,
                translate_to=translate_to,
                client=client,
                incremental=incremental,
            )
        dataset_file = Path(scraper_wiki.Config.OUTPUT_DIR) / "wikipedia_qa.json"
        if dataset_file.exists() and train:
            from training import pipeline

            pipeline.run_pipeline(dataset_file)
    else:
        from plugins import load_plugin, run_plugin

        plg = load_plugin(plugin)
        languages = lang or scraper_wiki.Config.LANGUAGES
        categories = cats or list(scraper_wiki.Config.CATEGORIES)
        run_plugin(plg, languages, categories, fmt, incremental=incremental)


@app.command()
def monitor():
    """Inicia o dashboard para monitoramento."""
    dashboard.main()


@app.command()
def queue(
    lang: Optional[List[str]] = typer.Option(
        None, "--lang", help="Idioma a processar", show_default=False
    ),
    category: Optional[List[str]] = typer.Option(
        None, "--category", help="Categoria específica", show_default=False
    ),
    fmt: str = typer.Option(
        "all",
        "--format",
        help="Formato de saída (json, jsonl, csv, parquet, tfrecord, qa, text)",
    ),
):
    """Enfileira um job de scraping."""
    from metrics import start_metrics_server, start_system_metrics_loop

    start_metrics_server(int(os.environ.get("METRICS_PORT", "8001")))
    start_system_metrics_loop()
    lang = lang or None
    category = category or None
    cats = (
        [scraper_wiki.normalize_category(c) or c for c in category]
        if category
        else None
    )
    job = {"lang": lang, "category": cats, "format": fmt}
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(job, ensure_ascii=False) + "\n")
    typer.echo(f"Job enfileirado: {job}")


@app.command()
def status():
    """Mostra arquivos gerados e configurações principais."""
    from scraper_wiki import Config

    output_dir = Path(Config.OUTPUT_DIR)
    typer.echo(f"Conteúdo de {output_dir}:")
    if output_dir.exists():
        for path in output_dir.iterdir():
            typer.echo(f"- {path.name}")
    else:
        typer.echo("(diretório não encontrado)")

    typer.echo("\nConfigurações chave:")
    settings = {
        "OUTPUT_DIR": Config.OUTPUT_DIR,
        "CACHE_DIR": Config.CACHE_DIR,
        "LOG_DIR": Config.LOG_DIR,
        "MAX_THREADS": Config.MAX_THREADS,
        "MAX_PROCESSES": Config.MAX_PROCESSES,
    }
    for key, value in settings.items():
        typer.echo(f"{key}: {value}")


@app.command("clear-cache")
def clear_cache_cmd():
    """Remove entradas expiradas do cache."""
    scraper_wiki.clear_cache()
    typer.echo("Cache limpo")


@app.command("search")
def search_cli(query: str):
    """Search indexed records using Elasticsearch."""
    results = indexer.query_index(query)
    typer.echo(json.dumps(results, ensure_ascii=False))


@app.command("auto-scrape")
def auto_scrape(
    start_url: List[str] = typer.Argument(..., help="URL inicial para coleta"),
    depth: int = typer.Option(1, "--depth", help="Profundidade de navegação"),
    threads: int = typer.Option(
        scraper_wiki.Config.MAX_THREADS,
        "--threads",
        help="Número de threads",
    ),
):
    """Rastreia páginas dinâmicas usando ``AutoLearnerScraper``."""
    from concurrent.futures import ThreadPoolExecutor
    from urllib.parse import urlparse

    from core import AutoLearnerScraper

    scraper_wiki.Config.MAX_THREADS = threads

    scrapers: dict[str, AutoLearnerScraper] = {}

    def get_scraper(url: str) -> AutoLearnerScraper:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        sc = scrapers.get(base)
        if sc is None:
            sc = AutoLearnerScraper(base, backend=scraper_wiki.Config.SCRAPER_BACKEND)
            scrapers[base] = sc
        return sc

    def process(url_depth: tuple[str, int]):
        url, cur_depth = url_depth
        sc = get_scraper(url)
        record = sc.fetch_page(url)
        links = []
        if cur_depth < depth:
            html = sc.driver.page_source
            links = scraper_wiki.extract_links(html, sc.base_url)
        return record, links, cur_depth

    results = []
    queue: list[tuple[str, int]] = [(u, 0) for u in start_url]
    visited: set[str] = set()

    with ThreadPoolExecutor(max_workers=threads) as executor:
        while queue:
            batch = []
            while queue and len(batch) < threads:
                batch.append(queue.pop(0))
            futures = [executor.submit(process, item) for item in batch]
            for fut in futures:
                record, links, cur_depth = fut.result()
                results.append(record)
                visited.add(record["url"])
                if cur_depth < depth:
                    for link in links:
                        if link not in visited:
                            queue.append((link, cur_depth + 1))

    for sc in scrapers.values():
        sc.close()

    typer.echo(json.dumps(results, ensure_ascii=False))


@app.command("process")
def process_pipeline(
    dataset: str = typer.Argument(..., help="Caminho do dataset"),
    pipeline: str = typer.Option("default", "--pipeline", help="Nome do pipeline"),
):
    """Run processing pipeline on an existing dataset."""
    from processing.pipeline import get_pipeline

    data_path = Path(dataset)
    records = json.loads(data_path.read_text(encoding="utf-8"))
    pipe = get_pipeline(pipeline)
    result = pipe(records)
    data_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    typer.echo(f"Processados {len(result)} registros")


@app.command("build-graph")
def build_graph_cmd(
    dataset: str = typer.Argument(..., help="Caminho para dataset de relações"),
    persist: bool = typer.Option(False, "--persist", help="Salva o grafo no Neo4j"),
):
    """Converte um dataset de relações em grafo."""
    from utils.compression import load_json_file
    from utils.relation import relations_to_graph

    data = load_json_file(dataset)
    g = relations_to_graph(data, persist=persist)
    typer.echo(
        f"Graph construido com {g.number_of_nodes()} nós e {g.number_of_edges()} arestas"
    )


@app.command("merge-datasets")
def merge_datasets_cmd(
    datasets: List[str] = typer.Argument(..., help="Caminhos para arquivos JSON"),
    output: str = typer.Option(
        "merged_dataset.json", "--output", "-o", help="Arquivo de saída"
    ),
):
    """Une múltiplos datasets em um único arquivo deduplicado."""
    from processing.aggregator import merge_datasets

    merged = merge_datasets(datasets)
    Path(output).write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    typer.echo(f"Salvo {len(merged)} registros em {output}")


@app.command("start-crawler")
def start_crawler_cmd(
    config: str = typer.Option(None, "--config", help="Path to cluster config")
):
    """Start the distributed crawler."""

    from crawling.distributed import start_crawler

    start_crawler(config)


@app.command("stop-crawler")
def stop_crawler_cmd():
    """Stop the distributed crawler."""

    from crawling.distributed import stop_crawler

    stop_crawler()


@app.command("update-parsers")
def update_parsers_cmd():
    """Refresh ANTLR grammars or ML models."""
    import time

    grammar_dir = Path("parsers")
    for g in grammar_dir.glob("*.g4"):
        typer.echo(f"Updating {g.name}")
        # Placeholder for real compilation or download logic
    metrics.parser_update_timestamp.set(time.time())
    typer.echo("Parsers updated")


@app.command("process-raw")
def process_raw_cmd(
    raw_dir: str = typer.Option(
        "datasets/raw", "--raw-dir", help="Diretório de dumps brutos"
    ),
    fmt: str = typer.Option("all", "--format", help="Formato de saída"),
):
    """Processa arquivos brutos gerados previamente."""
    from tqdm import tqdm

    builder = scraper_wiki.DatasetBuilder()
    path = Path(raw_dir)
    files = sorted(path.glob("*.json"))
    for fp in tqdm(files, desc="Processando dumps"):
        data = json.loads(fp.read_text(encoding="utf-8"))
        title = data.get("title", "")
        lang = data.get("lang", "")
        category = data.get("category", "")
        html = data.get("html", "")
        text = data.get("text", "")
        clean = scraper_wiki.advanced_clean_text(
            text, lang, remove_stopwords=scraper_wiki.Config.REMOVE_STOPWORDS
        )
        record = scraper_wiki.cpu_process_page(
            title,
            clean,
            lang,
            category,
            scraper_wiki.extract_images(html),
            scraper_wiki.extract_videos(html),
            None,
            builder.rev_limit,
        )
        builder.dataset.append(record)

    builder.enhance_with_clustering()
    builder.save_dataset(format=fmt)


if __name__ == "__main__":
    app()
