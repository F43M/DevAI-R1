"""Background worker that consumes page tasks and publishes results."""

import asyncio
import logging
import os
from task_queue import consume, publish
from scraper_wiki import DatasetBuilder, Config, get_base_url
from metrics import start_metrics_server, start_system_metrics_loop
from provenance.tracker import should_fetch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main_sync() -> None:
    """Run worker in synchronous mode."""
    start_metrics_server(int(os.environ.get("METRICS_PORT", "8001")))
    start_system_metrics_loop()
    builder = DatasetBuilder()
    for task, ack in consume("scrape_tasks", manual_ack=True):
        url = task.get("url")
        if not url and {"title", "lang"} <= task.keys():
            url = f"{get_base_url(task['lang'])}/wiki/{task['title'].replace(' ', '_')}"
        if url and not should_fetch(url):
            logger.info("Skipping %s", url)
            continue
        logger.info("Processing %s", task.get("title") or url)
        result = builder.process_page(task)
        if result:
            publish("scrape_results", result)
        ack()


async def _handle_task(
    task: dict, ack_fn, builder: DatasetBuilder, sem: asyncio.Semaphore
) -> None:
    async with sem:
        url = task.get("url")
        if not url and {"title", "lang"} <= task.keys():
            url = f"{get_base_url(task['lang'])}/wiki/{task['title'].replace(' ', '_')}"
        if url and not should_fetch(url):
            logger.info("Skipping %s", url)
            return
        logger.info("Processing %s", task.get("title") or url)
        result = await builder.process_page_async(task)
        if result:
            await asyncio.to_thread(publish, "scrape_results", result)
        ack_fn()


async def main_async() -> None:
    """Run worker using asynchronous scraping."""
    start_metrics_server(int(os.environ.get("METRICS_PORT", "8001")))
    start_system_metrics_loop()
    builder = DatasetBuilder()
    sem = asyncio.Semaphore(Config.WORKER_CONCURRENCY)
    iterator = consume("scrape_tasks", manual_ack=True)
    while True:
        task, ack = await asyncio.to_thread(next, iterator)
        asyncio.create_task(_handle_task(task, ack, builder, sem))


def main(async_mode: bool = False) -> None:
    """Entry point for the worker."""
    if async_mode:
        asyncio.run(main_async())
    else:
        main_sync()


if __name__ == "__main__":
    import sys

    main(async_mode="--async" in sys.argv)
