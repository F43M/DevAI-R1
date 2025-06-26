"""Utility to send alerts to maintainers."""

from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger("wiki_scraper")


def send_alert(message: str) -> None:
    """Send a notification using the configured webhook."""
    from scraper_wiki import Config

    if not Config.ALERT_WEBHOOK_URL:
        return
    try:
        requests.post(
            Config.ALERT_WEBHOOK_URL,
            json={"text": message},
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - network errors
        logger.error(f"Failed to send alert: {exc}")
