"""Helper to track dataset changes with DVC."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("wiki_scraper")


def track_path(path: str) -> None:
    """Add ``path`` to DVC and push updates."""
    if not Path(path).exists():
        return
    try:
        subprocess.run(["dvc", "add", path], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["dvc", "push"], check=True, stdout=subprocess.DEVNULL)
    except Exception as exc:  # pragma: no cover - dvc missing
        logger.error(f"Failed to push dataset to DVC: {exc}")
