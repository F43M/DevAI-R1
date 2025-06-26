"""Minimal SonarQube API integration for static analysis."""

from __future__ import annotations

import logging
import os
from typing import Dict

import requests

logger = logging.getLogger(__name__)


def analyze_code(code: str) -> Dict[str, float]:
    """Analyze ``code`` using a SonarQube server if configured."""
    url = os.getenv("SONARQUBE_URL")
    token = os.getenv("SONARQUBE_TOKEN")
    if not url or not token:
        return {}
    try:
        resp = requests.post(
            f"{url}/api/measure/analyze",
            headers={"Authorization": f"Bearer {token}"},
            json={"code": code},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("measures", {})
    except Exception as e:  # pragma: no cover - network issues
        logger.error("SonarQube analysis failed: %s", e)
        return {}
