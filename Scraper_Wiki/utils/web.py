"""Utilities for web crawling and navigation."""

from __future__ import annotations

import posixpath
import random
import re
from typing import Any, Dict
from urllib.parse import parse_qsl, urlparse, urlunparse, urlencode

import numpy as np


_TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
}


def normalize_url(url: str) -> str:
    """Return a normalized representation of ``url``.

    The function performs several transformations to ensure URLs can be
    compared reliably:

    * lowercase scheme and domain;
    * remove default ports (80 for HTTP, 443 for HTTPS);
    * remove fragments;
    * normalize path resolving ``."" and ``..`` segments;
    * remove duplicate slashes and trailing slash;
    * sort query parameters and drop common tracking keys;
    * convert IDN domains to ASCII using ``idna``.
    """

    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    host = parsed.hostname.lower() if parsed.hostname else ""
    if host.startswith("www."):
        host = host[4:]
    try:
        host = host.encode("idna").decode("ascii")
    except Exception:
        pass
    port = f":{parsed.port}" if parsed.port else ""
    if (scheme == "http" and parsed.port == 80) or (
        scheme == "https" and parsed.port == 443
    ):
        port = ""
    netloc = host + port

    path = posixpath.normpath(parsed.path)
    path = re.sub(r"/+", "/", path)
    if path == ".":
        path = "/"

    query_items = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if k not in _TRACKING_PARAMS
    ]
    query = urlencode(sorted(query_items))

    normalized = urlunparse((scheme, netloc, path, "", query, ""))
    if normalized.endswith("/") and path != "/":
        normalized = normalized[:-1]
    return normalized


def _match_domain_pattern(patterns: list[dict[str, Any]], classes: list[str]) -> int:
    return int(any(any(c in classes for c in p.get("classes", [])) for p in patterns))


def decide_navigation_action(
    link_info: Dict[str, Any],
    domain: str,
    knowledge_base: Dict[str, Any],
    navigation_model: Any,
    config: Dict[str, Any],
) -> int:
    """Choose how to handle ``link_info`` discovered while crawling.

    The ``navigation_model`` is expected to expose a ``predict`` method taking
    a ``numpy`` array of shape ``(n_samples, n_features)`` and returning
    probabilities for three actions: skip, follow, or extract. ``knowledge_base``
    may contain site specific patterns used to enrich feature vectors.
    """

    domain_patterns = knowledge_base.get("site_patterns", {}).get(domain, {})
    classes = link_info.get("element", [])

    features = [
        len(link_info.get("url", "")),
        len(link_info.get("text", "")),
        int(
            any(
                k in link_info.get("url", "").lower()
                for k in config.get("priority_urls", [])
            )
        ),
        int(
            any(
                k in link_info.get("text", "").lower()
                for k in config.get("content_keywords", [])
            )
        ),
        int(
            any(
                k in link_info.get("text", "").lower()
                for k in config.get("navigation_keywords", [])
            )
        ),
        int("#" in link_info.get("url", "")),
        int(link_info.get("url", "").endswith("/")),
        link_info.get("url", "").count("/"),
        int(re.search(r"\d", link_info.get("url", "")) is not None),
        int(domain in link_info.get("url", "")),
    ]

    features.extend(
        [
            _match_domain_pattern(domain_patterns.get("content_elements", []), classes),
            _match_domain_pattern(
                domain_patterns.get("navigation_elements", []), classes
            ),
            _match_domain_pattern(domain_patterns.get("data_elements", []), classes),
            _match_domain_pattern(domain_patterns.get("anti_patterns", []), classes),
        ]
    )

    arr = np.array([features], dtype=float)
    try:
        action_probs = navigation_model.predict(arr)[0]
        action = int(np.argmax(action_probs))
    except Exception:
        action = 1

    if random.random() < config.get("exploration_rate", 0.0):
        action = random.randint(0, 2)

    rate = config.get("exploration_rate", 0.0) * config.get("exploration_decay", 1.0)
    config["exploration_rate"] = max(rate, config.get("min_exploration_rate", 0.0))

    return action


__all__ = ["normalize_url", "decide_navigation_action"]
