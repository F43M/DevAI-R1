"""License compliance utilities."""

from __future__ import annotations

import json
from typing import Iterable
from urllib.parse import urlparse

import requests

ALLOWED_LICENSES = {
    "CC BY-SA 3.0",
    "CC BY-SA 4.0",
    "MIT",
    "Apache-2.0",
}


def _github_api_license(owner: str, repo: str) -> str:
    """Return the license identifier for a GitHub repository."""
    url = f"https://api.github.com/repos/{owner}/{repo}/license"
    try:
        resp = requests.get(url, timeout=5)
        if resp.ok:
            data = resp.json()
            lic = data.get("license", {}).get("spdx_id") or data.get("license", {}).get(
                "name"
            )
            if lic:
                return lic
    except Exception:
        pass
    return "unknown"


def check_license(url: str, allowed: Iterable[str] = ALLOWED_LICENSES) -> str:
    """Check the license for ``url`` and ensure it is allowed.

    Parameters
    ----------
    url:
        Source URL.
    allowed:
        Iterable of accepted license identifiers.

    Returns
    -------
    str
        Detected license identifier or ``"unknown"``.

    Raises
    ------
    ValueError
        If a license is detected but not in ``allowed``.
    """

    domain = urlparse(url).netloc.lower()
    license_id = "unknown"
    if "wikipedia.org" in domain:
        license_id = "CC BY-SA 3.0"
    elif "github.com" in domain:
        parts = urlparse(url).path.strip("/").split("/")
        if len(parts) >= 2:
            license_id = _github_api_license(parts[0], parts[1])
    if license_id != "unknown" and allowed and license_id not in set(allowed):
        raise ValueError(f"License {license_id} for {url} not allowed")
    return license_id


__all__ = ["check_license"]
