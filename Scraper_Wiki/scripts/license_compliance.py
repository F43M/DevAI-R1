#!/usr/bin/env python
"""Verify scraped datasets include proper licensing information."""

from __future__ import annotations

import sys
from pathlib import Path

LICENSE_KEYWORDS = ["CC BY-SA", "Creative Commons"]


def check_license(dataset_dir: str) -> int:
    """Return 0 if the directory contains a license mentioning CC BY-SA."""
    path = Path(dataset_dir)
    if not path.exists():
        print(f"Dataset directory {dataset_dir} not found", file=sys.stderr)
        return 1
    license_files = [p for p in path.rglob("LICENSE*") if p.is_file()] + [
        p for p in path.rglob("license*") if p.is_file()
    ]
    if not license_files:
        print(f"No license file found in {dataset_dir}", file=sys.stderr)
        return 1
    for lf in license_files:
        content = lf.read_text(errors="ignore")
        if any(keyword in content for keyword in LICENSE_KEYWORDS):
            return 0
    print(
        f"License files found in {dataset_dir} but none reference 'CC BY-SA'",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    directory = sys.argv[1] if len(sys.argv) > 1 else "datasets_wikipedia_pro"
    return check_license(directory)


if __name__ == "__main__":
    raise SystemExit(main())
