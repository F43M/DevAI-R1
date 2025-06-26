#!/usr/bin/env python
"""Generate dataset metadata JSON."""
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def main(dataset_dir: str, output_file: str, version: str) -> None:
    path = Path(dataset_dir)
    size = 0
    if path.exists():
        for f in path.rglob("*"):
            if f.is_file():
                size += f.stat().st_size
    metadata = {
        "version": version,
        "size_bytes": size,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(output_file, "w") as fh:
        json.dump(metadata, fh)


if __name__ == "__main__":
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "datasets_wikipedia_pro"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "dataset_metadata.json"
    version = os.getenv("VERSION", "0.0.0")
    main(dataset_dir, output_file, version)
