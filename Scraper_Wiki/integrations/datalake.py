"""Utility helpers for writing datasets to a lake."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List


def write_parquet(
    data: List[dict],
    path: str,
    compression: str = "none",
) -> None:
    """Write records to partitioned Parquet dataset.

    The output is compatible with Delta Lake and Apache Iceberg as it uses
    ``pyarrow.dataset.write_dataset`` to create partitioned directories
    for ``lang`` and ``domain`` columns.
    """

    if not data:
        return

    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("pyarrow is required for datalake operations") from exc

    table = pa.Table.from_pylist(data)
    ds.write_dataset(
        table,
        base_dir=path,
        format="parquet",
        partitioning=["lang", "domain"],
        existing_data_behavior="overwrite_or_ignore",
    )

    if compression != "none":
        from utils.compression import compress_bytes
        import glob

        ext = ".zst" if compression == "zstd" else ".gz"
        for file in glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True):
            raw = Path(file).read_bytes()
            comp = compress_bytes(raw, compression)
            Path(file + ext).write_bytes(comp)
            os.remove(file)
