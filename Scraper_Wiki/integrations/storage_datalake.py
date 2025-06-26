import os
from typing import List
from integrations.storage import StorageBackend
from integrations import datalake


class DatalakeStorage(StorageBackend):
    """Store datasets in a local lake partitioned by ``lang`` and ``domain``."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(path, exist_ok=True)

    def save_dataset(
        self,
        data: List[dict],
        fmt: str = "all",
        version: str | None = None,
        compression: str = "none",
    ) -> None:
        if fmt not in ["all", "parquet"]:
            return

        datalake.write_parquet(data, self.path, compression)

        if version:
            with open(
                os.path.join(self.path, "dataset_version.txt"), "w", encoding="utf-8"
            ) as vf:
                vf.write(version)
