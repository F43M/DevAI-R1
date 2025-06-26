"""Helpers to load distributed cluster configuration."""

import os
import yaml

DEFAULT_CONFIG = "cluster.yaml"


def load_config(path: str | None = None) -> dict:
    path = path or DEFAULT_CONFIG
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_client(config: dict | None = None):
    cfg = config or load_config()
    backend = cfg.get("cluster", {}).get("backend")
    scheduler = cfg.get("cluster", {}).get("scheduler")
    if backend == "dask":
        try:
            from dask.distributed import Client

            return Client(scheduler)
        except Exception:
            raise RuntimeError("Dask not available")
    elif backend == "ray":
        try:
            import ray

            ray.init(address=scheduler)

            class _RayFuture:
                def __init__(self, ref):
                    self._ref = ref

                def result(self):
                    return ray.get(self._ref)

            class _RayClient:
                def submit(self, fn, *args, **kwargs):
                    remote_fn = ray.remote(fn)
                    ref = remote_fn.remote(*args, **kwargs)
                    return _RayFuture(ref)

            return _RayClient()
        except Exception:
            raise RuntimeError("Ray not available")
    return None
