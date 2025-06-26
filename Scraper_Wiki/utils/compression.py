import gzip
import io
from pathlib import Path


def compress_bytes(data: bytes, algorithm: str) -> bytes:
    """Return ``data`` compressed with the given ``algorithm``."""
    alg = algorithm.lower() if algorithm else "none"
    if alg == "gzip":
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as fh:
            fh.write(data)
        return buf.getvalue()
    if alg == "zstd":
        import zstandard as zstd

        return zstd.ZstdCompressor().compress(data)
    return data


def decompress_bytes(data: bytes, algorithm: str) -> bytes:
    """Decompress ``data`` previously compressed with ``algorithm``."""
    alg = algorithm.lower() if algorithm else "none"
    if alg == "gzip":
        return gzip.decompress(data)
    if alg == "zstd":
        import zstandard as zstd

        return zstd.ZstdDecompressor().decompress(data)
    return data


def load_json_file(path: str | Path) -> list[dict]:
    """Load JSON dataset possibly compressed by extension."""
    p = Path(path)
    data = p.read_bytes()
    if p.suffix == ".gz":
        data = gzip.decompress(data)
    elif p.suffix == ".zst":
        import zstandard as zstd

        data = zstd.ZstdDecompressor().decompress(data)
    import json

    return json.loads(data.decode("utf-8"))
