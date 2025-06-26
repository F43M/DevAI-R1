import os
import json
import csv
from pathlib import Path
from typing import List


def _maybe_upload_large(data: List[dict], name: str) -> None:
    """Upload dataset to cloud storage if it exceeds the configured threshold."""
    threshold = int(os.environ.get("UPLOAD_THRESHOLD_MB", "10")) * 1024 * 1024
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    if len(payload) < threshold:
        return

    bucket = os.environ.get("S3_UPLOAD_BUCKET")
    if bucket:
        try:
            import boto3

            boto3.client("s3").put_object(Bucket=bucket, Key=name, Body=payload)
        except Exception:  # pragma: no cover - missing deps
            return
    bucket = os.environ.get("GCS_UPLOAD_BUCKET")
    if bucket:
        try:
            from google.cloud import storage as gcs

            client = gcs.Client()
            blob = client.bucket(bucket).blob(name)
            blob.upload_from_string(payload)
        except Exception:  # pragma: no cover - missing deps
            return


class StorageBackend:
    """Interface for storage backends."""

    def save_dataset(
        self,
        data: List[dict],
        fmt: str = "all",
        version: str | None = None,
        compression: str = "none",
    ) -> None:
        raise NotImplementedError


class LocalStorage(StorageBackend):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_dataset(
        self,
        data: List[dict],
        fmt: str = "all",
        version: str | None = None,
        compression: str = "none",
    ) -> None:
        if fmt in ["all", "json"]:
            json_file = os.path.join(self.output_dir, "wikipedia_qa.json")
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            if compression != "none":
                from utils.compression import compress_bytes

                raw = Path(json_file).read_bytes()
                comp = compress_bytes(raw, compression)
                ext = ".zst" if compression == "zstd" else ".gz"
                comp_path = json_file + ext
                Path(comp_path).write_bytes(comp)
                os.remove(json_file)
                json_file = comp_path
        if fmt in ["all", "jsonl"]:
            jsonl_file = os.path.join(self.output_dir, "wikipedia_qa.jsonl")
            with open(jsonl_file, "w", encoding="utf-8") as f:
                for row in data:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if compression != "none":
                from utils.compression import compress_bytes

                raw = Path(jsonl_file).read_bytes()
                comp = compress_bytes(raw, compression)
                ext = ".zst" if compression == "zstd" else ".gz"
                comp_path = jsonl_file + ext
                Path(comp_path).write_bytes(comp)
                os.remove(jsonl_file)
                jsonl_file = comp_path
        if fmt in ["all", "csv"]:
            csv_file = os.path.join(self.output_dir, "wikipedia_qa.csv")
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                rows = []
                for row in data:
                    converted = {
                        k: (
                            json.dumps(v, ensure_ascii=False)
                            if isinstance(v, (list, dict))
                            else v
                        )
                        for k, v in row.items()
                    }
                    rows.append(converted)
                writer.writerows(rows)
            if compression != "none":
                from utils.compression import compress_bytes

                raw = Path(csv_file).read_bytes()
                comp = compress_bytes(raw, compression)
                ext = ".zst" if compression == "zstd" else ".gz"
                comp_path = csv_file + ext
                Path(comp_path).write_bytes(comp)
                os.remove(csv_file)
                csv_file = comp_path
        if fmt in ["all", "parquet"]:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq

                parquet_file = os.path.join(self.output_dir, "wikipedia_qa.parquet")
                table = pa.Table.from_pylist(data)
                pq.write_table(table, parquet_file)
                if compression != "none":
                    from utils.compression import compress_bytes

                    raw = Path(parquet_file).read_bytes()
                    comp = compress_bytes(raw, compression)
                    ext = ".zst" if compression == "zstd" else ".gz"
                    comp_path = parquet_file + ext
                    Path(comp_path).write_bytes(comp)
                    os.remove(parquet_file)
                    parquet_file = comp_path
            except Exception:
                pass
        if fmt in ["all", "tfrecord"]:
            try:
                import tensorflow as tf

                tf_path = os.path.join(self.output_dir, "wikipedia_qa.tfrecord")
                with tf.io.TFRecordWriter(tf_path) as writer:
                    for row in data:
                        writer.write(
                            json.dumps(row, ensure_ascii=False).encode("utf-8")
                        )
                if compression != "none":
                    from utils.compression import compress_bytes

                    raw = Path(tf_path).read_bytes()
                    comp = compress_bytes(raw, compression)
                    ext = ".zst" if compression == "zstd" else ".gz"
                    comp_path = tf_path + ext
                    Path(comp_path).write_bytes(comp)
                    os.remove(tf_path)
                    tf_path = comp_path
            except Exception:
                pass
        _maybe_upload_large(data, "wikipedia_qa.json")
        if version:
            with open(
                os.path.join(self.output_dir, "dataset_version.txt"),
                "w",
                encoding="utf-8",
            ) as vf:
                vf.write(version)


class S3Storage(StorageBackend):
    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        endpoint_url: str | None = None,
        client=None,
    ):
        try:
            import boto3
        except Exception as e:
            raise ImportError("boto3 is required for S3Storage") from e
        self.bucket = bucket
        self.prefix = prefix
        self.s3 = client or boto3.client("s3", endpoint_url=endpoint_url)

    def _key(self, name: str) -> str:
        return f"{self.prefix}/{name}" if self.prefix else name

    def save_dataset(
        self,
        data: List[dict],
        fmt: str = "all",
        version: str | None = None,
        compression: str = "none",
    ) -> None:
        if fmt in ["all", "json"]:
            body = json.dumps(data, ensure_ascii=False, indent=4).encode("utf-8")
            from utils.compression import compress_bytes

            body = compress_bytes(body, compression)
            ext = (
                ".zst"
                if compression == "zstd"
                else ".gz" if compression != "none" else ""
            )
            self.s3.put_object(
                Bucket=self.bucket,
                Key=self._key(f"wikipedia_qa.json{ext}"),
                Body=body,
            )
        if fmt in ["all", "jsonl"]:
            lines = "\n".join(json.dumps(row, ensure_ascii=False) for row in data)
            body = compress_bytes(lines.encode("utf-8"), compression)
            ext = (
                ".zst"
                if compression == "zstd"
                else ".gz" if compression != "none" else ""
            )
            self.s3.put_object(
                Bucket=self.bucket,
                Key=self._key(f"wikipedia_qa.jsonl{ext}"),
                Body=body,
            )
        if fmt in ["all", "csv"]:
            import io

            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=data[0].keys())
            writer.writeheader()
            for row in data:
                writer.writerow(
                    {
                        k: (
                            json.dumps(v, ensure_ascii=False)
                            if isinstance(v, (list, dict))
                            else v
                        )
                        for k, v in row.items()
                    }
                )
            body = compress_bytes(buffer.getvalue().encode("utf-8"), compression)
            ext = (
                ".zst"
                if compression == "zstd"
                else ".gz" if compression != "none" else ""
            )
            self.s3.put_object(
                Bucket=self.bucket,
                Key=self._key(f"wikipedia_qa.csv{ext}"),
                Body=body,
            )
        if fmt in ["all", "parquet"]:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq

                table = pa.Table.from_pylist(data)
                buf = pa.BufferOutputStream()
                pq.write_table(table, buf)
                body = compress_bytes(buf.getvalue().to_pybytes(), compression)
                ext = (
                    ".zst"
                    if compression == "zstd"
                    else ".gz" if compression != "none" else ""
                )
                self.s3.put_object(
                    Bucket=self.bucket,
                    Key=self._key(f"wikipedia_qa.parquet{ext}"),
                    Body=body,
                )
            except Exception:
                pass
        if fmt in ["all", "tfrecord"]:
            import tempfile

            try:
                import tensorflow as tf

                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    with tf.io.TFRecordWriter(tmp.name) as writer:
                        for row in data:
                            writer.write(
                                json.dumps(row, ensure_ascii=False).encode("utf-8")
                            )
                    tmp.flush()
                    tmp.seek(0)
                    body = tmp.read()
                from utils.compression import compress_bytes

                body = compress_bytes(body, compression)
                ext = (
                    ".zst"
                    if compression == "zstd"
                    else ".gz" if compression != "none" else ""
                )
                self.s3.put_object(
                    Bucket=self.bucket,
                    Key=self._key(f"wikipedia_qa.tfrecord{ext}"),
                    Body=body,
                )
            except Exception:
                pass
        if version:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=self._key("dataset_version.txt"),
                Body=version.encode("utf-8"),
            )


class MongoDBStorage(StorageBackend):
    def __init__(self, uri: str, db_name: str = "scraper", collection: str = "dataset"):
        try:
            from pymongo import MongoClient
        except Exception as e:
            raise ImportError("pymongo is required for MongoDBStorage") from e
        client = MongoClient(uri)
        self.collection = client[db_name][collection]

    def save_dataset(
        self, data: List[dict], fmt: str = "all", version: str | None = None
    ) -> None:
        if not data:
            return
        self.collection.insert_many(data)
        _maybe_upload_large(data, "mongodb_dataset.json")


class PostgreSQLStorage(StorageBackend):
    def __init__(self, dsn: str, table: str = "dataset"):
        try:
            import psycopg2
        except Exception as e:
            raise ImportError("psycopg2 is required for PostgreSQLStorage") from e
        self.conn = psycopg2.connect(dsn)
        self.table = table
        with self.conn, self.conn.cursor() as cur:
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table} (id SERIAL PRIMARY KEY, data JSONB NOT NULL)"
            )

    def save_dataset(
        self, data: List[dict], fmt: str = "all", version: str | None = None
    ) -> None:
        import json as _json

        with self.conn, self.conn.cursor() as cur:
            for row in data:
                cur.execute(
                    f"INSERT INTO {self.table} (data) VALUES (%s)", [_json.dumps(row)]
                )
        _maybe_upload_large(data, "postgres_dataset.json")


try:  # pragma: no cover - optional backends
    from integrations.storage_datalake import DatalakeStorage
except Exception:  # pragma: no cover - missing deps
    DatalakeStorage = None

try:  # pragma: no cover - optional backends
    from integrations.storage_graph import GraphStorage
except Exception:
    GraphStorage = None

try:  # pragma: no cover - optional backends
    from integrations.vector_storage import (
        MilvusVectorStore,
        WeaviateVectorStore,
    )
except Exception:
    MilvusVectorStore = WeaviateVectorStore = None


def get_backend(name: str, output_dir: str):
    name = (name or "local").lower()
    if name in ["s3", "minio"]:
        bucket = os.environ.get("S3_BUCKET", "datasets")
        prefix = os.environ.get("S3_PREFIX", "")
        endpoint = (
            os.environ.get("S3_ENDPOINT")
            if name == "s3"
            else os.environ.get("MINIO_ENDPOINT")
        )
        return S3Storage(bucket, prefix=prefix, endpoint_url=endpoint)
    if name in ["iceberg", "delta", "datalake"]:
        path = os.environ.get("DATALAKE_PATH", output_dir)
        return DatalakeStorage(path)
    if name == "mongodb":
        uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
        db = os.environ.get("MONGODB_DB", "scraper")
        col = os.environ.get("MONGODB_COLLECTION", "dataset")
        return MongoDBStorage(uri, db, col)
    if name == "postgres":
        dsn = os.environ.get("POSTGRES_DSN", "dbname=scraper user=postgres")
        table = os.environ.get("POSTGRES_TABLE", "dataset")
        return PostgreSQLStorage(dsn, table)
    if name == "neo4j":
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        pwd = os.environ.get("NEO4J_PASSWORD", "test")
        return GraphStorage(uri, user, pwd)
    if name in ["milvus", "weaviate"]:
        if name == "milvus":
            uri = os.environ.get("MILVUS_URI", "http://localhost:19530")
            col = os.environ.get("MILVUS_COLLECTION", "embeddings")
            return MilvusVectorStore(uri, col)
        uri = os.environ.get("WEAVIATE_URI", "http://localhost:8080")
        return WeaviateVectorStore(uri)
    return LocalStorage(output_dir)
