# Setup

## Installation

```bash
pip install -r requirements.txt
```

Heavy optional dependencies such as `tensorflow` and `transformers` can be skipped if not needed.

## Environment Variables

- `STORAGE_BACKEND`: `local`, `s3`, `mongodb`, `postgres`, `iceberg`, `delta`, `datalake`, `neo4j`, `milvus`, or `weaviate`.
- `S3_BUCKET`, `S3_ENDPOINT`: Configure S3/MinIO storage.
- `MONGODB_URI`: MongoDB connection string.
- `POSTGRES_DSN`: PostgreSQL DSN.
- `DATALAKE_PATH`: Path for Iceberg/Delta Lake.
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`: Neo4j connection.
- `MILVUS_URI`, `MILVUS_COLLECTION`: Milvus vector store.
- `WEAVIATE_URI`: Weaviate endpoint.
- `USE_GPU`: Set to `1` or `true` to force GPU usage, `0` to disable. When unset,
  GPU availability is detected automatically via `torch.cuda.is_available()`.

