from integrations.vector_storage import MilvusVectorStore, WeaviateVectorStore


def test_vector_store_clients():
    milvus = MilvusVectorStore()
    assert hasattr(milvus.client, "insert")

    weaviate = WeaviateVectorStore()
    assert hasattr(weaviate.client, "insert")
