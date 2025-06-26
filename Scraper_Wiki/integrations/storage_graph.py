from typing import List
from integrations.storage import StorageBackend


class GraphStorage(StorageBackend):
    """Store relationships in a Neo4j graph."""

    def __init__(self, uri: str, user: str, password: str):
        try:
            from neo4j import GraphDatabase
        except Exception as e:  # pragma: no cover - missing deps
            raise ImportError("neo4j is required for GraphStorage") from e
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def save_dataset(
        self, data: List[dict], fmt: str = "all", version: str | None = None
    ) -> None:
        if not data:
            return
        with self.driver.session() as session:
            for row in data:
                session.run(
                    "MERGE (a:Entity {id:$from}) MERGE (b:Entity {id:$to}) "
                    "MERGE (a)-[:RELATED {type:$rel}]->(b)",
                    {
                        "from": row.get("from"),
                        "to": row.get("to"),
                        "rel": row.get("relation"),
                    },
                )
