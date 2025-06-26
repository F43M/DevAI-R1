from __future__ import annotations

import os
from typing import Optional

import networkx as nx


def save_graph(
    graph: nx.DiGraph,
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> None:
    """Persist a NetworkX graph to Neo4j."""
    if graph.number_of_edges() == 0:
        return
    try:  # pragma: no cover - optional dependency
        from neo4j import GraphDatabase
    except Exception as exc:  # pragma: no cover - missing deps
        raise ImportError("neo4j is required for graph persistence") from exc

    uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = user or os.getenv("NEO4J_USER", "neo4j")
    password = password or os.getenv("NEO4J_PASSWORD", "test")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            for src, dst, data in graph.edges(data=True):
                session.run(
                    "MERGE (a:Entity {id:$from}) "
                    "MERGE (b:Entity {id:$to}) "
                    "MERGE (a)-[:RELATED {type:$rel}]->(b)",
                    {"from": src, "to": dst, "rel": data.get("relation")},
                )
    finally:
        driver.close()
