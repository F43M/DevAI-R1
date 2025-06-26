# Knowledge Graph

This project can transform scraped datasets into a graph stored in Neo4j.

## Building the Graph

Use the CLI to convert a JSON file of triples into a graph:

```bash
python cli.py build-graph path/to/dataset_triples.json --persist
```

The `--persist` flag sends all edges to the database using the connection
information from the `NEO4J_URI`, `NEO4J_USER` and `NEO4J_PASSWORD`
environment variables.

## Example Queries

In the Neo4j browser you can explore the data with Cypher:

```cypher
// List first relations
MATCH (a:Entity)-[r:RELATED]->(b:Entity)
RETURN a,b,r LIMIT 25;

// Find everything related to Python
MATCH (p:Entity {id:"Python"})<-[:RELATED]-(n)
RETURN n;
```

## Visualization

Neo4j Browser provides a simple visualization. After running a query, switch to
*Graph* view to see nodes and edges. For larger exports consider tools like
Neo4j Bloom or `networkx` drawing utilities.
