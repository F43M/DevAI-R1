import json
from typing import Iterable, List, Dict

import requests


DEFAULT_URL = "http://localhost:9200"
DEFAULT_INDEX = "records"


def bulk_index(
    records: Iterable[Dict],
    *,
    es_url: str = DEFAULT_URL,
    index: str = DEFAULT_INDEX,
) -> Dict:
    """Index ``records`` using the Elasticsearch bulk API."""

    actions = []
    for rec in records:
        actions.append(json.dumps({"index": {"_index": index, "_id": rec.get("id")}}))
        actions.append(json.dumps(rec, ensure_ascii=False))
    payload = "\n".join(actions) + "\n"
    resp = requests.post(
        f"{es_url}/_bulk",
        data=payload.encode("utf-8"),
        headers={"Content-Type": "application/x-ndjson"},
    )
    resp.raise_for_status()
    return resp.json()


def query_index(
    query: str,
    *,
    es_url: str = DEFAULT_URL,
    index: str = DEFAULT_INDEX,
    size: int = 10,
) -> List[Dict]:
    """Return records from ``index`` matching ``query``."""

    params = {"q": query, "size": size}
    resp = requests.get(f"{es_url}/{index}/_search", params=params)
    resp.raise_for_status()
    hits = resp.json().get("hits", {}).get("hits", [])
    return [h.get("_source", {}) for h in hits]
