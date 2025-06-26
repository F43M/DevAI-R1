# -*- coding: utf-8 -*-
"""Simple relation extraction utilities."""

from __future__ import annotations

from typing import List, Dict
import os
import re
import networkx as nx


def _token_to_ent_text(token, ents):
    """Return entity text covering the token if present."""
    for ent in ents:
        if ent.start <= token.i < ent.end:
            return ent.text
    return token.text


def extract_relations(text: str, lang: str = "en") -> List[Dict[str, str]]:
    """Extract basic subject-verb-object relations using spaCy."""
    try:
        from scraper_wiki import NLPProcessor

        nlp = NLPProcessor.get_instance(lang)
        doc = nlp(text)
    except Exception:
        return []

    relations: List[Dict[str, str]] = []
    for sent in doc.sents:
        root = next((t for t in sent if t.dep_ == "ROOT"), None)
        if not root:
            continue
        subj = next(
            (c for c in root.children if c.dep_ in {"nsubj", "nsubjpass"}), None
        )
        obj = next(
            (c for c in root.children if c.dep_ in {"dobj", "attr", "pobj", "obj"}),
            None,
        )
        if subj and obj:
            relations.append(
                {
                    "subject": _token_to_ent_text(subj, doc.ents),
                    "relation": root.lemma_,
                    "object": _token_to_ent_text(obj, doc.ents),
                }
            )
    return relations


def extract_relations_regex(text: str) -> List[Dict[str, str]]:
    """Extract simple relations based on a regex pattern."""
    pattern = re.compile(
        r"([A-Z][a-zA-Z]*) (worked at|studied at|discovered) ([A-Z][a-zA-Z]*)"
    )
    relations: List[Dict[str, str]] = []
    for match in pattern.finditer(text):
        subject, relation, obj = match.groups()
        relations.append({"subject": subject, "relation": relation, "object": obj})
    return relations


def relations_to_graph(
    relations: List[Dict[str, str]], persist: bool | None = None
) -> nx.DiGraph:
    """Convert a list of relations to a directed NetworkX graph.

    Parameters
    ----------
    relations:
        List of extracted relations with ``subject``, ``relation`` and ``object``.
    persist:
        If ``True`` the graph is also persisted to Neo4j. Defaults to ``False``
        unless the ``PERSIST_GRAPH`` environment variable is set.
    """
    graph = nx.DiGraph()
    for rel in relations:
        subj = rel.get("subject")
        obj = rel.get("object")
        graph.add_node(subj)
        graph.add_node(obj)
        graph.add_edge(subj, obj, relation=rel.get("relation"))

    do_persist = persist if persist is not None else os.getenv("PERSIST_GRAPH") == "1"
    if do_persist:
        try:
            from integrations.neo4j_backend import save_graph

            save_graph(graph)
        except Exception:  # pragma: no cover - optional backend
            pass

    return graph
