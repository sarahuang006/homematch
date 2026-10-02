"""Two retrieval modes, so we can compare them in the evaluation step."""
from dataclasses import dataclass

from openai import OpenAI

from homematch.filters import build_where
from homematch.index import embed, get_collection
from homematch.query_parser import ParsedQuery, parse_query


@dataclass
class SearchResult:
    listing_id: str
    distance: float  # smaller = more similar
    metadata: dict
    document: str


def _query(client, embedding_model, text: str, k: int, where: dict | None) -> list[SearchResult]:
    collection = get_collection()
    kwargs = {"query_embeddings": embed(client, embedding_model, [text]), "n_results": k}
    if where:
        kwargs["where"] = where
    res = collection.query(**kwargs)
    return [
        SearchResult(i, d, m, doc)
        for i, d, m, doc in zip(res["ids"][0], res["distances"][0], res["metadatas"][0], res["documents"][0])
    ]


def vector_search(client: OpenAI, embedding_model: str, query: str, k: int = 5) -> list[SearchResult]:
    """Baseline: semantic similarity only, no filters."""
    return _query(client, embedding_model, query, k, where=None)


def hybrid_search(
    client: OpenAI, chat_model: str, embedding_model: str, query: str, k: int = 5
) -> tuple[ParsedQuery, list[SearchResult]]:
    """LLM extracts exact filters, then vector search ranks only the listings that pass them."""
    parsed = parse_query(client, chat_model, query)
    results = _query(client, embedding_model, parsed.semantic_query, k, where=build_where(parsed))
    return parsed, results
