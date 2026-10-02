"""Compare vector search vs hybrid search for one query.
Usage: PYTHONPATH=src python scripts/search_demo.py "4 bedroom house with fireplace under $800k"
"""
import sys

from openai import OpenAI

from homematch.config import OPENAI_EMBEDDING_MODEL, OPENAI_MODEL, require_api_key
from homematch.search import hybrid_search, vector_search


def show(results) -> None:
    for r in results:
        m = r.metadata
        print(
            f"  {r.listing_id}  ${m['price']:>9,}  {m['bedrooms']}bd  {m['property_type']:<9}"
            f"  {m['neighborhood']}  (distance {r.distance:.3f})"
        )
    if not results:
        print("  (no listings match these filters)")


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "4 bedroom house with a fireplace under $800k"
    client = OpenAI(api_key=require_api_key())
    print(f'Query: "{query}"\n')

    print("VECTOR SEARCH (baseline):")
    show(vector_search(client, OPENAI_EMBEDDING_MODEL, query))

    parsed, results = hybrid_search(client, OPENAI_MODEL, OPENAI_EMBEDDING_MODEL, query)
    print("\nHYBRID SEARCH:")
    print(f"  parsed filters: {parsed.model_dump(exclude_none=True)}")
    show(results)


if __name__ == "__main__":
    main()
