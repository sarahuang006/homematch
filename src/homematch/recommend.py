"""Generate a grounded recommendation from retrieved listings, streamed token by token."""
from collections.abc import Iterator

from openai import OpenAI

from homematch.search import SearchResult

SYSTEM_PROMPT = """You are a real-estate assistant. Recommend homes ONLY from the listings provided.
For each recommendation, mention its id (e.g. L0012), price, and why it fits the request.
If a listing is missing something the buyer wants (e.g. no fireplace), say so honestly.
If no listings are provided, say nothing matched and suggest which requirement to relax.
Never invent listings or details. Keep it concise."""


def build_messages(query: str, results: list[SearchResult]) -> list[dict]:
    context = "\n\n".join(f"[{r.listing_id}]\n{r.document}" for r in results)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Buyer request: {query}\n\nListings:\n{context or '(no listings matched)'}"},
    ]


def stream_recommendation(client: OpenAI, model: str, query: str, results: list[SearchResult]) -> Iterator[str]:
    stream = client.chat.completions.create(
        model=model, temperature=0.3, stream=True, messages=build_messages(query, results)
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
