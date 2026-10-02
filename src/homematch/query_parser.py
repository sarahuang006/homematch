"""Turn a natural-language request into exact filters + a semantic query."""
import json

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from homematch.schemas import PropertyType


class ParsedQuery(BaseModel):
    min_price: int | None = None
    max_price: int | None = None
    min_bedrooms: int | None = None
    max_bedrooms: int | None = None
    min_bathrooms: float | None = None
    min_size_sqft: int | None = None
    max_size_sqft: int | None = None
    property_type: PropertyType | None = None
    semantic_query: str  # the "soft" preferences, used for vector search


PROMPT = """Extract search filters from a home buyer's request.

Return JSON with these keys (use null when the request doesn't mention it):
min_price, max_price (int, USD), min_bedrooms, max_bedrooms (int), min_bathrooms (float),
min_size_sqft, max_size_sqft (int), property_type ("house" | "condo" | "townhouse" | null),
semantic_query (str).

Rules:
- Only set property_type if the user says house, condo, or townhouse. "home" or "place" = null.
- "N bedrooms" means min_bedrooms = N. Only set max_bedrooms if the user sets an upper limit.
- "under $800k" means max_price = 800000. "around $500k" means min_price 450000, max_price 550000.
- semantic_query: the remaining preferences (style, features, neighborhood feel), written as a
  short search phrase. If there are none, use the original request.

Request: {query}"""


def parse_response(raw_json: str, original_query: str) -> ParsedQuery:
    """Validate the LLM output. If it's unusable, fall back to plain semantic search."""
    try:
        return ParsedQuery.model_validate(json.loads(raw_json))
    except (json.JSONDecodeError, ValidationError):
        return ParsedQuery(semantic_query=original_query)


def parse_query(client: OpenAI, model: str, query: str) -> ParsedQuery:
    response = client.chat.completions.create(
        model=model,
        temperature=0,  # we want the same answer every time
        max_tokens=300,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": PROMPT.format(query=query)}],
    )
    return parse_response(response.choices[0].message.content, query)
