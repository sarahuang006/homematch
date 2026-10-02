"""Generate varied, structured real-estate listings with an LLM."""
import json

from openai import OpenAI
from pydantic import ValidationError

from homematch.schemas import Listing, ListingDraft

# Each batch gets a different "scenario" so listings don't all sound the same.
SCENARIOS = [
    "dense downtown area with high-rise condos, transit, and nightlife",
    "quiet suburb with good schools, larger yards, and family homes",
    "rural area with land, privacy, and long drives to town",
    "waterfront neighborhood near a lake or the ocean",
    "historic district with older homes and walkable streets",
    "mountain or hillside area with views and hiking trails",
    "budget-friendly area for first-time buyers",
    "luxury area with high-end homes and amenities",
    "college town with a younger population and townhouses",
    "fast-growing area with new construction and modern, energy-efficient homes",
]

PROMPT = """Generate {n} realistic, varied US real-estate listings for this setting: {scenario}.
Vary prices, sizes, bedroom counts (1-6), property types, and features. Do not reuse
neighborhood names or copy phrases between listings. Some homes should have drawbacks.

Return JSON: {{"listings": [ ... ]}} where each listing has exactly these keys:
neighborhood (str), property_type ("house" | "condo" | "townhouse"), price (int, USD),
bedrooms (int), bathrooms (float), size_sqft (int), features (list of 3-6 short tags),
description (3-4 sentences), neighborhood_description (2-3 sentences)."""


def parse_listings(raw_json: str, start_index: int) -> list[Listing]:
    """Validate LLM output and assign ids. Invalid listings are skipped, not crashed on."""
    items = json.loads(raw_json).get("listings", [])
    listings = []
    for item in items:
        try:
            draft = ListingDraft.model_validate(item)
        except ValidationError as err:
            print(f"Skipping invalid listing: {err.errors()[0]['msg']}")
            continue
        listing_id = f"L{start_index + len(listings):04d}"
        listings.append(Listing(id=listing_id, **draft.model_dump()))
    return listings


def generate_batch(
    client: OpenAI, model: str, scenario: str, n: int, start_index: int, max_retries: int = 3
) -> list[Listing]:
    """Ask the LLM for n listings. Retries if the response is broken or cut off."""
    for attempt in range(1, max_retries + 1):
        response = client.chat.completions.create(
            model=model,
            temperature=0.8,
            max_tokens=4000,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": PROMPT.format(n=n, scenario=scenario)}],
        )
        choice = response.choices[0]
        if choice.finish_reason == "length":
            print(f"  attempt {attempt}: response was cut off (too long), retrying")
            continue
        try:
            return parse_listings(choice.message.content, start_index)
        except json.JSONDecodeError as err:
            print(f"  attempt {attempt}: invalid JSON ({err.msg}), retrying")
    print(f"  giving up on this batch after {max_retries} attempts")
    return []
