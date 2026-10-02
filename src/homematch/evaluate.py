"""Evaluation: compare search results against ground truth computed from structured data."""
from pydantic import BaseModel

from homematch.query_parser import ParsedQuery
from homematch.schemas import Listing

FILTER_FIELDS = [
    "min_price", "max_price", "min_bedrooms", "max_bedrooms",
    "min_bathrooms", "min_size_sqft", "max_size_sqft", "property_type",
]


class Constraints(BaseModel):
    """What a correct result must satisfy. Written by hand, NOT by the LLM (avoids circular grading)."""

    min_price: int | None = None
    max_price: int | None = None
    min_bedrooms: int | None = None
    max_bedrooms: int | None = None
    min_bathrooms: float | None = None
    min_size_sqft: int | None = None
    max_size_sqft: int | None = None
    property_type: str | None = None
    features: list[str] = []  # each must appear in a listing's feature tags (substring match)


class EvalQuery(BaseModel):
    query: str
    constraints: Constraints


def meets_hard_constraints(listing: Listing, c: Constraints) -> bool:
    checks = [
        c.min_price is None or listing.price >= c.min_price,
        c.max_price is None or listing.price <= c.max_price,
        c.min_bedrooms is None or listing.bedrooms >= c.min_bedrooms,
        c.max_bedrooms is None or listing.bedrooms <= c.max_bedrooms,
        c.min_bathrooms is None or listing.bathrooms >= c.min_bathrooms,
        c.min_size_sqft is None or listing.size_sqft >= c.min_size_sqft,
        c.max_size_sqft is None or listing.size_sqft <= c.max_size_sqft,
        c.property_type is None or listing.property_type == c.property_type,
    ]
    return all(checks)


def has_features(listing: Listing, c: Constraints) -> bool:
    tags = " | ".join(listing.features).lower()
    return all(feature.lower() in tags for feature in c.features)


def is_relevant(listing: Listing, c: Constraints) -> bool:
    return meets_hard_constraints(listing, c) and has_features(listing, c)


def score_query(returned_ids: list[str], listings: dict[str, Listing], c: Constraints, k: int) -> dict:
    returned = [listings[i] for i in returned_ids[:k]]
    relevant_ids = {lid for lid, l in listings.items() if is_relevant(l, c)}
    hits = sum(1 for l in returned if l.id in relevant_ids)

    return {
        "n_returned": len(returned),
        "n_relevant_total": len(relevant_ids),
        # Precision is measured over k slots: returning nothing counts as 0, not as "perfect".
        "constraint_precision": sum(meets_hard_constraints(l, c) for l in returned) / k,
        "full_precision": hits / k,
        # Recall: of what we COULD have found (at most k), how much did we find?
        "recall": hits / min(k, len(relevant_ids)) if relevant_ids else None,
    }


def parser_is_correct(parsed: ParsedQuery, c: Constraints) -> bool:
    return all(getattr(parsed, f) == getattr(c, f) for f in FILTER_FIELDS)
