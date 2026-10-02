"""Convert parsed filters into Chroma's `where` syntax."""
from homematch.query_parser import ParsedQuery

# (field on ParsedQuery, metadata field in Chroma, operator)
RULES = [
    ("min_price", "price", "$gte"),
    ("max_price", "price", "$lte"),
    ("min_bedrooms", "bedrooms", "$gte"),
    ("max_bedrooms", "bedrooms", "$lte"),
    ("min_bathrooms", "bathrooms", "$gte"),
    ("min_size_sqft", "size_sqft", "$gte"),
    ("max_size_sqft", "size_sqft", "$lte"),
    ("property_type", "property_type", "$eq"),
]


def build_where(parsed: ParsedQuery) -> dict | None:
    conditions = []
    for attr, field, op in RULES:
        value = getattr(parsed, attr)
        if value is not None:
            conditions.append({field: {op: value}})

    if not conditions:
        return None
    if len(conditions) == 1:  # Chroma requires "$and" to have at least 2 items
        return conditions[0]
    return {"$and": conditions}
