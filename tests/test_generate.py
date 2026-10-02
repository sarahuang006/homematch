import json

from homematch.generate import parse_listings

VALID = {
    "neighborhood": "Cedar Flats",
    "property_type": "house",
    "price": 650000,
    "bedrooms": 3,
    "bathrooms": 2.0,
    "size_sqft": 1800,
    "features": ["fireplace", "garage"],
    "description": "A bright three-bedroom home with an updated kitchen.",
    "neighborhood_description": "Quiet streets close to parks and schools.",
}


def test_parse_assigns_sequential_ids():
    raw = json.dumps({"listings": [VALID, VALID]})
    listings = parse_listings(raw, start_index=5)
    assert [l.id for l in listings] == ["L0005", "L0006"]


def test_parse_skips_invalid_listing():
    bad = {**VALID, "price": -100}
    raw = json.dumps({"listings": [VALID, bad]})
    assert len(parse_listings(raw, start_index=0)) == 1


def test_document_contains_structured_fields():
    listing = parse_listings(json.dumps({"listings": [VALID]}), 0)[0]
    doc = listing.to_document()
    assert "$650,000" in doc and "3 bedrooms" in doc and "fireplace" in doc


class FakeClient:
    """Pretends to be the OpenAI client, so we can test without calling the API."""

    def __init__(self, contents):
        self.contents = list(contents)
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        from types import SimpleNamespace

        message = SimpleNamespace(content=self.contents.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


def test_generate_batch_retries_after_invalid_json():
    from homematch.generate import generate_batch

    good = json.dumps({"listings": [VALID]})
    client = FakeClient(['{"listings": [{broken', good])
    listings = generate_batch(client, "any-model", "any scenario", n=1, start_index=0)
    assert len(listings) == 1


def test_generate_batch_gives_up_after_max_retries():
    from homematch.generate import generate_batch

    client = FakeClient(["{bad"] * 3)
    assert generate_batch(client, "any-model", "s", n=1, start_index=0, max_retries=3) == []
