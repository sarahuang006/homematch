from homematch.filters import build_where
from homematch.query_parser import ParsedQuery, parse_response


def test_no_filters_returns_none():
    assert build_where(ParsedQuery(semantic_query="cozy home")) is None


def test_single_filter_is_not_wrapped_in_and():
    parsed = ParsedQuery(max_price=800000, semantic_query="x")
    assert build_where(parsed) == {"price": {"$lte": 800000}}


def test_multiple_filters_use_and():
    parsed = ParsedQuery(max_price=800000, min_bedrooms=4, property_type="house", semantic_query="x")
    assert build_where(parsed) == {
        "$and": [
            {"price": {"$lte": 800000}},
            {"bedrooms": {"$gte": 4}},
            {"property_type": {"$eq": "house"}},
        ]
    }


def test_parser_reads_valid_json():
    raw = '{"max_price": 800000, "min_bedrooms": 4, "semantic_query": "fireplace"}'
    parsed = parse_response(raw, "original")
    assert parsed.max_price == 800000 and parsed.semantic_query == "fireplace"


def test_parser_falls_back_on_bad_json():
    parsed = parse_response("{not json", "4 bed house")
    assert parsed.semantic_query == "4 bed house" and parsed.max_price is None


def test_parser_falls_back_on_invalid_property_type():
    parsed = parse_response('{"property_type": "castle", "semantic_query": "x"}', "orig")
    assert parsed.property_type is None and parsed.semantic_query == "orig"
