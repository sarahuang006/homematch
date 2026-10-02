from homematch.evaluate import Constraints, parser_is_correct, score_query
from homematch.query_parser import ParsedQuery
from homematch.schemas import Listing


def make(lid, price, beds, ptype="house", features=("fireplace",)):
    return Listing(
        id=lid, neighborhood="N", property_type=ptype, price=price, bedrooms=beds,
        bathrooms=2, size_sqft=2000, features=list(features),
        description="A nice home with plenty of light.", neighborhood_description="A quiet, friendly area.",
    )


LISTINGS = {
    "A": make("A", 700_000, 4),
    "B": make("B", 900_000, 4),             # too expensive
    "C": make("C", 600_000, 4, features=["pool"]),  # no fireplace
}
C = Constraints(max_price=800_000, min_bedrooms=4, features=["fireplace"])


def test_perfect_result():
    s = score_query(["A"], LISTINGS, C, k=1)
    assert s["constraint_precision"] == 1 and s["full_precision"] == 1 and s["recall"] == 1


def test_constraint_vs_full_precision():
    s = score_query(["A", "B", "C"], LISTINGS, C, k=3)
    assert s["constraint_precision"] == 2 / 3  # A and C pass price/bedrooms
    assert s["full_precision"] == 1 / 3        # only A also has a fireplace


def test_empty_results_score_zero_not_perfect():
    s = score_query([], LISTINGS, C, k=5)
    assert s["full_precision"] == 0 and s["recall"] == 0


def test_recall_is_none_when_nothing_is_relevant():
    impossible = Constraints(min_price=10_000_000)
    assert score_query(["A"], LISTINGS, impossible, k=5)["recall"] is None


def test_parser_correctness():
    good = ParsedQuery(max_price=800_000, min_bedrooms=4, semantic_query="fireplace")
    wrong = ParsedQuery(max_price=800_000, min_bedrooms=4, property_type="house", semantic_query="x")
    assert parser_is_correct(good, C) and not parser_is_correct(wrong, C)
