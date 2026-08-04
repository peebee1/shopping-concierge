import pytest

from shopcon.catalog import generate_sample_catalog
from shopcon.retrieval import (
    Constraints,
    _keyword_hits,
    _normalize_categories,
    parse_constraints_rule_based,
    retrieve,
)


@pytest.fixture(scope="module")
def catalog():
    return generate_sample_catalog(seed=42, per_category=10)


def test_price_filter(catalog):
    c = Constraints(query="under 100", max_price=100.0)
    picks, c2 = retrieve(catalog, c, top_n=5)
    assert len(picks) == 5
    assert all(p.price <= 100 for p in picks)
    assert not c2.relaxed


def test_category_filter(catalog):
    c = Constraints(query="keyboard", categories=["mechanical_keyboard"])
    picks, _ = retrieve(catalog, c, top_n=10)
    assert picks
    assert all(p.category == "mechanical_keyboard" for p in picks)


def test_keyword_matching(catalog):
    c = Constraints(query="wireless", must_keywords=["wireless"])
    picks, _ = retrieve(catalog, c, top_n=10)
    assert picks
    # keyword hits match name + spec text (e.g. "Wireless Headphones" by name,
    # or wireless=yes in specs)
    for p in picks:
        assert "wireless" in (p.name + " " + p.spec_text).lower()


def test_relaxation_records_what_was_dropped(catalog):
    # A budget below every product's price must relax "price", not return nothing.
    c = Constraints(query="under 1", max_price=1.0)
    picks, c2 = retrieve(catalog, c, top_n=5)
    assert picks
    assert "price" in c2.relaxed


def test_keyword_token_and_fallback():
    """'32 gb ram' should match a product with spec ram=32 GB (token-AND matching)."""
    from shopcon.catalog import Product

    p = Product(id="x1", name="Laptop", brand="B", category="laptop", price=1000,
                specs={"ram": "32 GB", "cpu": "Intel Core i7"})
    hits = _keyword_hits(p, ["32 gb ram"])
    assert hits == ["32 gb ram"]


def test_keyword_does_not_match_negated_spec():
    """'hot-swap' must NOT match hot_swappable=no."""
    from shopcon.catalog import Product

    p = Product(id="x2", name="Keyboard", brand="B", category="mechanical_keyboard", price=50,
                specs={"hot_swappable": "no", "switch": "Cherry MX Red"})
    assert _keyword_hits(p, ["hot-swap"]) == []


def test_category_normalization_maps_offlist_names():
    """'gaming laptop' (off-list) -> category 'laptop' + keyword 'gaming'."""
    cats, extra = _normalize_categories(["gaming laptop"], ["laptop", "mouse"])
    assert cats == ["laptop"]
    assert "gaming" in extra


def test_rule_based_budget_parsing():
    assert parse_constraints_rule_based("keyboard under $100").max_price == 100.0
    assert parse_constraints_rule_based("over 500 dollars").min_price == 500.0
    r = parse_constraints_rule_based("between 100 and 200")
    assert r.min_price == 100.0 and r.max_price == 200.0
    assert parse_constraints_rule_based("hot-swap wireless kb").must_keywords
    assert "mechanical_keyboard" in parse_constraints_rule_based("mechanical keyboard").categories
