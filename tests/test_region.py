"""Tests for the region/currency layer."""

from datetime import datetime, timezone

import httpx
import pytest

from shopcon.catalog import generate_sample_catalog
from shopcon.eval import load_queries, run_query
from shopcon.llm import MockLLM
from shopcon.pipeline import recommend
from shopcon.region import (
    CURRENCY_TO_USD,
    Region,
    convert,
    default,
    detect_from_ip,
    detect_from_locale,
    from_code,
    fx_to_region,
)
from shopcon.retrieval import parse_constraints_rule_based


@pytest.fixture(scope="module")
def catalog():
    return generate_sample_catalog(seed=42)


def test_from_code_and_defaults():
    assert from_code("in").code == "IN"
    assert from_code("DE").currency == "EUR"
    assert from_code("XX").code == "US"  # unknown falls back
    assert from_code(None).code == "US"
    assert default().code == "US"
    assert "India" in from_code("IN").display_name()


def test_convert_math():
    assert convert(100, "USD", "USD") == 100.0
    assert convert(100, "INR", "USD") == pytest.approx(1.2)  # 100 * 0.012
    assert convert(36, "USD", "INR") == pytest.approx(3000.0)  # 36 / 0.012
    assert convert(150, "EUR", "USD") == pytest.approx(162.0)
    assert convert(50, "USD", "GBP") == pytest.approx(50 / 1.27)
    assert convert(10, "XYZ", "USD") == 10.0  # unknown currency: passthrough


def test_fx_to_region():
    assert fx_to_region("USD", from_code("US")) is None  # same currency
    fx = fx_to_region("USD", from_code("IN"))
    assert fx is not None and fx == pytest.approx(1 / 0.012)
    assert fx_to_region("USD", from_code("DE")) == pytest.approx(1 / 1.08)


def test_detect_from_locale(monkeypatch):
    monkeypatch.setenv("LC_ALL", "")
    monkeypatch.setenv("LANG", "de_DE.UTF-8")
    assert detect_from_locale().code == "DE"
    monkeypatch.setenv("LANG", "en_IN")
    assert detect_from_locale().code == "IN"
    monkeypatch.setenv("LANG", "")
    assert detect_from_locale().code == "US"


def test_detect_from_ip(monkeypatch):
    def fake_get(url, timeout):
        return httpx.Response(200, json={"country_code": "DE"}, request=httpx.Request("GET", url))

    monkeypatch.setattr("shopcon.region.httpx.get", fake_get)
    region = detect_from_ip()
    assert region is not None and region.code == "DE"

    def fail_get(url, timeout):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr("shopcon.region.httpx.get", fail_get)
    from shopcon.region import _ip_cache

    _ip_cache["at"] = 0  # bust cache
    assert detect_from_ip() is None


def test_rule_based_budget_with_currencies():
    # explicit ₹ -> converted to USD source
    c = parse_constraints_rule_based("wireless mouse under ₹4000", region=from_code("IN"))
    assert c.max_price == pytest.approx(48.0)
    # explicit €
    c = parse_constraints_rule_based("headphones under €150", region=from_code("DE"))
    assert c.max_price == pytest.approx(162.0)
    # no symbol, non-US region -> assume region currency
    c = parse_constraints_rule_based("headphones under 5000", region=from_code("JP"))
    assert c.max_price == pytest.approx(33.5)
    # no symbol, US region -> USD
    c = parse_constraints_rule_based("keyboard under 100", region=from_code("US"))
    assert c.max_price == 100.0
    # "$100" explicit USD beats region
    c = parse_constraints_rule_based("keyboard under $100", region=from_code("IN"))
    assert c.max_price == 100.0
    # word currency + range
    c = parse_constraints_rule_based("between 1000 and 2000 rupees", region=from_code("IN"))
    assert c.min_price == pytest.approx(12.0) and c.max_price == pytest.approx(24.0)


def test_mock_extraction_uses_region_markers(catalog):
    """The mock LLM reads REGION/SOURCE_CURRENCY markers from the system prompt."""
    result = recommend("wireless mouse under ₹4000", catalog, MockLLM(), top_n=3, region=from_code("IN"))
    assert result.constraints.max_price == pytest.approx(48.0)
    assert result.region_code == "IN"
    assert result.region_currency == "INR"
    assert result.fx_to_region == pytest.approx(1 / 0.012)
    # prices in the ranked JSON are converted to INR
    ranked = result.to_dict()["ranked"]
    assert ranked[0]["price_local"] == pytest.approx(ranked[0]["price"] / 0.012, abs=0.5)
    assert ranked[0]["currency_local"] == "INR"


def test_region_in_trace_and_no_fx_for_same_currency(catalog):
    result = recommend("wireless mouse", catalog, MockLLM(), top_n=3, region=from_code("US"))
    assert result.fx_to_region is None
    assert any("region: US (USD)" in step for step in result.trace)


def test_eval_query_with_region_runs(catalog):
    queries = load_queries()
    inr = next(q for q in queries if q.region == "IN")
    assert inr.max_price == pytest.approx(48.0)
    qr, _ = run_query(inr, catalog, MockLLM(), top_n=5)
    assert qr.constraint_pass
    assert qr.budget_violations == 0
    assert qr.gold_count > 0, "catalog must contain wireless mice under $48 for the INR query"
