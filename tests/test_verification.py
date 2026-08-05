"""Tests for the evidence & provenance layer (verification, freshness, confidence)."""

import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from shopcon.catalog import Product, generate_sample_catalog
from shopcon.llm import MockLLM
from shopcon.pipeline import recommend
from shopcon.sources import FakeStoreSource, JsonSource, SyntheticSource
from shopcon.verification import (
    VerificationResult,
    confidence_label,
    freshness_label,
    freshness_line,
    human_age,
    verification_notes,
)


@pytest.fixture(scope="module")
def catalog():
    return generate_sample_catalog(seed=42)


def test_human_age_and_freshness():
    now = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)
    assert human_age(now - timedelta(minutes=30), now) == "30m old"
    assert human_age(now - timedelta(hours=5), now) == "5h old"
    assert human_age(now - timedelta(days=3), now) == "3d old"
    assert freshness_label(now - timedelta(hours=1), now) == "fresh"
    assert freshness_label(now - timedelta(days=30), now) == "stale"
    line = freshness_line(now - timedelta(hours=2))
    assert line and "2026-08-05 10:00" in line and "fresh" in line
    assert freshness_line(None) is None


def test_verification_notes():
    r_ok = VerificationResult("a", "verified", 10.0, 10.0)
    r_ch = VerificationResult("b", "changed", 10.0, 12.5)
    assert "prices unchanged" in (verification_notes({"a": r_ok}) or "")
    assert "$10.00 → $12.50" in (verification_notes({"a": r_ok, "b": r_ch}) or "")
    assert verification_notes({"b": VerificationResult("b", "unverifiable")}) is None


def test_confidence_label():
    assert confidence_label(0.9) == "high"
    assert confidence_label(0.7) == "medium"
    assert confidence_label(0.4) == "low"


def test_synthetic_source_verify_is_honest():
    src = SyntheticSource()
    products = src.load()
    assert src.as_of is not None
    results = src.verify(products[:2])
    assert all(v.status == "unverifiable" for v in results.values())
    assert "synthetic" in results[products[0].id].note


def test_json_source_verify_file(tmp_path):
    products = [Product(id="a1", name="Widget", brand="Acme", category="gadgets", price=10.0)]
    path = tmp_path / "cat.json"
    path.write_text(json.dumps({"products": [asdict(p) for p in products]}))
    src = JsonSource(path)
    loaded = src.load()
    assert src.as_of is not None  # file mtime

    # unchanged -> verified
    results = src.verify(loaded)
    assert results["a1"].status == "verified"

    # price moved -> changed
    products[0].price = 12.5
    path.write_text(json.dumps({"products": [asdict(p) for p in products]}))
    results = src.verify(loaded)
    assert results["a1"].status == "changed"
    assert results["a1"].price_before == 10.0 and results["a1"].price_after == 12.5

    # removed -> unavailable
    path.write_text(json.dumps({"products": []}))
    results = src.verify(loaded)
    assert results["a1"].status == "unavailable"


def test_fakestore_verify(monkeypatch):
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        price = 99.99 if url.endswith("/2") else 64.0
        return httpx.Response(200, json={"id": int(url.rsplit("/", 1)[1]), "price": price}, request=httpx.Request("GET", url))

    monkeypatch.setattr("shopcon.sources.httpx.get", fake_get)
    src = FakeStoreSource()
    p1 = Product(id="fs-1", name="A", brand="FakeStore", category="electronics", price=64.0)
    p2 = Product(id="fs-2", name="B", brand="FakeStore", category="electronics", price=64.0)
    results = src.verify([p1, p2])
    assert results["fs-1"].status == "verified"
    assert results["fs-2"].status == "changed"
    assert results["fs-2"].price_after == 99.99
    assert calls == ["https://fakestoreapi.com/products/1", "https://fakestoreapi.com/products/2"]


def test_fakestore_verify_network_error(monkeypatch):
    def fake_get(url, timeout):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr("shopcon.sources.httpx.get", fake_get)
    results = FakeStoreSource().verify([Product(id="fs-1", name="A", brand="B", category="c", price=1.0)])
    assert results["fs-1"].status == "unverifiable"


class StubSource:
    """A source that reports one price change on verify."""

    name = "stub"
    as_of = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)

    def load(self):
        return []

    def verify(self, products):
        return {
            p.id: VerificationResult(p.id, "changed", p.price, p.price + 10, note="price moved")
            for p in products
        }


def test_pipeline_verify_stage_and_confidence(catalog):
    result = recommend("wireless keyboard under $100", catalog, MockLLM(), top_n=3, source=StubSource())
    # provenance
    assert result.data_as_of == "2026-08-05T10:00:00+00:00"
    assert result.freshness and "stub" not in result.freshness
    # verification
    assert result.verifications
    assert all(v["status"] == "changed" for v in result.verifications.values())
    assert any("verified" in step for step in result.trace)
    assert "[verification]" in result.summary
    # confidence
    for r in result.ranked:
        assert 0.0 <= r.confidence <= 1.0
        assert r.confidence_label in {"high", "medium", "low"}


def test_pipeline_without_source_has_no_verification(catalog):
    result = recommend("wireless keyboard", catalog, MockLLM(), top_n=3)
    assert result.verifications == {}
    assert result.data_as_of is None
    assert result.freshness is None
