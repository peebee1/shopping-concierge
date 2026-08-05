"""Tests for the catalog source layer — including its LLM-agnostic guarantee."""

import json
import subprocess
import sys
from dataclasses import asdict

import httpx
import pytest

from shopcon.catalog import Product, generate_sample_catalog, load_catalog, resolve_source
from shopcon.sources import CatalogError, FakeStoreSource, JsonSource, SyntheticSource

SAMPLE_FAKESTORE_ITEM = {
    "id": 1,
    "title": "Fjallraven Backpack",
    "price": 109.95,
    "description": "A durable laptop backpack with padded sleeves and USB port.",
    "category": "men's clothing",
    "image": "https://fakestoreapi.com/img/81fPKd-2AYL._AC_SL1500_.jpg",
    "rating": {"rate": 3.9, "count": 120},
}


def test_synthetic_source_deterministic():
    a = SyntheticSource(seed=42).load()
    b = SyntheticSource(seed=42).load()
    assert [p.id for p in a] == [p.id for p in b]
    assert len(a) == 243
    assert len({p.id for p in a}) == len(a)


def test_generate_sample_catalog_backcompat():
    products = generate_sample_catalog(seed=7, per_category=5)
    assert len(products) == 9 * 5
    assert isinstance(products[0], Product)


def test_json_source_from_file(tmp_path):
    products = [Product(id="a1", name="Widget", brand="Acme", category="gadgets", price=10.0)]
    path = tmp_path / "cat.json"
    path.write_text(json.dumps({"products": [asdict(p) for p in products]}))
    loaded = JsonSource(path).load()
    assert loaded[0].__dict__ == products[0].__dict__


def test_json_source_missing_file_raises():
    with pytest.raises(CatalogError, match="not found"):
        JsonSource("/nonexistent/catalog.json").load()


def test_fake_store_mapping():
    p = FakeStoreSource._map(SAMPLE_FAKESTORE_ITEM)
    assert p.id == "fs-1"
    assert p.name == "Fjallraven Backpack"
    assert p.category == "mens_clothing"
    assert p.price == 109.95
    assert p.rating == 3.9
    assert p.review_count == 120
    assert "USB" in p.specs["description"]


def test_fake_store_load(monkeypatch):
    def fake_get(url, timeout):
        return httpx.Response(200, json=[SAMPLE_FAKESTORE_ITEM], request=httpx.Request("GET", url))

    monkeypatch.setattr("shopcon.sources.httpx.get", fake_get)
    products = FakeStoreSource().load()
    assert len(products) == 1
    assert products[0].name.startswith("Fjallraven")


def test_fake_store_load_network_error(monkeypatch):
    def fake_get(url, timeout):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr("shopcon.sources.httpx.get", fake_get)
    with pytest.raises(CatalogError, match="FakeStoreAPI"):
        FakeStoreSource().load()


def test_resolve_source_dispatch(tmp_path):
    assert isinstance(resolve_source("synthetic"), SyntheticSource)
    assert isinstance(resolve_source("fakestore"), FakeStoreSource)
    assert isinstance(resolve_source("https://example.com/cat.json"), JsonSource)
    path = tmp_path / "cat.json"
    path.write_text(json.dumps({"products": []}))
    assert isinstance(resolve_source(path), JsonSource)
    assert isinstance(resolve_source(None), (JsonSource, SyntheticSource))  # default path or generated


def test_load_catalog_via_spec(tmp_path):
    products = [Product(id="b1", name="Thing", brand="B", category="c", price=5.0)]
    path = tmp_path / "cat.json"
    path.write_text(json.dumps({"products": [asdict(p) for p in products]}))
    assert load_catalog(str(path))[0].name == "Thing"


def test_catalog_layer_is_llm_agnostic():
    """Importing the catalog/retrieval/sources modules must NOT pull in the LLM layer."""
    code = (
        "import sys; "
        "import shopcon.catalog, shopcon.sources, shopcon.retrieval; "
        "sys.exit(0 if 'shopcon.llm' not in sys.modules else 1)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"LLM module leaked into catalog layer:\n{result.stderr}"
