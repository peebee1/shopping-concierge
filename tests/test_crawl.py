"""Tests for the website crawler (discovery, extraction, ingest) — no network."""

import json
from pathlib import Path

import httpx
import pytest

from shopcon.catalog import Product, load_catalog
from shopcon.crawl import (
    RobotsRules,
    crawl_site,
    discover_urls,
    extract_products,
    extract_products_jsonld,
    extract_products_meta,
    fetch_robots,
    ingest,
)

SITEMAP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://shop.example/p/blue-widget</loc></url>
  <url><loc>https://shop.example/p/red-widget</loc></url>
  <url><loc>https://shop.example/about</loc></url>
  <url><loc>https://shop.example/admin/hidden</loc></url>
</urlset>"""

SITEMAP_INDEX_XML = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://shop.example/sitemap2.xml</loc></sitemap>
</sitemapindex>"""

PRODUCT_HTML = """<html><head>
<script type="application/ld+json">{
  "@context": "https://schema.org",
  "@type": "Product",
  "name": "Blue Widget Pro",
  "brand": {"name": "Acme"},
  "description": "A sturdy blue widget with extra bolts.",
  "offers": {"price": "49.99", "priceCurrency": "USD", "availability": "https://schema.org/InStock", "url": "https://shop.example/p/blue-widget"}
}</script>
</head><body><h1>Blue Widget Pro</h1></body></html>"""

GRAPH_HTML = """<html><head>
<script type="application/ld+json">{
  "@context": "https://schema.org",
  "@graph": [
    {"@type": "Product", "name": "Widget A", "offers": {"price": 10}},
    {"@type": "Product", "name": "Widget B", "offers": {"price": "20.5"}}
  ]
}</script>
</head></html>"""

META_HTML = """<html><head>
<meta property="og:title" content="Meta Widget">
<meta property="og:price:amount" content="15.75">
<meta property="og:url" content="https://shop.example/p/meta-widget">
<meta property="og:description" content="A meta-tagged widget.">
</head><body><h1>Meta Widget</h1></body></html>"""

JUNK_HTML = "<html><body><p>nothing here</p></body></html>"


def test_robots_rules():
    r = RobotsRules("User-agent: *\nDisallow: /admin/\nDisallow: /private\n")
    assert r.allows("https://shop.example/p/blue-widget")
    assert not r.allows("https://shop.example/admin/hidden")
    assert not r.allows("https://shop.example/private/data")
    assert RobotsRules(None).allows("https://shop.example/x")


class _FakeClient:
    """Duck-typed httpx.Client stand-in for offline tests."""

    def __init__(self, handler=None, **kwargs):
        self._handler = handler

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, **kw):
        assert self._handler is not None
        return self._handler(url, kw.get("timeout"))


def test_discover_urls_respects_robots_and_indexes(monkeypatch):
    def fake_get(url, timeout):
        if url.endswith("/robots.txt"):
            return httpx.Response(200, text="Disallow: /admin/", request=httpx.Request("GET", url))
        if url.endswith("/sitemap.xml"):
            return httpx.Response(200, text=SITEMAP_INDEX_XML, request=httpx.Request("GET", url))
        if url.endswith("/sitemap2.xml"):
            return httpx.Response(200, text=SITEMAP_XML, request=httpx.Request("GET", url))
        return httpx.Response(404, request=httpx.Request("GET", url))

    client = _FakeClient(fake_get)
    robots = fetch_robots(client, "https://shop.example")  # type: ignore[arg-type]
    urls = discover_urls(client, "https://shop.example", robots, max_urls=10)  # type: ignore[arg-type]
    assert "https://shop.example/p/blue-widget" in urls
    assert "https://shop.example/p/red-widget" in urls
    assert "https://shop.example/admin/hidden" not in urls  # robots-disallowed
    assert "https://shop.example/about" not in urls  # blocklisted (never a product page)


def test_extract_jsonld_single_product():
    products = extract_products_jsonld(PRODUCT_HTML, "https://shop.example/p/blue-widget")
    assert len(products) == 1
    p = products[0]
    assert p.name == "Blue Widget Pro"
    assert p.brand == "Acme"
    assert p.price == 49.99
    assert p.url == "https://shop.example/p/blue-widget"
    assert p.specs["availability"] == "in stock"
    assert p.id.startswith("crawl-")
    # id is stable across crawls
    assert extract_products_jsonld(PRODUCT_HTML, "https://shop.example/p/blue-widget")[0].id == p.id


def test_extract_jsonld_graph_multiple():
    products = extract_products_jsonld(GRAPH_HTML, "https://shop.example/category")
    assert len(products) == 2
    assert {p.name for p in products} == {"Widget A", "Widget B"}


def test_extract_meta_fallback():
    products = extract_products_meta(META_HTML, "https://shop.example/p/meta-widget")
    assert len(products) == 1
    assert products[0].name == "Meta Widget"
    assert products[0].price == 15.75


def test_extract_products_fallback_chain():
    assert len(extract_products(PRODUCT_HTML, "https://shop.example/p/blue-widget")) == 1  # JSON-LD
    assert len(extract_products(META_HTML, "https://shop.example/p/meta-widget")) == 1  # meta
    assert extract_products(JUNK_HTML, "https://shop.example/junk") == []  # nothing

    # heuristic requires product-page markers: an <h1> + "$75" banner must NOT
    # become a product
    banner = '<html><body><h1>Accessibility Statement</h1><p>Free shipping over $75!</p></body></html>'
    assert extract_products(banner, "https://shop.example/accessibility-statement") == []
    product_like = '<html><body><h1>Maple Syrup</h1><p>$12.99 <button>Add to Cart</button></p></body></html>'
    picks = extract_products(product_like, "https://shop.example/p/maple-syrup")
    assert len(picks) == 1 and picks[0].name == "Maple Syrup" and picks[0].price == 12.99


def test_ingest_upserts_by_stable_id(tmp_path):
    out = tmp_path / "cat.json"
    p1 = Product(id="crawl-aaa", name="Widget", brand="Acme", category="scraped", price=10.0, url="https://x/p/1")
    merged, stats = ingest([p1], out, "USD")
    assert stats.added == 1 and stats.updated == 0
    assert len(merged) == 1

    # same id, new price -> updated, not duplicated
    p1.price = 12.0
    merged, stats = ingest([p1], out, "USD")
    assert stats.added == 0 and stats.updated == 1
    assert len(merged) == 1
    assert merged[0].price == 12.0

    # meta reflects crawl
    data = json.loads(out.read_text())
    assert data["_meta"]["synthetic"] is False
    assert data["_meta"]["currency"] == "USD"

    # loadable by the normal pipeline loader
    loaded = load_catalog(out, autogenerate=False)
    assert loaded[0].price == 12.0


def test_crawl_site_end_to_end(monkeypatch, tmp_path):
    """Full crawl against a fake site: robots -> sitemap -> pages -> ingest."""
    pages = {
        "https://shop.example/p/blue-widget": PRODUCT_HTML,
        "https://shop.example/p/red-widget": PRODUCT_HTML.replace("Blue Widget Pro", "Red Widget Deluxe").replace("p/blue-widget", "p/red-widget"),
        "https://shop.example/p/meta-widget": META_HTML,
        "https://shop.example/p/junk": JUNK_HTML,
    }

    def handler(url, timeout):
        if url.endswith("/robots.txt"):
            return httpx.Response(200, text="Disallow: /admin/", request=httpx.Request("GET", url))
        if url.endswith("/sitemap.xml"):
            return httpx.Response(200, text=SITEMAP_XML, request=httpx.Request("GET", url))
        body = pages.get(url)
        if body is not None:
            return httpx.Response(200, text=body, request=httpx.Request("GET", url))
        return httpx.Response(404, request=httpx.Request("GET", url))

    monkeypatch.setattr("shopcon.crawl.httpx.Client", lambda **kw: _FakeClient(handler))
    out = tmp_path / "scraped.json"
    stats = crawl_site("https://shop.example", out, max_urls=10, max_products=10, delay=0, quiet=True)  # type: ignore[arg-type]
    assert stats.pages_fetched >= 2
    assert stats.added >= 2  # blue + red (both JSON-LD, both in the sitemap)
    loaded = load_catalog(out, autogenerate=False)
    names = {p.name for p in loaded}
    assert "Blue Widget Pro" in names
    assert "Red Widget Deluxe" in names
    assert all(p.category == "scraped" for p in loaded)
    assert all(p.url for p in loaded)
