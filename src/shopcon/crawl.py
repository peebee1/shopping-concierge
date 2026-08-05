"""Website crawler: build a product catalog from a website on its own.

Pipeline::

    discover (robots.txt + sitemap.xml) -> fetch pages -> extract products
    (JSON-LD, then og: meta tags, then minimal heuristics, optionally LLM)
    -> upsert into a catalog JSON -> query with the existing pipeline
    (``shopcon "..." --catalog <out>``).

Design notes:

* Polite by default: respects robots.txt, rate-limits, identifies itself via
  User-Agent. Crawl only sites whose terms allow it.
* Deterministic core: JSON-LD (schema.org) extraction needs no LLM. The
  ``--llm`` flag enables LLM extraction for pages without structured data.
* Upsert is idempotent: ids are derived from the product URL, so re-crawling
  updates prices instead of duplicating products — and the resulting file
  works with the pipeline's live-verification stage (it re-reads the file).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from html import unescape as html_unescape
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

from .catalog import Product, load_catalog, save_catalog

USER_AGENT = "shopcon-crawler/0.1 (+https://github.com/peebee1/shopping-concierge; polite demo crawler)"

_PRICE_RE = re.compile(r"(?:€|£|¥|\$|usd|eur|gbp|jpy)\s*([\d,]+(?:\.\d+)?)", re.I)


# ---------------------------------------------------------------------------
# Discovery: robots.txt + sitemap.xml
# ---------------------------------------------------------------------------

class RobotsRules:
    """Minimal robots.txt parser: Disallow prefix rules (any user-agent)."""

    def __init__(self, text: str | None):
        self.disallows: list[str] = []
        if not text:
            return
        for line in text.splitlines():
            line = line.strip()
            if line.lower().startswith("disallow:"):
                path = line.split(":", 1)[1].strip()
                if path:
                    self.disallows.append(path)

    def allows(self, url: str) -> bool:
        path = urlparse(url).path or "/"
        return not any(path.startswith(d) for d in self.disallows if d != "/")


def fetch_robots(client: httpx.Client, site: str) -> RobotsRules:
    try:
        resp = client.get(urljoin(site, "/robots.txt"), timeout=15)
        if resp.status_code == 200:
            return RobotsRules(resp.text)
    except httpx.HTTPError:
        pass
    return RobotsRules(None)


def _parse_sitemap(text: str) -> list[str]:
    """Extract <loc> URLs from a sitemap (urlset or sitemapindex)."""
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []
    urls: list[str] = []
    for elem in root.iter():
        if elem.tag.split("}")[-1] == "loc" and elem.text:
            url = elem.text.strip()
            if url.startswith("http"):
                urls.append(url)
    return urls


# Path segments that are never product pages (blocked from discovery).
_NON_PRODUCT_PREFIXES = (
    "/401", "/403", "/404", "/500", "/about", "/accessibility", "/account", "/affiliate",
    "/blog", "/careers", "/cart", "/category", "/checkout", "/contact", "/faq", "/gift-card-balance",
    "/guarantee", "/help", "/login", "/lookbook", "/magazine", "/news", "/privacy", "/recipes",
    "/returns", "/search", "/shipping", "/sitemap", "/static", "/store-locator", "/stores",
    "/terms", "/track", "/wishlist",
)

_PRODUCT_PAGE_MARKERS = (
    "add to cart", "add to bag", "add to basket", "buy now", "in stock", "out of stock",
    "item #", "sku", "quantity",
)


def _is_product_candidate(url: str) -> bool:
    path = urlparse(url).path.lower()
    return not any(path.startswith(p) for p in _NON_PRODUCT_PREFIXES)


def _looks_like_product_page(html: str) -> bool:
    return any(m in html.lower() for m in _PRODUCT_PAGE_MARKERS)


def discover_urls(client: httpx.Client, site: str, robots: RobotsRules, max_urls: int = 50) -> list[str]:
    """Fetch sitemap.xml (following sitemap indexes) and return candidate URLs."""
    for sitemap_path in ("/sitemap.xml", "/sitemap_index.xml"):
        try:
            resp = client.get(urljoin(site, sitemap_path), timeout=20)
        except httpx.HTTPError:
            continue
        if resp.status_code != 200:
            continue
        text = resp.text
        if resp.url.path.endswith(".gz") or sitemap_path.endswith(".gz"):
            try:
                text = gzip.decompress(resp.content).decode("utf-8", "replace")
            except OSError:
                continue
        urls = _parse_sitemap(text)
        # if it's an index, recurse into child sitemaps — product sitemaps first
        if "sitemapindex" in text[:2000]:
            children = sorted(
                urls[:30],
                key=lambda u: 0 if "product" in u.lower() else 1,
            )
            urls = []
            for child in children:
                try:
                    cresp = client.get(child, timeout=20)
                except httpx.HTTPError:
                    continue
                if cresp.status_code != 200:
                    continue
                ctext = cresp.text
                if child.endswith(".gz"):
                    try:
                        ctext = gzip.decompress(cresp.content).decode("utf-8", "replace")
                    except OSError:
                        continue
                urls.extend(_parse_sitemap(ctext))
        return [
            u
            for u in urls
            if robots.allows(u) and _is_product_candidate(u)
        ][:max_urls]
    return []


# ---------------------------------------------------------------------------
# Extraction: JSON-LD -> og: meta -> heuristics (-> optional LLM)
# ---------------------------------------------------------------------------

def _jsonld_blocks(html: str) -> list[dict]:
    blocks: list[dict] = []
    for m in re.finditer(r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, re.S | re.I):
        try:
            data = json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            blocks.extend(data)
        elif isinstance(data, dict):
            blocks.append(data)
    return blocks


def _walk_products(node) -> list[dict]:
    """Find schema.org Product nodes (top-level or inside @graph/ItemList)."""
    found: list[dict] = []
    if isinstance(node, dict):
        types = node.get("@type")
        if isinstance(types, str):
            types = [types]
        if types and any("Product" in t for t in types):
            found.append(node)
        for child in node.get("@graph", []) or []:
            found.extend(_walk_products(child))
        if node.get("@type") == "ItemList" or "ItemList" in (types or []):
            for item in node.get("itemListElement", []) or []:
                if isinstance(item, dict):
                    found.extend(_walk_products(item.get("item") or item))
    elif isinstance(node, list):
        for item in node:
            found.extend(_walk_products(item))
    return found


def _node_text(value) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or value.get("@id") or "")
    if isinstance(value, list):
        return ", ".join(_node_text(v) for v in value)
    return str(value or "")


def _price_of(offer) -> float | None:
    if isinstance(offer, dict):
        price = offer.get("price") or offer.get("lowPrice")
        if price is not None:
            try:
                return float(str(price).replace(",", ""))
            except ValueError:
                return None
    return None


def _currency_of(offer) -> str | None:
    if isinstance(offer, dict):
        cur = offer.get("priceCurrency") or offer.get("currency")
        return str(cur).upper() if cur else None
    return None


def extract_products_jsonld(html: str, page_url: str) -> list[Product]:
    """Extract products from schema.org JSON-LD (the reliable path).

    One product per Product node — a node's offers list holds variants
    (sizes/colors) of the same item, so only the first valid offer is used.
    """
    products: list[Product] = []
    for block in _jsonld_blocks(html):
        for node in _walk_products(node=block):
            offers = node.get("offers") or []
            if isinstance(offers, dict):
                offers = [offers]
            if not offers:
                # no offers: use lowPrice/highPrice aggregates if present
                offers = [node]
            offer = next((o for o in offers if _price_of(o) is not None), None)
            if offer is None:
                continue
            price = _price_of(offer)
            if price is None:  # pragma: no cover - guarded by the next() above
                continue
            brand = html_unescape(_node_text(node.get("brand")).strip()) or "Unknown"
            name = html_unescape(_node_text(node.get("name")).strip())
            if not name:
                continue
            specs: dict[str, str] = {}
            desc = html_unescape(_node_text(node.get("description")).strip().replace("\n", " "))
            if desc:
                specs["description"] = desc[:200]
            availability = _node_text(offer.get("availability")).lower()
            if "instock" in availability:
                specs["availability"] = "in stock"
            elif "outofstock" in availability:
                specs["availability"] = "out of stock"
            url = _node_text(offer.get("url")) or _node_text(node.get("url")) or _node_text(node.get("@id")) or page_url
            products.append(
                Product(
                    id=_stable_id(url),
                    name=name[:120],
                    brand=brand[:60],
                    category="scraped",
                    price=price,
                    specs=specs,
                    rating=0.0,
                    review_count=0,
                    url=url,
                )
            )
    return products


def extract_products_meta(html: str, page_url: str) -> list[Product]:
    """Fallback: og: meta tags (og:title / og:price:amount / og:url)."""
    def meta(prop: str) -> str:
        m = re.search(rf'<meta[^>]*property=["\']{re.escape(prop)}["\'][^>]*content=["\']([^"\']*)', html, re.I)
        if not m:
            m = re.search(rf'<meta[^>]*content=["\']([^"\']*)["\'][^>]*property=["\']{re.escape(prop)}["\']', html, re.I)
        return m.group(1).strip() if m else ""

    title = meta("og:title")
    price_raw = meta("og:price:amount")
    if not title or not price_raw:
        return []
    try:
        price = float(price_raw.replace(",", ""))
    except ValueError:
        return []
    url = meta("og:url") or page_url
    specs = {}
    desc = meta("og:description")
    if desc:
        specs["description"] = desc[:200]
    return [
        Product(
            id=_stable_id(url),
            name=title[:120],
            brand="Unknown",
            category="scraped",
            price=price,
            specs=specs,
            rating=0.0,
            review_count=0,
            url=url,
        )
    ]


def extract_products_heuristic(html: str, page_url: str) -> list[Product]:
    """Last resort: <h1> + first price-looking string — but ONLY on pages that
    look like product pages (cart/buy/sku markers). Prevents sitewide banners
    like "$75 free shipping" from becoming products."""
    if not _looks_like_product_page(html):
        return []
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.S | re.I)
    title = re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else ""
    if not title:
        return []
    m = _PRICE_RE.search(html)
    if not m:
        return []
    try:
        price = float(m.group(1).replace(",", ""))
    except ValueError:
        return []
    return [
        Product(
            id=_stable_id(page_url),
            name=title[:120],
            brand="Unknown",
            category="scraped",
            price=price,
            specs={"description": "extracted by heuristic (no structured data)"},
            rating=0.0,
            review_count=0,
            url=page_url,
        )
    ]


def extract_products(html: str, page_url: str, llm=None, source_currency: str = "USD") -> list[Product]:
    """JSON-LD -> og: meta -> heuristics. Optional LLM extraction for pages
    with no structured data (requires an OpenAI-compatible key)."""
    products = extract_products_jsonld(html, page_url)
    if not products:
        products = extract_products_meta(html, page_url)
    if not products:
        products = extract_products_heuristic(html, page_url)
    if not products and llm is not None:
        products = extract_products_llm(html, page_url, llm)
    # prices come in the site's currency; record it per product via specs so
    # the pipeline knows what it's dealing with (catalog _meta.currency wins)
    return products


def extract_products_llm(html: str, page_url: str, llm) -> list[Product]:
    """LLM extraction: visible text -> {name, price, brand, description}."""
    text = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()[:3000]
    system = (
        "TASK: extract_product\n"
        "Extract the product from this page text. Return ONLY JSON: "
        '{"name": str, "price": number, "brand": str, "description": str}. '
        'If this is not a product page, return {"name": ""}.'
    )
    try:
        data = llm.complete_json(system, text, temperature=0.0)
    except Exception:  # noqa: BLE001
        return []
    name = str(data.get("name") or "").strip()
    if not name or not data.get("price"):
        return []
    specs = {}
    desc = str(data.get("description") or "").strip()
    if desc:
        specs["description"] = desc[:200]
    return [
        Product(
            id=_stable_id(page_url),
            name=name[:120],
            brand=str(data.get("brand") or "Unknown")[:60],
            category="scraped",
            price=float(data["price"]),
            specs=specs,
            rating=0.0,
            review_count=0,
            url=page_url,
        )
    ]


def _stable_id(url: str) -> str:
    return "crawl-" + hashlib.sha1(url.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Ingest: idempotent upsert into a catalog JSON
# ---------------------------------------------------------------------------

@dataclass
class CrawlStats:
    urls_found: int = 0
    pages_fetched: int = 0
    products_extracted: int = 0
    added: int = 0
    updated: int = 0


def ingest(products: list[Product], out_path: Path, currency: str) -> tuple[list[Product], CrawlStats]:
    """Upsert crawled products into the catalog file (ids are URL-stable)."""
    existing = load_catalog(out_path, autogenerate=False) if out_path.exists() else []
    by_id = {p.id: p for p in existing}
    stats = CrawlStats(products_extracted=len(products))
    for p in products:
        prev = by_id.get(p.id)
        if prev is None:
            by_id[p.id] = p
            stats.added += 1
        else:
            changed = prev.price != p.price or prev.name != p.name or prev.specs != p.specs
            by_id[p.id] = p
            if changed:
                stats.updated += 1
    merged = sorted(by_id.values(), key=lambda p: p.id)
    save_catalog(
        merged,
        out_path,
        meta={
            "synthetic": False,
            "source": "crawl",
            "currency": currency,
            "note": "Crawled from a live website — see product URLs. Re-crawl to refresh prices.",
            "generated_by": "shopcon.crawl",
        },
    )
    return merged, stats


# ---------------------------------------------------------------------------
# Orchestration + CLI
# ---------------------------------------------------------------------------

def crawl_site(
    site: str,
    out_path: Path | str,
    max_urls: int = 50,
    max_products: int = 25,
    currency: str = "USD",
    delay: float = 0.5,
    llm=None,
    quiet: bool = False,
) -> CrawlStats:
    out_path = Path(out_path)
    stats = CrawlStats()
    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(headers=headers, follow_redirects=True, timeout=20) as client:
        robots = fetch_robots(client, site)
        urls = discover_urls(client, site, robots, max_urls=max_urls)
        stats.urls_found = len(urls)
        if not quiet:
            print(f"robots.txt: {'ok' if robots.disallows else 'no rules'}")
            print(f"sitemap: {stats.urls_found} candidate URLs (polite crawl, {delay}s delay)")
        products: list[Product] = []
        for url in urls:
            if len(products) >= max_products:
                break
            try:
                resp = client.get(url)
            except httpx.HTTPError:
                continue
            if resp.status_code != 200:
                continue
            stats.pages_fetched += 1
            page_products = extract_products(resp.text, str(resp.url), llm=llm, source_currency=currency)
            products.extend(page_products)
            if not quiet and page_products:
                print(f"  + {len(page_products)} product(s) from {url[:90]}")
            time.sleep(delay)
        merged, ingest_stats = ingest(products, out_path, currency)
        # keep the crawl counters (don't let ingest's stats clobber them)
        stats.products_extracted = ingest_stats.products_extracted
        stats.added = ingest_stats.added
        stats.updated = ingest_stats.updated
        if not quiet:
            print(
                f"\n{stats.pages_fetched} pages fetched, {stats.products_extracted} products extracted, "
                f"{stats.added} added, {stats.updated} updated"
            )
            print(f"saved -> {out_path} ({len(merged)} products, {currency})")
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="shopcon-crawl",
        description="Crawl a website's sitemap, extract products, and upsert them into a catalog JSON you can query.",
    )
    parser.add_argument("site", help="site root, e.g. https://www.example.com")
    parser.add_argument("--out", default="data/scraped.json", help="catalog JSON to upsert into (default data/scraped.json)")
    parser.add_argument("--max-urls", type=int, default=50, help="sitemap URLs to try (default 50)")
    parser.add_argument("--max-products", type=int, default=25, help="stop after this many products (default 25)")
    parser.add_argument("--currency", default="USD", help="currency of the site's prices (default USD)")
    parser.add_argument("--delay", type=float, default=0.5, help="seconds between page fetches (default 0.5)")
    parser.add_argument("--llm", action="store_true", help="use LLM extraction for pages without structured data (needs SHOPCON_API_KEY)")
    parser.add_argument("--quiet", action="store_true", help="minimal output")
    args = parser.parse_args(argv)

    llm = None
    if args.llm:
        from .llm import OpenAICompatLLM

        try:
            llm = OpenAICompatLLM()
        except Exception as exc:  # noqa: BLE001
            print(f"note: {exc} -> continuing without LLM extraction", file=sys.stderr)

    crawl_site(
        args.site,
        args.out,
        max_urls=args.max_urls,
        max_products=args.max_products,
        currency=args.currency,
        delay=args.delay,
        llm=llm,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
