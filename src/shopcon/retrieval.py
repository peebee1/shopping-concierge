"""Retrieval stage: turn the raw query into structured Constraints, then
deterministically score the catalog to surface top candidates.

This stage is deliberately rule-based (no LLM cost): the LLM only handles
*understanding* (constraint extraction) and *judgment* (final ranking).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from .catalog import CATEGORY_SYNONYMS, Product
from .region import Region, convert, from_code, default as default_region

# Feature-ish terms often present in shopping queries, matched against name+specs.
_FEATURE_KEYWORDS = [
    "hot-swap", "hotswap", "wireless", "rgb", "backlit", "mechanical", "anc",
    "noise-cancelling", "noise cancel", "4k", "144hz", "240hz", "oled", "portable",
    "bluetooth", "usb-c", "usb c", "ergonomic", "gaming", "lightweight", "silent",
    "touchscreen", "tactile", "linear", "macbook", "hdr", "ultrawide", "waterproof",
    # spec-level terms the rule-based fallback (mock) can also catch
    "ryzen 7", "ryzen 5", "i7", "i9", "1tb", "32 gb", "ssd",
]


class ConstraintExtractor(Protocol):
    """Anything with an LLM-style ``complete_json``.

    Declared here instead of importing the LLM module so retrieval (and the
    whole catalog side) stays LLM-agnostic — the mock LLM, any OpenAI-
    compatible client, or a future provider all satisfy this protocol.
    """

    def complete_json(self, system: str, user: str, temperature: float = 0.2) -> dict: ...


# Keyword -> spec keys where a *positive* value satisfies the keyword even
# when the string doesn't appear in the text ("noise-cancelling" == anc=yes).
_SPEC_KEY_ALIASES: dict[str, list[str]] = {
    "anc": ["anc"],
    "noise-cancelling": ["anc"],
    "noise cancel": ["anc"],
    "bluetooth": ["bluetooth"],
    "wireless": ["wireless"],
    "hot-swap": ["hot_swappable"],
    "hotswap": ["hot_swappable"],
    "rgb": ["backlight"],
    "backlit": ["backlight"],
    "gps": ["gps"],
    "mic": ["mic"],
    "microphone": ["mic"],
    "usb": ["interface"],
    "usb-c": ["interface"],
}

# Keyword -> {spec key: accepted exact values} ("4k" == resolution 3840x2160).
_SPEC_VALUE_ALIASES: dict[str, dict[str, set[str]]] = {
    "4k": {"resolution": {"3840x2160"}},
    "hdr": {"hdr": {"HDR400", "HDR600", "HDR1000"}},
    "portable": {"type": {"portable", "party"}},
}

_AMOUNT = r"([\d,]+(?:\.\d+)?)"
_SYMBOL_CUR = r"(₹|€|£|¥|\$|usd|inr|eur|gbp|jpy)"
_WORD_CUR = r"(rupees?|euros?|pounds?|yen|dollars?|usd|inr|eur|gbp|jpy)"
_BUDGET_PATTERNS = [
    (
        rf"(?:under|below|less than|max(?:imum)?|budget(?:\s+of)?|within)"
        rf"\s*(?:{_SYMBOL_CUR}\s*)?{_AMOUNT}(?:\s*{_WORD_CUR})?",
        "max",
    ),
    (
        rf"(?:over|above|more than|min(?:imum)?|at least)"
        rf"\s*(?:{_SYMBOL_CUR}\s*)?{_AMOUNT}(?:\s*{_WORD_CUR})?",
        "min",
    ),
    (
        rf"between\s*(?:{_SYMBOL_CUR}\s*)?{_AMOUNT}\s*(?:and|to|-)\s*"
        rf"(?:{_SYMBOL_CUR}\s*)?{_AMOUNT}(?:\s*{_WORD_CUR})?",
        "range",
    ),
]

_CURRENCY_LOOKUP = {
    "₹": "INR", "€": "EUR", "£": "GBP", "¥": "JPY", "$": "USD",
    "usd": "USD", "inr": "INR", "eur": "EUR", "gbp": "GBP", "jpy": "JPY",
    "rupee": "INR", "rupees": "INR", "euro": "EUR", "euros": "EUR",
    "pound": "GBP", "pounds": "GBP", "yen": "JPY", "dollar": "USD", "dollars": "USD",
}


def _currency_of(symbol: str | None, word: str | None) -> str | None:
    cur = symbol or word
    return _CURRENCY_LOOKUP.get(cur.lower()) if cur else None


@dataclass
class Constraints:
    query: str
    max_price: float | None = None
    min_price: float | None = None
    categories: list[str] = field(default_factory=list)
    must_keywords: list[str] = field(default_factory=list)
    nice_keywords: list[str] = field(default_factory=list)
    brands: list[str] = field(default_factory=list)
    relaxed: list[str] = field(default_factory=list)  # constraints dropped to find *some* match

    def describe(self) -> str:
        parts = []
        if self.max_price is not None:
            parts.append(f"max ${self.max_price:g}")
        if self.min_price is not None:
            parts.append(f"min ${self.min_price:g}")
        if self.categories:
            parts.append("category: " + "/".join(self.categories))
        if self.must_keywords:
            parts.append("must: " + ", ".join(self.must_keywords))
        if self.nice_keywords:
            parts.append("nice: " + ", ".join(self.nice_keywords))
        if self.brands:
            parts.append("brand: " + "/".join(self.brands))
        return "; ".join(parts) if parts else "no explicit constraints"


def parse_constraints_rule_based(query: str, region: Region | None = None, source_currency: str = "USD") -> Constraints:
    """Keyword/regex constraint extraction — used by MockLLM and as the
    fallback when a real LLM call fails.

    Budgets may carry any supported currency (₹, €, £, ¥, $, words); amounts
    are converted to the catalog's source currency. Amounts with no stated
    currency are assumed to be in the region's currency when it differs from
    the source (a German user's "under 100" means €100), else the source
    currency.
    """
    q = query.lower()
    c = Constraints(query=query)
    region = region or default_region()

    # Budgets
    for pattern, kind in _BUDGET_PATTERNS:
        m = re.search(pattern, q)
        if not m:
            continue
        if kind == "range":
            cur = _currency_of(m.group(1), None) or _currency_of(m.group(3), m.group(5))
            lo = _to_source_currency(float(m.group(2).replace(",", "")), cur, region, source_currency)
            hi = _to_source_currency(float(m.group(4).replace(",", "")), cur, region, source_currency)
            c.min_price, c.max_price = lo, hi
            break
        cur = _currency_of(m.group(1), m.group(3))
        value = _to_source_currency(float(m.group(2).replace(",", "")), cur, region, source_currency)
        if kind == "max":
            c.max_price = value
        else:
            c.min_price = value
        break

    # Categories: word-boundary match ("mic" must not match inside "ergonomic"),
    # and pick the category whose synonym appears EARLIEST in the query —
    # "webcam with built-in microphone" means webcam, not microphone.
    best: tuple[int, str] | None = None
    for cat, words in CATEGORY_SYNONYMS.items():
        for w in words:
            m = re.search(rf"\b{re.escape(w)}\b", q)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), cat)
    if best:
        c.categories.append(best[1])

    # Feature keywords: "hot-swap" -> "hot-swap"/"hotswap"
    for kw in _FEATURE_KEYWORDS:
        if kw in q:
            c.must_keywords.append(kw.replace("usb c", "usb-c"))
    c.must_keywords = list(dict.fromkeys(c.must_keywords))
    return c


def _to_source_currency(amount: float, currency: str | None, region: Region, source_currency: str) -> float:
    """Convert a budget amount into the catalog's source currency."""
    if currency is None:
        # No stated currency: assume the region's currency when it differs
        # from the source, else the source currency.
        currency = region.currency if region.currency != source_currency else source_currency
    if currency == source_currency:
        return amount
    return convert(amount, currency, source_currency)


def extract_constraints(
    query: str,
    llm: ConstraintExtractor,
    known_categories: list[str],
    known_brands: list[str],
    region: Region | None = None,
    source_currency: str = "USD",
) -> Constraints:
    """LLM-first constraint extraction with a rule-based fallback."""
    region = region or default_region()
    try:
        cats = ", ".join(known_categories) or "any"
        brands = ", ".join(known_brands) or "any"
        system = (
            f"REGION: {region.code}\nSOURCE_CURRENCY: {source_currency}\n"
            "TASK: constraints\n"
            f"Catalog prices are in {source_currency}. The user is in {region.country} "
            f"({region.currency}). Budget amounts in the request are in the user's "
            f"currency — return max_price/min_price converted to {source_currency}.\n"
            "Extract shopping constraints from the user request into strict JSON. "
            "Return ONLY a JSON object with keys: "
            '"max_price" (number or null, in ' + source_currency + '), '
            '"min_price" (number or null, in ' + source_currency + '), '
            '"categories" (array of strings, from this list if any: ' + cats + '), '
            '"must_keywords" (array of lowercase feature/spec terms the user explicitly requires, '
            'e.g. "hot-swap", "wireless", "rgb", "anc", "4k", "144hz"), '
            '"nice_keywords" (array of lowercase preference terms, not hard requirements), '
            '"brands" (array from this list if any: ' + brands + '). '
            'Use null / empty arrays for unspecified constraints.'
        )
        data = llm.complete_json(system, query)
        c = Constraints(query=query)
        c.max_price = _as_float(data.get("max_price"))
        c.min_price = _as_float(data.get("min_price"))
        c.brands = _dedupe(s for s in data.get("brands") or [] if isinstance(s, str))
        # Normalize categories against the known list; fold unknowns into keywords
        # ("gaming laptop" -> category "laptop" + must-keyword "gaming").
        cats, extra = _normalize_categories(data.get("categories") or [], known_categories)
        c.categories = cats
        c.must_keywords = _clean_keywords((data.get("must_keywords") or []) + extra)
        c.nice_keywords = _clean_keywords(data.get("nice_keywords") or [])
        return c
    except Exception:
        # Rule-based fallback (also what MockLLM uses)
        return parse_constraints_rule_based(query, region=region, source_currency=source_currency)


def _clean_keywords(items) -> list[str]:
    """Normalize extracted keywords: lowercase, dedupe, drop filler words
    ("built-in microphone" -> "microphone"; "with noise cancelling" -> "noise cancelling")."""
    filler = {
        "built", "in", "with", "feature", "featuring", "support", "supports",
        "integrated", "including", "include", "having", "has",
    }
    out: list[str] = []
    for k in items:
        if not isinstance(k, str):
            continue
        k = k.lower().strip()
        if not k or k in _stop_keywords():
            continue
        words = [w for w in re.split(r"[^a-z0-9]+", k) if w and w not in filler]
        k2 = " ".join(words) if words else k
        if k2 and k2 not in out:
            out.append(k2)
    return out


def _normalize_categories(raw: list, known_categories: list[str]) -> tuple[list[str], list[str]]:
    """Map extracted categories onto the known list; return (known cats, extra words)."""
    known_lower = {k.lower(): k for k in known_categories}
    cats: list[str] = []
    extra: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if not s:
            continue
        if s.lower() in known_lower:
            cats.append(known_lower[s.lower()])
            continue
        # e.g. "gaming laptop" -> known "laptop", leftover word "gaming"
        hit = next((k for k in known_categories if k.lower() in s.lower()), None)
        if hit:
            cats.append(hit)
            leftover = [w for w in re.split(r"[^a-z0-9]+", s.lower()) if w and w != hit.lower() and len(w) >= 2]
            extra.extend(leftover)
        else:
            extra.append(s.lower())
    return _dedupe(cats), _dedupe(extra)


def _stop_keywords() -> set[str]:
    return {"good", "best", "cheap", "affordable", "nice", "great", "recommend", "under", "budget"}


def _as_float(v: object) -> float | None:
    if v is None:
        return None
    try:
        return float(str(v))
    except (TypeError, ValueError):
        return None


def _dedupe(items) -> list[str]:
    return list(dict.fromkeys(str(x).strip() for x in items if str(x).strip()))


def _normalize(s: str) -> str:
    """Lowercase, strip non-alphanumerics so 'hot-swap' == 'hotswap' == 'hot_swappable'."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _match_text(product: Product) -> str:
    """Name + specs, excluding spec entries whose value negates the feature
    (hot_swappable=no, anc=none ...) so 'hot-swap' doesn't match 'hot_swappable=no'."""
    parts = [product.name]
    for k, v in product.specs.items():
        if v.strip().lower() not in {"no", "none", "-", ""}:
            parts.append(f"{k}={v}")
    return " ".join(parts)


def _keyword_hits(product: Product, keywords: list[str]) -> list[str]:
    if not keywords:
        return []
    text = _normalize(_match_text(product))
    hits = []
    for k in keywords:
        nk = _normalize(k)
        if nk and nk in text:
            hits.append(k)
            continue
        # token-AND fallback: "32 gb ram" matches "ram=32 GB" because
        # normalized "32gbram" isn't in "ram32gb..." but tokens 32/gb/ram all are.
        tokens = [t for t in re.split(r"[^a-z0-9]+", k.lower()) if len(t) >= 2]
        if len(tokens) >= 2 and all(t in text for t in tokens):
            hits.append(k)
            continue
        # spec aliases: positive spec values count ("noise-cancelling" == anc=yes)
        neg_values = {"no", "none", "-", ""}
        if any(
            (v := product.specs.get(key)) is not None and v.strip().lower() not in neg_values
            for key in _SPEC_KEY_ALIASES.get(k, [])
        ):
            hits.append(k)
            continue
        if any(
            product.specs.get(key, "").strip().lower() in {x.lower() for x in values}
            for key, values in _SPEC_VALUE_ALIASES.get(k, {}).items()
        ):
            hits.append(k)
    return hits


def retrieve(products: list[Product], constraints: Constraints, top_n: int = 8) -> tuple[list[Product], Constraints]:
    """Score the catalog against constraints; return top_n candidates.

    Relaxes constraints (category -> price -> keywords) only when a stage
    would otherwise return zero matches, and records what was relaxed so the
    pipeline can surface it in the trace.
    """
    # --- category filter ---
    pool = products
    if constraints.categories:
        pool = [p for p in products if p.category in constraints.categories]
        if not pool:
            constraints.relaxed.append("category")
            pool = products

    # --- price filter ---
    price_pool = pool
    if constraints.min_price is not None or constraints.max_price is not None:
        price_pool = [
            p
            for p in pool
            if (constraints.min_price is None or p.price >= constraints.min_price)
            and (constraints.max_price is None or p.price <= constraints.max_price)
        ]
        if not price_pool:
            constraints.relaxed.append("price")
            price_pool = pool

    # --- keyword handling: must-keywords FILTER the pool (relaxed if nothing matches) ---
    must = list(constraints.must_keywords)
    hits: dict[str, list[str]] = {}
    if must:
        hits = {p.id: _keyword_hits(p, must) for p in price_pool}
        matching = [p for p in price_pool if hits[p.id]]
        if matching:
            price_pool = matching
        else:
            constraints.relaxed.append("keywords")
            must = []
            hits = {}

    # --- scoring (deterministic) ---
    scored: list[tuple[float, Product]] = []
    for p in price_pool:
        must_hits = hits.get(p.id, []) if must else []
        nice_hits = _keyword_hits(p, constraints.nice_keywords)
        brand_ok = any(p.brand.lower() == b.lower() for b in constraints.brands)
        budget_bonus = 0.0
        if constraints.max_price:
            ratio = p.price / constraints.max_price
            budget_bonus = 2.0 if ratio <= 0.8 else max(0.0, 2.0 - (ratio - 0.8) * 5)
        rating_score = (p.rating - 3.5) * 1.5
        popularity = min(p.review_count / 2000.0, 1.0)
        score = (
            len(must_hits) * 3.0
            + len(nice_hits) * 1.5
            + (2.0 if brand_ok else 0.0)
            + budget_bonus
            + rating_score
            + popularity
        )
        scored.append((score, p))

    scored.sort(key=lambda t: (-t[0], -t[1].rating))
    return [p for _, p in scored[:top_n]], constraints
