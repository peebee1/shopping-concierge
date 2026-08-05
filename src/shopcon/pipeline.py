"""Pipeline: understand -> retrieve -> rank.

The "agentic" loop is deliberately small and observable:

1. understand   LLM parses the query into structured Constraints
2. retrieve     deterministic catalog scoring (no LLM cost)
3. rank         LLM judges the top candidates: order + rationale + verdict

Every step appends to `trace` so callers can show *why* the agent did what
it did (transparency is the point of the demo).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .catalog import CATEGORY_SPEC_KEYS, Product
from .llm import LLM, LLMError
from .region import Region, default as default_region, fx_to_region
from .retrieval import Constraints, _keyword_hits, _normalize, extract_constraints, retrieve
from .verification import freshness_line, verification_notes


@dataclass
class RankedItem:
    product: Product
    rank: int
    rationale: str
    confidence: float = 0.0
    confidence_label: str = ""


@dataclass
class RecommendationResult:
    query: str
    constraints: Constraints
    candidates: list[Product]
    ranked: list[RankedItem]
    summary: str
    trace: list[str]
    model: str
    data_as_of: str | None = None
    freshness: str | None = None
    verifications: dict[str, dict] = field(default_factory=dict)
    region_code: str | None = None
    region_currency: str | None = None
    source_currency: str = "USD"
    fx_to_region: float | None = None  # 1 source-currency unit in region currency

    def to_dict(self) -> dict:
        fx = self.fx_to_region
        return {
            "query": self.query,
            "constraints": {
                "max_price": self.constraints.max_price,
                "min_price": self.constraints.min_price,
                "categories": self.constraints.categories,
                "must_keywords": self.constraints.must_keywords,
                "nice_keywords": self.constraints.nice_keywords,
                "brands": self.constraints.brands,
                "relaxed": self.constraints.relaxed,
            },
            "candidates": [p.__dict__ for p in self.candidates],
            "ranked": [
                {
                    "rank": r.rank,
                    **r.product.__dict__,
                    "rationale": r.rationale,
                    "confidence": r.confidence,
                    "confidence_label": r.confidence_label,
                    "price_local": round(r.product.price * fx, 2) if fx else None,
                    "currency_local": self.region_currency if fx else None,
                }
                for r in self.ranked
            ],
            "summary": self.summary,
            "trace": self.trace,
            "model": self.model,
            "data_as_of": self.data_as_of,
            "freshness": self.freshness,
            "verifications": self.verifications,
            "region": self.region_code,
            "region_currency": self.region_currency,
            "source_currency": self.source_currency,
            "fx_to_region": fx,
        }


def recommend(
    query: str,
    products: list[Product],
    llm: LLM,
    top_n: int = 5,
    candidate_pool: int = 8,
    source=None,
    verify_n: int = 3,
    region: Region | None = None,
) -> RecommendationResult:
    region = region or default_region()
    source_currency = getattr(source, "currency", "USD") if source is not None else "USD"
    trace: list[str] = []

    # 1) understand
    constraints = extract_constraints(
        query, llm, _cats(products), _brands(products), region=region, source_currency=source_currency
    )
    trace.append(f"understood query -> {constraints.describe()}")
    trace.append(f"region: {region.code} ({region.currency}); catalog prices in {source_currency}")

    # 2) retrieve
    candidates, constraints = retrieve(products, constraints, top_n=candidate_pool)
    trace.append(f"retrieved {len(candidates)} candidates from {len(products)} products")
    if constraints.relaxed:
        trace.append("relaxed constraints: " + ", ".join(constraints.relaxed))

    # 3) rank
    ranked, summary = _rank_with_llm(query, constraints, candidates, llm, top_n, region, source_currency)
    trace.append(f"ranked with {llm.name}")

    # Honesty rule: if must-keywords were relaxed (NOTHING matched them) and
    # none of the ranked picks satisfy any of them, return an empty shortlist
    # instead of padding with irrelevant products.
    if "keywords" in constraints.relaxed and constraints.must_keywords:
        satisfying = [r for r in ranked if _keyword_hits(r.product, constraints.must_keywords)]
        if not satisfying:
            ranked = []
            trace.append("no product matches the requested features — returned an honest empty shortlist")
            if not any(m in summary.lower() for m in ("none of", "no products", "no valid", "does not", "no match", "nothing")):
                summary = f"No products in the catalog match your request ('{query}'). Nothing recommended — no padding."

    # Confidence: deterministic heuristic (no LLM call).
    for r in ranked:
        r.confidence = round(_confidence(r.product, constraints), 2)
        r.confidence_label = "high" if r.confidence >= 0.8 else ("medium" if r.confidence >= 0.6 else "low")

    # 4) verify: evidence & provenance
    as_of = getattr(source, "as_of", None) if source is not None else None
    data_as_of = as_of.isoformat() if as_of else None
    freshness = freshness_line(as_of) if as_of else None
    verifications: dict[str, dict] = {}
    if source is not None and verify_n > 0 and ranked and hasattr(source, "verify"):
        try:
            results = source.verify([r.product for r in ranked[:verify_n]]) or {}
        except Exception as exc:  # noqa: BLE001 - a verification failure must not break the answer
            trace.append(f"live verification failed: {exc}")
            results = {}
        verifications = {k: v.to_dict() for k, v in results.items()}
        if results:
            changed = [v for v in results.values() if v.status == "changed"]
            if changed:
                detail = "; ".join(f"{v.product_id} ${v.price_before:.2f}→${v.price_after:.2f}" for v in changed)
                trace.append(f"verified {len(results)} pick(s) live: {len(changed)} price change(s) — {detail}")
            else:
                trace.append(f"verified {len(results)} pick(s) live: prices unchanged")
            notes = verification_notes(results)
            if notes:
                summary = f"{summary}\n\n{notes}"
        else:
            trace.append("live verification unavailable")
    elif source is not None and ranked and verify_n > 0:
        trace.append("source does not support live verification")

    return RecommendationResult(
        query=query,
        constraints=constraints,
        candidates=candidates,
        ranked=ranked,
        summary=summary,
        trace=trace,
        model=llm.name,
        data_as_of=data_as_of,
        freshness=freshness,
        verifications=verifications,
        region_code=region.code,
        region_currency=region.currency,
        source_currency=source_currency,
        fx_to_region=fx_to_region(source_currency, region),
    )


def _confidence(p: Product, constraints: Constraints) -> float:
    """Deterministic confidence heuristic: how strongly does this pick satisfy
    the extracted constraints? 0..1 (no LLM involved)."""
    must = list(constraints.must_keywords)
    if must:
        matched = {_normalize(h) for h in _keyword_hits(p, must)}
        expected = {_normalize(k) for k in must}
        must_ratio = len(matched & expected) / len(expected)
    else:
        must_ratio = 1.0
    budget = 1.0 if constraints.max_price is None or p.price <= constraints.max_price else 0.5
    cat_ok = 1.0 if not constraints.categories or p.category in constraints.categories else 0.6
    rating = max(0.0, min(1.0, (p.rating - 3.0) / 2.0))
    return 0.45 * must_ratio + 0.25 * budget + 0.15 * cat_ok + 0.15 * rating


def _cats(products: list[Product]) -> list[str]:
    return sorted({p.category for p in products})


def _brands(products: list[Product]) -> list[str]:
    return sorted({p.brand for p in products})


def _rank_with_llm(
    query: str,
    constraints: Constraints,
    candidates: list[Product],
    llm: LLM,
    top_n: int,
    region: Region,
    source_currency: str,
) -> tuple[list[RankedItem], str]:
    # Pre-score so the mock (and fallbacks) have signal without extra LLM calls.
    scored = _rule_pre_score(query, constraints, candidates)
    cand_json = [
        {
            **p.__dict__,
            "_score": round(score, 2),
            "_price_fit": bool(constraints.max_price is None or p.price <= constraints.max_price),
            "_matched_keywords": _matched(query, constraints, p),
        }
        for score, p in scored
    ]
    system = (
        "TASK: rank\n"
        "You are a shopping advisor ranking candidate products for a user request. "
        "Candidates are given as a JSON array with id, name, brand, category, price, "
        "specs, rating, review_count. Return ONLY a JSON object with two keys:\n"
        '"ranked": array of {"id": <candidate id>, "rank": <int 1..N>, '
        '"rationale": <1-2 sentence reason, referencing the user\'s constraints and '
        "price/value, not generic praise>},\n"
        '"summary": <2-3 sentence verdict naming the best pick and a strong alternative, '
        "with one caveat if relevant>.\n"
        "Every ranked id MUST come from the candidates array. Use every candidate exactly once."
    )
    user = (
        f"Request: {query}\n"
        f"Prices are in {source_currency}; user region: {region.code} ({region.currency}). "
        "You may mention converted prices in the user's currency.\n"
        f"Constraints: {constraints.describe()}\n"
        f"Candidates:\n{json.dumps(cand_json, indent=1)}"
    )

    fallback_ranked = [p for _, p in scored]
    exc: Exception | None = None
    try:
        data = llm.complete_json(system, user)
        if not data:
            raise LLMError("empty LLM response")
    except Exception as exc2:  # noqa: BLE001 - network flakiness must degrade, not crash
        exc = exc2
        # one retry before giving up to the deterministic fallback
        try:
            data = llm.complete_json(system, user)
        except Exception as exc3:  # noqa: BLE001
            exc = exc3
            data = None
    if data is None:
        ranked = [
            RankedItem(product=p, rank=i, rationale=_fallback_rationale(query, constraints, p))
            for i, p in enumerate(fallback_ranked[:top_n], start=1)
        ]
        return ranked, _fallback_summary(ranked) + f" (LLM ranking failed: {exc or 'unknown error'})"

    # Tolerate both {"ranked": [{"id", "rank", "rationale"}...]} and
    # {"ranked_ids": ["id", ...]} output shapes.
    raw = data.get("ranked") or data.get("ranked_ids") or data.get("ids") or []
    summary = str(data.get("summary") or "").strip()
    by_id = {p.id: p for p in candidates}
    ordered: list[Product] = []
    rationales: dict[str, str] = {}
    for item in raw:
        if isinstance(item, str):
            pid, rat = item, ""
        elif isinstance(item, dict):
            pid = item.get("id")
            rat = str(item.get("rationale") or "")
        else:
            continue
        if pid in by_id and by_id[pid] not in ordered:
            ordered.append(by_id[pid])
            rationales[pid] = rat
    # Fill anything the LLM skipped, preserving rule order
    for p in fallback_ranked:
        if p not in ordered:
            ordered.append(p)
    ranked = [
        RankedItem(
            product=p,
            rank=i,
            rationale=rationales.get(p.id) or _fallback_rationale(query, constraints, p),
        )
        for i, p in enumerate(ordered[:top_n], start=1)
    ]
    if not summary:
        summary = _fallback_summary(ranked)
    return ranked, summary


def _rule_pre_score(query: str, constraints: Constraints, candidates: list[Product]) -> list[tuple[float, Product]]:
    from .retrieval import _keyword_hits  # private but shared module

    scored: list[tuple[float, Product]] = []
    for p in candidates:
        must = _keyword_hits(p, constraints.must_keywords)
        nice = _keyword_hits(p, constraints.nice_keywords)
        score = len(must) * 3.0 + len(nice) * 1.5 + p.rating + min(p.review_count / 2000.0, 1.0)
        scored.append((score, p))
    scored.sort(key=lambda t: -t[0])
    return scored


def _matched(query: str, constraints: Constraints, p: Product) -> list[str]:
    from .retrieval import _keyword_hits

    return _keyword_hits(p, list(constraints.must_keywords) + list(constraints.nice_keywords))


def _fallback_rationale(query: str, constraints: Constraints, p: Product) -> str:
    from .retrieval import _keyword_hits

    must = _keyword_hits(p, constraints.must_keywords)
    nice = _keyword_hits(p, constraints.nice_keywords)
    bits = []
    if must:
        bits.append(f"matches {len(must)} requested feature(s): {', '.join(must)}")
    if constraints.max_price is not None:
        bits.append("within budget" if p.price <= constraints.max_price else "slightly above budget")
    bits.append(f"rated {p.rating} from {p.review_count} reviews")
    return "; ".join(bits) + "."


def _fallback_summary(ranked: list[RankedItem]) -> str:
    if not ranked:
        return "No matches found."
    top = ranked[0]
    return f"Best overall pick: {top.product.name} (${top.product.price:.2f}) — {top.rationale}"


def spec_keys_for(product: Product) -> list[str]:
    return CATEGORY_SPEC_KEYS.get(product.category, [])
