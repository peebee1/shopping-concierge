"""CLI: `shopcon "query"` -> ranked, reasoned shortlist as markdown."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .catalog import resolve_source
from .llm import LLM, LLMError, MockLLM, OpenAICompatLLM
from .pipeline import recommend, spec_keys_for
from .region import Region, detect_from_locale, from_code


def _make_llm(mock: bool) -> tuple[LLM, bool]:
    if mock:
        return MockLLM(), True
    try:
        return OpenAICompatLLM(), False
    except LLMError as exc:
        print(f"note: {exc} -> falling back to mock LLM", file=sys.stderr)
        return MockLLM(), True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="shopcon",
        description="Shopping Concierge: ask for a product in plain English, get a ranked, reasoned shortlist.",
    )
    parser.add_argument("query", help='e.g. "hot-swap mechanical keyboard under $100"')
    parser.add_argument("--top", type=int, default=5, help="how many picks to show (default 5)")
    parser.add_argument(
        "--catalog",
        default=None,
        help="catalog source: synthetic | fakestore | path/to.json | https://... (default: data/catalog.json, auto-generated)",
    )
    parser.add_argument("--mock", action="store_true", help="force the deterministic mock LLM (no API key)")
    parser.add_argument("--verify", type=int, default=3, help="live-verify the top N picks against the source (0 disables)")
    parser.add_argument("--region", default=None, help="region code (US, IN, DE, GB, JP, ...) — default: from locale or SHOPCON_REGION")
    parser.add_argument("--json", action="store_true", help="print full result as JSON")
    parser.add_argument("--quiet", action="store_true", help="skip the trace output")
    args = parser.parse_args(argv)

    region: Region = from_code(args.region or os.environ.get("SHOPCON_REGION") or detect_from_locale().code)
    if os.environ.get("SHOPCON_LIVE_FX") == "1":
        from .region import refresh_rates

        refresh_rates()

    src = resolve_source(args.catalog)
    products = src.load()
    llm, using_mock = _make_llm(args.mock)
    if using_mock and not args.quiet:
        print("note: using mock LLM (set SHOPCON_API_KEY for real ranking)", file=sys.stderr)

    result = recommend(args.query, products, llm, top_n=args.top, source=src, verify_n=args.verify, region=region)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print(f"\nQuery: {result.query}")
    print(
        f"Catalog: {src.name} ({len(products)} products) · {result.source_currency}"
        f" · region {result.region_code} ({result.region_currency})"
        + (f" · {result.freshness}" if result.freshness else "")
    )
    print(f"Picks ({len(result.ranked)}):\n")
    if result.ranked:
        header = "| # | Product | Price | Rating | Key specs | Conf | Why this one |"
        sep = "|---|---------|-------|--------|-----------|------|--------------|"
        rows = []
        for item in result.ranked:
            p = item.product
            price = f"${p.price:,.2f}"
            if result.fx_to_region and result.region_currency:
                local = p.price * result.fx_to_region
                price += f" (~{result.region_currency} {local:,.0f})"
            specs = p.spec_line(spec_keys_for(p))
            if len(specs) > 60:
                specs = specs[:57] + "..."
            rows.append(
                f"| {item.rank} | **{p.name}** | {price} | {p.rating} ({p.review_count}) | {specs} "
                f"| {item.confidence_label} ({item.confidence:.2f}) | {item.rationale} |"
            )
        print("\n".join([header, sep] + rows))

    if result.verifications:
        icons = {"verified": "✓", "changed": "⚠", "unavailable": "✗", "unverifiable": "–"}
        print(f"\nVerification (live re-check of top picks):")
        for vid, v in result.verifications.items():
            if v["status"] == "changed":
                print(f"  ⚠ {vid} — price changed: ${v['price_before']:.2f} → ${v['price_after']:.2f}")
            elif v["status"] == "verified":
                print(f"  ✓ {vid} — unchanged (${v['price_after']:.2f})")
            else:
                print(f"  {icons.get(v['status'], '·')} {vid} — {v['note']}")
        print()

    print(f"\n**Summary:** {result.summary}\n")

    if not args.quiet:
        print("How it decided (trace):")
        for i, step in enumerate(result.trace, start=1):
            print(f"  {i}. {step}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
