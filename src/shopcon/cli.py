"""CLI: `shopcon "query"` -> ranked, reasoned shortlist as markdown."""

from __future__ import annotations

import argparse
import json
import sys

from .catalog import DEFAULT_CATALOG, load_catalog
from .llm import LLM, LLMError, MockLLM, OpenAICompatLLM
from .pipeline import recommend, spec_keys_for


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
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG), help="path to catalog JSON (generated if missing)")
    parser.add_argument("--mock", action="store_true", help="force the deterministic mock LLM (no API key)")
    parser.add_argument("--json", action="store_true", help="print full result as JSON")
    parser.add_argument("--quiet", action="store_true", help="skip the trace output")
    args = parser.parse_args(argv)

    products = load_catalog(args.catalog)
    llm, using_mock = _make_llm(args.mock)
    if using_mock and not args.quiet:
        print("note: using mock LLM (set SHOPCON_API_KEY for real ranking)", file=sys.stderr)

    result = recommend(args.query, products, llm, top_n=args.top)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print(f"\nQuery: {result.query}")
    print(f"Picks ({len(result.ranked)}):\n")
    if result.ranked:
        header = "| # | Product | Price | Rating | Key specs | Why this one |"
        sep = "|---|---------|-------|--------|-----------|--------------|"
        rows = []
        for item in result.ranked:
            p = item.product
            price = f"${p.price:,.2f}"
            specs = p.spec_line(spec_keys_for(p))
            if len(specs) > 60:
                specs = specs[:57] + "..."
            rows.append(f"| {item.rank} | **{p.name}** | {price} | {p.rating} ({p.review_count}) | {specs} | {item.rationale} |")
        print("\n".join([header, sep] + rows))
    print(f"\n**Summary:** {result.summary}\n")

    if not args.quiet:
        print("How it decided (trace):")
        for i, step in enumerate(result.trace, start=1):
            print(f"  {i}. {step}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
