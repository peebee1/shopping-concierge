"""Evaluation harness for the Shopping Concierge pipeline.

Holds out a set of queries with hand-written expected constraints, runs the
pipeline end-to-end, and scores it on two axes:

* **behavior** (deterministic, LLM-agnostic): did the picks respect the
  budget / category / keywords, and how much of the ground-truth gold set
  (products genuinely satisfying the expected constraints) landed in the
  top-k?
* **quality** (optional judge LLM): a 1-5 rating of ranking quality and
  honesty, with notes.

Also reports cost/latency per query (calls, tokens, approx USD).

Run keyless with the mock LLM (CI-safe) or with a real LLM::

    shopcon-eval                          # mock LLM, default query set
    shopcon-eval --judge                  # also LLM-judge ranking quality
    shopcon-eval --catalog fakestore --top 5
    shopcon-eval --json-out eval/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .catalog import Product, load_catalog
from .llm import LLM, LLMError, MockLLM, OpenAICompatLLM
from .pipeline import RecommendationResult, recommend
from .region import from_code
from .retrieval import _keyword_hits, _normalize

DEFAULT_QUERIES = Path(__file__).resolve().parent.parent.parent / "data" / "eval_queries.json"

# Approximate USD per 1M tokens (input, output) for known cheap models — used
# for a rough cost column only; unknown models report $0.00.
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "deepseek-v4-flash": (0.14, 0.28),
}

JUDGE_SYSTEM = (
    "TASK: judge\n"
    "You are evaluating a shopping recommendation agent. The user asked for "
    "products; the agent returned a ranked shortlist with a reason per pick "
    "and a summary. Score the response 1-5 on: "
    "(1) how well the picks satisfy the request (budget, features, category), "
    "(2) ranking quality (best value first, worse picks later), "
    "(3) honesty (calls out compromises; admits when nothing matches). "
    'Return ONLY JSON: {"score": <int 1-5>, "notes": "<one sentence>"}.'
)

_HONEST_MARKERS = (
    "none of", "no valid match", "does not satisfy", "no exact match",
    "no products", "nothing matches", "no direct match", "no match",
)


@dataclass
class EvalQuery:
    query: str
    max_price: float | None = None
    min_price: float | None = None
    categories: list[str] = field(default_factory=list)
    must_keywords: list[str] = field(default_factory=list)
    expect_none: bool = False
    region: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "EvalQuery":
        return cls(
            query=str(d["query"]),
            max_price=d.get("max_price"),
            min_price=d.get("min_price"),
            categories=list(d.get("categories") or []),
            must_keywords=[str(k).lower() for k in (d.get("must_keywords") or [])],
            expect_none=bool(d.get("expect_none")),
            region=d.get("region"),
        )

    def expected(self) -> dict:
        return {
            "max_price": self.max_price,
            "min_price": self.min_price,
            "categories": self.categories,
            "must_keywords": self.must_keywords,
            "expect_none": self.expect_none,
            "region": self.region,
        }


def load_queries(path: str | Path | None = None) -> list[EvalQuery]:
    path = Path(path) if path else DEFAULT_QUERIES
    data = json.loads(path.read_text())
    return [EvalQuery.from_dict(d) for d in data["queries"]]


def gold_products(q: EvalQuery, products: list[Product]) -> list[Product]:
    """Ground truth: every product that genuinely satisfies the expected
    constraints (budget, category, ALL must-keywords)."""
    gold = []
    for p in products:
        if q.categories and p.category not in q.categories:
            continue
        if q.max_price is not None and p.price > q.max_price:
            continue
        if q.min_price is not None and p.price < q.min_price:
            continue
        if q.must_keywords:
            hits = _keyword_hits(p, q.must_keywords)
            hit_norm = {_normalize(h) for h in hits}
            if not all(_normalize(k) in hit_norm for k in q.must_keywords):
                continue
        gold.append(p)
    return gold


@dataclass
class QueryResult:
    query: str
    expected: dict
    extracted: dict
    constraint_pass: bool
    budget_violations: int
    category_purity: float
    keyword_recall: float
    gold_count: int
    gold_recall_at_k: float | None
    ranked_ids: list[str]
    summary: str
    trace: list[str]
    judge_score: float | None = None
    judge_notes: str = ""
    calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def run_query(q: EvalQuery, products: list[Product], llm: LLM, top_n: int = 5) -> tuple[QueryResult, RecommendationResult]:
    before = (llm.calls, dict(llm.usage))
    t0 = time.monotonic()
    result = recommend(q.query, products, llm, top_n=top_n, region=from_code(q.region))
    seconds = time.monotonic() - t0
    calls = llm.calls - before[0]
    used = {k: llm.usage[k] - before[1][k] for k in before[1]}
    tokens = used["prompt_tokens"] + used["completion_tokens"]
    price = _PRICING.get(getattr(llm, "model", ""))
    cost = (
        used["prompt_tokens"] / 1e6 * price[0] + used["completion_tokens"] / 1e6 * price[1]
        if price
        else 0.0
    )

    ranked = result.ranked[:top_n]
    budget_violations = sum(1 for r in ranked if q.max_price is not None and r.product.price > q.max_price)

    # Empty shortlists are a feature (honest no-match): purity/recall are
    # vacuous then, not zero.
    if q.categories:
        category_purity = (
            sum(1 for r in ranked if r.product.category in q.categories) / len(ranked) if ranked else 1.0
        )
    else:
        category_purity = 1.0

    if q.must_keywords:
        matched = sum(
            1
            for k in q.must_keywords
            if any(_keyword_hits(r.product, [k]) for r in ranked)
        )
        keyword_recall = matched / len(q.must_keywords) if ranked else 1.0
    else:
        keyword_recall = 1.0

    passes: list[bool] = []
    if q.max_price is not None:
        passes.append(budget_violations == 0)
    if q.categories and not q.expect_none:
        passes.append(category_purity >= 0.8)
    if q.must_keywords and not q.expect_none:
        passes.append(keyword_recall >= 0.5)

    gold_ids: set[str] = set()
    gold_count = 0
    gold_recall: float | None = None
    if q.expect_none:
        s = result.summary.lower()
        honest = (not ranked) or any(m in s for m in _HONEST_MARKERS)
        passes.append(honest)
    else:
        gold = gold_products(q, products)
        gold_ids = {p.id for p in gold}
        gold_count = len(gold)
        if gold_ids:
            gold_recall = len(gold_ids & set(r.product.id for r in ranked)) / len(gold_ids)

    extracted = {
        "max_price": result.constraints.max_price,
        "min_price": result.constraints.min_price,
        "categories": result.constraints.categories,
        "must_keywords": result.constraints.must_keywords,
        "nice_keywords": result.constraints.nice_keywords,
        "relaxed": result.constraints.relaxed,
    }

    qr = QueryResult(
        query=q.query,
        expected=q.expected(),
        extracted=extracted,
        constraint_pass=all(passes) if passes else True,
        budget_violations=budget_violations,
        category_purity=round(category_purity, 2),
        keyword_recall=round(keyword_recall, 2),
        gold_count=gold_count,
        gold_recall_at_k=round(gold_recall, 3) if gold_recall is not None else None,
        ranked_ids=[r.product.id for r in ranked],
        summary=result.summary,
        trace=result.trace,
        calls=calls,
        tokens=tokens,
        cost_usd=round(cost, 4),
        seconds=round(seconds, 2),
    )
    return qr, result


def judge_query(q: EvalQuery, result: RecommendationResult, judge: LLM, top_n: int = 5) -> tuple[float | None, str]:
    ranked_lines = []
    for r in result.ranked[:top_n]:
        p = r.product
        ranked_lines.append(
            f"- {p.name} (${p.price:.2f}, {p.rating}/5, {p.review_count} reviews): {r.rationale}"
        )
    user = f"Request: {q.query}\n\nRanked picks:\n" + "\n".join(ranked_lines) + f"\n\nSummary: {result.summary}"
    try:
        data = judge.complete_json(JUDGE_SYSTEM, user, temperature=0.0)
        score = int(data.get("score", 3))
        return max(1, min(5, score)), str(data.get("notes", ""))
    except Exception:  # noqa: BLE001 - a judge failure must never break the eval run
        return None, ""


@dataclass
class EvalReport:
    model: str
    catalog: str
    products_count: int
    top_n: int
    results: list[QueryResult]
    total_calls: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    total_seconds: float = 0.0

    def aggregate(self) -> dict:
        n = len(self.results) or 1
        golds = [r.gold_recall_at_k for r in self.results if r.gold_recall_at_k is not None]
        judges = [r.judge_score for r in self.results if r.judge_score is not None]
        return {
            "constraint_pass_rate": round(sum(r.constraint_pass for r in self.results) / n, 3),
            "avg_gold_recall_at_k": round(sum(golds) / len(golds), 3) if golds else None,
            "avg_judge_score": round(sum(judges) / len(judges), 2) if judges else None,
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 4),
            "total_seconds": round(self.total_seconds, 2),
        }

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "catalog": self.catalog,
            "products_count": self.products_count,
            "top_n": self.top_n,
            "aggregate": self.aggregate(),
            "results": [r.to_dict() for r in self.results],
        }

    def to_markdown(self) -> str:
        agg = self.aggregate()
        lines = [
            "# Shopping Concierge — evaluation report",
            "",
            f"- model: `{self.model}`  ·  catalog: `{self.catalog}` ({self.products_count} products)  ·  top-N: {self.top_n}",
            f"- queries: {len(self.results)}  ·  **constraint pass rate: {agg['constraint_pass_rate']:.0%}**"
            + (f"  ·  **avg gold recall@{self.top_n}: {agg['avg_gold_recall_at_k']:.0%}**" if agg["avg_gold_recall_at_k"] is not None else "")
            + (f"  ·  **avg judge score: {agg['avg_judge_score']}/5**" if agg["avg_judge_score"] is not None else ""),
            f"- cost: {agg['total_calls']} LLM calls, {agg['total_tokens']} tokens, ~${agg['cost_usd']:.4f}, {agg['total_seconds']}s",
            "",
            "| Query | Pass | $viol | Cat% | Kw% | Gold@N | Judge | Sec |",
            "|-------|------|-------|------|-----|--------|-------|-----|",
        ]
        for r in self.results:
            lines.append(
                f"| {r.query} | {'✅' if r.constraint_pass else '❌'} | {r.budget_violations} "
                f"| {r.category_purity:.0%} | {r.keyword_recall:.0%} "
                f"| {f'{r.gold_recall_at_k:.0%}' if r.gold_recall_at_k is not None else '—'} "
                f"| {f'{r.judge_score}/5' if r.judge_score is not None else '—'} | {r.seconds} |"
            )
        failed = [r for r in self.results if not r.constraint_pass]
        if failed:
            lines += ["", "### Failures", ""]
            for r in failed:
                lines += [
                    f"**{r.query}** — extracted: {r.extracted}",
                    f"- ranked: {', '.join(r.ranked_ids) or '(empty)'}",
                    f"- summary: {r.summary}",
                    f"- trace: {'; '.join(r.trace)}",
                    "",
                ]
        if agg["avg_judge_score"] is not None:
            notes = [r for r in self.results if r.judge_score is not None and r.judge_score <= 2]
            if notes:
                lines += ["### Lowest-judged", ""]
                for r in notes:
                    lines.append(f"- **{r.query}** ({r.judge_score}/5): {r.judge_notes}")
        return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="shopcon-eval",
        description="Evaluate the Shopping Concierge pipeline against held-out queries.",
    )
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES), help="eval queries JSON")
    parser.add_argument("--catalog", default=None, help="catalog source (synthetic | fakestore | file | URL)")
    parser.add_argument("--top", type=int, default=5, help="ranked picks per query (default 5)")
    parser.add_argument("--mock", action="store_true", help="force the mock LLM (no API key, CI-safe)")
    parser.add_argument("--judge", action="store_true", help="also rate ranking quality with a judge LLM")
    parser.add_argument("--json-out", default=None, help="write full results JSON to this path")
    parser.add_argument("--max-queries", type=int, default=0, help="limit to first N queries")
    parser.add_argument("--skip", type=int, default=0, help="skip the first N queries (chunked runs)")
    args = parser.parse_args(argv)

    products = load_catalog(args.catalog)
    queries = load_queries(args.queries)
    if args.skip:
        queries = queries[args.skip :]
    if args.max_queries:
        queries = queries[: args.max_queries]

    if args.mock:
        llm: LLM = MockLLM()
    else:
        try:
            llm = OpenAICompatLLM()
        except LLMError as exc:
            print(f"note: {exc} -> using mock LLM", file=sys.stderr)
            llm = MockLLM()
    judge = llm if args.judge else None

    report = EvalReport(
        model=llm.name,
        catalog=str(args.catalog) if args.catalog else "default (data/catalog.json)",
        products_count=len(products),
        top_n=args.top,
        results=[],
    )
    for q in queries:
        qr, result = run_query(q, products, llm, top_n=args.top)
        if judge:
            qr.judge_score, qr.judge_notes = judge_query(q, result, judge, top_n=args.top)
        report.results.append(qr)
        report.total_calls += qr.calls
        report.total_tokens += qr.tokens
        report.cost_usd += qr.cost_usd
        report.total_seconds += qr.seconds
        flag = "✅" if qr.constraint_pass else "❌"
        print(f"{flag} {q.query}  (judge {qr.judge_score}/5)" if qr.judge_score is not None else f"{flag} {q.query}")

    print()
    print(report.to_markdown())
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(report.to_dict(), indent=2))
        print(f"full results -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
