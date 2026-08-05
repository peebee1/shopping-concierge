"""Tests for the evaluation harness."""

import json

import pytest

from shopcon.catalog import Product, generate_sample_catalog
from shopcon.eval import (
    EvalQuery,
    gold_products,
    judge_query,
    load_queries,
    run_query,
)
from shopcon.llm import MockLLM
from shopcon.pipeline import recommend


@pytest.fixture(scope="module")
def catalog():
    return generate_sample_catalog(seed=42)


@pytest.fixture(scope="module")
def queries():
    return load_queries()


def test_query_set_loads():
    qs = load_queries()
    assert len(qs) >= 10
    for q in qs:
        assert q.query
        assert isinstance(q.expect_none, bool)


def test_gold_products_respect_all_constraints(catalog):
    q = EvalQuery(
        query="hot-swap mechanical keyboard under $100",
        max_price=100.0,
        categories=["mechanical_keyboard"],
        must_keywords=["hot-swap", "mechanical"],
    )
    gold = gold_products(q, catalog)
    assert gold, "synthetic catalog must contain satisfying products"
    for p in gold:
        assert p.category == "mechanical_keyboard"
        assert p.price <= 100.0
        assert p.specs.get("hot_swappable") == "yes"
        assert "mechanical" in p.name.lower()


def test_run_query_mock_metrics(catalog):
    q = EvalQuery(
        query="wireless noise-cancelling headphones under $150",
        max_price=150.0,
        categories=["headphones"],
        must_keywords=["wireless", "noise-cancelling"],
    )
    llm = MockLLM()
    qr, result = run_query(q, catalog, llm, top_n=5)
    assert qr.constraint_pass
    assert qr.budget_violations == 0
    assert qr.category_purity == 1.0
    assert qr.keyword_recall == 1.0
    assert qr.gold_count > 0
    assert qr.gold_recall_at_k is not None and qr.gold_recall_at_k > 0
    assert qr.calls == 2  # constraints + rank
    assert qr.tokens == 0  # mock has no token usage
    assert qr.seconds >= 0
    assert len(result.ranked) == 5


def test_run_query_expect_none_is_honest(catalog):
    q = EvalQuery(query="smart fridge under $100", max_price=100.0, expect_none=True)
    qr, _ = run_query(q, catalog, MockLLM(), top_n=5)
    assert qr.gold_count == 0
    assert qr.gold_recall_at_k is None
    # the honesty marker is in the summary (mock falls back to a rule-based
    # summary, so this asserts the plumbing, not the quality)
    assert isinstance(qr.summary, str)


def test_judge_query_falls_back_on_unknown_llm(catalog):
    q = EvalQuery(query="wireless keyboard")
    result = recommend(q.query, catalog, MockLLM(), top_n=3)
    score, notes = judge_query(q, result, MockLLM())
    assert score is None  # mock LLM can't judge -> graceful fallback
    assert notes == ""


def test_report_renders_markdown_and_json(catalog):
    from shopcon.eval import EvalReport

    llm = MockLLM()
    report = EvalReport(model="mock", catalog="synthetic", products_count=len(catalog), top_n=5, results=[])
    for q in [EvalQuery(query="wireless keyboard"), EvalQuery(query="smart fridge under $100", max_price=100.0, expect_none=True)]:
        qr, _ = run_query(q, catalog, llm, top_n=5)
        report.results.append(qr)
        report.total_calls += qr.calls
        report.total_tokens += qr.tokens
        report.cost_usd += qr.cost_usd
        report.total_seconds += qr.seconds

    md = report.to_markdown()
    assert "evaluation report" in md
    assert "wireless keyboard" in md
    assert "constraint pass rate" in md
    json.dumps(report.to_dict())  # must be serializable
