"""Tests for the committed evaluation gate.

The defect these pin: the README's keyless numbers were produced once by hand,
no CI job re-ran them, and `shopcon-eval` exited 0 whatever the result. A gate
that cannot fail is a claim, not a measurement.
"""

import json
from pathlib import Path

import pytest

from shopcon.eval import (
    EvalReport,
    QueryResult,
    drift_paths,
    gate_failures,
    main as eval_main,
    stable_record,
)

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "eval" / "mock_gate.json"


def _report() -> EvalReport:
    qr = QueryResult(
        query="wireless mouse under $20",
        expected={"max_price": 20.0},
        extracted={"max_price": 20.0},
        constraint_pass=True,
        budget_violations=0,
        category_purity=1.0,
        keyword_recall=1.0,
        gold_count=2,
        gold_recall_at_k=0.5,
        ranked_ids=["mouse-001", "mouse-002"],
        summary="Mock ranker (deterministic, no LLM).",
        trace=["understood query -> max $20"],
        calls=2,
        tokens=0,
        cost_usd=0.0,
        seconds=0.0,
    )
    return EvalReport(
        model="mock",
        catalog="default (data/catalog.json)",
        products_count=243,
        top_n=5,
        results=[qr],
        total_calls=2,
        total_tokens=0,
        cost_usd=0.0,
        total_seconds=0.0,
    )


def test_stable_record_drops_wall_clock_only():
    """Two runs that differ only in timing must produce the same record."""
    fast, slow = _report(), _report()
    fast.results[0].seconds = 0.0
    slow.results[0].seconds = 4.2
    fast.total_seconds, slow.total_seconds = 0.0, 61.5
    assert stable_record(fast) == stable_record(slow)
    assert "total_seconds" not in stable_record(slow)["aggregate"]
    assert "seconds" not in stable_record(slow)["results"][0]


def test_committed_record_reproduces_on_the_committed_catalog(capsys):
    """The record in the repo is the number the README quotes, and it holds."""
    assert RECORD.exists(), "eval/mock_gate.json is the gate; without it CI proves nothing"
    code = eval_main(["--mock", "--check", str(RECORD), "--min-pass", "0.85", "--min-gold", "0.75"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "gate: PASS" in out


def test_tampered_record_fails_and_names_the_field(tmp_path, capsys):
    recorded = json.loads(RECORD.read_text())
    recorded["results"][7]["ranked_ids"] = ["not-a-real-product"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(recorded))

    from shopcon.eval import main as eval_main

    assert eval_main(["--mock", "--check", str(bad)]) == 1
    out = capsys.readouterr().out
    assert "FAIL: drift at results[7]" in out
    assert "ranked_ids" in out


def test_floor_trips_below_the_threshold():
    report = _report()
    assert gate_failures(report, 0.85, 0.40) == []
    # one passing query is a 1.0 pass rate, so only the gold floor can trip here
    only_gold = gate_failures(report, 0.99, 0.90)
    assert len(only_gold) == 1 and "avg_gold_recall_at_k" in only_gold[0]
    # a floor above 1.0 proves the pass floor is wired as well
    both = gate_failures(report, 1.01, 0.90)
    assert len(both) == 2
    assert "constraint_pass_rate" in both[0]


def test_drift_counts_a_dropped_query():
    recorded = stable_record(_report())
    shorter = stable_record(_report())
    shorter["results"] = []
    messages = drift_paths(recorded, shorter)
    assert any("recorded 1 entries, current run has 0" in m for m in messages)


def test_record_mode_refuses_a_live_run(tmp_path):
    """A gate record must be the keyless run, or the number depends on a model."""
    from shopcon.eval import main as eval_main

    with pytest.raises(SystemExit) as exc:
        eval_main(["--record", str(tmp_path / "x.json")])
    assert exc.value.code == 2
