#!/usr/bin/env python3
"""Merge chunked eval reports (report_p1..p3.json) into eval/report.json."""
import json
import sys

sys.path.insert(0, "src")
from shopcon.eval import EvalReport, QueryResult  # noqa: E402

parts = []
for i in (1, 2, 3):
    with open(f"eval/report_p{i}.json") as f:
        parts.append(json.load(f))

report = EvalReport(
    model=parts[0]["model"],
    catalog=parts[0]["catalog"],
    products_count=parts[0]["products_count"],
    top_n=parts[0]["top_n"],
    results=[],
)
fields = set(QueryResult.__dataclass_fields__)
for p in parts:
    for r in p["results"]:
        report.results.append(QueryResult(**{k: v for k, v in r.items() if k in fields}))
        report.total_calls += r["calls"]
        report.total_tokens += r["tokens"]
        report.cost_usd += r["cost_usd"]
        report.total_seconds += r["seconds"]

with open("eval/report.json", "w") as f:
    json.dump(report.to_dict(), f, indent=2)
print(report.to_markdown())
