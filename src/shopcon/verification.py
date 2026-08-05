"""Evidence & provenance: freshness labeling, per-pick verification results,
and deterministic confidence heuristics.

No LLM calls here — this is the *trust* layer: where the data came from, how
old it is, whether it was re-checked live, and how sure the agent is of a
pick. Everything is computed, never generated.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

# Source data older than this many days gets labeled "stale".
def max_age_days() -> int:
    return int(os.environ.get("SHOPCON_MAX_AGE_DAYS", "7"))


@dataclass
class VerificationResult:
    """Outcome of re-checking one product against its live source.

    status:
        verified     — still present, price unchanged
        changed      — still present, price moved (price_before -> price_after)
        unavailable  — no longer present in the source
        unverifiable — no live source, or the re-check failed (network, ...)
    """

    product_id: str
    status: str
    price_before: float | None = None
    price_after: float | None = None
    fetched_at: str | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return self.__dict__


def human_age(as_of: datetime, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    secs = max(0, int((now - as_of).total_seconds()))
    if secs < 3600:
        return f"{secs // 60}m old"
    if secs < 86400:
        return f"{secs // 3600}h old"
    return f"{secs // 86400}d old"


def freshness_label(as_of: datetime, now: datetime | None = None) -> str:
    age = (now or datetime.now(timezone.utc)) - as_of
    return "fresh" if age.days < max_age_days() else "stale"


def freshness_line(as_of: datetime | None) -> str | None:
    """Human line like 'data as of 2026-08-05 14:02 UTC (fresh, 2h old)'."""
    if as_of is None:
        return None
    return (
        f"data as of {as_of:%Y-%m-%d %H:%M} UTC "
        f"({freshness_label(as_of)}, {human_age(as_of)})"
    )


def verification_notes(results: dict[str, VerificationResult]) -> str | None:
    """Deterministic annotation appended to the summary after a live re-check."""
    changed = [v for v in results.values() if v.status == "changed"]
    verified = [v for v in results.values() if v.status == "verified"]
    if changed:
        detail = "; ".join(
            f"{v.product_id}: ${v.price_before:.2f} → ${v.price_after:.2f}" for v in changed
        )
        return f"[verification] {len(changed)} pick(s) changed since ranking: {detail}"
    if verified:
        return "[verification] top picks re-checked live — prices unchanged."
    return None


def confidence_label(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.6:
        return "medium"
    return "low"
