"""Regions, currencies, and FX — LLM-agnostic.

Three-layer region resolution (explicit choice wins, then persisted/env,
then auto-detection):

1. explicit: ``--region IN`` (CLI) / ``?region=IN`` (API) / ``SHOPCON_REGION``
2. locale:   LANG/LC_ALL env vars (CLI, deterministic, no network)
3. IP:       keyless ipwho.is, best-effort, cached 1h (server only)

FX uses approximate static rates so everything stays deterministic and
offline; ``refresh_rates()`` optionally pulls live ECB rates from the
keyless frankfurter.app API (enable with SHOPCON_LIVE_FX=1).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx

DEFAULT_REGION_CODE = "US"

# Approximate USD per 1 unit of currency (static fallback; refreshable).
CURRENCY_TO_USD: dict[str, float] = {
    "USD": 1.0,
    "INR": 0.0120,   # ~83 INR/USD
    "EUR": 1.08,
    "GBP": 1.27,
    "JPY": 0.0067,
    "AUD": 0.65,
    "CAD": 0.73,
    "SGD": 0.74,
}


@dataclass(frozen=True)
class Region:
    code: str
    country: str
    currency: str
    locale: str

    def display_name(self) -> str:
        return f"{self.country} ({self.currency})"


REGIONS: dict[str, Region] = {
    "US": Region("US", "United States", "USD", "en-US"),
    "IN": Region("IN", "India", "INR", "en-IN"),
    "DE": Region("DE", "Germany", "EUR", "de-DE"),
    "GB": Region("GB", "United Kingdom", "GBP", "en-GB"),
    "FR": Region("FR", "France", "EUR", "fr-FR"),
    "JP": Region("JP", "Japan", "JPY", "ja-JP"),
    "AU": Region("AU", "Australia", "AUD", "en-AU"),
    "CA": Region("CA", "Canada", "CAD", "en-CA"),
    "SG": Region("SG", "Singapore", "SGD", "en-SG"),
}


def from_code(code: str | None) -> Region:
    """Resolve a region code; unknown codes fall back to the default."""
    if not code:
        return REGIONS[DEFAULT_REGION_CODE]
    return REGIONS.get(code.strip().upper(), REGIONS[DEFAULT_REGION_CODE])


def default() -> Region:
    return REGIONS[DEFAULT_REGION_CODE]


def convert(amount: float, from_currency: str, to_currency: str) -> float:
    """Convert between any two known currencies (via USD)."""
    if from_currency == to_currency:
        return amount
    rate_from = CURRENCY_TO_USD.get(from_currency.upper())
    rate_to = CURRENCY_TO_USD.get(to_currency.upper())
    if rate_from is None or rate_to is None:
        return amount  # unknown currency: assume already in target
    return amount * rate_from / rate_to


def fx_to_region(source_currency: str, region: Region) -> float | None:
    """1 unit of source currency in region currency (None if same)."""
    if source_currency.upper() == region.currency:
        return None
    rate_source = CURRENCY_TO_USD.get(source_currency.upper())
    rate_region = CURRENCY_TO_USD.get(region.currency)
    if rate_source is None or rate_region is None:
        return None
    return rate_source / rate_region


def detect_from_locale() -> Region:
    """Region from LANG/LC_ALL env vars — deterministic, no network."""
    raw = (
        os.environ.get("LC_ALL")
        or os.environ.get("LC_MONETARY")
        or os.environ.get("LANG")
        or ""
    )
    part = raw.split(".", 1)[0]  # "de_DE.UTF-8" -> "de_DE"
    if "_" in part:
        cc = part.split("_", 1)[1].upper()
        if cc in REGIONS:
            return REGIONS[cc]
    return REGIONS[DEFAULT_REGION_CODE]


_ip_cache: dict = {"region": None, "at": 0.0}


def detect_from_ip(timeout: int = 3) -> Region | None:
    """Best-effort country detection from the request's IP (keyless ipwho.is).

    Cached for an hour; any failure returns None so callers can fall back.
    """
    now = time.time()
    if now - _ip_cache["at"] < 3600:
        return _ip_cache["region"]
    region: Region | None = None
    try:
        resp = httpx.get("https://ipwho.is/", timeout=timeout)
        resp.raise_for_status()
        cc = str(resp.json().get("country_code", "")).upper()
        region = REGIONS.get(cc)
    except Exception:  # noqa: BLE001 - detection must degrade, not crash
        region = None
    _ip_cache.update({"region": region, "at": now})
    return region


def refresh_rates() -> bool:
    """Pull live ECB rates from frankfurter.app (keyless) into the table.

    Call once at startup (e.g. when SHOPCON_LIVE_FX=1). Returns False on any
    failure so callers keep the static fallback.
    """
    targets = sorted({r.currency for r in REGIONS.values()} - {"USD"})
    try:
        resp = httpx.get(
            "https://api.frankfurter.app/latest",
            params={"from": "USD", "to": ",".join(targets)},
            timeout=10,
        )
        resp.raise_for_status()
        rates = resp.json().get("rates", {})
        for cur, rate in rates.items():
            cur = cur.upper()
            if cur in CURRENCY_TO_USD and isinstance(rate, (int, float)):
                CURRENCY_TO_USD[cur] = float(rate)
        return True
    except Exception:  # noqa: BLE001
        return False
