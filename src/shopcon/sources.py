"""Catalog sources — pluggable, and LLM-agnostic by design.

A *source* is anything that produces ``list[Product]``: the bundled synthetic
generator, a local JSON file, a JSON URL, or a live commerce API. None of this
module imports or knows about the LLM layer — swap sources freely and run the
pipeline with any LLM (or none).

To add a source: implement the ``CatalogSource`` protocol (``name`` +
``load()``) and register it in ``catalog.resolve_source``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Protocol

import httpx

from .catalog import Product, save_catalog


class CatalogError(RuntimeError):
    """Raised when a catalog source cannot produce products."""


class CatalogSource(Protocol):
    name: str

    def load(self) -> list[Product]: ...


# ---------------------------------------------------------------------------
# Synthetic source (seeded, deterministic, offline)
# ---------------------------------------------------------------------------

_FICTIONAL_BRANDS = [
    "Northline", "VoltEdge", "Aurio", "Craftly", "Nimbus", "Orbit", "PixelForge",
    "SoundHive", "Klavier", "Ember", "Zenith", "Drift", "Halcyon", "Vantage", "Relay",
]

_CATEGORY_TEMPLATES: dict[str, dict] = {
    "mechanical_keyboard": {
        "words": ["Mechanical Keyboard", "Pro Keyboard", "Gaming Keyboard"],
        "price": (45, 220),
        "specs": {
            "switch": ["Cherry MX Red", "Cherry MX Brown", "Cherry MX Blue", "Gateron Yellow", "Gateron Brown", "Razer-style Linear"],
            "layout": ["65%", "TKL", "Full-size", "75%", "40%"],
            "hot_swappable": ["yes", "no"],
            "wireless": ["yes", "no"],
            "backlight": ["RGB", "white", "none"],
        },
    },
    "laptop": {
        "words": ["Ultrabook", "Notebook", "Creator Laptop", "Workstation Laptop"],
        "price": (480, 2600),
        "specs": {
            "cpu": ["Intel Core i5", "Intel Core i7", "AMD Ryzen 5", "AMD Ryzen 7", "Intel Core i9", "AMD Ryzen 9"],
            "ram": ["8 GB", "16 GB", "32 GB", "64 GB"],
            "storage": ["256 GB SSD", "512 GB SSD", "1 TB SSD", "2 TB SSD"],
            "screen": ['13.3"', '14"', '15.6"', '16"', '17.3"'],
            "weight": ["1.2 kg", "1.4 kg", "1.8 kg", "2.1 kg", "2.5 kg"],
        },
    },
    "headphones": {
        "words": ["Wireless Headphones", "Over-Ear Headphones", "ANC Headphones", "In-Ear Monitors"],
        "price": (30, 420),
        "specs": {
            "type": ["over-ear", "in-ear", "on-ear", "earbud"],
            "wireless": ["yes", "no"],
            "anc": ["yes", "no"],
            "battery": ["20 h", "30 h", "40 h", "60 h"],
            "driver": ["40 mm", "50 mm", "10 mm", "13 mm"],
        },
    },
    "monitor": {
        "words": ["Monitor", "Gaming Monitor", "4K Monitor", "Ultrawide Monitor"],
        "price": (90, 950),
        "specs": {
            "size": ['24"', '27"', '32"', '34"'],
            "resolution": ["1920x1080", "2560x1440", "3840x2160", "3440x1440"],
            "refresh": ["60 Hz", "75 Hz", "144 Hz", "165 Hz", "240 Hz"],
            "panel": ["IPS", "VA", "TN", "OLED"],
            "hdr": ["none", "HDR400", "HDR600", "HDR1000"],
        },
    },
    "mouse": {
        "words": ["Gaming Mouse", "Wireless Mouse", "Ergonomic Mouse"],
        "price": (15, 160),
        "specs": {
            "sensor_dpi": ["8000", "16000", "26000", "32000"],
            "wireless": ["yes", "no"],
            "buttons": ["6", "8", "10", "12"],
            "weight": ["55 g", "70 g", "85 g", "100 g"],
        },
    },
    "webcam": {
        "words": ["Webcam", "Streaming Camera", "Conference Camera"],
        "price": (25, 210),
        "specs": {
            "resolution": ["1080p", "1440p", "4K"],
            "fps": ["30", "60"],
            "mic": ["yes", "no"],
            "fov": ["70 deg", "90 deg", "110 deg"],
        },
    },
    "speaker": {
        "words": ["Bluetooth Speaker", "Smart Speaker", "Soundbar"],
        "price": (20, 360),
        "specs": {
            "type": ["portable", "desktop", "soundbar", "party"],
            "watts": ["10 W", "20 W", "30 W", "50 W", "80 W"],
            "bluetooth": ["5.0", "5.1", "5.3"],
            "battery": ["8 h", "12 h", "20 h", "none"],
        },
    },
    "smartwatch": {
        "words": ["Smartwatch", "Fitness Watch", "Sports Watch"],
        "price": (40, 520),
        "specs": {
            "display": ["AMOLED", "LCD", "E-Ink"],
            "gps": ["yes", "no"],
            "battery": ["1 day", "3 days", "7 days", "14 days"],
            "water_resist": ["5 ATM", "10 ATM", "IP68"],
            "health": ["heart rate", "heart rate + SpO2", "heart rate + SpO2 + ECG"],
        },
    },
    "microphone": {
        "words": ["Condenser Microphone", "USB Microphone", "Dynamic Microphone"],
        "price": (25, 320),
        "specs": {
            "type": ["condenser", "dynamic", "shotgun"],
            "polar_pattern": ["cardioid", "supercardioid", "omnidirectional"],
            "interface": ["USB", "XLR"],
            "bit_depth": ["16-bit", "24-bit"],
        },
    },
}

_VARIANTS = ["Arc", "Prime", "Nova", "Core", "Lite", "Max", "Elite", "Sprint", "Zephyr", "Forge", "Pulse", "Titan"]


class SyntheticSource:
    """Deterministically generated catalog with invented brands/products.

    Same seed -> same catalog. Optionally persists to a JSON file (``save_to``)
    so the default data/catalog.json gets generated on first run.
    """

    name = "synthetic"

    def __init__(self, seed: int = 42, per_category: int = 27, save_to: Path | str | None = None):
        self.seed = seed
        self.per_category = per_category
        self.save_to = Path(save_to) if save_to else None

    def load(self) -> list[Product]:
        rng = random.Random(self.seed)
        products: list[Product] = []
        for cat, tpl in _CATEGORY_TEMPLATES.items():
            lo, hi = tpl["price"]
            for i in range(self.per_category):
                brand = rng.choice(_FICTIONAL_BRANDS)
                variant = rng.choice(_VARIANTS)
                name = f"{brand} {variant} {rng.choice(tpl['words'])}"
                specs = {k: rng.choice(v) for k, v in tpl["specs"].items()}
                products.append(
                    Product(
                        id=f"{cat}-{i + 1:03d}",
                        name=name,
                        brand=brand,
                        category=cat,
                        price=round(rng.uniform(lo, hi), 2),
                        specs=specs,
                        rating=round(rng.uniform(3.6, 4.9), 1),
                        review_count=rng.randint(10, 8000),
                    )
                )
        if self.save_to:
            save_catalog(products, self.save_to)
        return products


# ---------------------------------------------------------------------------
# JSON source (local file or any JSON endpoint)
# ---------------------------------------------------------------------------

class JsonSource:
    """Products from a local JSON file or a JSON URL.

    Expected shape — the same schema ``catalog.save_catalog`` writes::

        {"products": [{"id", "name", "brand", "category", "price",
                       "specs", "rating", "review_count", "url"}]}
    """

    def __init__(self, location: str | Path):
        self.location = str(location)
        self.name = "json-url" if self.location.startswith(("http://", "https://")) else "json"

    def load(self) -> list[Product]:
        loc = self.location
        if loc.startswith(("http://", "https://")):
            try:
                resp = httpx.get(loc, timeout=20)
                resp.raise_for_status()
                data = resp.json()
            except (httpx.HTTPError, ValueError) as exc:
                raise CatalogError(f"could not fetch catalog from {loc}: {exc}") from exc
        else:
            path = Path(loc)
            if not path.exists():
                raise CatalogError(
                    f"catalog file not found: {path} "
                    "(known sources: synthetic, fakestore; or a file path / URL)"
                )
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise CatalogError(f"could not read catalog {path}: {exc}") from exc
        try:
            return [Product(**p) for p in data["products"]]
        except (KeyError, TypeError) as exc:
            raise CatalogError(f"catalog at {loc} has no 'products' array: {exc}") from exc


# ---------------------------------------------------------------------------
# Live source: FakeStoreAPI (https://fakestoreapi.com) — keyless
# ---------------------------------------------------------------------------

class FakeStoreSource:
    """Live products from FakeStoreAPI — a real REST commerce API, no key.

    Categories are broad (electronics, jewelery, clothing) and descriptions
    are free text, so queries like "bluetooth speaker under $100" run against
    real data. Demonstrates mapping an arbitrary API to the Product schema.
    """

    name = "fakestore"
    url = "https://fakestoreapi.com/products"

    def load(self) -> list[Product]:
        try:
            resp = httpx.get(self.url, timeout=20)
            resp.raise_for_status()
            raw = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise CatalogError(f"FakeStoreAPI unavailable: {exc}") from exc
        return [self._map(item) for item in raw]

    @staticmethod
    def _map(item: dict) -> Product:
        desc = str(item.get("description") or "").strip().replace("\n", " ")
        rating = item.get("rating") or {}
        return Product(
            id=f"fs-{item.get('id', 0)}",
            name=str(item.get("title") or "Untitled"),
            brand="FakeStore",
            category=_fs_category(str(item.get("category") or "other")),
            price=float(item.get("price") or 0.0),
            specs={"description": desc[:140]},
            rating=float(rating.get("rate") or 0.0),
            review_count=int(rating.get("count") or 0),
        )


def _fs_category(cat: str) -> str:
    mapping = {
        "electronics": "electronics",
        "jewelery": "jewelery",
        "men's clothing": "mens_clothing",
        "women's clothing": "womens_clothing",
    }
    return mapping.get(cat, cat.replace(" ", "_").replace("'", ""))
