"""Product catalog: data model + source registry (LLM-agnostic by design).

The catalog layer never imports or talks to the LLM layer. Product data comes
from pluggable *sources* (see ``sources.py``): the bundled synthetic
generator, a JSON file, a JSON URL, or a live commerce API. The pipeline
consumes ``list[Product]`` and works with any LLM — or none (mock).

Catalog JSON shape (what ``save_catalog`` writes and ``JsonSource`` reads)::

    {
      "_meta": {"synthetic": true, "count": 243},
      "products": [
        {"id": "kb-001", "name": "...", "brand": "...", "category": "mechanical_keyboard",
         "price": 89.0, "specs": {"switch": "Cherry MX Brown", "layout": "TKL", ...},
         "rating": 4.6, "review_count": 1200, "url": ""}
      ]
    }
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_CATALOG = Path(__file__).resolve().parent.parent.parent / "data" / "catalog.json"

# Short spec keys worth showing in tables, per category.
CATEGORY_SPEC_KEYS: dict[str, list[str]] = {
    "mechanical_keyboard": ["switch", "layout", "hot_swappable", "wireless", "backlight"],
    "laptop": ["cpu", "ram", "storage", "screen", "weight"],
    "headphones": ["type", "wireless", "anc", "battery", "driver"],
    "monitor": ["size", "resolution", "refresh", "panel", "hdr"],
    "mouse": ["sensor_dpi", "wireless", "buttons", "weight"],
    "webcam": ["resolution", "fps", "mic", "fov"],
    "speaker": ["type", "watts", "bluetooth", "battery"],
    "smartwatch": ["display", "gps", "battery", "water_resist", "health"],
    "microphone": ["type", "polar_pattern", "interface", "bit_depth"],
}

CATEGORY_SYNONYMS: dict[str, list[str]] = {
    "mechanical_keyboard": ["keyboard", "keyboard", "kb", "mech"],
    "laptop": ["laptop", "notebook", "macbook", "ultrabook"],
    "headphones": ["headphone", "headphones", "earbud", "earbuds", "headset", "earphones"],
    "monitor": ["monitor", "display", "screen"],
    "mouse": ["mouse"],
    "webcam": ["webcam", "camera"],
    "speaker": ["speaker", "speakers", "soundbar", "bluetooth speaker"],
    "smartwatch": ["watch", "smartwatch", "fitness watch"],
    "microphone": ["mic", "microphone"],
}


@dataclass
class Product:
    id: str
    name: str
    brand: str
    category: str
    price: float
    specs: dict[str, str] = field(default_factory=dict)
    rating: float = 4.2
    review_count: int = 100
    url: str = ""

    @property
    def spec_text(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in self.specs.items())

    def spec_line(self, keys: list[str] | None = None) -> str:
        keys = keys or list(self.specs.keys())
        parts = [self.specs[k] for k in keys if k in self.specs]
        return ", ".join(parts) if parts else "-"


def save_catalog(
    products: list[Product],
    path: Path | str,
    meta: dict | None = None,
    as_of: datetime | None = None,
) -> None:
    """Write products to a JSON file in the catalog schema.

    ``meta`` keys override the defaults in ``_meta`` (e.g. crawl sources set
    ``synthetic: False`` and their own ``source``/``currency``).

    Every file gets ``_meta.data_as_of``: the moment the data was produced,
    written into the data. Freshness must be a property of the data, not of
    the file, because a git checkout stamps mtime with the clone time, so an
    mtime-derived label reports years-old data as fresh. ``as_of`` is
    injectable so tests and backfills never depend on the wall clock.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = as_of or datetime.now(timezone.utc)
    defaults: dict = {
        "generated_by": "shopcon.catalog.save_catalog",
        "count": len(products),
        "data_as_of": stamp.isoformat(timespec="seconds"),
    }
    if meta:
        defaults.update(meta)
    payload = {"_meta": defaults, "products": [asdict(p) for p in products]}
    path.write_text(json.dumps(payload, indent=2))


def generate_sample_catalog(seed: int = 42, per_category: int = 27) -> list[Product]:
    """Backwards-compatible convenience: the synthetic source, no save."""
    from .sources import SyntheticSource

    return SyntheticSource(seed=seed, per_category=per_category).load()


def resolve_source(spec: str | Path | None = None, autogenerate: bool = True):
    """Resolve a catalog spec to a ``CatalogSource`` (see sources.py).

    ``spec`` can be::

        None            -> data/catalog.json (auto-generated + saved if missing)
        "synthetic"     -> SyntheticSource (seeded, offline)
        "fakestore"     -> FakeStoreSource (live FakeStoreAPI, no key)
        "https://..."   -> JsonSource (any JSON endpoint)
        "path/to.json"  -> JsonSource (local file)
    """
    from .sources import CatalogError, FakeStoreSource, JsonSource, SyntheticSource

    if spec is None:
        if DEFAULT_CATALOG.exists():
            return JsonSource(DEFAULT_CATALOG)
        if autogenerate:
            return SyntheticSource(save_to=DEFAULT_CATALOG)
        raise CatalogError(f"catalog not found: {DEFAULT_CATALOG}")

    s = str(spec)
    low = s.lower()
    if low == "synthetic":
        return SyntheticSource()
    if low == "fakestore":
        return FakeStoreSource()
    return JsonSource(s)  # URL or file path; JsonSource raises if unusable


def load_catalog(source: str | Path | None = None, autogenerate: bool = True) -> list[Product]:
    """Resolve a catalog spec (name / path / URL) and load its products."""
    return resolve_source(source, autogenerate=autogenerate).load()


def known_categories(products: list[Product]) -> list[str]:
    return sorted({p.category for p in products})


def known_brands(products: list[Product]) -> list[str]:
    return sorted({p.brand for p in products})


def main(argv: list[str] | None = None) -> int:
    """Regenerate the sample catalog and print the date it records.

    The fixture written here is reproducible from committed code (seeded), and
    it now carries ``_meta.data_as_of`` so its age survives a checkout.

        uv run shopcon-catalog --out data/catalog.json
    """
    import argparse

    from .sources import SyntheticSource

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_CATALOG))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-category", type=int, default=27)
    args = ap.parse_args(argv)

    out = Path(args.out)
    products = SyntheticSource(
        seed=args.seed, per_category=args.per_category, save_to=out
    ).load()
    meta = json.loads(out.read_text())["_meta"]
    print(f"wrote {out} ({len(products)} products)")
    print(f"data_as_of: {meta['data_as_of']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
