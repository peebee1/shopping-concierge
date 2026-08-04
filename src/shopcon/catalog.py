"""Product catalog: data model, sample-data generator, load/save helpers.

The bundled catalog is SYNTHETIC: it is generated with a fixed random seed so
the repo is self-contained, deterministic, and free of scraping/ToS concerns.
All brand names and products are invented. Point the CLI/server at your own
JSON file (see README) to use real data.

Catalog JSON shape::

    {
      "_meta": {"synthetic": true, "generated_by": "shopcon.catalog", "count": 240},
      "products": [
        {"id": "kb-001", "name": "...", "brand": "...", "category": "mechanical_keyboard",
         "price": 89.0, "specs": {"switch": "Cherry MX Brown", "layout": "TKL", ...},
         "rating": 4.6, "review_count": 1200, "url": ""}
      ]
    }
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
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


# ---------------------------------------------------------------------------
# Synthetic sample catalog (seeded + deterministic)
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


def generate_sample_catalog(seed: int = 42, per_category: int = 27) -> list[Product]:
    """Deterministically generate a synthetic catalog (same seed -> same catalog)."""
    rng = random.Random(seed)
    products: list[Product] = []
    for cat, tpl in _CATEGORY_TEMPLATES.items():
        lo, hi = tpl["price"]
        for i in range(per_category):
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
    return products


def save_catalog(products: list[Product], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": {
            "synthetic": True,
            "note": "SYNTHETIC sample data - invented brands/products for demo purposes. Replace with a real catalog.",
            "generated_by": "shopcon.catalog.generate_sample_catalog",
            "count": len(products),
        },
        "products": [asdict(p) for p in products],
    }
    path.write_text(json.dumps(payload, indent=2))


def load_catalog(path: Path | str | None = None, autogenerate: bool = True) -> list[Product]:
    """Load a catalog JSON file; generate the bundled sample catalog if missing."""
    path = Path(path) if path else DEFAULT_CATALOG
    if not path.exists():
        if not autogenerate:
            raise FileNotFoundError(f"catalog not found: {path}")
        save_catalog(generate_sample_catalog(), path)
    data = json.loads(path.read_text())
    return [Product(**p) for p in data["products"]]


def known_categories(products: list[Product]) -> list[str]:
    return sorted({p.category for p in products})


def known_brands(products: list[Product]) -> list[str]:
    return sorted({p.brand for p in products})
