import json
import os
from datetime import datetime, timedelta, timezone

from shopcon.catalog import generate_sample_catalog, load_catalog, save_catalog
from shopcon.sources import JsonSource, parse_recorded_as_of
from shopcon.verification import freshness_label, freshness_line


def test_catalog_is_deterministic():
    a = generate_sample_catalog(seed=42)
    b = generate_sample_catalog(seed=42)
    assert [p.id for p in a] == [p.id for p in b]
    assert len(a) == len(b) == 9 * 27


def test_catalog_roundtrip(tmp_path):
    products = generate_sample_catalog(seed=7, per_category=3)
    path = tmp_path / "cat.json"
    save_catalog(products, path)
    loaded = load_catalog(path, autogenerate=False)
    assert len(loaded) == len(products)
    assert loaded[0].__dict__ == products[0].__dict__


def test_catalog_values_are_sane():
    products = generate_sample_catalog()
    assert len({p.id for p in products}) == len(products), "product ids must be unique"
    for p in products:
        assert p.price > 0
        assert 0 <= p.rating <= 5
        assert p.specs, "every product should have specs"
        assert p.name and p.brand


# ---------------------------------------------------------------------------
# Provenance: the data carries its own date, so a checkout cannot reset it.
# ---------------------------------------------------------------------------

STAMP = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)


def test_save_catalog_records_the_data_date(tmp_path):
    path = tmp_path / "cat.json"
    save_catalog(generate_sample_catalog(seed=7, per_category=2), path, as_of=STAMP)
    meta = json.loads(path.read_text())["_meta"]
    assert meta["data_as_of"] == "2026-08-05T10:00:00+00:00"

    src = JsonSource(path)
    src.load()
    assert src.as_of == STAMP
    assert src.as_of_source == "recorded"


def test_saving_twice_with_one_stamp_is_byte_identical(tmp_path):
    products = generate_sample_catalog(seed=7, per_category=2)
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    save_catalog(products, a, as_of=STAMP)
    save_catalog(products, b, as_of=STAMP)
    assert a.read_bytes() == b.read_bytes()


def test_recorded_date_beats_a_fresh_file_mtime(tmp_path, monkeypatch):
    """The defect: git stamps mtime with the clone time, so old data reads fresh."""
    monkeypatch.delenv("SHOPCON_MAX_AGE_DAYS", raising=False)
    path = tmp_path / "cat.json"
    save_catalog(generate_sample_catalog(seed=7, per_category=2), path, as_of=STAMP)
    # A checkout writes the file today.
    os.utime(path, None)

    src = JsonSource(path)
    src.load()
    assert src.as_of == STAMP, "mtime must not overwrite the recorded date"
    now = datetime.now(timezone.utc)
    assert src.as_of is not None
    assert freshness_label(src.as_of, now) == "stale"
    line = freshness_line(src.as_of, now)
    assert line is not None
    assert line.startswith("data as of 2026-08-05 10:00 UTC (stale")

    # The old behaviour, shown explicitly: the file alone says fresh.
    legacy = json.loads(path.read_text())
    del legacy["_meta"]["data_as_of"]
    path.write_text(json.dumps(legacy))
    legacy_src = JsonSource(path)
    legacy_src.load()
    assert legacy_src.as_of_source == "mtime"
    assert legacy_src.as_of is not None
    assert freshness_label(legacy_src.as_of, now) == "fresh"


def test_legacy_catalog_without_a_date_falls_back_to_mtime(tmp_path):
    path = tmp_path / "cat.json"
    save_catalog(generate_sample_catalog(seed=7, per_category=2), path, as_of=STAMP)
    payload = json.loads(path.read_text())
    del payload["_meta"]["data_as_of"]
    path.write_text(json.dumps(payload))

    src = JsonSource(path)
    src.load()
    assert src.as_of_source == "mtime"
    assert src.as_of is not None
    assert abs((src.as_of - datetime.now(timezone.utc)).total_seconds()) < 30


def test_bad_recorded_date_falls_back_instead_of_crashing(tmp_path):
    path = tmp_path / "cat.json"
    save_catalog(generate_sample_catalog(seed=7, per_category=2), path, as_of=STAMP)
    payload = json.loads(path.read_text())
    payload["_meta"]["data_as_of"] = "sometime last summer"
    path.write_text(json.dumps(payload))

    src = JsonSource(path)
    src.load()
    assert src.as_of_source == "mtime"


def test_parse_recorded_as_of_accepts_common_forms():
    assert parse_recorded_as_of("2026-08-05T10:00:00+00:00") == STAMP
    assert parse_recorded_as_of("2026-08-05T10:00:00Z") == STAMP
    assert parse_recorded_as_of("2026-08-05T10:00:00") == STAMP  # naive means UTC
    assert parse_recorded_as_of("2026-08-05") == datetime(2026, 8, 5, tzinfo=timezone.utc)
    for bad in (None, "", "   ", 12345, "not a date"):
        assert parse_recorded_as_of(bad) is None


def test_age_is_computed_from_the_recorded_date_only():
    """One stamp, one label: no wall clock in the arithmetic."""
    later = STAMP + timedelta(days=99)
    assert freshness_line(STAMP, later) == "data as of 2026-08-05 10:00 UTC (stale, 99d old)"
    assert freshness_line(STAMP, STAMP.replace(day=6)) == "data as of 2026-08-05 10:00 UTC (fresh, 1d old)"
