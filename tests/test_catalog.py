from shopcon.catalog import generate_sample_catalog, load_catalog, save_catalog


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
