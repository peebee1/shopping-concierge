import pytest

from shopcon.catalog import generate_sample_catalog
from shopcon.llm import MockLLM
from shopcon.pipeline import recommend


@pytest.fixture(scope="module")
def catalog():
    return generate_sample_catalog(seed=42)


class RankedIdsLLM(MockLLM):
    """Emulates a real LLM that returns the alternate {"ranked_ids": [...]} shape."""

    def complete_json(self, system, user, temperature=0.2):
        if "TASK: rank" in system:
            import json as _json

            start = user.find("[")
            depth = 0
            end = -1
            for i in range(start, len(user)):
                if user[i] == "[":
                    depth += 1
                elif user[i] == "]":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            cands = _json.loads(user[start:end])
            ids = [c["id"] for c in cands]
            return {"ranked_ids": ids, "summary": "ranked via ranked_ids shape"}
        return super().complete_json(system, user, temperature)


def test_full_pipeline_mock(catalog):
    result = recommend("wireless mechanical keyboard under $120", catalog, MockLLM(), top_n=5)
    assert len(result.ranked) == 5
    assert result.summary
    assert result.trace, "transparency trace should be populated"
    assert result.model == "mock"
    # ranks are 1..N, no duplicates
    assert [r.rank for r in result.ranked] == list(range(1, 6))
    # candidates are a superset of ranked
    cand_ids = {p.id for p in result.candidates}
    assert all(r.product.id in cand_ids for r in result.ranked)


def test_pipeline_relaxation_surfaces_in_trace(catalog):
    result = recommend("mechanical keyboard under $1", catalog, MockLLM(), top_n=3)
    assert result.constraints.relaxed, "impossible budget should relax"
    assert any("relaxed" in step for step in result.trace)


def test_pipeline_json_serializable(catalog):
    result = recommend("noise cancelling headphones", catalog, MockLLM(), top_n=3)
    import json

    payload = json.dumps(result.to_dict())
    assert '"ranked"' in payload
    assert '"summary"' in payload


def test_pipeline_tolerates_ranked_ids_shape(catalog):
    result = recommend("wireless keyboard", catalog, RankedIdsLLM(), top_n=3)
    assert len(result.ranked) == 3
    assert "ranked via ranked_ids shape" in result.summary


def test_pipeline_honesty_rule_returns_empty_for_impossible_queries(catalog):
    """When nothing matches the requested features, the agent must not pad
    the shortlist with irrelevant products (mock-parsable no-match query)."""
    result = recommend("smartwatch with touchscreen under $200", catalog, MockLLM(), top_n=5)
    assert result.ranked == []
    assert "no products" in result.summary.lower()
    assert any("honest empty shortlist" in step for step in result.trace)
