"""LLM plumbing: a tiny OpenAI-compatible client plus a deterministic mock.

The mock lets the whole pipeline run with zero API key / zero cost (used by
tests and as a fallback when no key is configured).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any


class LLMError(RuntimeError):
    pass


def _extract_json(text: str) -> dict[str, Any]:
    """Tolerant JSON extraction: fenced blocks -> first balanced {...} -> raw."""
    if not text:
        raise LLMError("empty LLM response")
    text = text.strip()
    # 1) fenced code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    # 2) first balanced { ... } block
    m = re.search(r"\{", text)
    if m:
        depth = 0
        start = m.start()
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    text = text[start : i + 1]
                    break
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"could not parse JSON from LLM response: {exc}\n---\n{text[:400]}")
    if not isinstance(obj, dict):
        raise LLMError(f"expected JSON object, got {type(obj).__name__}")
    return obj


class LLM:
    """Protocol: a JSON-in/JSON-out model wrapper.

    Subclasses expose `name` (human label), `model` (for cost lookup),
    `calls` (cumulative complete_json calls) and `usage` (cumulative token
    counts) so the eval harness can report cost/latency.
    """

    name: str = "base"
    model: str = "base"
    calls: int = 0
    usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

    def complete_json(self, system: str, user: str, temperature: float = 0.2) -> dict[str, Any]:
        raise NotImplementedError


class OpenAICompatLLM(LLM):
    """OpenAI-compatible chat completions (works with OpenAI, OpenRouter,
    Groq, local vLLM/Ollama servers, etc.) via the standard env vars:

    SHOPCON_API_KEY / SHOPCON_BASE_URL / SHOPCON_MODEL
    (falls back to OPENAI_API_KEY / OPENAI_BASE_URL / gpt-4o-mini)
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        from openai import OpenAI

        self.model = (
            model
            or os.environ.get("SHOPCON_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or "gpt-4o-mini"
        )
        key = api_key or os.environ.get("SHOPCON_API_KEY") or os.environ.get("OPENAI_API_KEY")
        url = base_url or os.environ.get("SHOPCON_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        if not key:
            raise LLMError("no API key: set SHOPCON_API_KEY (or OPENAI_API_KEY)")
        self._client = OpenAI(api_key=key, base_url=url or None)
        self.name = f"openai-compatible/{self.model}"
        # Some reasoning models spend output budget on hidden reasoning before
        # the visible answer; keep the budget generous (configurable).
        self.max_tokens = int(os.environ.get("SHOPCON_MAX_TOKENS", "6000"))

    def complete_json(self, system: str, user: str, temperature: float = 0.2) -> dict[str, Any]:
        self.calls += 1
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=self.max_tokens,
        )
        if resp.usage:
            self.usage["prompt_tokens"] += resp.usage.prompt_tokens or 0
            self.usage["completion_tokens"] += resp.usage.completion_tokens or 0
        return _extract_json(resp.choices[0].message.content or "")


class MockLLM(LLM):
    """Deterministic stand-in: rule-based constraint parsing and a
    template-based ranking. No network, no cost. Used by tests and when
    no API key is configured."""

    name = "mock"
    model = "mock"

    def complete_json(self, system: str, user: str, temperature: float = 0.2) -> dict[str, Any]:
        self.calls += 1
        if "TASK: constraints" in system:
            return _mock_constraints(user)
        if "TASK: rank" in system:
            return _mock_rank(user)
        raise LLMError("mock LLM does not understand this task")


def _mock_constraints(query: str) -> dict[str, Any]:
    """Shared rule-based constraint extraction (also the no-LLM fallback)."""
    import shopcon.retrieval as r

    # Scanned by retrieval.parse_constraints_rule_based via a stub call below.
    # We reuse the same implementation to keep mock and fallback identical.
    return r.parse_constraints_rule_based(query).__dict__  # type: ignore[attr-defined]


def _mock_rank(user: str) -> dict[str, Any]:
    # Reconstruct candidate list from the rank prompt (see pipeline.py):
    # it ends with "Candidates:\n<json array>". Parse the first balanced
    # [...] block instead of assuming a single-line payload.
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
    candidates = json.loads(user[start:end])
    scores = {c["id"]: c["_score"] for c in candidates}
    names = {c["id"]: c["name"] for c in candidates}
    ranked = []
    for c in candidates:
        price_fit = "within budget" if c["_price_fit"] else "outside stated budget"
        matched = ", ".join(c["_matched_keywords"]) if c["_matched_keywords"] else "no explicit keyword match"
        rationale = (
            f"Matches {len(c['_matched_keywords'])} requested feature(s) ({matched}); "
            f"{price_fit}; rated {c['rating']} from {c['review_count']} reviews."
        )
        ranked.append({"id": c["id"], "rank": 0, "rationale": rationale})
    ranked.sort(key=lambda x: scores[x["id"]], reverse=True)
    for i, item in enumerate(ranked, start=1):
        item["rank"] = i
    summary = f"Mock ranker (deterministic, no LLM). Top pick: {names[ranked[0]['id']]}."
    return {"ranked": ranked, "summary": summary}
