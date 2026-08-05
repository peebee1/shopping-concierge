# 🛍️ Shopping Concierge

Ask for a product in plain English. Get a ranked, reasoned shortlist.

A minimal, observable **agentic-commerce** demo: one LLM call to *understand* the request, a deterministic catalog retrieval, and one LLM call to *rank and justify* the picks. Every step is traced so you can see exactly why the agent chose what it chose.

```bash
$ shopcon "hot-swap mechanical keyboard under $100"
```

```
Query: hot-swap mechanical keyboard under $100
Picks (3):

| # | Product | Price | Rating | Key specs | Why this one |
|---|---------|-------|--------|-----------|--------------|
| 1 | **Klavier Prime Gaming Keyboard** | $85.72 | 4.4 (4607) | Gateron Yellow, 65%, yes, yes, RGB | Meets the hot-swap requirement under $100 with the highest rating (4.4) and includes wireless, RGB, and a 65% layout, offering the best feature-per-dollar value despite the higher price. |
| 2 | **VoltEdge Titan Gaming Keyboard** | $57.40 | 3.9 (5490) | Cherry MX Red, TKL, yes, yes, white | Affordable at $57.40, hot-swappable, and adds wireless connectivity with a white backlight, though its 3.9 rating is the lowest among the options. |
| 3 | **SoundHive Prime Mechanical Keyboard** | $56.12 | 4.1 (3844) | Cherry MX Red, TKL, yes, no, none | Cheapest option at $56.12 and hot-swappable with a solid 4.1 rating, but lacks wireless and backlighting, making it a basic value pick rather than a complete feature set. |

**Summary:** The best pick is the Klavier Prime Gaming Keyboard for its hot-swap functionality, wireless/RGB features, and top 4.4 rating, all under $100. A strong budget alternative is the VoltEdge Titan if you want wireless at a much lower price, but its lower rating is a caveat; the SoundHive is the cheapest yet lacks wireless and backlighting.
```

*(Example output generated with `deepseek-v4-flash` via an OpenAI-compatible endpoint — 2 LLM calls, a few hundred tokens total.)*

It handles messy, multi-constraint requests too:

```bash
$ shopcon "gaming laptop with 32 GB RAM and a good screen, between 1000 and 1500"
```

> **Summary:** The best pick is the PixelForge Pulse Notebook: it offers a larger 17.3" screen, an Intel Core i9, and a lower price, making it the stronger gaming value within your budget. The Zenith Titan Creator Laptop is a good alternative if you prefer a highly rated, reliable machine and can accept a smaller 14" screen. One caveat is the PixelForge's lower 3.9 rating, so check user reviews for display quality and thermals before buying.

Note that the agent *noticed* the 32 GB RAM requirement filtered the catalog down to two real matches — no fake fillers.

---

## Why this design

Most "agentic shopping" repos glue an LLM onto a scraper and call it a day. This one is deliberately **hybrid**:

| Stage | What runs | Cost |
|-------|-----------|------|
| 1. Understand | LLM turns the query into structured constraints (budget, category, must-have features) | 1 call |
| 2. Retrieve | Deterministic scoring over the catalog — price fit, keyword match, rating, popularity | **0 calls** |
| 3. Rank | LLM orders the top 8 candidates and writes a rationale + verdict per pick | 1 call |

So a full recommendation costs **2 LLM calls** — cheap flash-tier models work great (default is `gpt-4o-mini`, any OpenAI-compatible endpoint works).

The two "agentic" properties people care about are built in:

- **Judgment** — the LLM decides what the user actually meant (is "hot-swap" a hard requirement? does $100 include tax?) and why one product beats another.
- **Transparency** — the trace shows the extracted constraints, what was retrieved, and what got relaxed when nothing matched (e.g. impossible budget). No black box.

## Quickstart

```bash
git clone https://github.com/peebee1/shopping-concierge.git
cd shopping-concierge
uv sync                     # or: pip install -e . && pip install pytest
uv run shopcon "wireless noise-cancelling headphones under $150"
```

Works out of the box with **no API key** — it falls back to a deterministic mock LLM (used by the tests too). For real ranking:

```bash
export SHOPCON_API_KEY=sk-...        # any OpenAI-compatible key
export SHOPCON_BASE_URL=             # optional, e.g. https://api.openai.com/v1
export SHOPCON_MODEL=gpt-4o-mini     # cheap flash-tier model recommended
```

Or copy `.env.example` → `.env`. The CLI also supports `--mock`, `--json`, `--top N`, `--quiet`, and `--catalog <source>`.

Want live data instead of the bundled synthetic catalog?

```bash
uv run shopcon "external hard drive under $100" --catalog fakestore
```

FakeStoreAPI is a real, keyless REST commerce API — see "Catalog sources" below.

### Web playground

```bash
uv run uvicorn shopcon.server:app --port 8000
# open http://localhost:8000
```

## Catalog sources (LLM-agnostic)

The catalog layer is **fully LLM-agnostic** — it never imports or talks to the LLM layer (a test enforces this: importing `catalog`/`sources`/`retrieval` never pulls in `shopcon.llm`). Product data comes from pluggable *sources*; pick one with `--catalog` (CLI) or `SHOPCON_CATALOG` (server):

| Source | Spec | Notes |
|--------|------|-------|
| Synthetic (default) | `synthetic` or omit | 243 invented products, 9 categories, seeded + deterministic, offline |
| JSON file | `path/to/catalog.json` | same schema `save_catalog` writes — drop in real data |
| JSON URL | `https://...` | any endpoint returning the catalog schema |
| FakeStoreAPI (live) | `fakestore` | real REST commerce API, no key — 20 live products |

```bash
$ shopcon "external hard drive under 100" --catalog fakestore
Catalog: fakestore (20 products)
Picks (1):
| 1 | **WD 2TB Elements Portable External Hard Drive - USB 3.0** | $64.00 | ... |
**Summary:** The best pick is the WD 2TB Elements ... fits all your stated requirements: external, hard drive, electronics, and under $100 ...
```

When nothing genuinely matches, the agent **says so instead of padding the list** — retrieval records what it relaxed in the trace (`relaxed constraints: keywords`), and the LLM verdict names the gap (e.g. "no Bluetooth speaker under $100 exists in this catalog").

### Custom source

Any class with `name` + `load() -> list[Product]` works; register it in `catalog.resolve_source`. Example (Best Buy API):

```python
# sources.py
class BestBuySource:
    name = "bestbuy"

    def load(self) -> list[Product]:
        resp = httpx.get("https://api.bestbuy.com/v1/products", params={...})
        return [
            Product(id=p["sku"], name=p["name"], brand="", category="electronics",
                    price=float(p["salePrice"]), specs={"description": p.get("shortDescription", "")})
            for p in resp.json()["products"]
        ]
```

The pipeline, CLI, and server never change — they only ever see `list[Product]`.

## Roadmap

- [x] Understand → Retrieve → Rank pipeline with trace
- [x] Deterministic mock LLM (keyless runs, testable)
- [x] CLI + FastAPI playground
- [x] Live catalog adapter (FakeStoreAPI keyless, JSON file/URL, custom sources pluggable)
- [ ] Evaluation harness: held-out queries scored by a judge LLM
- [ ] Human-in-the-loop: confirm before "purchasing" (see sibling project ideas)
- [ ] Price-alert agent loop on top of the ranker

## Project layout

```
src/shopcon/
  catalog.py     Product model + source registry (LLM-agnostic)
  sources.py     Pluggable catalog sources: synthetic, JSON file/URL, FakeStoreAPI
  llm.py         OpenAI-compatible client + deterministic MockLLM
  retrieval.py   constraint parsing + deterministic catalog scoring (LLM-agnostic)
  pipeline.py    understand -> retrieve -> rank (+ trace)
  cli.py         terminal UI
  server.py      FastAPI app
tests/           pytest suite (mock LLM, no network)
```

## License

MIT
