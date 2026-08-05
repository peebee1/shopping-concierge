"""Minimal FastAPI server: POST /recommend (JSON) + a small HTML playground at /.

Catalog source is configurable via the SHOPCON_CATALOG env var
(synthetic | fakestore | path/to.json | https://...). Defaults to the
bundled data/catalog.json (auto-generated on first run).
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .catalog import resolve_source
from .llm import MockLLM, OpenAICompatLLM
from .pipeline import recommend
from .sources import CatalogError, SyntheticSource

app = FastAPI(title="Shopping Concierge", version="0.1.0")

try:
    _source = resolve_source(os.environ.get("SHOPCON_CATALOG"))
    _products = _source.load()
except CatalogError:
    _source = SyntheticSource()
    _products = _source.load()
try:
    _llm = OpenAICompatLLM()
except Exception:
    _llm = MockLLM()


class RecommendRequest(BaseModel):
    query: str
    top: int = 5
    verify: int = 3


@app.post("/recommend")
def recommend_endpoint(req: RecommendRequest) -> dict:
    result = recommend(req.query, _products, _llm, top_n=req.top, source=_source, verify_n=req.verify)
    return result.to_dict()


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Shopping Concierge</title>
<style>
 body{font-family:system-ui;max-width:860px;margin:40px auto;padding:0 16px;background:#0f1115;color:#e6e6e6}
 input{width:70%;padding:10px;font-size:15px;border-radius:8px;border:1px solid #333;background:#1a1d24;color:#fff}
 button{padding:10px 18px;border-radius:8px;border:0;background:#4f8cff;color:#fff;font-size:15px;cursor:pointer}
 table{border-collapse:collapse;width:100%;margin-top:16px;font-size:14px}
 th,td{border:1px solid #2a2e38;padding:8px;text-align:left;vertical-align:top}
 th{background:#1a1d24}
 .trace{color:#8b93a7;font-size:13px;margin-top:20px;white-space:pre-wrap}
 .summary{margin-top:16px;padding:12px;background:#12241c;border-radius:8px;border:1px solid #1f4a36}
</style></head><body>
<h2>🛍️ Shopping Concierge</h2>
<p>Ask in plain English. The agent extracts constraints, retrieves candidates, and ranks them with reasons.</p>
<form id="f"><input id="q" placeholder='e.g. "wireless noise-cancelling headphones under $150"' autofocus>
<button type="submit">Recommend</button></form>
<div id="out"></div>
<script>
const f=document.getElementById('f'),q=document.getElementById('q'),out=document.getElementById('out');
f.onsubmit=async e=>{e.preventDefault();
 out.innerHTML='<p>thinking…</p>';
 const r=await fetch('/recommend',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q.value})});
 const d=await r.json();
 let rows=(d.ranked||[]).map(x=>`<tr><td>${x.rank}</td><td><b>${x.name}</b><br><small>${x.brand} · ${x.category}</small></td><td>$${x.price}</td><td>${x.rating}</td><td>${x.confidence_label||''}</td><td>${x.rationale}</td></tr>`).join('');
 let vf='';
 const icons={verified:'✓',changed:'⚠',unavailable:'✗',unverifiable:'–'};
 if(d.verifications&&Object.keys(d.verifications).length){
  vf='<div class="trace">Verification (live re-check):<br>'+Object.values(d.verifications).map(v=>`${icons[v.status]||'·'} ${v.product_id} — ${v.status==='changed'?`price changed: $${v.price_before} → $${v.price_after}`:v.note}`).join('<br>')+'</div>';
 }
 out.innerHTML=`<div class="summary"><b>${d.summary}</b></div>${vf}<table><tr><th>#</th><th>Product</th><th>Price</th><th>Rating</th><th>Conf</th><th>Why</th></tr>${rows}</table><div class="trace">${(d.trace||[]).map((t,i)=>`${i+1}. ${t}`).join('\\n')}</div>`;
};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE
