#!/usr/bin/env python3
"""BM25 retrieval service over the wiki-18 corpus (DeepSearch environment).

Runs inside the `agentenv-webshop` conda env (pyserini + JDK11 already there).
Deterministic by construction: fixed Lucene index + BM25 + stable tie-break on
docid, so identical queries always return identical passages — this is what
keeps the DeepSearch environment fully replayable (teacher entry / analysis).

Launch (see start_env_deepsearch.sh):
    python env_service/launch_script/retrieval_server.py \
        --index /projects_vol/gp_wangwy/qisheng/duet_h200/deepsearch/bm25_wiki18 \
        --port 25011
"""
import argparse
import re
from typing import List

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn
except ImportError as e:  # pragma: no cover
    raise SystemExit(f"fastapi/uvicorn required in this env: {e}")

from pyserini.search.lucene import LuceneSearcher

app = FastAPI(title="wiki18-bm25")
SEARCHER: LuceneSearcher = None  # set in main()


class SearchRequest(BaseModel):
    query: str
    k: int = 3


def _split_title(raw_contents: str):
    # corpus format: "\"Title\"\nbody text"
    m = re.match(r'^"(.*?)"\n(.*)$', raw_contents, flags=re.S)
    if m:
        return m.group(1), m.group(2)
    return "", raw_contents


@app.get("/health")
def health():
    return {"status": "ok", "num_docs": SEARCHER.num_docs}


@app.post("/search")
def search(req: SearchRequest):
    query = (req.query or "").strip()
    if not query:
        return {"query": query, "results": []}
    k = max(1, min(int(req.k), 10))
    hits = SEARCHER.search(query, k=k)
    results: List[dict] = []
    for h in hits:
        doc = SEARCHER.doc(h.docid)
        import json as _json
        contents = _json.loads(doc.raw()).get("contents", "")
        title, text = _split_title(contents)
        results.append(
            {
                "docid": h.docid,
                "score": round(float(h.score), 4),
                "title": title,
                "text": text.strip(),
            }
        )
    return {"query": query, "results": results}


def main():
    global SEARCHER
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--port", type=int, default=25011)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    SEARCHER = LuceneSearcher(args.index)
    print(f"loaded index: {args.index}  docs={SEARCHER.num_docs}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
