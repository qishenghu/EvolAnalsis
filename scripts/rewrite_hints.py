"""CATALYST P-A 试点 A2 臂:把教师 think 摘要用学生自己的腔调重写(HD 汇率检验)。

输入 hints json({task_id:{"raw":...}}),对每条 raw 调学生 vLLM 重写,
写出含 "rewritten" 键的新 json。幂等:已有 rewritten 的任务跳过(--resume)。
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

PROMPT = (
    "Below is a reference approach for solving a task, written by someone else.\n"
    "Rewrite it in your own natural style. Keep ALL key steps, facts, object names "
    "and specifics exactly as they are; do not add new information; do not omit steps; "
    "output only the rewritten approach, nothing else.\n\n"
    "Reference approach:\n{raw}"
)


def rewrite_one(args, task_id, raw):
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": PROMPT.format(raw=raw)}],
        "temperature": 0.3,
        "max_tokens": args.max_tokens,
    }
    for attempt in range(4):
        try:
            r = requests.post(
                f"{args.api_base}/chat/completions",
                json=body,
                headers={"Authorization": "Bearer EMPTY"},
                timeout=600,
            )
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            text = (msg.get("content") or "").strip()
            if text:
                return task_id, text
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {task_id} attempt {attempt}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    return task_id, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hints-in", required=True)
    ap.add_argument("--hints-out", required=True)
    ap.add_argument("--api-base", required=True, help="e.g. http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    hints = json.load(open(a.hints_in))
    outp = Path(a.hints_out)
    if outp.exists():  # resume
        prev = json.load(open(outp))
        for k, v in prev.items():
            if v.get("rewritten"):
                hints.setdefault(k, {}).update(v)
    todo = [(k, v["raw"]) for k, v in hints.items() if not v.get("rewritten")]
    print(f"rewrite: total={len(hints)} todo={len(todo)}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for tid, text in ex.map(lambda kv: rewrite_one(a, *kv), todo):
            if text is None:
                print(f"[fail] {tid}: keeping raw as fallback", flush=True)
                hints[tid]["rewritten"] = hints[tid]["raw"]
                hints[tid]["rewrite_failed"] = True
            else:
                hints[tid]["rewritten"] = text
    json.dump(hints, open(outp, "w"), ensure_ascii=False, indent=0)
    n_ok = sum(1 for v in hints.values() if v.get("rewritten") and not v.get("rewrite_failed"))
    print(f"done: rewritten_ok={n_ok}/{len(hints)} -> {outp}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
