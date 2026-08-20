#!/usr/bin/env python3
"""Acceptance gates for the Qwen3.5-122B teacher-collection pilot.

Modes:
  traces  — validate pilot success records: every sampled decision must carry a
            complete <think>...</think> block AND a valid <action>...</action>
            block; report SR / turns / token stats from the attempt ledger.
  webshop-determinism — create the same WebShop tasks twice via env_service and
            compare instruction text; run once before and once after a stack
            restart with --phase {a,b} to also pin cross-restart determinism.

Exit code 0 = gate passed. Non-zero = do not launch the full campaign.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
ACTION_RE = re.compile(r"<action>\s*(.+?)\s*</action>", re.DOTALL | re.IGNORECASE)


def check_traces(args: argparse.Namespace) -> int:
    path = Path(args.output)
    if not path.exists() or path.stat().st_size == 0:
        print(f"FAIL: no pilot output at {path}")
        return 1
    n_records = 0
    n_decisions = 0
    bad = []
    for line_no, line in enumerate(path.open(), start=1):
        rec = json.loads(line)
        n_records += 1
        trace = rec.get("decision_trace") or []
        if not trace:
            bad.append((line_no, "empty decision_trace"))
            continue
        for d_idx, dec in enumerate(trace):
            n_decisions += 1
            content = str(dec.get("completion_content", ""))
            if not THINK_RE.search(content):
                bad.append((line_no, f"decision {d_idx}: no complete <think>...</think>"))
            m = ACTION_RE.search(content.split("</think>")[-1])
            if not (m and m.group(1).strip()):
                bad.append((line_no, f"decision {d_idx}: no valid post-think <action>"))
        # messages must contain exactly as many action-bearing assistant turns
        # as there are sampled decisions (preamble assistant turns carry none).
        msgs = rec.get("messages") or []
        action_turns = sum(
            1 for m_ in msgs
            if m_.get("role") == "assistant" and ACTION_RE.search(str(m_.get("content", "")))
        )
        if action_turns != len(trace):
            bad.append((line_no, f"messages action-turns {action_turns} != decisions {len(trace)}"))

    # SR from the attempt ledger (successes / distinct attempted slots).
    attempts_path = Path(str(path) + ".attempts.jsonl")
    slots_started: set = set()
    slots_succeeded: set = set()
    if attempts_path.exists():
        for line in attempts_path.open():
            ev = json.loads(line)
            if ev.get("event") == "attempt_started":
                slots_started.add(ev["rollout_id"])
            if ev.get("event") == "attempt_finished" and ev.get("success"):
                slots_succeeded.add(ev["rollout_id"])
    sr = (len(slots_succeeded) / len(slots_started)) if slots_started else 0.0

    print(f"records={n_records} decisions={n_decisions} "
          f"slots_started={len(slots_started)} slots_succeeded={len(slots_succeeded)} "
          f"slot_SR={sr:.3f}")
    for line_no, msg in bad[:20]:
        print(f"  BAD line {line_no}: {msg}")
    if bad:
        print(f"FAIL: {len(bad)} think/action integrity violations")
        return 1
    if n_records == 0:
        print("FAIL: zero successful trajectories in pilot")
        return 1
    if sr < args.min_sr:
        print(f"FAIL: slot SR {sr:.3f} < required {args.min_sr}")
        return 1
    print("PASS: trace integrity + SR gate")
    return 0


def _instruction_fingerprint(client, task_id: str) -> str:
    instance_id = uuid.uuid4().hex
    resp = client.create_instance(
        env_type="webshop",
        task_id=task_id,
        instance_id=instance_id,
        params={"is_open_query": False, "action_format": "react_tags"},
    )
    try:
        state = resp["state"]
        text = json.dumps(state, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()
    finally:
        try:
            client.release_instance(instance_id)
        except Exception:
            pass


def check_webshop_determinism(args: argparse.Namespace) -> int:
    from agentevolver.client.env_client import EnvClient

    client = EnvClient(base_url=args.env_url)
    task_ids = [t for t in args.tasks.split(",") if t]
    store = Path(args.fingerprint_store)
    results = {}
    ok = True
    for tid in task_ids:
        f1 = _instruction_fingerprint(client, tid)
        f2 = _instruction_fingerprint(client, tid)
        same_boot = f1 == f2
        if not same_boot:
            ok = False
        results[tid] = f1
        print(f"task {tid}: within-boot {'OK' if same_boot else 'MISMATCH'} ({f1[:12]})")

    if args.phase == "a":
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text(json.dumps(results, indent=2))
        print(f"phase a fingerprints stored -> {store}")
    else:
        if not store.exists():
            print("FAIL: phase-a fingerprint store missing")
            return 1
        prev = json.loads(store.read_text())
        for tid, fp in results.items():
            if prev.get(tid) != fp:
                ok = False
                print(f"task {tid}: CROSS-RESTART MISMATCH {prev.get(tid, '?')[:12]} != {fp[:12]}")
            else:
                print(f"task {tid}: cross-restart OK")
    if not ok:
        print("FAIL: WebShop instruction determinism violated")
        return 1
    print(f"PASS: WebShop determinism phase {args.phase}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    t = sub.add_parser("traces")
    t.add_argument("--output", required=True)
    t.add_argument("--min-sr", type=float, default=0.0)

    w = sub.add_parser("webshop-determinism")
    w.add_argument("--env-url", default="http://127.0.0.1:8083")
    w.add_argument("--tasks", required=True, help="comma-separated numeric webshop task ids")
    w.add_argument("--phase", choices=["a", "b"], required=True)
    w.add_argument("--fingerprint-store", required=True)

    args = parser.parse_args()
    if args.mode == "traces":
        return check_traces(args)
    return check_webshop_determinism(args)


if __name__ == "__main__":
    raise SystemExit(main())
