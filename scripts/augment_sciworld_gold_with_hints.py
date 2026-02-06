#!/usr/bin/env python3
"""
Augment SciWorld gold trajectories with per-step action hints (possible_actions/possible_objects).

Why:
- Training prompt in `env_service/environments/sciworld/sciworld_env.py` appends action hints
  to the user message after reset and after each step.
- Previously collected gold trajectories (sciworld_gold*.jsonl) do not include those hints.

How:
- For each data_idx, reset env on AgentGym SciWorld HTTP server.
- Replay the stored gold actions.
- After reset and after each step, call GET /action_hint?id=...
- Save an augmented JSONL record containing:
  - init_hints: possible_actions/possible_objects + formatted hint_str
  - steps_augmented: step info + hints AFTER the action (aligns with training replay)

Usage:
  # Ensure an AgentGym SciWorld server is running (recommended port: 36010)
  # (must include /action_hint; gold endpoint not required for augmentation)
  python scripts/augment_sciworld_gold_with_hints.py \
    --server_url http://127.0.0.1:36010 \
    --inputs data/teacher_trajectories/sciworld_gold.jsonl data/teacher_trajectories/sciworld_gold_retry_404.jsonl \
    --output data/teacher_trajectories/sciworld_gold_augmented.jsonl \
    --resume
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


def _post(base_url: str, path: str, payload: Dict[str, Any], timeout: float = 300.0) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _get(base_url: str, path: str, params: Dict[str, Any], timeout: float = 300.0) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _iter_jsonl(paths: List[str]) -> Iterable[Dict[str, Any]]:
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    yield json.loads(s)
                except Exception:
                    continue


def _best_record(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Pick the better gold record for the same data_idx."""
    a_err = "error" in a
    b_err = "error" in b
    if a_err != b_err:
        return b if a_err else a
    a_score = a.get("final_score")
    b_score = b.get("final_score")
    if isinstance(a_score, (int, float)) and isinstance(b_score, (int, float)) and a_score != b_score:
        return a if a_score > b_score else b
    a_steps = len(a.get("steps", []) or [])
    b_steps = len(b.get("steps", []) or [])
    if a_steps != b_steps:
        return a if a_steps > b_steps else b
    return b


def load_gold_records(inputs: List[str]) -> List[Dict[str, Any]]:
    by_id: Dict[int, Dict[str, Any]] = {}
    for rec in _iter_jsonl(inputs):
        if "data_idx" not in rec:
            continue
        try:
            did = int(rec["data_idx"])
        except Exception:
            continue
        by_id[did] = rec if did not in by_id else _best_record(by_id[did], rec)
    return [by_id[k] for k in sorted(by_id)]


def load_completed_data_idxs(output_path: str) -> set:
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if "data_idx" in obj and "error" not in obj:
                try:
                    done.add(int(obj["data_idx"]))
                except Exception:
                    pass
    return done


def format_action_hints(possible_actions: List[Any], possible_objects: List[Any]) -> str:
    # Mirror `SciworldEnv._get_action_hints` formatting.
    pa = (possible_actions or [])[:10]
    po = (possible_objects or [])[:10]
    hint_str = ""
    if pa:
        hint_str += f"Suggested actions: {pa}\n"
    if po:
        hint_str += f"Nearby objects: {po}"
    return hint_str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augment SciWorld gold trajectories with action hints")
    p.add_argument("--server_url", type=str, default=os.environ.get("SCIWORLD_SERVER_URL", "http://127.0.0.1:36010"))
    p.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/teacher_trajectories/sciworld_gold.jsonl",
            "data/teacher_trajectories/sciworld_gold_retry_404.jsonl",
        ],
        help="Input gold JSONL files",
    )
    p.add_argument("--output", type=str, default="data/teacher_trajectories/sciworld_gold_augmented.jsonl")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max_tasks", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None, help="Cap replay steps per task")
    p.add_argument("--verify_observation", action="store_true", default=True, help="Compare replay obs with saved obs")
    p.add_argument("--no_verify_observation", action="store_false", dest="verify_observation")
    p.add_argument("--verify_score", action="store_true", default=True, help="Compare replay score with saved score")
    p.add_argument("--no_verify_score", action="store_false", dest="verify_score")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    gold_records = load_gold_records(args.inputs)
    # Only augment records that already have gold actions and no error.
    gold_records = [r for r in gold_records if "error" not in r and (r.get("gold_action_sequence") or [])]

    if args.max_tasks is not None and args.max_tasks > 0:
        gold_records = gold_records[: args.max_tasks]

    completed = set()
    if args.resume:
        completed = load_completed_data_idxs(args.output)
        if completed:
            gold_records = [r for r in gold_records if int(r["data_idx"]) not in completed]
            print(f"Resume enabled: remaining={len(gold_records)}")

    created = _post(args.server_url, "/create", {})
    if "error" in created:
        raise RuntimeError(f"Failed to create SciWorld env: {created['error']}")
    env_id = int(created["id"])
    print(f"Created remote SciWorld env id={env_id} at {args.server_url}")

    mode = "a" if (args.resume and os.path.exists(args.output)) else "w"
    ok = 0
    err = 0
    with open(args.output, mode, encoding="utf-8") as wf:
        for rec in gold_records:
            data_idx = int(rec["data_idx"])
            out: Dict[str, Any] = {
                "env": "sciworld",
                "data_idx": data_idx,
                "augmented_at": datetime.now().isoformat(),
                "server_url": args.server_url,
                # Keep important original fields
                "task_name": rec.get("task_name", ""),
                "var_num": rec.get("var_num", None),
                "task_description": rec.get("task_description", ""),
                "initial_observation": rec.get("initial_observation", ""),
                "gold_action_sequence": rec.get("gold_action_sequence", []),
                "final_score": rec.get("final_score", None),
                "done": rec.get("done", None),
                "success": rec.get("success", None),
                "source_record": {
                    "collected_at": rec.get("collected_at"),
                    "source_server_url": rec.get("server_url"),
                    "simplification_str": rec.get("simplification_str", ""),
                },
            }
            try:
                reset = _post(args.server_url, "/reset", {"id": env_id, "data_idx": data_idx})
                if "error" in reset:
                    raise RuntimeError(reset["error"])

                # Initial hints (after reset + the server auto did "look around" in reset handler)
                hints0 = _get(args.server_url, "/action_hint", {"id": env_id})
                if "error" in hints0:
                    raise RuntimeError(hints0["error"])
                pa0 = hints0.get("possible_actions", []) or []
                po0 = hints0.get("possible_objects", []) or []
                out["init_hints"] = {
                    "possible_actions": pa0[:10],
                    "possible_objects": po0[:10],
                    "hint_str": format_action_hints(pa0, po0),
                }

                steps_aug: List[Dict[str, Any]] = []
                mismatches: List[Dict[str, Any]] = []

                gold_steps = rec.get("steps", []) or []
                actions = (rec.get("gold_action_sequence", []) or [])
                max_steps = min(len(actions), int(args.max_steps)) if args.max_steps else len(actions)

                for t in range(max_steps):
                    action = str(actions[t])
                    step = _post(args.server_url, "/step", {"id": env_id, "action": action})
                    if "error" in step:
                        raise RuntimeError(step["error"])

                    # Hints AFTER action (this is what training replay appends in user message)
                    hints = _get(args.server_url, "/action_hint", {"id": env_id})
                    if "error" in hints:
                        raise RuntimeError(hints["error"])
                    pa = hints.get("possible_actions", []) or []
                    po = hints.get("possible_objects", []) or []

                    steps_aug.append(
                        {
                            "t": t,
                            "action": action,
                            "observation": step.get("observation", ""),
                            "reward": step.get("reward", None),
                            "score": step.get("score", None),
                            "done": step.get("done", None),
                            "hints": {
                                "possible_actions": pa[:10],
                                "possible_objects": po[:10],
                                "hint_str": format_action_hints(pa, po),
                            },
                        }
                    )

                    # Verification vs saved record (best-effort; may differ due to minor formatting)
                    if t < len(gold_steps):
                        if args.verify_observation:
                            saved_obs = str(gold_steps[t].get("observation", ""))
                            replay_obs = str(step.get("observation", ""))
                            if saved_obs and replay_obs and saved_obs != replay_obs:
                                mismatches.append({"t": t, "type": "observation", "saved": saved_obs[:2000], "replay": replay_obs[:2000]})
                        if args.verify_score:
                            saved_score = gold_steps[t].get("score", None)
                            replay_score = step.get("score", None)
                            if saved_score is not None and replay_score is not None and saved_score != replay_score:
                                mismatches.append({"t": t, "type": "score", "saved": saved_score, "replay": replay_score})

                    if bool(step.get("done", False)):
                        break

                out["steps_augmented"] = steps_aug
                out["replay_final_score"] = steps_aug[-1]["score"] if steps_aug else None
                out["replay_done"] = bool(steps_aug[-1]["done"]) if steps_aug else False
                out["replay_success"] = bool(out["replay_final_score"] == 100) if out["replay_final_score"] is not None else False
                out["mismatches"] = mismatches
                out["verified"] = len(mismatches) == 0

                wf.write(json.dumps(out, ensure_ascii=False) + "\n")
                wf.flush()
                ok += 1
                print(f"[ok={ok} err={err}] data_idx={data_idx} replay_success={out['replay_success']} mismatches={len(mismatches)}")

            except Exception as e:
                out["error"] = str(e)
                wf.write(json.dumps(out, ensure_ascii=False) + "\n")
                wf.flush()
                err += 1
                print(f"[ok={ok} err={err}] data_idx={data_idx} ERROR: {e}", file=sys.stderr)

    # Best-effort close
    try:
        _post(args.server_url, "/close", {"id": env_id})
    except Exception:
        pass


if __name__ == "__main__":
    main()

