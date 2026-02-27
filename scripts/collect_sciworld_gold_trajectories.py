#!/usr/bin/env python3
"""
Collect ScienceWorld "gold path" trajectories via AgentGym SciWorld HTTP server.

Prerequisites:
- Start the AgentGym SciWorld server (in the `agentenv-sciworld` conda env):
    sciworld --host 0.0.0.0 --port 36004

This script will:
- POST /create once to get an env id
- For each data_idx (task variation index), POST /reset with generate_gold_path=true
- GET /gold_action_sequence to fetch the gold actions
- Replay each action with POST /step and record observations/scores
- After reset and after each step, GET /action_hint to save possible_actions/possible_objects
- Save one JSON object per line (JSONL)

Note:
- ScienceWorld's gold action sequence is not guaranteed to be optimal.
- Gold path generation may fail for some tasks; we record the error.
- Use --workers N to run N tasks in parallel (each worker uses a separate env).
"""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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


def load_task_ids(task_file: str, start: int = 0, end: Optional[int] = None, max_tasks: Optional[int] = None) -> List[int]:
    # Support multiple formats:
    # 1) txt: one integer per line
    # 2) json: list of {"item_id": "sciworld_123"} or list of ints
    # 3) jsonl: one json per line, containing data_idx/int-like fields
    with open(task_file, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    ids: List[int] = []
    if not raw:
        return ids

    # JSON / JSONL
    if raw[0] in ("[", "{"):
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                for it in data:
                    if isinstance(it, int):
                        ids.append(int(it))
                        continue
                    if isinstance(it, str) and it.isdigit():
                        ids.append(int(it))
                        continue
                    if isinstance(it, dict):
                        item_id = str(it.get("item_id", it.get("data_idx", it.get("task_id", ""))))
                        # Accept "sciworld_606" or "606"
                        if item_id.startswith("sciworld_"):
                            item_id = item_id.split("_", 1)[1]
                        if str(item_id).isdigit():
                            ids.append(int(item_id))
                            continue
                # done
            elif isinstance(data, dict):
                # single object
                v = data.get("data_idx", data.get("task_id", data.get("item_id", "")))
                item_id = str(v)
                if item_id.startswith("sciworld_"):
                    item_id = item_id.split("_", 1)[1]
                if item_id.isdigit():
                    ids.append(int(item_id))
        except json.JSONDecodeError:
            # maybe JSONL
            for line in raw.splitlines():
                s = line.strip()
                if not s:
                    continue
                obj = json.loads(s)
                v = obj.get("data_idx", obj.get("task_id", obj.get("item_id", "")))
                item_id = str(v)
                if item_id.startswith("sciworld_"):
                    item_id = item_id.split("_", 1)[1]
                if item_id.isdigit():
                    ids.append(int(item_id))
    else:
        # Plain text: one id per line
        for line in raw.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ids.append(int(s))

    if end is None:
        end = len(ids)
    ids = ids[start:end]
    if max_tasks is not None and max_tasks > 0:
        ids = ids[:max_tasks]
    return ids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect ScienceWorld gold path trajectories (JSONL)")
    p.add_argument("--server_url", type=str, default=os.environ.get("SCIWORLD_SERVER_URL", "http://127.0.0.1:36004"))
    p.add_argument("--task_file", type=str, required=True, help="Path to data_idx list file (one int per line)")
    p.add_argument("--output", type=str, required=True, help="Output JSONL path")
    p.add_argument("--task_start", type=int, default=0)
    p.add_argument("--task_end", type=int, default=None)
    p.add_argument("--max_tasks", type=int, default=None)
    p.add_argument("--simplification_str", type=str, default="easy", help='e.g. "easy" or "teleportAction,openDoors"')
    p.add_argument("--max_steps", type=int, default=200, help="Max replay steps (cap gold action length)")
    p.add_argument("--sleep_sec", type=float, default=0.0, help="Optional sleep between steps (debug/throttle)")
    p.add_argument("--resume", action="store_true", help="Resume: skip data_idx already present in output")
    p.add_argument(
        "--resume_policy",
        type=str,
        default="no_error",
        choices=["any", "no_error", "success"],
        help="When --resume is enabled, which records count as completed: "
        "'any' (skip if any record exists), "
        "'no_error' (skip only records without error), "
        "'success' (skip only records with success=true). Default: no_error.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (each uses a separate env). Default: 1 (sequential).",
    )
    return p.parse_args()


def _collect_one_task(
    server_url: str,
    env_id: int,
    data_idx: int,
    args: argparse.Namespace,
) -> Tuple[int, Dict[str, Any]]:
    """Collect gold trajectory for one data_idx. Returns (data_idx, record)."""
    record: Dict[str, Any] = {
        "env": "sciworld",
        "data_idx": data_idx,
        "collected_at": datetime.now().isoformat(),
        "server_url": server_url,
        "simplification_str": args.simplification_str,
    }
    try:
        reset = _post(
            server_url,
            "/reset",
            {
                "id": env_id,
                "data_idx": int(data_idx),
                "generate_gold_path": True,
                "simplification_str": args.simplification_str,
            },
        )
        if "error" in reset:
            raise RuntimeError(reset["error"])

        record.update(
            {
                "task_name": reset.get("task_name", ""),
                "var_num": reset.get("var_num", None),
                "task_description": reset.get("task_description", ""),
                "initial_observation": reset.get("observation", ""),
            }
        )

        try:
            init_hints = _get(server_url, "/action_hint", {"id": env_id})
            if "error" not in init_hints:
                record["init_hints"] = {
                    "possible_actions": init_hints.get("possible_actions", []) or [],
                    "possible_objects": init_hints.get("possible_objects", []) or [],
                }
        except Exception:
            pass

        gold_resp = _get(server_url, "/gold_action_sequence", {"id": env_id})
        if "error" in gold_resp:
            raise RuntimeError(gold_resp["error"])
        gold_actions = gold_resp.get("gold_action_sequence", [])
        record["gold_action_sequence"] = gold_actions

        steps: List[Dict[str, Any]] = []
        final_score = None
        done = False
        for i, action in enumerate(gold_actions[: max(0, int(args.max_steps))]):
            step = _post(server_url, "/step", {"id": env_id, "action": action})
            if "error" in step:
                raise RuntimeError(step["error"])

            hints: Dict[str, Any] = {}
            try:
                hints_resp = _get(server_url, "/action_hint", {"id": env_id})
                if "error" not in hints_resp:
                    hints = {
                        "possible_actions": hints_resp.get("possible_actions", []) or [],
                        "possible_objects": hints_resp.get("possible_objects", []) or [],
                    }
            except Exception:
                pass

            step_record: Dict[str, Any] = {
                "t": i,
                "action": action,
                "observation": step.get("observation", ""),
                "reward": step.get("reward", None),
                "score": step.get("score", None),
                "done": step.get("done", None),
            }
            if hints:
                step_record["possible_actions"] = hints["possible_actions"]
                step_record["possible_objects"] = hints["possible_objects"]
            steps.append(step_record)

            final_score = step.get("score", final_score)
            done = bool(step.get("done", False))
            if args.sleep_sec and args.sleep_sec > 0:
                time.sleep(args.sleep_sec)
            if done:
                break

        record["steps"] = steps
        record["final_score"] = final_score
        record["done"] = done
        record["success"] = bool(final_score == 100) if final_score is not None else False

    except Exception as e:
        record["error"] = str(e)

    return (data_idx, record)


def _worker_process_tasks(
    server_url: str,
    env_id: int,
    task_ids: List[int],
    args: argparse.Namespace,
    output_path: str,
    write_lock: threading.Lock,
    stats: Dict[str, Any],
) -> None:
    """Worker: process assigned task_ids on env_id, write results with lock."""
    for data_idx in task_ids:
        _, record = _collect_one_task(server_url, env_id, data_idx, args)
        ok = "error" not in record
        with write_lock:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["n_ok"] += 1 if ok else 0
            stats["n_err"] += 0 if ok else 1
            print(f"[{stats['n_ok']}/{stats['n_err']}] data_idx={data_idx} ok={ok}")


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    task_ids = load_task_ids(args.task_file, args.task_start, args.task_end, args.max_tasks)
    if not task_ids:
        print("No tasks to process.")
        return

    completed = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as rf:
            for line in rf:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if "data_idx" not in obj:
                    continue
                data_idx = int(obj["data_idx"])
                if args.resume_policy == "any":
                    completed.add(data_idx)
                elif args.resume_policy == "success":
                    if obj.get("success") is True:
                        completed.add(data_idx)
                else:  # no_error
                    if "error" not in obj:
                        completed.add(data_idx)
        if completed:
            before = len(task_ids)
            task_ids = [i for i in task_ids if int(i) not in completed]
            print(
                f"Resume enabled (policy={args.resume_policy}): "
                f"skip {before - len(task_ids)} completed, remaining {len(task_ids)}"
            )

    n_workers = max(1, int(args.workers))
    if n_workers > 1:
        # Create N envs for parallel workers
        env_ids: List[int] = []
        for _ in range(n_workers):
            created = _post(args.server_url, "/create", {})
            if "error" in created:
                raise RuntimeError(f"Failed to create SciWorld env: {created['error']}")
            env_ids.append(int(created["id"]))
        print(f"Created {n_workers} envs (ids={env_ids}) at {args.server_url}")

        # Split tasks among workers
        chunks: List[List[int]] = [[] for _ in range(n_workers)]
        for i, tid in enumerate(task_ids):
            chunks[i % n_workers].append(tid)

        mode = "a" if (args.resume and os.path.exists(args.output)) else "w"
        if mode == "w":
            open(args.output, "w").close()

        write_lock = threading.Lock()
        stats: Dict[str, Any] = {"n_ok": 0, "n_err": 0}

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [
                ex.submit(
                    _worker_process_tasks,
                    args.server_url,
                    env_ids[w],
                    chunks[w],
                    args,
                    args.output,
                    write_lock,
                    stats,
                )
                for w in range(n_workers)
            ]
            for fut in as_completed(futures):
                fut.result()

        for eid in env_ids:
            try:
                _post(args.server_url, "/close", {"id": eid})
            except Exception:
                pass
        print(f"Done. ok={stats['n_ok']}, err={stats['n_err']}, output={args.output}")

    else:
        # Sequential (original behavior)
        created = _post(args.server_url, "/create", {})
        if "error" in created:
            raise RuntimeError(f"Failed to create SciWorld env: {created['error']}")
        env_id = int(created["id"])
        print(f"Created remote SciWorld env id={env_id} at {args.server_url}")

        n_ok = 0
        n_err = 0
        mode = "a" if (args.resume and os.path.exists(args.output)) else "w"
        with open(args.output, mode, encoding="utf-8") as f:
            for data_idx in task_ids:
                _, record = _collect_one_task(args.server_url, env_id, data_idx, args)
                ok = "error" not in record
                n_ok += 1 if ok else 0
                n_err += 0 if ok else 1
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{n_ok}/{n_err}] data_idx={data_idx} ok={ok}")

        try:
            _post(args.server_url, "/close", {"id": env_id})
        except Exception:
            pass
        print(f"Done. ok={n_ok}, err={n_err}, output={args.output}")


if __name__ == "__main__":
    main()

