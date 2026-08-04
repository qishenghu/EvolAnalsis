#!/usr/bin/env python3
"""Verify WebShop teacher trajectories by replaying them in the live env.

For each sampled trajectory:
  1. Extract the action sequence from <action>...</action> tags in assistant messages
  2. Create a fresh env instance for that task_id
  3. Step through the actions
  4. Compare env-returned final reward with stored reward (should be 1.0)
  5. Release instance
"""
from __future__ import annotations

import argparse
import pickle
import random
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentevolver.client.env_client import EnvClient


ACTION_RE = re.compile(r"<action>\s*(.+?)\s*</action>", re.DOTALL)


def extract_actions(messages: list[dict]) -> list[str]:
    """Pull all <action>...</action> contents from assistant messages."""
    actions = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        content = m.get("content", "")
        for match in ACTION_RE.finditer(content):
            actions.append(match.group(1).strip())
    return actions


def replay_one(client: EnvClient, traj: dict, env_url: str) -> dict:
    """Returns dict with: task_id, stored_reward, env_reward, n_actions, status, error."""
    task_id = traj["task_id"]
    stored_reward = float(traj["reward"])
    actions = extract_actions(traj["messages"])
    n_actions = len(actions)
    out = {
        "task_id": task_id,
        "stored_reward": stored_reward,
        "env_reward": None,
        "n_actions": n_actions,
        "status": "pending",
        "error": None,
    }
    if n_actions == 0:
        out["status"] = "no_actions"
        out["error"] = "no <action> blocks in trajectory"
        return out

    instance_id = f"verify_{task_id}_{int(time.time()*1000)}"
    create_resp = None
    try:
        create_resp = client.create_instance(
            env_type="webshop",
            task_id=task_id,
            instance_id=instance_id,
            params={
                "action_format": "react_tags",
                "enable_action_sanitizer": True,
                "invalid_action_penalty": -0.05,
                "invalid_action_penalty_cap": -0.1,
                "max_consecutive_invalid_actions": 2,
                "terminate_on_invalid_action": False,
                "invalid_action_final_reward": -0.1,
            },
        )
        # create_instance returns the "data" field directly; instance_id at top level
        returned_iid = create_resp.get("instance_id", "")
        if not returned_iid:
            out["status"] = "create_failed"
            out["error"] = f"no instance_id in resp keys={list(create_resp.keys())[:8]}"
            return out

        last_reward = 0.0
        terminated = False
        step_idx = 0
        for action_str in actions:
            # webshop_env.step expects {"content": "<action>...</action>"}
            step_resp = client.step(
                instance_id=instance_id,
                action={"content": f"<action>{action_str}</action>"},
            )
            last_reward = float(step_resp.get("reward", last_reward))
            terminated = bool(step_resp.get("is_terminated", False))
            step_idx += 1
            if terminated:
                break

        out["env_reward"] = last_reward
        if abs(last_reward - stored_reward) < 1e-3:
            out["status"] = "MATCH"
        else:
            out["status"] = "MISMATCH"
            out["error"] = f"env={last_reward} vs stored={stored_reward}, terminated={terminated}, steps_used={step_idx}/{n_actions}"
    except Exception as e:
        out["status"] = "EXCEPTION"
        out["error"] = f"{type(e).__name__}: {e}"
    finally:
        try:
            client.release_instance(instance_id)
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="data/teacher_trajectories/qwen72b/webshop_qwen72b_filtered.pkl")
    ap.add_argument("--env_url", default="http://127.0.0.1:8083")
    ap.add_argument("--n_sample", type=int, default=30, help="number of random trajectories to verify")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--unique_tasks", action="store_true", help="sample unique task_ids only")
    args = ap.parse_args()

    print(f"Loading {args.pkl}...")
    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    print(f"  loaded {len(data)} trajectories")

    random.seed(args.seed)
    if args.unique_tasks:
        # Group by task_id and sample one per
        by_task = {}
        for d in data:
            by_task.setdefault(d["task_id"], []).append(d)
        chosen_tasks = random.sample(sorted(by_task.keys()), args.n_sample)
        sampled = [random.choice(by_task[t]) for t in chosen_tasks]
    else:
        sampled = random.sample(data, args.n_sample)
    print(f"  sampled {len(sampled)} trajectories")

    client = EnvClient(base_url=args.env_url)

    results = []
    print(f"\nReplaying against {args.env_url}...")
    print(f"{'#':>3} {'task_id':>8} {'#act':>5} {'stored':>7} {'env':>7} {'status':>10}  details")
    for i, traj in enumerate(sampled, 1):
        r = replay_one(client, traj, args.env_url)
        results.append(r)
        env_reward_str = f"{r['env_reward']:.3f}" if r["env_reward"] is not None else "—"
        details = (r["error"] or "")[:80]
        print(f"{i:>3} {r['task_id']:>8} {r['n_actions']:>5} {r['stored_reward']:>7.3f} {env_reward_str:>7} {r['status']:>10}  {details}")

    # Summary
    n = len(results)
    n_match = sum(1 for r in results if r["status"] == "MATCH")
    n_mismatch = sum(1 for r in results if r["status"] == "MISMATCH")
    n_exception = sum(1 for r in results if r["status"] == "EXCEPTION")
    n_no_actions = sum(1 for r in results if r["status"] == "no_actions")

    print(f"\n=== SUMMARY ===")
    print(f"  total replayed:  {n}")
    print(f"  MATCH (env=stored=1.0):  {n_match}  ({n_match/n*100:.1f}%)")
    print(f"  MISMATCH:        {n_mismatch}")
    print(f"  EXCEPTION:       {n_exception}")
    print(f"  no_actions:      {n_no_actions}")

    # Show all mismatch details
    if n_mismatch > 0:
        print(f"\n--- mismatch details ---")
        for r in results:
            if r["status"] == "MISMATCH":
                print(f"  task {r['task_id']}: {r['error']}")
    if n_exception > 0:
        print(f"\n--- exception details ---")
        for r in results:
            if r["status"] == "EXCEPTION":
                print(f"  task {r['task_id']}: {r['error']}")

    return 0 if (n_mismatch + n_exception + n_no_actions) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
