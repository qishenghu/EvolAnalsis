#!/usr/bin/env python3
"""
Synthesize teacher trajectories for WebShop from verified gold action sequences.

Stage 1:
    scripts/collect_webshop_gold_trajectories.py

Stage 2 (this script):
    - Read verified gold action trajectories
    - Use a teacher LLM to synthesize Thought text for each gold action
    - Export training-ready JSONL + PKL files

The output message format is aligned with `env_service/environments/webshop/webshop_env.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env_service.environments.webshop.webshop_env import WebshopEnv


WEBSHOP_SYSTEM_PROMPT = WebshopEnv(task_id="0", instance_id="dummy")._get_system_prompt()

SYNTH_SYSTEM_PROMPT = """You will be given a WebShop shopping instruction, previous interactions, the current observation, and the next action that will be taken.

Your job: write ONLY the Thought that causally justifies taking that action right now.
The Thought should read like online decision making by an agent browsing a shopping website.

Output format:
- Output ONLY the Thought text.
- Do NOT include "Thought:" or "Action:".
- 1-3 sentences total.

Hard constraints:
1) Evidence binding: mention 1-2 concrete details from the current observation, such as a visible product title, an option value, the presence of "Buy Now", or the current clickable elements.
2) Action grounding: explain why this exact next action is appropriate now, not a generic shopping plan.
3) Minimal planning: include one short expectation of what this action should reveal or achieve next.
4) Do NOT mention that the next action was given to you.
5) Avoid boilerplate like "I should take the next step" unless tied to specific evidence.
6) Do NOT claim you can already see information that is not present in the current observation. In particular, do not mention search results, option values, or page elements from future pages.

Examples:
- Good: The search results already show the target pillow listing "Ambesonne Abstract Throw Pillow Cushion Cover" and the matching product id is clickable. Opening that item page should let me select the required size before purchasing.
- Good: On the product page, the required option "3x-large" is explicitly listed among the clickable size choices, while "Buy Now" is already available. Selecting the correct size first will align the purchase with the instruction before I finalize it.
- Bad: I need to continue shopping, so I will click the right thing.
"""


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
    a_err = "error" in a
    b_err = "error" in b
    if a_err != b_err:
        return b if a_err else a

    a_success = bool(a.get("success"))
    b_success = bool(b.get("success"))
    if a_success != b_success:
        return b if b_success else a

    a_instr = bool(a.get("instruction_match"))
    b_instr = bool(b.get("instruction_match"))
    if a_instr != b_instr:
        return b if b_instr else a

    a_reward = float(a.get("final_reward", 0.0) or 0.0)
    b_reward = float(b.get("final_reward", 0.0) or 0.0)
    if a_reward != b_reward:
        return a if a_reward > b_reward else b

    a_steps = len(a.get("steps", []) or [])
    b_steps = len(b.get("steps", []) or [])
    if a_steps != b_steps:
        return a if a_steps < b_steps else b

    return b


def load_gold_records(inputs: List[str]) -> List[Dict[str, Any]]:
    by_rollout: Dict[str, Dict[str, Any]] = {}
    for rec in _iter_jsonl(inputs):
        if rec.get("record_kind", "rollout") != "rollout":
            continue
        rollout_id = rec.get("rollout_id")
        if rollout_id is None:
            continue
        key = str(rollout_id)
        by_rollout[key] = rec if key not in by_rollout else _best_record(by_rollout[key], rec)
    return [
        by_rollout[k]
        for k in sorted(by_rollout, key=lambda x: (int(str(x).split("_")[1]), x))
    ]


def load_completed(output_path: str, resume_policy: str) -> set[str]:
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
            rollout_id = obj.get("rollout_id")
            if rollout_id is None:
                continue
            rollout_id = str(rollout_id)
            if resume_policy == "any":
                done.add(rollout_id)
            elif resume_policy == "success":
                if obj.get("success") is True:
                    done.add(rollout_id)
            else:
                if "error" not in obj:
                    done.add(rollout_id)
    return done


def _strip_thought_markers(text: str) -> str:
    t = (text or "").strip()
    m = re.search(r"Thought:\s*(.*?)(\n\s*\n\s*Action:|\Z)", t, flags=re.I | re.S)
    if m:
        return m.group(1).strip()
    if t.lower().startswith("thought:"):
        return t.split(":", 1)[-1].strip()
    m2 = re.search(r"Action:", t, flags=re.I)
    if m2:
        return t[: m2.start()].strip()
    return t


def _build_synth_context(
    instruction: str,
    initial_observation: str,
    steps: List[Dict[str, Any]],
    actions: List[str],
    thoughts_so_far: List[str],
    current_step_idx: int,
    next_action: str,
    history_steps: int = 8,
) -> str:
    t = current_step_idx
    start = max(0, t - history_steps)

    lines: List[str] = []
    lines.append(f"Instruction:\n{instruction}")
    lines.append("\nInteraction history:")

    if start == 0:
        lines.append(f"Obs#0:\n{initial_observation}")
    else:
        obs_prev = str(steps[start - 1].get("observation", ""))
        lines.append(f"Obs#{start}:\n{obs_prev}")

    for j in range(start, t):
        thought = (thoughts_so_far[j] or "").strip() if j < len(thoughts_so_far) else ""
        action = str(actions[j]).strip() if j < len(actions) else ""
        obs_after = str(steps[j].get("observation", "")) if j < len(steps) else ""
        if thought:
            lines.append(f"Thought#{j}: {thought}")
        lines.append(f"Action#{j}: {action}")
        lines.append(f"Obs#{j + 1}: {obs_after}")

    current_observation = initial_observation if t == 0 else str(steps[t - 1].get("observation", ""))
    if f"Obs#{t}:" not in "\n".join(lines[-2:]):
        lines.append(f"Obs#{t}: {current_observation}")

    lines.append("\nCurrent observation you must ground on:")
    lines.append(current_observation)

    lines.append(f"\nNext action:\n{next_action}")
    return "\n\n".join(lines)


def _get_training_task_subset(n_tasks: int, seed: int) -> set[str]:
    import random

    all_ids = WebshopEnv.get_query_list("train")
    rng = random.Random(seed)
    rng.shuffle(all_ids)
    return set(all_ids[:n_tasks])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthesize WebShop teacher trajectories with Thought reasoning")
    p.add_argument(
        "--inputs",
        nargs="+",
        default=["analysis_outputs/webshop_gold_train_verified.jsonl"],
    )
    p.add_argument(
        "--output",
        default="data/teacher_trajectories/webshop_gold_qwen7b_synth.jsonl",
    )
    p.add_argument("--export_base", type=str, default=None)
    p.add_argument("--export_threshold", type=float, default=1.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume_policy", choices=["any", "no_error", "success"], default="no_error")
    p.add_argument("--max_tasks", type=int, default=None)
    p.add_argument("--max_steps_per_task", type=int, default=None)
    p.add_argument("--success_only", action="store_true", default=True)
    p.add_argument("--no_success_only", action="store_false", dest="success_only")
    p.add_argument("--instruction_match_only", action="store_true", default=True)
    p.add_argument("--no_instruction_match_only", action="store_false", dest="instruction_match_only")
    p.add_argument("--model_path", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--collect_log_prob", choices=["true", "false"], default="true")
    p.add_argument("--history_steps", type=int, default=8)
    p.add_argument("--task_subset", type=int, default=None)
    p.add_argument("--task_seed", type=int, default=2026)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    gold = load_gold_records(args.inputs)
    gold = [r for r in gold if "error" not in r]
    if args.success_only:
        gold = [r for r in gold if r.get("success") is True]
    if args.instruction_match_only:
        gold = [r for r in gold if r.get("instruction_match") is True]
    if args.task_subset and args.task_subset > 0:
        subset_ids = _get_training_task_subset(args.task_subset, args.task_seed)
        before = len(gold)
        gold = [r for r in gold if str(r["task_id"]) in subset_ids]
        print(f"Task subset (n={args.task_subset}, seed={args.task_seed}): {before} -> {len(gold)} records")
    if args.resume:
        done = load_completed(args.output, args.resume_policy)
        if done:
            gold = [r for r in gold if str(r["rollout_id"]) not in done]
            print(f"Resume (policy={args.resume_policy}): remaining={len(gold)}")
    if args.max_tasks and args.max_tasks > 0:
        gold = gold[: args.max_tasks]

    total_steps = sum(len(r.get("action_sequence", []) or []) for r in gold)
    print(f"Gold records to process: {len(gold)}, total steps: {total_steps}")

    from agentevolver.module.teacher import create_teacher_llm

    llm = create_teacher_llm(
        {
            "type": "vllm",
            "model_path": args.model_path,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "collect_log_prob": (args.collect_log_prob == "true"),
        }
    )

    mode = "a" if (args.resume and os.path.exists(args.output)) else "w"
    ok = 0
    err = 0
    all_trajectory_dicts: List[Dict[str, Any]] = []

    with open(args.output, mode, encoding="utf-8") as wf:
        for rec in gold:
            task_id = str(rec["task_id"])
            source_rollout_id = str(rec["rollout_id"])
            instruction = str(rec.get("instruction", rec.get("expected_instruction", "")))
            initial_observation = str(rec.get("initial_observation", ""))
            actions = rec.get("action_sequence", []) or []
            steps = rec.get("steps", []) or []
            n = min(len(actions), len(steps))
            if args.max_steps_per_task and args.max_steps_per_task > 0:
                n = min(n, int(args.max_steps_per_task))

            out: Dict[str, Any] = {
                "task_id": task_id,
                "data_id": task_id,
                "rollout_id": f"{source_rollout_id}_synth",
                "teacher_model": getattr(llm, "model_name", args.model_path.split("/")[-1]),
                "metadata": {
                    "is_teacher": True,
                    "is_synthesized": True,
                    "source": "webshop_gold",
                    "source_files": args.inputs,
                    "collected_at": datetime.now().isoformat(),
                    "instruction_match": bool(rec.get("instruction_match", False)),
                    "initial_gold_success": bool(rec.get("success", False)),
                    "trajectory_type": rec.get("trajectory_type"),
                    "num_search_actions": int(rec.get("num_search_actions", 0) or 0),
                    "search_policy": rec.get("search_policy"),
                    "source_rollout_id": source_rollout_id,
                },
            }

            try:
                messages: List[Dict[str, str]] = [
                    {"role": "system", "content": WEBSHOP_SYSTEM_PROMPT},
                    {
                        "role": "assistant",
                        "content": "OK. I'll help you find and purchase the item according to your requirements.",
                    },
                    {"role": "user", "content": initial_observation},
                ]

                acc_lp: List[float] = []
                turn_lp: List[Dict[str, Any]] = []
                has_lp = True
                thoughts_so_far: List[str] = []

                for i in range(n):
                    action = str(actions[i])
                    synth_user = _build_synth_context(
                        instruction=instruction,
                        initial_observation=initial_observation,
                        steps=steps,
                        actions=actions,
                        thoughts_so_far=thoughts_so_far,
                        current_step_idx=i,
                        next_action=action,
                        history_steps=int(args.history_steps),
                    )
                    synth_msgs = [
                        {"role": "system", "content": SYNTH_SYSTEM_PROMPT},
                        {"role": "user", "content": synth_user},
                    ]
                    resp, meta = llm(synth_msgs)

                    thought = _strip_thought_markers(resp or "")
                    if not thought:
                        thought = "The current page shows the relevant evidence for this action, and taking it should move me closer to completing the purchase correctly."

                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"Thought:\n{thought}\n\nAction:\n{action}",
                        }
                    )
                    thoughts_so_far.append(thought)

                    obs_after = str(steps[i].get("observation", ""))
                    messages.append({"role": "user", "content": obs_after})

                    if meta and "log_probs" in meta and has_lp:
                        turn_lp.append(
                            {
                                "turn_idx": i,
                                "log_probs": meta.get("log_probs", []),
                                "token_ids": meta.get("token_ids", []),
                                "tokens": meta.get("tokens", []),
                            }
                        )
                        acc_lp.extend(meta.get("log_probs", []))

                final_reward = float(rec.get("final_reward", 0.0) or 0.0)
                done = bool(rec.get("done", False))
                reward = final_reward

                out.update(
                    {
                        "messages": messages,
                        "reward": reward,
                        "success": reward > 0,
                        "is_terminated": done,
                    }
                )
                out["metadata"]["has_log_prob"] = bool(has_lp and len(acc_lp) > 0)
                out["metadata"]["num_turns"] = n
                out["metadata"]["total_generated_tokens"] = len(acc_lp) if has_lp else 0
                out["metadata"]["reward_source"] = "verified_webshop_replay"
                if out["metadata"]["has_log_prob"]:
                    out["metadata"]["log_prob_scope"] = "thought_only"
                    out["log_probs"] = acc_lp
                    out["log_probs_per_turn"] = turn_lp
                    out["thought_log_probs"] = acc_lp
                    out["thought_log_probs_per_turn"] = turn_lp

                wf.write(json.dumps(out, ensure_ascii=False) + "\n")
                wf.flush()
                all_trajectory_dicts.append(out)
                ok += 1
                print(
                    f"[ok={ok} err={err}] task_id={task_id} rollout_id={source_rollout_id} "
                    f"turns={n} reward={reward}"
                )

            except Exception as e:
                out["error"] = str(e)
                wf.write(json.dumps(out, ensure_ascii=False) + "\n")
                wf.flush()
                err += 1
                print(
                    f"[ok={ok} err={err}] task_id={task_id} rollout_id={source_rollout_id} ERROR: {e}",
                    file=sys.stderr,
                )

    pkl_path = os.path.splitext(args.output)[0] + ".pkl"
    if args.resume and os.path.exists(pkl_path):
        existing_dicts = pickle.load(open(pkl_path, "rb"))
        all_trajectory_dicts = existing_dicts + all_trajectory_dicts
    with open(pkl_path, "wb") as pf:
        pickle.dump(all_trajectory_dicts, pf)
    print(f"\nDone. ok={ok}, err={err}, output={args.output}")
    print(f"PKL saved: {pkl_path} ({len(all_trajectory_dicts)} trajectories)")

    if args.export_base:
        try:
            from scripts.filter_teacher_trajectories import filter_trajectories

            print(f"\nExporting filtered teacher trajectories to {args.export_base} ...")
            filter_trajectories(
                input_path=args.output,
                output_base=args.export_base,
                reward_threshold=float(args.export_threshold),
                verbose=True,
            )
        except Exception as e:
            print(f"WARNING: export failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
