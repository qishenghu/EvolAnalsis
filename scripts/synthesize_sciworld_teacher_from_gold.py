#!/usr/bin/env python3
"""
Synthesize teacher trajectories for ScienceWorld DR3 training.

Reads gold trajectories (from collect_sciworld_gold_trajectories.py),
uses a teacher LLM to generate Thought reasoning for each gold action,
and outputs multi-turn trajectories aligned with the training format
(system prompt, hint format, reward scheme all match sciworld_env.py).

Usage:
    python scripts/synthesize_sciworld_teacher_from_gold.py \
        --model_path Qwen/Qwen2.5-7B-Instruct \
        --inputs data/sciworld/gold_trajectories.jsonl \
        --output data/teacher_trajectories/sciworld_gold_qwen7b_synth.jsonl
"""

import argparse, json, os, pickle, re, sys
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# System prompt – kept in sync with sciworld_env.py._get_system_prompt()
# ---------------------------------------------------------------------------
SCIWORLD_SYSTEM_PROMPT = '''You are a scientific experiment assistant in a text-based simulation environment. Your task is to perform scientific experiments by interacting with objects in the environment.

At each step, you will receive:
1. The task description (what experiment you need to perform)
2. Your current observation (what you can see/do)
3. OBJ candidates (the objects that can be interacted with in the current state).

Available actions:
[
{"action": "open OBJ", "description": "open a container"},
{"action": "close OBJ", "description": "close a container"},
{"action": "activate OBJ", "description": "activate a device"},
{"action": "deactivate OBJ", "description": "deactivate a device"},
{"action": "connect OBJ to OBJ", "description": "connect electrical components"},
{"action": "disconnect OBJ", "description": "disconnect electrical components"},
{"action": "use OBJ [on OBJ]", "description": "use a device/item"},
{"action": "look around", "description": "describe the current room"},
{"action": "look at OBJ", "description": "describe an object in detail"},
{"action": "look in OBJ", "description": "describe a container's contents"},
{"action": "read OBJ", "description": "read a note or book"},
{"action": "move OBJ to OBJ", "description": "move an object to a container"},
{"action": "pick up OBJ", "description": "move an object to the inventory"},
{"action": "put down OBJ", "description": "drop an inventory item"},
{"action": "pour OBJ into OBJ", "description": "pour a liquid into a container"},
{"action": "dunk OBJ into OBJ", "description": "dunk a container into a liquid"},
{"action": "mix OBJ", "description": "chemically mix a container"},
{"action": "go to LOC", "description": "move to a new location"},
{"action": "eat OBJ", "description": "eat a food"},
{"action": "flush OBJ", "description": "flush a toilet"},
{"action": "focus on OBJ", "description": "signal intent on a task object"},
{"action": "wait", "description": "take no action for 10 iterations"},
{"action": "wait1", "description": "take no action for 1 iteration"},
{"action": "task", "description": "describe current task"},
{"action": "inventory", "description": "list your inventory"}
]

Important:
1. Read the task description carefully.
2. Plan your experiment steps logically.
3. Pay attention to the objects and locations available.
4. OBJ in the selected action should be replaced with one of the OBJ candidates using the exact string as provided.
5. If the environment returns "No known action matches that input.", that means your previous action is invalid and you should try more options.

In each turn, you should choose from two answer formats: "THOUGHT" or "ACTION".
- If you choose "THOUGHT", first analyze the task and current state, then output your action.
  Format: "Thought:\\nyour thoughts.\\n\\nAction:\\nyour next action"
- If you choose "ACTION", directly output the action.
  Format: "Action:\\nyour next action"
'''

# ---------------------------------------------------------------------------
# Thought-synthesis prompt (sent to the teacher LLM)
# ---------------------------------------------------------------------------
SYNTH_SYSTEM_PROMPT = """You will be given a ScienceWorld task, previous interactions, the current observation, and the next action that will be taken.

Your job: write ONLY the Thought that *causally justifies* taking that action in the current state.
The Thought should read like online reasoning, not a generic template.

Output format:
- Output ONLY the Thought text (do NOT include "Thought:" and do NOT include "Action:").
- 1-4 sentences total.

Hard constraints:
1) Evidence binding: explicitly cite 1-2 concrete details from the observation/hints (exact object names, room name, or a state like "door is open/closed", "in inventory", etc.).
2) Minimal planning: include one short checkpoint about what you expect to learn/achieve right after this action.
3) Avoid boilerplate and vague filler.
4) Do NOT mention that the next action was given to you.

Bad example (too generic):
I need to find orange juice, so I should go to the kitchen.

Good example (evidence-bound + checkpoint):
I am currently in the bathroom, which does not contain any appliances or containers suitable for melting orange juice. The kitchen is more likely to have heat sources like a stove and is also more likely to store orange juice. Since the door to the kitchen is visible but currently closed, I need to open it in order to access the kitchen and verify the next step.

Bad example:
I need to find the orange juice to melt it. I need to open the fridge.

Good example (reflective reasoning):
I already checked the counter drawer and it didn't contain orange juice. Since fridge commonly store liquids like juice, the most plausible next step is to look inside the fridge. Opening it should reveal whether orange juice is present so I can proceed to heat it next.
"""


# ===================================================================
# Hint rendering – matches sciworld_env.py._get_action_hints() format
# ===================================================================

def render_action_hints(possible_actions: Any, possible_objects: Any) -> str:
    """Render action hints in the same format as sciworld_env.py._get_action_hints()."""
    hint = ""
    if possible_actions:
        hint += f"Available actions: {possible_actions}\n"
    if possible_objects:
        hint += f"OBJ must be replaced with exactly one of the following candidates, using the exact string as provided: {possible_objects}."
    return hint.strip()


def _get_init_hints(rec: Dict[str, Any]) -> str:
    ih = rec.get("init_hints")
    if not isinstance(ih, dict):
        return ""
    return render_action_hints(ih.get("possible_actions"), ih.get("possible_objects"))


def _get_step_hints(step: Dict[str, Any]) -> str:
    return render_action_hints(step.get("possible_actions"), step.get("possible_objects"))


# ===================================================================
# Gold record loading
# ===================================================================

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


def _best_record(a: Dict, b: Dict) -> Dict:
    a_err = "error" in a
    b_err = "error" in b
    if a_err != b_err:
        return b if a_err else a
    a_score = a.get("final_score")
    b_score = b.get("final_score")
    if isinstance(a_score, (int, float)) and isinstance(b_score, (int, float)) and a_score != b_score:
        return a if a_score > b_score else b
    if len(a.get("steps", []) or []) != len(b.get("steps", []) or []):
        return a if len(a.get("steps", []) or []) > len(b.get("steps", []) or []) else b
    return b


def load_gold_records(inputs: List[str]) -> List[Dict[str, Any]]:
    by_id: Dict[int, Dict] = {}
    for rec in _iter_jsonl(inputs):
        if "data_idx" not in rec:
            continue
        try:
            did = int(rec["data_idx"])
        except Exception:
            continue
        by_id[did] = rec if did not in by_id else _best_record(by_id[did], rec)
    return [by_id[k] for k in sorted(by_id)]


def load_completed(output_path: str, resume_policy: str) -> set:
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
            tid = obj.get("task_id")
            if tid is None:
                continue
            try:
                tid = int(tid)
            except Exception:
                continue
            if resume_policy == "any":
                done.add(tid)
            elif resume_policy == "success":
                if obj.get("success") is True:
                    done.add(tid)
            else:
                if "error" not in obj:
                    done.add(tid)
    return done


# ===================================================================
# Thought synthesis helpers
# ===================================================================

def _strip_thought_markers(text: str) -> str:
    """Remove Thought:/Action: markers if the model included them."""
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
    task_desc: str,
    steps: List[Dict],
    actions: List[str],
    thoughts_so_far: List[str],
    init_obs: str,
    init_hint: str,
    current_step_idx: int,
    next_action: str,
    history_steps: int = 10,
) -> str:
    """Build the user message for the thought-synthesis LLM call."""
    t = current_step_idx
    start = max(0, t - history_steps)

    lines: List[str] = []
    lines.append(f"Task:\n{task_desc}")

    if start == 0:
        obs0 = init_obs
        hint0 = init_hint
    else:
        obs0 = str(steps[start - 1].get("observation", "")) if (start - 1) < len(steps) else ""
        hint0 = ""

    lines.append(f"\nInteraction history:")
    lines.append(f"Obs#{start}:\n{obs0}")
    if hint0:
        lines.append(hint0)

    for j in range(start, t):
        th = (thoughts_so_far[j] or "").strip() if j < len(thoughts_so_far) else ""
        act = str(actions[j]).strip() if j < len(actions) else ""
        obs_after = str(steps[j].get("observation", "")) if j < len(steps) else ""
        if th:
            lines.append(f"Thought#{j}: {th}")
        lines.append(f"Action#{j}: {act}")
        lines.append(f"Obs#{j + 1}: {obs_after}")

    # Current observation (before the action we're synthesising thought for)
    if t == 0:
        cur_obs = init_obs
        cur_hint = init_hint
    else:
        cur_obs = str(steps[t - 1].get("observation", "")) if (t - 1) < len(steps) else ""
        cur_hint = _get_step_hints(steps[t - 1]) if (t - 1) < len(steps) else ""

    if f"Obs#{t}:" not in "\n".join(lines[-2:]):
        lines.append(f"Obs#{t}: {cur_obs}")
        if cur_hint:
            lines.append(cur_hint)

    lines.append(f"\nNext action:\n{next_action}")

    return "\n\n".join(lines)


# ===================================================================
# CLI
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Synthesize SciWorld teacher trajectories with Thought reasoning")
    p.add_argument("--inputs", nargs="+", default=["data/sciworld/gold_trajectories.jsonl"])
    p.add_argument("--output", default="data/teacher_trajectories/sciworld_gold_qwen7b_synth.jsonl")
    p.add_argument("--export_base", type=str, default=None)
    p.add_argument("--export_threshold", type=float, default=1.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume_policy", choices=["any", "no_error", "success"], default="no_error")
    p.add_argument("--max_tasks", type=int, default=None)
    p.add_argument("--max_steps_per_task", type=int, default=None)
    p.add_argument("--success_only", action="store_true", default=False)
    p.add_argument("--model_path", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--tensor_parallel_size", type=int, default=4)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--collect_log_prob", choices=["true", "false"], default="true")
    p.add_argument("--history_steps", type=int, default=10)
    # Task subset filtering (match training config)
    p.add_argument(
        "--task_subset",
        type=int,
        default=None,
        help="Only synthesize for the first N training task_ids (after shuffle with --task_seed). "
             "Matches the training config's max_train_tasks + seed.",
    )
    p.add_argument("--task_seed", type=int, default=2026, help="Seed for task shuffling (match training config data.seed)")
    return p.parse_args()


def _get_training_task_subset(n_tasks: int, seed: int) -> set:
    """Reproduce the exact task subset used by training (max_train_tasks + seed)."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from env_service.environments.sciworld.sciworld_env import SciworldEnv
    import random

    all_ids = SciworldEnv.get_query_list("train")
    rng = random.Random(seed)
    rng.shuffle(all_ids)
    return set(all_ids[:n_tasks])


# ===================================================================
# Main
# ===================================================================

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    gold = load_gold_records(args.inputs)
    gold = [r for r in gold if "error" not in r]
    if args.success_only:
        gold = [r for r in gold if r.get("success") is True]
    # Filter to training task subset (reproduces max_train_tasks + seed from config)
    if args.task_subset and args.task_subset > 0:
        subset_ids = _get_training_task_subset(args.task_subset, args.task_seed)
        before = len(gold)
        gold = [r for r in gold if str(r["data_idx"]) in subset_ids]
        print(f"Task subset (n={args.task_subset}, seed={args.task_seed}): {before} -> {len(gold)} records")
    if args.resume:
        done = load_completed(args.output, args.resume_policy)
        if done:
            gold = [r for r in gold if int(r["data_idx"]) not in done]
            print(f"Resume (policy={args.resume_policy}): remaining={len(gold)}")
    if args.max_tasks and args.max_tasks > 0:
        gold = gold[: args.max_tasks]

    total_steps = sum(min(len(r.get("steps", [])), len(r.get("gold_action_sequence", []))) for r in gold)
    print(f"Gold records to process: {len(gold)}, total steps: {total_steps}")

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agentevolver.module.teacher import create_teacher_llm

    llm = create_teacher_llm({
        "type": "vllm",
        "model_path": args.model_path,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "collect_log_prob": (args.collect_log_prob == "true"),
    })

    system_prompt = SCIWORLD_SYSTEM_PROMPT

    mode = "a" if (args.resume and os.path.exists(args.output)) else "w"
    ok = 0
    err = 0
    all_trajectory_dicts: List[Dict[str, Any]] = []

    with open(args.output, mode, encoding="utf-8") as wf:
        for rec in gold:
            data_idx = int(rec["data_idx"])
            task_desc = rec.get("task_description", "")
            init_obs = rec.get("initial_observation", "")
            init_hint = _get_init_hints(rec)
            actions = rec.get("gold_action_sequence", []) or []
            steps = rec.get("steps", []) or []
            n = min(len(actions), len(steps))
            if args.max_steps_per_task and args.max_steps_per_task > 0:
                n = min(n, int(args.max_steps_per_task))

            out: Dict[str, Any] = {
                "task_id": str(data_idx),
                "data_id": str(data_idx),
                "rollout_id": f"{data_idx}_gold_synth_0",
                "teacher_model": getattr(llm, "model_name", args.model_path.split("/")[-1]),
                "metadata": {
                    "is_teacher": True,
                    "is_synthesized": True,
                    "source": "sciworld_gold",
                    "source_files": args.inputs,
                    "collected_at": datetime.now().isoformat(),
                },
            }

            try:
                # Build multi-turn messages matching training format
                messages: List[Dict[str, str]] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "assistant", "content": "OK. I'll help you complete this scientific experiment step by step."},
                    {"role": "user", "content": f"Task: {task_desc}\n\nCurrent observation:\n{init_obs}\n\n{init_hint}"},
                ]

                acc_lp: List[float] = []
                turn_lp: List[Dict] = []
                has_lp = True
                thoughts_so_far: List[str] = []

                for i in range(n):
                    action = str(actions[i])

                    # Ask LLM to generate Thought for this action
                    synth_user = _build_synth_context(
                        task_desc=task_desc,
                        steps=steps,
                        actions=actions,
                        thoughts_so_far=thoughts_so_far,
                        init_obs=init_obs,
                        init_hint=init_hint,
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
                        thought = "To make progress on the task, I will take the next action."

                    # Append assistant turn (Thought + Action)
                    messages.append({
                        "role": "assistant",
                        "content": f"Thought:\n{thought}\n\nAction:\n{action}",
                    })
                    thoughts_so_far.append(thought)

                    # Append user turn (env observation + action hints)
                    obs_after = str(steps[i].get("observation", ""))
                    hint_after = _get_step_hints(steps[i])
                    user_content = f"{obs_after}\n\n{hint_after}" if hint_after else obs_after
                    messages.append({"role": "user", "content": user_content})

                    # Collect log probs
                    if meta and "log_probs" in meta and has_lp:
                        turn_lp.append({
                            "turn_idx": i,
                            "log_probs": meta.get("log_probs", []),
                            "token_ids": meta.get("token_ids", []),
                            "tokens": meta.get("tokens", []),
                        })
                        acc_lp.extend(meta.get("log_probs", []))

                # Binary reward (EPO-style: done AND score > 0)
                final_score = rec.get("final_score")
                done = bool(rec.get("done", False))
                reward = 1.0 if (done and isinstance(final_score, (int, float)) and final_score > 0) else 0.0

                out.update({
                    "messages": messages,
                    "reward": reward,
                    "success": reward > 0,
                    "is_terminated": done,
                })
                out["metadata"]["has_log_prob"] = bool(has_lp and len(acc_lp) > 0)
                out["metadata"]["num_turns"] = n
                out["metadata"]["total_generated_tokens"] = len(acc_lp) if has_lp else 0
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
                print(f"[ok={ok} err={err}] task_id={data_idx} turns={n} reward={reward}")

            except Exception as e:
                out["error"] = str(e)
                wf.write(json.dumps(out, ensure_ascii=False) + "\n")
                wf.flush()
                err += 1
                print(f"[ok={ok} err={err}] task_id={data_idx} ERROR: {e}", file=sys.stderr)

    # Save pkl alongside jsonl (for training configs that reference .pkl)
    pkl_path = os.path.splitext(args.output)[0] + ".pkl"
    if args.resume and os.path.exists(pkl_path):
        existing_dicts = pickle.load(open(pkl_path, "rb"))
        all_trajectory_dicts = existing_dicts + all_trajectory_dicts
    with open(pkl_path, "wb") as pf:
        pickle.dump(all_trajectory_dicts, pf)
    print(f"\nDone. ok={ok}, err={err}, output={args.output}")
    print(f"PKL saved: {pkl_path} ({len(all_trajectory_dicts)} trajectories)")

    # Optional export
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
