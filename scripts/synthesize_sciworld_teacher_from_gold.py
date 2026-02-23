#!/usr/bin/env python3
import argparse, json, os, re, sys
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCIWORLD_BASE_SYSTEM_PROMPT = '''You are a scientific experiment assistant in a text-based simulation environment. Your task is to perform scientific experiments by interacting with objects in the environment.

At each step, you will receive:
1. The task description (what experiment you need to perform)
2. Your current observation (what you can see/do)
3. Hints about available actions and objects

Common action patterns:
- look around: observe your surroundings
- look at [object]: examine an object closely
- pick up [object]: take an object
- put [object] in [container]: place an object somewhere
- open [object]: open a container/door
- close [object]: close a container/door
- move to [location]: go to another location
- activate [object]: turn on a device
- deactivate [object]: turn off a device
- use [object] on [target]: use one object on another
- focus on [object]: focus your attention
- wait: wait for something to happen

You should choose from two actions: "THOUGHT" or "ACTION".
- If you choose "THOUGHT", first analyze the task and current state, then output your action.
  Format: "Thought:\nyour thoughts.\n\nAction:\nyour next action"
- If you choose "ACTION", directly output the action.
  Format: "Action:\nyour next action"

Important:
1. Read the task description carefully.
2. Plan your experiment steps logically.
3. Pay attention to the objects and locations available.
4. Some experiments may require multiple steps.'''


SYNTH_SYSTEM_PROMPT = """You will be given a ScienceWorld task, the current observation, and the next action that will be taken.

Write ONLY the Thought(1-3 sentences) that would plausibly lead to taking that action.

Hard constraints:
- Output ONLY the Thought text. Do NOT include "Action:".
- Do NOT mention that the next action was given to you."""

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inputs",
        nargs="+",
        default=[
            # Prefer augmented file if present; otherwise fall back to raw gold files.
            "data/teacher_trajectories/sciworld_gold_augmented.jsonl",
            "data/teacher_trajectories/sciworld_gold.jsonl",
            "data/teacher_trajectories/sciworld_gold_retry_404.jsonl",
        ],
    )
    p.add_argument("--output", default="data/teacher_trajectories/sciworld_gold_qwen7b_synth.jsonl")
    # Optional: after synthesis, export a filtered {base}.jsonl/{base}.pkl like AlfWorld teacher pipeline.
    # This mirrors `scripts/filter_teacher_trajectories.py`.
    p.add_argument(
        "--export_base",
        type=str,
        default=None,
        help="If set, run post-filtering and export {export_base}.jsonl and {export_base}.pkl",
    )
    p.add_argument(
        "--export_threshold",
        type=float,
        default=1.0,
        help="Reward threshold for export filtering (default: 1.0)",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume_policy", choices=["any","no_error","success"], default="no_error")
    p.add_argument("--max_tasks", type=int, default=None)
    p.add_argument("--max_steps_per_task", type=int, default=None)
    p.add_argument("--success_only", action="store_true", default=False)
    p.add_argument("--model_path", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--tensor_parallel_size", type=int, default=4)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--collect_log_prob", choices=["true","false"], default="true")
    return p.parse_args()

def _iter_jsonl(paths: List[str]) -> Iterable[Dict[str, Any]]:
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s=line.strip()
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
    a_score = a.get("final_score"); b_score = b.get("final_score")
    if isinstance(a_score,(int,float)) and isinstance(b_score,(int,float)) and a_score!=b_score:
        return a if a_score>b_score else b
    if len(a.get("steps",[]) or []) != len(b.get("steps",[]) or []):
        return a if len(a.get("steps",[]) or []) > len(b.get("steps",[]) or []) else b
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


def _get_hint_str_from_record(rec: Dict[str, Any]) -> str:
    # augmented format
    if isinstance(rec.get("init_hints"), dict) and isinstance(rec["init_hints"].get("hint_str"), str):
        return rec["init_hints"]["hint_str"]
    # raw format
    return ""


def _get_step_hint_str(rec: Dict[str, Any], step_idx: int) -> str:
    # augmented format: steps_augmented[t].hints.hint_str (after action)
    steps_aug = rec.get("steps_augmented")
    if isinstance(steps_aug, list) and step_idx < len(steps_aug):
        h = steps_aug[step_idx].get("hints", {})
        if isinstance(h, dict) and isinstance(h.get("hint_str"), str):
            return h["hint_str"]
    return ""

def load_completed(output_path: str, resume_policy: str) -> set:
    done=set()
    if not os.path.exists(output_path):
        return done
    with open(output_path,"r",encoding="utf-8") as f:
        for line in f:
            s=line.strip()
            if not s: 
                continue
            try:
                obj=json.loads(s)
            except Exception:
                continue
            tid=obj.get("task_id")
            if tid is None: 
                continue
            try:
                tid=int(tid)
            except Exception:
                continue
            if resume_policy=="any":
                done.add(tid)
            elif resume_policy=="success":
                if obj.get("success") is True:
                    done.add(tid)
            else:
                if "error" not in obj:
                    done.add(tid)
    return done

def _strip_thought_action(text: str) -> Tuple[str, Optional[str]]:
    t=(text or "").strip()
    if not t:
        return "", None
    m=re.search(r"Thought:\s*(.*?)\n\s*\n\s*Action:\s*(.*)$", t, flags=re.I|re.S)
    if m:
        thought=m.group(1).strip()
        act=m.group(2).strip().splitlines()[0].strip() if m.group(2).strip() else ""
        return thought, act or None
    m2=re.search(r"Action:\s*(.*)$", t, flags=re.I|re.S)
    if m2:
        act=m2.group(1).strip().splitlines()[0].strip()
        return "", act or None
    return t, None

def main():
    args=parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    gold=load_gold_records(args.inputs)
    gold=[r for r in gold if "error" not in r]
    if args.success_only:
        gold=[r for r in gold if r.get("success") is True]
    if args.resume:
        done=load_completed(args.output, args.resume_policy)
        if done:
            gold=[r for r in gold if int(r["data_idx"]) not in done]
            print(f"Resume enabled (policy={args.resume_policy}): remaining={len(gold)}")
    if args.max_tasks and args.max_tasks>0:
        gold=gold[:args.max_tasks]

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agentevolver.module.teacher import create_teacher_llm
    llm=create_teacher_llm({
        "type":"vllm",
        "model_path": args.model_path,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "collect_log_prob": (args.collect_log_prob=="true"),
    })

    mode="a" if (args.resume and os.path.exists(args.output)) else "w"
    ok=0; err=0
    with open(args.output, mode, encoding="utf-8") as wf:
        for rec in gold:
            data_idx=int(rec["data_idx"])
            task_desc=rec.get("task_description","")
            init_obs=rec.get("initial_observation","")
            actions = rec.get("gold_action_sequence", []) or []
            # Support both raw gold format (`steps`) and augmented format (`steps_augmented`).
            steps = rec.get("steps", None)
            if not isinstance(steps, list):
                steps_aug = rec.get("steps_augmented", []) or []
                if isinstance(steps_aug, list) and steps_aug:
                    # Normalize to the raw `steps` shape expected by the synthesizer.
                    steps = [
                        {
                            "t": s.get("t", i),
                            "action": s.get("action", ""),
                            "observation": s.get("observation", ""),
                            "reward": s.get("reward", None),
                            "score": s.get("score", None),
                            "done": s.get("done", None),
                        }
                        for i, s in enumerate(steps_aug)
                    ]
                else:
                    steps = []
            n=min(len(actions), len(steps))
            if args.max_steps_per_task and args.max_steps_per_task>0:
                n=min(n, int(args.max_steps_per_task))

            out={
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
                messages=[
                    {"role":"system","content":SCIWORLD_BASE_SYSTEM_PROMPT},
                    {"role":"assistant","content":"OK. I'll help you complete this scientific experiment step by step."},
                    {"role":"user","content":f"Task: {task_desc}\n\nCurrent observation:\n{init_obs}\n\n{_get_hint_str_from_record(rec)}"},
                ]
                turn_lp=[]; acc_lp=[]; has_lp=True
                for i in range(n):
                    action=str(actions[i])
                    pre_obs = init_obs if i==0 else str(steps[i-1].get("observation",""))
                    # In training replay, each user message also includes action hints.
                    pre_hint = _get_hint_str_from_record(rec) if i==0 else _get_step_hint_str(rec, i-1)
                    pre_user_content = f"{pre_obs}\n\n{pre_hint}" if pre_hint else pre_obs
                    synth_msgs=[
                        {"role":"system","content":SYNTH_SYSTEM_PROMPT},
                        {"role":"user","content":f"Task:\n{task_desc}\n\nObservation:\n{pre_user_content}\n\nNext action:\n{action}\n"},
                    ]
                    resp, meta = llm(synth_msgs)
                    # Model is instructed to output Thought only.
                    thought = (resp or "").strip()
                    # If model still returns Thought:/Action: blocks, strip them.
                    thought2, _ = _strip_thought_action(thought)
                    thought = thought2.strip()
                    if thought.lower().startswith("thought:"):
                        thought = thought.split(":", 1)[-1].strip()
                    if not thought:
                        # Safe fallback: ensure non-empty thought for downstream consumers.
                        thought = "To make progress on the task, I will take the next action."

                    messages.append({"role":"assistant","content":f"Thought:\n{thought}\n\nAction:\n{action}"})
                    # user replay content aligns with sciworld_env.py: "<observation>\n\n<hints>"
                    obs_after = str(steps[i].get("observation",""))
                    hint_after = _get_step_hint_str(rec, i)
                    user_after = f"{obs_after}\n\n{hint_after}" if hint_after else obs_after
                    messages.append({"role":"user","content":user_after})
                    if meta and "log_probs" in meta and has_lp:
                        turn_lp.append({"turn_idx": i, "log_probs": meta.get("log_probs",[]), "token_ids": meta.get("token_ids",[]), "tokens": meta.get("tokens",[])})
                        acc_lp.extend(meta.get("log_probs",[]))

                final_score=rec.get("final_score")
                done=bool(rec.get("done", False))
                outcome=max(0.0, min(100.0, float(final_score)))/100.0 if isinstance(final_score,(int,float)) else 0.0
                out.update({"messages": messages, "reward": outcome, "success": bool(outcome==1.0), "is_terminated": done})
                out["metadata"]["has_log_prob"] = bool(has_lp and len(acc_lp)>0)
                out["metadata"]["num_turns"] = n
                out["metadata"]["total_generated_tokens"] = len(acc_lp) if has_lp else 0
                # Log-probs here (if any) cover ONLY the synthesized Thought, not the appended Action.
                if out["metadata"]["has_log_prob"]:
                    out["metadata"]["log_prob_scope"] = "thought_only"
                    out["thought_log_probs"] = acc_lp
                    out["thought_log_probs_per_turn"] = turn_lp

                wf.write(json.dumps(out, ensure_ascii=False)+"\n"); wf.flush()
                ok += 1
                print(f"[ok={ok} err={err}] task_id={data_idx} turns={n} success={out['success']}")
            except Exception as e:
                out["error"]=str(e)
                wf.write(json.dumps(out, ensure_ascii=False)+"\n"); wf.flush()
                err += 1
                print(f"[ok={ok} err={err}] task_id={data_idx} ERROR: {e}", file=sys.stderr)

    # Optional export in AlfWorld-style filtered JSONL+PKL.
    if args.export_base:
        try:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from scripts.filter_teacher_trajectories import filter_trajectories

            print(f"\nExporting filtered teacher trajectories to base={args.export_base} ...")
            filter_trajectories(
                input_path=args.output,
                output_base=args.export_base,
                reward_threshold=float(args.export_threshold),
                verbose=True,
            )
        except Exception as e:
            print(f"WARNING: export failed: {e}", file=sys.stderr)

if __name__=="__main__":
    main()

