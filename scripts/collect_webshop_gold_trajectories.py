#!/usr/bin/env python3
"""
Collect verified WebShop gold trajectories for the actual task ids used by
AgentEvolver training.

This version supports multiple successful rollouts per task:
- 1 canonical `single_search` rollout
- up to N additional `multi_search` rollouts that explicitly contain >= 2
  search actions

All rollouts are verified on the live EnvService, and each rollout attempt
releases its remote environment in a `finally` block.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import re
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WEBSHOP_ROOT = PROJECT_ROOT / "AgentGym" / "agentenv-webshop" / "webshop"
if str(WEBSHOP_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBSHOP_ROOT))

from agentevolver.client.env_client import EnvClient
from env_service.environments.webshop.webshop_env import WebshopEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect verified WebShop gold trajectories aligned to training task ids."
    )
    parser.add_argument(
        "--env_url",
        type=str,
        default=os.environ.get("WEBSHOP_ENV_SERVICE_URL", "http://127.0.0.1:8083"),
        help="EnvService base URL.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "val", "dev"],
        help="Task split to collect from when --task_file is not provided.",
    )
    parser.add_argument(
        "--task_file",
        type=str,
        default=None,
        help="Optional explicit task-id file. Supports txt/json/jsonl formats.",
    )
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path.")
    parser.add_argument("--task_start", type=int, default=0)
    parser.add_argument("--task_end", type=int, default=None)
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument(
        "--task_subset",
        type=int,
        default=None,
        help="Shuffle train task ids with --task_seed and keep the first N, matching training config.",
    )
    parser.add_argument("--task_seed", type=int, default=2026)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tasks that already have a completed task_summary record in output.",
    )
    parser.add_argument("--num_products", type=int, default=1000)
    parser.add_argument(
        "--product_file",
        type=str,
        default=WebshopEnv._get_default_product_file(),
    )
    parser.add_argument("--human_goals", action="store_true")
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--max_pages", type=int, default=8)
    parser.add_argument(
        "--max_queries",
        type=int,
        default=10,
        help="Maximum distinct query templates used to build search plans.",
    )
    parser.add_argument("--success_threshold", type=float, default=1.0)
    parser.add_argument(
        "--instruction_match_policy",
        choices=["strict", "prefer", "ignore"],
        default="strict",
        help=(
            "'strict': only keep rollouts with instruction_match=true; "
            "'prefer': prefer matched rollouts but allow mismatched to fill gaps; "
            "'ignore': keep both."
        ),
    )
    parser.add_argument("--target_rollouts_per_task", type=int, default=5)
    parser.add_argument("--target_multisearch_rollouts", type=int, default=4)
    parser.add_argument(
        "--max_rollout_plans_per_task",
        type=int,
        default=16,
        help="Upper bound on rollout plans attempted per task.",
    )
    parser.add_argument("--sleep_sec", type=float, default=0.0)
    return parser.parse_args()


def parse_task_id(value: Any) -> str:
    parsed = WebshopEnv._parse_session_id(value)
    if parsed is None:
        raise ValueError(f"Unexpected WebShop task id: {value}")
    return str(parsed)


def load_task_ids_from_file(path: str) -> List[str]:
    raw = Path(path).read_text(encoding="utf-8").strip()
    if not raw:
        return []

    task_ids: List[str] = []
    if raw[0] in ("[", "{"):
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        task_ids.append(
                            parse_task_id(item.get("item_id", item.get("task_id", item.get("data_idx"))))
                        )
                    else:
                        task_ids.append(parse_task_id(item))
            elif isinstance(data, dict):
                task_ids.append(parse_task_id(data.get("item_id", data.get("task_id", data.get("data_idx")))))
        except json.JSONDecodeError:
            for line in raw.splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                task_ids.append(parse_task_id(obj.get("item_id", obj.get("task_id", obj.get("data_idx")))))
    else:
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            task_ids.append(parse_task_id(stripped))
    return task_ids


def fetch_task_ids(client: EnvClient, split: str) -> List[str]:
    return [parse_task_id(task_id) for task_id in client.get_env_profile("webshop", split)]


def slice_task_ids(
    task_ids: List[str],
    start: int,
    end: Optional[int],
    max_tasks: Optional[int],
) -> List[str]:
    sliced = task_ids[start:end]
    if max_tasks is not None and max_tasks > 0:
        sliced = sliced[:max_tasks]
    return sliced


def sample_training_subset(task_ids: List[str], n_tasks: Optional[int], seed: int) -> List[str]:
    if not n_tasks or n_tasks <= 0:
        return task_ids
    shuffled = list(task_ids)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled[:n_tasks]


def deterministic_price_for_product(asin: str, pricing: Sequence[float]) -> float:
    if not pricing:
        return 100.0
    if len(pricing) == 1:
        return float(pricing[0])

    low = float(pricing[0])
    high = float(pricing[1])
    if high <= low:
        return low

    key = f"{asin}|{low:.8f}|{high:.8f}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float((1 << 64) - 1)
    return low + (high - low) * fraction


def load_completed_task_ids(path: str) -> set[str]:
    completed: set[str] = set()
    output_path = Path(path)
    if not output_path.exists():
        return completed
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("record_kind") == "task_summary" and record.get("completed") is True:
                task_id = record.get("task_id")
                if task_id is not None:
                    completed.add(str(task_id))
    return completed


def rebuild_goals(product_file: str, num_products: int, human_goals: bool) -> List[Dict[str, Any]]:
    if human_goals:
        raise NotImplementedError(
            "This collector currently supports the synthetic-goal setup used by "
            "training. Re-run without --human_goals."
        )

    with open(product_file, "r", encoding="utf-8") as f:
        products = json.load(f)
    with open(WebshopEnv._get_default_attribute_file(), "r", encoding="utf-8") as f:
        attributes = json.load(f)

    def parse_price(pricing: Any) -> List[float]:
        if pricing is None or not pricing:
            return [100.0]
        values = [
            float(Decimal(re.sub(r"[^\d.]", "", price)))
            for price in str(pricing).split("$")[1:]
        ]
        return values[:2] if values else [100.0]

    price_range_base = [10.0 * i for i in range(1, 100)]
    seen_asins: set[str] = set()
    goals: List[Dict[str, Any]] = []

    random.seed(233)
    for product in products[:num_products]:
        asin = str(product.get("asin", ""))
        if asin == "nan" or not asin or len(asin) > 10 or asin in seen_asins:
            continue
        seen_asins.add(asin)

        attr_entry = attributes.get(asin)
        if not attr_entry:
            continue

        instruction_text = attr_entry.get("instruction")
        instruction_attributes = attr_entry.get("instruction_attributes")
        if instruction_text is None or not instruction_attributes:
            continue

        pricing = parse_price(product.get("pricing"))
        product_price = deterministic_price_for_product(asin, pricing)
        price_range = [p for p in price_range_base if p > product_price][:4]
        if len(price_range) >= 2:
            _, price_upper = sorted(random.sample(price_range, 2))
            price_text = f", and price lower than {price_upper:.2f} dollars"
        else:
            price_upper = 1000000
            price_text = ""

        raw_options = product.get("customization_options") or {}
        option_names: List[str] = []
        option_values: List[List[str]] = []
        for raw_option_name, option_contents in sorted(
            raw_options.items(), key=lambda item: str(item[0]).lower()
        ):
            if option_contents is None:
                continue
            normalized_values = [
                option_content.get("value", "").strip().replace("/", " | ").lower()
                for option_content in option_contents
                if option_content.get("value")
            ]
            if not normalized_values:
                continue
            option_names.append(str(raw_option_name).lower())
            option_values.append(normalized_values)

        combinations = list(itertools.product(*option_values)) if option_values else [()]
        for combination in combinations:
            goal_options = {
                option_names[i]: combination[i]
                for i in range(len(combination))
            }
            option_text = ", and ".join(f"{k}: {v}" for k, v in goal_options.items())
            option_text = f" with {option_text}" if option_text else ""
            goals.append(
                {
                    "asin": asin,
                    "category": product.get("category"),
                    "query": str(product.get("query", "")).lower().strip(),
                    "name": product.get("name", ""),
                    "product_category": product.get("product_category"),
                    "instruction_text": f"{instruction_text}{option_text}{price_text}",
                    "attributes": instruction_attributes,
                    "price_upper": price_upper,
                    "goal_options": goal_options,
                }
            )

    random.seed(233)
    random.shuffle(goals)
    return goals


def normalize_instruction(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("Instruction: "):
        return text[len("Instruction: ") :].strip()
    return text


def sanitize_query(text: str) -> str:
    cleaned = (text or "").replace("[", " ").replace("]", " ").replace("\n", " ")
    return " ".join(cleaned.split())


def build_query_catalog(goal: Dict[str, Any], max_queries: int) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    seen: set[str] = set()

    def add(name: str, candidate: str) -> None:
        candidate = sanitize_query(candidate)
        if not candidate or candidate in seen:
            return
        seen.add(candidate)
        pairs.append((name, candidate))

    title = str(goal.get("name", ""))
    query = str(goal.get("query", ""))
    instruction = normalize_instruction(str(goal.get("instruction_text", "")))
    instruction_no_price = instruction.split(", and price lower than", 1)[0].strip()
    attributes = [sanitize_query(str(x)) for x in goal.get("attributes", [])]
    option_values = [sanitize_query(str(v)) for v in (goal.get("goal_options") or {}).values()]

    add("title", title)
    add("short_title", " ".join(title.split()[:8]))
    add("query", query)
    add("instruction", instruction_no_price)

    for idx, attr in enumerate(attributes[:3]):
        add(f"attribute_{idx}", f"{attr} {query}")
    for idx, option_value in enumerate(option_values[:2]):
        add(f"option_{idx}", f"{option_value} {query}")

    return pairs[:max_queries]


def build_search_plans(goal: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    query_catalog = build_query_catalog(goal, args.max_queries)
    query_map = dict(query_catalog)
    plans: List[Dict[str, Any]] = []
    seen_signatures: set[Tuple[str, Tuple[str, ...], Tuple[int, ...]]] = set()

    def add_plan(
        trajectory_type: str,
        search_policy: str,
        query_keys: Sequence[str],
        page_budgets: Sequence[int],
    ) -> None:
        queries = [query_map[key] for key in query_keys if key in query_map]
        if len(queries) != len(query_keys):
            return
        signature = (trajectory_type, tuple(queries), tuple(page_budgets))
        if signature in seen_signatures:
            return
        seen_signatures.add(signature)
        plans.append(
            {
                "trajectory_type": trajectory_type,
                "search_policy": search_policy,
                "query_sequence": queries,
                "query_keys": list(query_keys),
                "page_budgets": list(page_budgets),
            }
        )

    add_plan("single_search", "canonical_title", ["title"], [args.max_pages])
    add_plan("single_search", "canonical_short_title", ["short_title"], [args.max_pages])
    add_plan("single_search", "canonical_query", ["query"], [args.max_pages])
    add_plan("single_search", "canonical_instruction", ["instruction"], [args.max_pages])

    multisearch_specs: List[Tuple[str, List[str], List[int]]] = [
        ("query_then_title", ["query", "title"], [1, args.max_pages]),
        ("instruction_then_title", ["instruction", "title"], [1, args.max_pages]),
        ("short_title_then_title", ["short_title", "title"], [1, args.max_pages]),
        ("query_then_short_title_then_title", ["query", "short_title", "title"], [1, 1, args.max_pages]),
    ]
    if "attribute_0" in query_map:
        multisearch_specs.append(("attribute_then_title", ["attribute_0", "title"], [1, args.max_pages]))
        multisearch_specs.append(
            ("query_then_attribute_then_title", ["query", "attribute_0", "title"], [1, 1, args.max_pages])
        )
    if "option_0" in query_map:
        multisearch_specs.append(("option_then_title", ["option_0", "title"], [1, args.max_pages]))

    for search_policy, query_keys, page_budgets in multisearch_specs:
        add_plan("multi_search", search_policy, query_keys, page_budgets)

    return plans[: max(1, int(args.max_rollout_plans_per_task))]


def format_action_message(action: str) -> Dict[str, str]:
    return {"role": "assistant", "content": f"Action:\n{action}"}


def extract_clickables(payload: Dict[str, Any]) -> List[str]:
    clickables = payload.get("available_actions", {}).get("clickables")
    if clickables is None:
        clickables = payload.get("info", {}).get("available_actions", {}).get("clickables", [])
    return [str(x) for x in clickables or []]


def build_step_record(step_index: int, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "t": step_index,
        "action": action,
        "reward": payload.get("reward"),
        "done": payload.get("is_terminated"),
        "observation": payload.get("state", [])[-1]["content"] if payload.get("state") else "",
        "available_actions": payload.get("info", {}).get("available_actions", {}),
    }


def _trajectory_signature(record: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(record.get("action_sequence", []) or [])


def _record_search_count(record: Dict[str, Any]) -> int:
    return sum(1 for action in record.get("action_sequence", []) if str(action).startswith("search["))


def _should_keep_rollout(record: Dict[str, Any], policy: str) -> bool:
    if policy == "ignore":
        return True
    if policy == "strict":
        return bool(record.get("instruction_match"))
    return True


def _select_rollouts(
    candidates: List[Dict[str, Any]],
    target_rollouts_per_task: int,
    target_multisearch_rollouts: int,
    instruction_match_policy: str,
) -> List[Dict[str, Any]]:
    matched = [rec for rec in candidates if rec.get("instruction_match")]
    mismatched = [rec for rec in candidates if not rec.get("instruction_match")]
    source_order: List[Dict[str, Any]]
    if instruction_match_policy == "strict":
        source_order = matched
    elif instruction_match_policy == "prefer":
        source_order = matched + mismatched
    else:
        source_order = candidates

    chosen: List[Dict[str, Any]] = []
    seen_signatures: set[Tuple[str, ...]] = set()
    single_taken = False
    multisearch_taken = 0

    for record in source_order:
        signature = _trajectory_signature(record)
        if signature in seen_signatures:
            continue
        traj_type = record.get("trajectory_type")
        if traj_type == "single_search":
            if single_taken:
                continue
            single_taken = True
        elif traj_type == "multi_search":
            if multisearch_taken >= target_multisearch_rollouts:
                continue
            multisearch_taken += 1
        else:
            continue

        seen_signatures.add(signature)
        chosen.append(record)
        if len(chosen) >= target_rollouts_per_task:
            break

    return chosen


def run_rollout_plan(
    client: EnvClient,
    task_id: str,
    goal: Dict[str, Any],
    plan: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    init = client.create_instance("webshop", task_id)
    instance_id = init["instance_id"]
    expected_instruction = normalize_instruction(str(goal["instruction_text"]))
    live_instruction = normalize_instruction(init.get("instruction", ""))
    initial_observation = init.get("state", [])[-1]["content"] if init.get("state") else ""
    instruction_match = live_instruction == expected_instruction
    target_click = f"click[{str(goal['asin']).lower()}]"
    action_sequence: List[str] = []
    steps: List[Dict[str, Any]] = []
    step_index = 0
    payload = {"state": init.get("state", []), "info": {"available_actions": init.get("available_actions", {})}}

    def execute_action(action: str) -> Dict[str, Any]:
        nonlocal step_index
        response = client.step(instance_id, format_action_message(action))
        action_sequence.append(action)
        steps.append(build_step_record(step_index, action, response))
        step_index += 1
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)
        return response

    try:
        for stage_idx, query in enumerate(plan["query_sequence"]):
            if step_index >= args.max_steps:
                return {
                    "success": False,
                    "error": "max_steps_reached_before_search",
                    "query_sequence": plan["query_sequence"],
                    "trajectory_type": plan["trajectory_type"],
                    "search_policy": plan["search_policy"],
                    "steps": steps,
                    "action_sequence": action_sequence,
                    "live_instruction": live_instruction,
                    "expected_instruction": expected_instruction,
                    "instruction_match": instruction_match,
                    "initial_observation": initial_observation,
                }

            if stage_idx > 0:
                clickables = [x.lower() for x in extract_clickables(payload)]
                back_action = "click[back to search]"
                if back_action not in clickables:
                    return {
                        "success": False,
                        "error": "missing_back_to_search_before_refine",
                        "query_sequence": plan["query_sequence"],
                        "trajectory_type": plan["trajectory_type"],
                        "search_policy": plan["search_policy"],
                        "steps": steps,
                        "action_sequence": action_sequence,
                        "live_instruction": live_instruction,
                        "expected_instruction": expected_instruction,
                        "instruction_match": instruction_match,
                        "initial_observation": initial_observation,
                    }
                payload = execute_action(back_action)

            payload = execute_action(f"search[{query}]")
            page_budget = max(1, int(plan["page_budgets"][stage_idx]))
            is_final_stage = stage_idx == len(plan["query_sequence"]) - 1

            found_target = False
            for _ in range(page_budget):
                clickables = [x.lower() for x in extract_clickables(payload)]
                if target_click in clickables:
                    found_target = True
                    if not is_final_stage:
                        break
                    payload = execute_action(target_click)
                    break

                if "click[next >]" not in clickables or step_index >= args.max_steps:
                    break
                payload = execute_action("click[next >]")

            if is_final_stage:
                if not found_target:
                    return {
                        "success": False,
                        "error": "target_asin_not_found_in_final_search_stage",
                        "query_sequence": plan["query_sequence"],
                        "trajectory_type": plan["trajectory_type"],
                        "search_policy": plan["search_policy"],
                        "steps": steps,
                        "action_sequence": action_sequence,
                        "live_instruction": live_instruction,
                        "expected_instruction": expected_instruction,
                        "instruction_match": instruction_match,
                        "initial_observation": initial_observation,
                    }

        goal_options = goal.get("goal_options", {})
        option_values = goal_options.values() if isinstance(goal_options, dict) else goal_options
        for option_value in option_values:
            if step_index >= args.max_steps:
                return {
                    "success": False,
                    "error": "max_steps_reached_before_option_selection",
                    "query_sequence": plan["query_sequence"],
                    "trajectory_type": plan["trajectory_type"],
                    "search_policy": plan["search_policy"],
                    "steps": steps,
                    "action_sequence": action_sequence,
                    "live_instruction": live_instruction,
                    "expected_instruction": expected_instruction,
                    "instruction_match": instruction_match,
                    "initial_observation": initial_observation,
                }
            option_click = f"click[{str(option_value).lower()}]"
            clickables = [x.lower() for x in extract_clickables(payload)]
            if option_click not in clickables:
                return {
                    "success": False,
                    "error": f"missing_option_click:{option_click}",
                    "query_sequence": plan["query_sequence"],
                    "trajectory_type": plan["trajectory_type"],
                    "search_policy": plan["search_policy"],
                    "steps": steps,
                    "action_sequence": action_sequence,
                    "live_instruction": live_instruction,
                    "expected_instruction": expected_instruction,
                    "instruction_match": instruction_match,
                    "initial_observation": initial_observation,
                }
            payload = execute_action(option_click)

        if step_index >= args.max_steps:
            return {
                "success": False,
                "error": "max_steps_reached_before_buy_now",
                "query_sequence": plan["query_sequence"],
                "trajectory_type": plan["trajectory_type"],
                "search_policy": plan["search_policy"],
                "steps": steps,
                "action_sequence": action_sequence,
                "live_instruction": live_instruction,
                "expected_instruction": expected_instruction,
                "instruction_match": instruction_match,
                "initial_observation": initial_observation,
            }

        payload = execute_action("click[buy now]")
        final_reward = float(client.evaluate(instance_id))

        return {
            "success": final_reward >= args.success_threshold,
            "final_reward": final_reward,
            "done": bool(payload.get("is_terminated", False)),
            "trajectory_type": plan["trajectory_type"],
            "search_policy": plan["search_policy"],
            "query_sequence": list(plan["query_sequence"]),
            "query_keys": list(plan["query_keys"]),
            "num_search_actions": _record_search_count({"action_sequence": action_sequence}),
            "steps": steps,
            "action_sequence": action_sequence,
            "live_instruction": live_instruction,
            "expected_instruction": expected_instruction,
            "instruction_match": instruction_match,
            "initial_observation": initial_observation,
        }
    finally:
        client.release_instance(instance_id)


def build_rollout_record(
    task_id: str,
    split: str,
    goal: Dict[str, Any],
    result: Dict[str, Any],
    rollout_index: int,
) -> Dict[str, Any]:
    rollout_id = f"webshop_{task_id}_{result['trajectory_type']}_{rollout_index}"
    return {
        "record_kind": "rollout",
        "env": "webshop",
        "split": split,
        "task_id": task_id,
        "data_id": task_id,
        "item_id": f"webshop_{task_id}",
        "rollout_id": rollout_id,
        "trajectory_type": result["trajectory_type"],
        "search_policy": result["search_policy"],
        "query_sequence": result["query_sequence"],
        "query_keys": result["query_keys"],
        "num_search_actions": result["num_search_actions"],
        "goal": {
            "asin": goal.get("asin"),
            "query": goal.get("query"),
            "name": goal.get("name"),
            "instruction_text": goal.get("instruction_text"),
            "attributes": goal.get("attributes"),
            "goal_options": goal.get("goal_options"),
        },
        "success": True,
        "instruction": f"Instruction: {result['live_instruction']}",
        "instruction_match": result["instruction_match"],
        "expected_instruction": result["expected_instruction"],
        "initial_observation": result["initial_observation"],
        "action_sequence": result["action_sequence"],
        "steps": result["steps"],
        "final_reward": result["final_reward"],
        "done": result["done"],
    }


def build_task_summary(
    task_id: str,
    split: str,
    goal: Dict[str, Any] | None,
    rollouts: List[Dict[str, Any]],
    attempted_plans: List[Dict[str, Any]],
    error: Optional[str] = None,
) -> Dict[str, Any]:
    single_successes = sum(1 for r in rollouts if r.get("trajectory_type") == "single_search")
    multi_successes = sum(1 for r in rollouts if r.get("trajectory_type") == "multi_search")
    return {
        "record_kind": "task_summary",
        "env": "webshop",
        "split": split,
        "task_id": task_id,
        "item_id": f"webshop_{task_id}",
        "completed": True,
        "success": len(rollouts) > 0,
        "num_success_rollouts": len(rollouts),
        "num_single_search_rollouts": single_successes,
        "num_multi_search_rollouts": multi_successes,
        "rollout_ids": [r["rollout_id"] for r in rollouts],
        "instruction_match_rollouts": sum(1 for r in rollouts if r.get("instruction_match")),
        "attempted_plans": attempted_plans,
        "goal_asin": goal.get("asin") if goal else None,
        "error": error,
    }


def collect_task_rollouts(
    client: EnvClient,
    task_id: str,
    goal: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    attempted_plans: List[Dict[str, Any]] = []
    rollout_index = 0

    for plan in build_search_plans(goal, args):
        if len(attempted_plans) >= args.max_rollout_plans_per_task:
            break
        try:
            result = run_rollout_plan(client, task_id, goal, plan, args)
        except Exception as exc:
            result = {
                "success": False,
                "error": str(exc),
                "trajectory_type": plan["trajectory_type"],
                "search_policy": plan["search_policy"],
                "query_sequence": list(plan["query_sequence"]),
            }

        attempted_plans.append(
            {
                "trajectory_type": plan["trajectory_type"],
                "search_policy": plan["search_policy"],
                "query_sequence": list(plan["query_sequence"]),
                "success": result.get("success", False),
                "instruction_match": result.get("instruction_match"),
                "final_reward": result.get("final_reward"),
                "error": result.get("error"),
            }
        )

        if not result.get("success"):
            continue
        if result.get("trajectory_type") == "multi_search" and result.get("num_search_actions", 0) < 2:
            continue
        if not _should_keep_rollout(result, args.instruction_match_policy):
            continue

        rollout_record = build_rollout_record(
            task_id=task_id,
            split=args.split,
            goal=goal,
            result=result,
            rollout_index=rollout_index,
        )
        rollout_index += 1
        candidates.append(rollout_record)

    chosen = _select_rollouts(
        candidates=candidates,
        target_rollouts_per_task=args.target_rollouts_per_task,
        target_multisearch_rollouts=args.target_multisearch_rollouts,
        instruction_match_policy=args.instruction_match_policy,
    )

    summary_error = None
    if not chosen:
        summary_error = attempted_plans[-1].get("error") if attempted_plans else "no_rollout_plans_generated"
    summary = build_task_summary(task_id, args.split, goal, chosen, attempted_plans, summary_error)
    return chosen, summary


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def append_records(path: str, records: List[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    client = EnvClient(args.env_url)

    if args.task_file:
        task_ids = load_task_ids_from_file(args.task_file)
    else:
        task_ids = fetch_task_ids(client, args.split)

    if args.split == "train" and args.task_subset and args.task_subset > 0:
        task_ids = sample_training_subset(task_ids, args.task_subset, args.task_seed)
    task_ids = slice_task_ids(task_ids, args.task_start, args.task_end, args.max_tasks)

    if args.resume:
        completed = load_completed_task_ids(args.output)
        task_ids = [task_id for task_id in task_ids if task_id not in completed]

    goals = rebuild_goals(
        product_file=args.product_file,
        num_products=args.num_products,
        human_goals=args.human_goals,
    )

    print(f"Loaded {len(task_ids)} task ids for split={args.split}.")
    print(f"Rebuilt {len(goals)} WebShop goals from {args.product_file}.")

    tasks_with_success = 0
    total_rollouts = 0
    for idx, task_id in enumerate(task_ids, start=1):
        goal_index = int(task_id)
        if goal_index < 0 or goal_index >= len(goals):
            summary = build_task_summary(
                task_id=task_id,
                split=args.split,
                goal=None,
                rollouts=[],
                attempted_plans=[],
                error=f"task_id_out_of_range_for_rebuilt_goals:{goal_index}",
            )
            append_records(args.output, [summary])
            print(f"[{idx}/{len(task_ids)}] task_id={task_id} -> failed (out_of_range)")
            continue

        rollouts, summary = collect_task_rollouts(client, task_id, goals[goal_index], args)
        records_to_write = rollouts + [summary]
        append_records(args.output, records_to_write)

        if rollouts:
            tasks_with_success += 1
            total_rollouts += len(rollouts)
        print(
            f"[{idx}/{len(task_ids)}] task_id={task_id} -> "
            f"success_rollouts={len(rollouts)} "
            f"(single={summary['num_single_search_rollouts']}, multi={summary['num_multi_search_rollouts']})"
        )

    print(
        f"Finished. tasks_with_success={tasks_with_success}/{len(task_ids)}, "
        f"total_success_rollouts={total_rollouts}. Output written to {args.output}"
    )


if __name__ == "__main__":
    main()
