# -*- coding: utf-8 -*-
"""DeepSearch environment: multi-hop QA over a fixed wiki-18 corpus.

Self-contained BaseEnv (no AgentGym server): the episode logic lives here.
The agent alternates <think>/<action> turns; actions are
    search[query]   -> top-k BM25 passages from the retrieval service
    answer[text]    -> terminal; reward = strict EM against golden answers

Task source: MuSiQue-Ans 3-4 hop (see scripts/build_deepsearch_splits.py and
data/deepsearch/SPLIT_MANIFEST.json). Determinism: fixed Lucene index + BM25
means identical action sequences always reproduce identical observations,
so the replay contract holds by construction.

Reward protocol (pre-registered, 2026-08-05):
  - main metric  : strict EM after SQuAD-style normalization (lowercase,
    strip articles/punctuation/extra whitespace), any golden answer matches;
  - auxiliary    : token-level F1 (recorded in info, not used as reward);
  - anti-hacking : only the text inside answer[...] is evaluated.
"""

import json
import os
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from env_service.base import BaseEnv
from env_service.registry import Registry

_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "deepsearch"
_TASK_FILES = {
    "train": _DATA_DIR / "tasks_train_pool.jsonl",
    "test": _DATA_DIR / "tasks_test_dev34.jsonl",
}
_TASKS: Dict[str, Dict[str, Any]] = {}


def _load_tasks() -> Dict[str, Dict[str, Any]]:
    global _TASKS
    if not _TASKS:
        tasks: Dict[str, Dict[str, Any]] = {}
        for split, path in _TASK_FILES.items():
            if not path.exists():
                raise FileNotFoundError(f"deepsearch task file missing: {path}")
            for line in path.open():
                r = json.loads(line)
                r["split"] = split
                tasks[r["task_id"]] = r
        _TASKS = tasks
    return _TASKS


# ---------------- answer normalization (SQuAD-style) ----------------

def normalize_answer(s: str) -> str:
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: List[str]) -> float:
    p = normalize_answer(pred)
    return 1.0 if any(p == normalize_answer(g) for g in golds) else 0.0


def f1_score(pred: str, golds: List[str]) -> float:
    p_tokens = normalize_answer(pred).split()
    best = 0.0
    for g in golds:
        g_tokens = normalize_answer(g).split()
        common = Counter(p_tokens) & Counter(g_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        prec = overlap / len(p_tokens)
        rec = overlap / len(g_tokens)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


@Registry.register("deepsearch")
class DeepsearchEnv(BaseEnv):  # 类名必须匹配 env_service 的 capitalize() 约定,勿改为 DeepSearchEnv
    """Multi-hop QA search agent environment over wiki-18 (BM25)."""

    def __init__(self, task_id: str = None, instance_id: str = None,
                 params: Dict[str, Any] = None):
        self.task_id = task_id
        self.instance_id = instance_id
        self.params = params or {}
        self.retrieval_url: str = (
            self.params.get("retrieval_url")
            or os.environ.get("DEEPSEARCH_RETRIEVAL_URL", "http://127.0.0.1:25011")
        ).rstrip("/")
        self.top_k = int(self.params.get("top_k", os.environ.get("DEEPSEARCH_TOP_K", 3)))
        self.action_format = str(self.params.get("action_format", "react_tags"))

        self.question: Optional[str] = None
        self.golden_answers: List[str] = []
        self.hop: Optional[int] = None
        self.search_count = 0
        self.is_done = False
        self.current_reward = 0.0
        self.last_f1 = 0.0
        self.last_pred: Optional[str] = None

    # ---------------- prompts ----------------

    def _get_system_prompt(self) -> str:
        return (
            "You are a research assistant that answers questions by searching a "
            "Wikipedia corpus. Questions typically require combining evidence from "
            "multiple documents (multi-hop): break the question into sub-questions "
            "and search for one fact at a time.\n\n"
            "Available actions:\n"
            "[\n"
            '{"action": "search[query]", "description": "search the corpus; returns the top passages for your query"},\n'
            '{"action": "answer[final answer]", "description": "submit your final answer and finish the task"}\n'
            "]\n\n"
            "Rules:\n"
            "1. Search for ONE specific fact per query; refine your query if results are unhelpful.\n"
            "2. Only submit answer[...] once the evidence supports it.\n"
            "3. The final answer must be a short exact phrase (an entity, name, date or number), "
            "not a sentence.\n\n"
            "At each turn you must respond in this format:\n"
            "<think>\nyour reasoning about what to do next.\n</think>\n"
            "<action>\nsearch[your query] or answer[your final answer]\n</action>"
        )

    # ---------------- BaseEnv interface ----------------

    def get_init_state(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        tasks = _load_tasks()
        task = tasks.get(str(self.task_id))
        if task is None:
            raise KeyError(f"unknown deepsearch task_id: {self.task_id}")
        self.question = task["question"]
        self.golden_answers = list(task["golden_answers"])
        self.hop = task.get("hop")
        self.search_count = 0
        self.is_done = False
        self.current_reward = 0.0
        self.last_f1 = 0.0
        self.last_pred = None

        init_messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "assistant",
             "content": "OK. I'll research this question step by step, searching for one fact at a time."},
            {"role": "user",
             "content": f"Question: {self.question}\n\nStart by thinking about what to search first."},
        ]
        return {
            "instance_id": self.instance_id,
            "state": init_messages,
            "task_name": f"musique_{self.hop}hop",
            "task_description": self.question,
        }

    def _parse_action(self, text: str) -> Optional[str]:
        if self.action_format == "react_tags":
            m = list(re.finditer(r"<action>(.*?)</action>", text, re.S))
            if m:
                return m[-1].group(1).strip()
            return None
        m = re.search(r"Action:\s*(.+)", text, re.S)
        return m.group(1).strip() if m else None

    def step(self, action: Dict[str, Any], params: Dict[str, Any] = None) -> Dict[str, Any]:
        if isinstance(action, dict):
            action_str = action.get("content", action.get("action", ""))
        elif isinstance(action, str):
            action_str = action
        else:
            raise ValueError(f"Invalid action format: {action}")

        parsed = self._parse_action(action_str or "")
        if parsed is None:
            return {
                "state": [{"role": "user", "content":
                           "Invalid action format. Please use '<action>\nsearch[query] or answer[final answer]\n</action>'."}],
                "reward": 0.0,
                "is_terminated": False,
                "info": {"invalid_action": True},
                "instance_id": self.instance_id,
            }

        m_search = re.match(r"^search\[(.*)\]$", parsed, re.S)
        m_answer = re.match(r"^answer\[(.*)\]$", parsed, re.S)

        if m_answer:
            pred = m_answer.group(1).strip()
            self.last_pred = pred
            self.current_reward = exact_match(pred, self.golden_answers)
            self.last_f1 = f1_score(pred, self.golden_answers)
            self.is_done = True
            return {
                "state": [{"role": "user", "content": "Answer submitted. Episode finished."}],
                "reward": self.current_reward,
                "is_terminated": True,
                "info": {
                    "prediction": pred,
                    "em": self.current_reward,
                    "f1": self.last_f1,
                    "search_count": self.search_count,
                    "hop": self.hop,
                },
                "instance_id": self.instance_id,
            }

        if m_search:
            query = m_search.group(1).strip()
            self.search_count += 1
            try:
                resp = requests.post(
                    f"{self.retrieval_url}/search",
                    json={"query": query, "k": self.top_k},
                    timeout=60.0,
                )
                resp.raise_for_status()
                results = resp.json().get("results", [])
            except Exception as e:
                raise RuntimeError(f"retrieval service error: {e}") from e
            if results:
                blocks = []
                for i, r in enumerate(results, 1):
                    blocks.append(f"[Doc {i}] {r['title']}\n{r['text']}")
                obs = f"Search results for '{query}':\n\n" + "\n\n".join(blocks)
            else:
                obs = f"No results found for '{query}'. Try a different query."
            return {
                "state": [{"role": "user", "content": obs}],
                "reward": 0.0,
                "is_terminated": False,
                "info": {"search_count": self.search_count, "query": query},
                "instance_id": self.instance_id,
            }

        return {
            "state": [{"role": "user", "content":
                       f"Unknown action '{parsed}'. Valid actions: search[query] or answer[final answer]."}],
            "reward": 0.0,
            "is_terminated": False,
            "info": {"invalid_action": True},
            "instance_id": self.instance_id,
        }

    def evaluate(self, messages: Dict[str, Any] = None, params: Dict[str, Any] = None) -> float:
        return float(self.current_reward)

    def get_info(self, messages: Dict[str, Any] = None, params: Dict[str, Any] = None) -> Dict[str, Any]:
        return {
            "env_id": "deepsearch",
            "task_id": self.task_id,
            "question": self.question,
            "hop": self.hop,
            "done": self.is_done,
            "reward": self.current_reward,
            "em": self.current_reward,
            "f1": self.last_f1,
            "prediction": self.last_pred,
            "search_count": self.search_count,
        }

    def close(self):
        pass

    @staticmethod
    def get_query_list(split: str = "train", params: Dict[str, Any] = None) -> List[str]:
        tasks = _load_tasks()
        if split in ("train",):
            ids = [t for t, r in tasks.items() if r["split"] == "train"]
        elif split in ("val", "dev"):
            val_file = _DATA_DIR / "task_ids_val200_seed2026.txt"
            ids = val_file.read_text().split()
        else:
            ids = [t for t, r in tasks.items() if r["split"] == "test"]
        return sorted(ids)
