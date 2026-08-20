#!/usr/bin/env python3
"""Collect auditable OpenRouter teacher trajectories on the student curriculum.

The collector intentionally reuses the production ``EnvWorker`` and
``AgentFlow``.  The only rollout-config override is
``context_management.snapshot_training=false``: an API teacher cannot return
the student's raw vLLM token/log-prob objects, and that switch does not change
the messages presented to the model.  Every other model-facing and environment
field comes from the resolved student experiment config.
"""

from __future__ import annotations

import argparse
import ast
import copy
import fcntl
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from agentevolver.module.agent_flow.agent_flow import AgentFlow
from agentevolver.module.context_manager.cmt_base import chat_template_ids
from agentevolver.module.env_manager.env_worker import EnvWorker
from agentevolver.module.exp_manager.exp_manager import TrajExpConfig
from agentevolver.module.teacher.openai_teacher_llm import OpenAITeacherLLM
from agentevolver.schema.task import Task


SCHEMA_VERSION = "openrouter_teacher_trajectory_v2"
ATTEMPT_SCHEMA_VERSION = "openrouter_teacher_attempt_v1"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
# GaaS-cluster (H200) artifact pins, 2026-08-04. The stock /projects_vol
# Qwen3.5-4B chat template is thinking-on by default and strips historical
# <think> blocks — semantically identical to the A100 "-think" patched dir
# this contract originally pinned (tokenizer.json is byte-identical to the
# A100 pin; chat_template.jinja/tokenizer_config.json bytes differ because
# the A100 dir patched an older HF snapshot). Cross-machine reconciliation
# must re-pin these hashes explicitly.
DEFAULT_STUDENT_TOKENIZER = "/projects_vol/gp_wangwy/models/Qwen3.5-4B"
EXPECTED_TOKENIZER_HASHES = {
    "chat_template.jinja": (
        "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
    ),
    "tokenizer_config.json": (
        "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"
    ),
    "tokenizer.json": (
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
    ),
}
EXPECTED_CONTEXT = {
    "alfworld": {"recent_turns": 2, "history_observation_max_tokens": 160},
    "webshop": {"recent_turns": 4, "history_observation_max_tokens": 512},
    # sciworld:上下文压缩沿 ALFWorld 风格(recent 2 / 历史观测 160 tok),
    # 32K 契约不变;该组参数待 GPU 冒烟确认(军规),确认前仅限试点采集。
    "sciworld": {"recent_turns": 2, "history_observation_max_tokens": 160},
    # deepsearch:20 步 × top-3 检索观测 ≈ 15K < 22528,合同内全量无损;
    # recent_turns=20 覆盖全部轮次,1024 tok 历史上限仅作病态兜底(正常
    # 永不触发)。见 docs/design/DEEPSEARCH_ENV.md §4。
    "deepsearch": {"recent_turns": 20, "history_observation_max_tokens": 1024},
}
# 各环境 multi_turn.max_steps 契约值。AF/WS 维持 30(历史契约,不可动);
# sciworld 长程实验取 100;deepsearch 取 20(论证见 DEEPSEARCH_ENV.md §3)。
EXPECTED_MAX_STEPS = {
    "alfworld": 30,
    "webshop": 30,
    "sciworld": 100,
    "deepsearch": 20,
}
CANONICAL_TASK_SOURCES = {
    "alfworld": PROJECT_ROOT
    / "AgentGym/agentenv-alfworld/configs/mappings_train.json",
    "webshop": PROJECT_ROOT
    / "env_service/environments/webshop/webshop_train.json",
    # sciworld:规范池由 scripts/build_sciworld_task_pool.py 从
    # SciworldEnv.get_query_list("train")(= eval 200 题的补集)一次性固化。
    "sciworld": PROJECT_ROOT / "data/sciworld/canonical_task_pool.json",
    # deepsearch:MuSiQue 3-4 hop 训练池(5562 条,jsonl 每行含 task_id,
    # 落盘时已按 task_id 字典序排序;见 scripts/build_deepsearch_splits.py
    # 与 data/deepsearch/SPLIT_MANIFEST.json)。
    "deepsearch": PROJECT_ROOT / "data/deepsearch/tasks_train_pool.jsonl",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return json_safe(model_dump(mode="json", exclude_none=True))
    return str(value)


def safe_error_text(error: BaseException) -> str:
    text = str(error)
    text = re.sub(r"(?i)bearer\s+[^\s,;]+", "Bearer [REDACTED]", text)
    text = re.sub(r"sk-[A-Za-z0-9_-]{12,}", "sk-[REDACTED]", text)
    return text[:2000]


def parse_api_key_from_python(path: Path) -> str:
    """Read a literal ``api_key=...`` without importing/executing the file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    candidates: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if (
                    keyword.arg == "api_key"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                ):
                    candidates.add(keyword.value.value.strip())
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                continue
            for target in targets:
                if isinstance(target, ast.Name) and target.id in {
                    "api_key",
                    "openrouter_api_key",
                }:
                    candidates.add(value.value.strip())
    candidates.discard("")
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one literal api_key in {path}; found {len(candidates)}"
        )
    return next(iter(candidates))


def load_api_key(source: Optional[Path], env_name: str) -> str:
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value.strip()
    if source is None:
        raise RuntimeError(
            f"{env_name} is unset and no --api-key-source was provided"
        )
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"API key source does not exist: {source}")
    if source.suffix == ".py":
        value = parse_api_key_from_python(source)
    else:
        value = source.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError(f"API key source is empty: {source}")
    return value


def _numeric_webshop_id(value: Any) -> str:
    text = str(value).strip()
    match = re.fullmatch(r"(?:webshop_)?(\d+)", text, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"invalid WebShop task id: {value!r}")
    return str(int(match.group(1)))


def _canonical_task_id(env_name: str, value: Any) -> str:
    """按环境把任务 id 规范成课程文件中的字符串形态。

    - alfworld/sciworld:十进制整数字符串(int 往返剔除前导零等非规范形态);
    - webshop:"webshop_123"/"123" 统一为 "123";
    - deepsearch:id 本身是字符串(如 "musique_train_13900"),原样保留,
      仅做字符集校验——绝不能过 int()。
    """
    if env_name == "webshop":
        return _numeric_webshop_id(value)
    if env_name == "deepsearch":
        text = str(value).strip()
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+", text):
            raise ValueError(f"invalid deepsearch task id: {value!r}")
        return text
    return str(int(value))


def _membership_sort_key(env_name: str):
    """sorted_membership 的排序键(per-env)。

    alfworld/webshop/sciworld 维持 int 键——AF/WS 既有 manifest 里的
    sorted_membership_sha256 必须逐字节不变;deepsearch 的字符串 id 用
    字典序,与 scripts/build_deepsearch_splits.py 的 sorted() 约定一致。
    """
    if env_name == "deepsearch":
        return lambda item: item
    return lambda item: int(item)


def canonical_task_pool(env_name: str) -> tuple[List[str], Path]:
    source = CANONICAL_TASK_SOURCES[env_name]
    if env_name == "deepsearch":
        # jsonl 每行一个任务对象;build_deepsearch_splits.py 落盘时已按
        # task_id 字典序排序,这里复核以保证 shuffle 前的池顺序确定。
        task_ids = [
            _canonical_task_id(env_name, json.loads(line)["task_id"])
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if task_ids != sorted(task_ids):
            raise RuntimeError(
                "canonical deepsearch task pool must be sorted by task_id"
            )
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if env_name == "alfworld":
            task_ids = [str(item["item_id"]) for item in payload]
        elif env_name == "webshop":
            task_ids = [_numeric_webshop_id(item["item_id"]) for item in payload]
        elif env_name == "sciworld":
            # 固化文件由 build_sciworld_task_pool.py 生成,task_ids 字段保序
            # 保存 get_query_list("train") 的输出(0..max_eval 去掉 eval 集)。
            task_ids = [
                _canonical_task_id(env_name, item) for item in payload["task_ids"]
            ]
        else:  # pragma: no cover - argparse/config validation prevents this
            raise ValueError(f"unsupported environment: {env_name}")
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError(f"canonical {env_name} task pool contains duplicates")
    return task_ids, source


def expected_curriculum(env_name: str, task_seed: int, count: int) -> Dict[str, Any]:
    pool, source = canonical_task_pool(env_name)
    if count <= 0 or count > len(pool):
        raise ValueError(f"invalid curriculum count {count} for pool size {len(pool)}")
    ordered = list(pool)
    random.Random(task_seed).shuffle(ordered)
    ordered = ordered[:count]
    newline_payload = ("\n".join(ordered) + "\n").encode("utf-8")
    # 排序键 per-env:AF/WS/SciWorld 维持 int 键(AF/WS 历史 sha 逐字节
    # 不变),deepsearch 字符串 id 用字典序。
    sorted_payload = (
        "\n".join(sorted(ordered, key=_membership_sort_key(env_name))) + "\n"
    ).encode("utf-8")
    return {
        "environment": env_name,
        "algorithm": "python_random_seed_then_shuffle_then_prefix_without_replacement",
        "task_seed": task_seed,
        "pool_count": len(pool),
        "pool_unique_count": len(set(pool)),
        "count": count,
        "source_path": str(source.relative_to(PROJECT_ROOT)),
        "source_sha256": sha256_file(source),
        "ordered_newline_sha256": sha256_bytes(newline_payload),
        "ordered_json_sha256": canonical_json_hash(ordered),
        "sorted_membership_sha256": sha256_bytes(sorted_payload),
        "task_ids": ordered,
    }


def load_and_validate_task_file(
    path: Path,
    *,
    env_name: str,
    task_seed: int,
    expected_count: int,
) -> tuple[List[str], Dict[str, Any]]:
    path = path.resolve()
    raw = path.read_text(encoding="utf-8")
    if not raw.endswith("\n"):
        raise RuntimeError(f"task file must end with a newline: {path}")
    lines = raw.splitlines()
    if any(not line.strip() or line.strip() != line for line in lines):
        raise RuntimeError(f"task file contains blank or non-canonical lines: {path}")
    # per-env 规范化:deepsearch 的字符串 id 绝不能过 int()。
    task_ids = [_canonical_task_id(env_name, line) for line in lines]
    if len(task_ids) != expected_count or len(set(task_ids)) != expected_count:
        raise RuntimeError(
            f"task file must contain exactly {expected_count} unique IDs; "
            f"got {len(task_ids)} lines/{len(set(task_ids))} unique"
        )
    expected = expected_curriculum(env_name, task_seed, expected_count)
    if task_ids != expected["task_ids"]:
        overlap = len(set(task_ids) & set(expected["task_ids"]))
        raise RuntimeError(
            "task file is not the current student curriculum: "
            f"ordered_match=false, membership_overlap={overlap}/{expected_count}, "
            f"expected_sha256={expected['ordered_newline_sha256']}, "
            f"actual_sha256={sha256_bytes(raw.encode('utf-8'))}"
        )
    manifest = {key: value for key, value in expected.items() if key != "task_ids"}
    manifest.update(
        {
            "task_file": str(path),
            "task_file_sha256": sha256_bytes(raw.encode("utf-8")),
        }
    )
    return task_ids, manifest


def compose_student_config(path: Path) -> DictConfig:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"student config does not exist: {path}")
    with initialize_config_dir(config_dir=str(path.parent), version_base=None):
        config = compose(config_name=path.stem)
    OmegaConf.resolve(config)
    return config


def validate_student_contract(config: DictConfig) -> Dict[str, Any]:
    rollout = config.actor_rollout_ref.rollout
    data = config.data
    env_name = str(config.env_service.env_type).lower()
    if env_name not in EXPECTED_CONTEXT:
        raise RuntimeError(f"unsupported student environment: {env_name}")
    checks = {
        "thinking_mode": (str(rollout.thinking_mode), "native_qwen35"),
        "prompt_length": (int(rollout.prompt_length), 22528),
        "response_length": (int(rollout.response_length), 10240),
        "max_model_len": (int(rollout.max_model_len), 32768),
        "data.max_prompt_length": (int(data.max_prompt_length), 22528),
        "data.max_response_length": (int(data.max_response_length), 10240),
        # per-env 契约步数(AF/WS 恒 30;sciworld 100;deepsearch 20)。
        "multi_turn.max_steps": (
            int(rollout.multi_turn.max_steps),
            EXPECTED_MAX_STEPS[env_name],
        ),
        "context_template": (str(rollout.context_template), "linear"),
        "context_management.enabled": (
            bool(rollout.context_management.enabled),
            True,
        ),
        "context_management.max_prompt_tokens": (
            int(rollout.context_management.max_prompt_tokens),
            22528,
        ),
        "context_management.min_recent_turns": (
            int(rollout.context_management.min_recent_turns),
            1,
        ),
        "context_management.recent_observation_max_tokens": (
            int(rollout.context_management.recent_observation_max_tokens),
            -1,
        ),
        "context_management.allow_current_observation_truncation": (
            bool(rollout.context_management.allow_current_observation_truncation),
            False,
        ),
        "context_management.reasoning_history_tokens": (
            int(rollout.context_management.reasoning_history_tokens),
            0,
        ),
        "context_management.snapshot_training": (
            bool(rollout.context_management.snapshot_training),
            True,
        ),
        "env_params.action_format": (
            str(config.env_service.env_params.action_format),
            "react_tags",
        ),
        "exp_manager.train_rollout_mode": (
            str(config.exp_manager.train_rollout_mode),
            "woexp",
        ),
    }
    expected_context = EXPECTED_CONTEXT[env_name]
    checks["context_management.recent_turns"] = (
        int(rollout.context_management.recent_turns),
        expected_context["recent_turns"],
    )
    checks["context_management.history_observation_max_tokens"] = (
        int(rollout.context_management.history_observation_max_tokens),
        expected_context["history_observation_max_tokens"],
    )
    failed = {
        key: {"actual": actual, "expected": expected}
        for key, (actual, expected) in checks.items()
        if actual != expected
    }
    if failed:
        raise RuntimeError(
            "student config does not satisfy the Qwen3.5 v5 collection contract: "
            + json.dumps(failed, sort_keys=True)
        )
    if int(rollout.prompt_length) + int(rollout.response_length) != int(
        rollout.max_model_len
    ):
        raise RuntimeError("prompt_length + response_length must equal max_model_len")
    model_path = str(config.actor_rollout_ref.model.path)
    if Path(model_path).resolve() != Path(DEFAULT_STUDENT_TOKENIZER).resolve():
        raise RuntimeError(
            f"unexpected student tokenizer/model path: {model_path}; "
            f"expected {DEFAULT_STUDENT_TOKENIZER}"
        )
    return {
        "environment": env_name,
        "student_model_path": model_path,
        "prompt_length": int(rollout.prompt_length),
        "response_length": int(rollout.response_length),
        "max_model_len": int(rollout.max_model_len),
        "max_steps": int(rollout.multi_turn.max_steps),
        "thinking_mode": str(rollout.thinking_mode),
        "context_management": OmegaConf.to_container(
            rollout.context_management, resolve=True
        ),
        "env_params": OmegaConf.to_container(config.env_service.env_params, resolve=True),
    }


def collection_config(student_config: DictConfig, env_url: str) -> DictConfig:
    result = OmegaConf.create(
        copy.deepcopy(OmegaConf.to_container(student_config, resolve=True))
    )
    result.env_service.env_url = str(env_url).rstrip("/")
    # API responses do not include the vLLM token objects/log-probabilities that
    # PPO snapshots require.  This flag changes only training capture; the
    # StructuredContextPolicy messages are byte-identical.
    result.actor_rollout_ref.rollout.context_management.snapshot_training = False
    result.actor_rollout_ref.rollout.debug_llm_io = False
    return result


def load_student_tokenizer(model_path: str):
    path = Path(model_path).resolve()
    for filename, expected_hash in EXPECTED_TOKENIZER_HASHES.items():
        candidate = path / filename
        if not candidate.exists():
            raise FileNotFoundError(f"required tokenizer artifact missing: {candidate}")
        actual_hash = sha256_file(candidate)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"tokenizer drift detected for {filename}: "
                f"{actual_hash} != {expected_hash}"
            )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(path), trust_remote_code=True, local_files_only=True
    )
    template_hash = sha256_bytes(
        str(getattr(tokenizer, "chat_template", "") or "").encode("utf-8")
    )
    if template_hash != EXPECTED_TOKENIZER_HASHES["chat_template.jinja"]:
        raise RuntimeError(
            f"loaded chat template drift: {template_hash} != "
            f"{EXPECTED_TOKENIZER_HASHES['chat_template.jinja']}"
        )
    return tokenizer, {
        "path": str(path),
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
        "chat_template_sha256": template_hash,
        "artifact_sha256": dict(EXPECTED_TOKENIZER_HASHES),
    }


def verify_live_task_profile(
    env_url: str,
    env_name: str,
    expected: Sequence[str],
    task_seed: int,
) -> None:
    import requests

    response = requests.post(
        f"{env_url.rstrip('/')}/get_env_profile",
        json={"env_type": env_name, "params": {"split": "train"}},
        timeout=120,
    )
    response.raise_for_status()
    body = response.json()
    if not body.get("success"):
        raise RuntimeError(f"live task profile request failed: {json_safe(body)}")
    # per-env 规范化:deepsearch 的字符串 id 绝不能过 int()。
    live = [_canonical_task_id(env_name, item) for item in body["data"]]
    random.Random(task_seed).shuffle(live)
    if live[: len(expected)] != list(expected):
        live_prefix_payload = ("\n".join(live[: len(expected)]) + "\n").encode()
        raise RuntimeError(
            "live environment profile does not match the frozen curriculum: "
            f"live_prefix_sha256={sha256_bytes(live_prefix_payload)}"
        )


class ExclusiveOutputLock:
    def __init__(self, output: Path):
        self.path = Path(str(output) + ".lock")
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another collector holds {self.path}") from error
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f"pid={os.getpid()} acquired_at={utc_now()}\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
        return False


class JsonlJournal:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: Mapping[str, Any]) -> None:
        payload = json.dumps(
            json_safe(record), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        if "\n" in payload:
            raise AssertionError("JSONL serialization unexpectedly contains a raw newline")
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(payload + "\n")
                handle.flush()
                os.fsync(handle.fileno())


def scan_success_output(
    path: Path,
    *,
    contract_sha256: str,
    allowed_tasks: set[str],
) -> Dict[str, Dict[str, Any]]:
    completed: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n") or not line.strip():
                raise RuntimeError(
                    f"malformed or incomplete JSONL line {line_number} in {path}"
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if record.get("schema_version") != SCHEMA_VERSION:
                raise RuntimeError(f"schema mismatch at {path}:{line_number}")
            if record.get("contract_sha256") != contract_sha256:
                raise RuntimeError(f"contract mismatch at {path}:{line_number}")
            if not bool(record.get("success")):
                raise RuntimeError(f"non-success trajectory in success dataset at line {line_number}")
            task_id = str(record.get("task_id"))
            if task_id not in allowed_tasks:
                raise RuntimeError(f"extraneous task {task_id} at {path}:{line_number}")
            rollout_id = str(record.get("rollout_id", ""))
            if not rollout_id or rollout_id in completed:
                raise RuntimeError(f"missing/duplicate rollout_id at {path}:{line_number}")
            completed[rollout_id] = record
    return completed


def scan_attempt_ledger(path: Path, contract_sha256: str) -> Dict[str, int]:
    """Return the next never-before-started attempt index for each rollout.

    A started attempt consumes budget even if the process dies before it can
    append ``attempt_finished``.  Counting only finished attempts would let a
    repeatedly interrupted ``--resume`` exceed ``--max-attempts-per-rollout``.
    """
    next_attempt: Dict[str, int] = {}
    if not path.exists():
        return next_attempt
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n") or not line.strip():
                raise RuntimeError(f"malformed attempt ledger line {line_number}: {path}")
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid attempt JSON at {path}:{line_number}: {error}"
                ) from error
            if event.get("schema_version") != ATTEMPT_SCHEMA_VERSION:
                raise RuntimeError(f"attempt schema mismatch at {path}:{line_number}")
            if event.get("contract_sha256") != contract_sha256:
                raise RuntimeError(f"attempt contract mismatch at {path}:{line_number}")
            if event.get("event") not in {"attempt_started", "attempt_finished"}:
                continue
            rollout_id = str(event["rollout_id"])
            attempt_index = int(event["attempt_index"])
            if not rollout_id or attempt_index < 0:
                raise RuntimeError(
                    f"invalid attempt identity at {path}:{line_number}"
                )
            next_attempt[rollout_id] = max(
                next_attempt.get(rollout_id, 0), attempt_index + 1
            )
    return next_attempt


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _usage_number(usage: Any, *paths: str) -> float:
    if not isinstance(usage, Mapping):
        return 0.0
    for path in paths:
        current: Any = usage
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                current = None
                break
            current = current[part]
        if isinstance(current, (int, float)):
            return float(current)
    return 0.0


def aggregate_trace(trace: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "decision_count": len(trace),
        "api_prompt_tokens": 0.0,
        "api_completion_tokens": 0.0,
        "api_total_tokens": 0.0,
        "api_cost": 0.0,
        "api_latency_ms": 0.0,
        "api_retry_count": 0,
        "max_raw_prompt_tokens": 0,
        "max_managed_prompt_tokens": 0,
        "total_dropped_turns": 0,
        "total_clipped_observations": 0,
        "length_truncated_decisions": 0,
    }
    for decision in trace:
        api = decision.get("api", {})
        usage = api.get("usage") or {}
        result["api_prompt_tokens"] += _usage_number(usage, "prompt_tokens")
        result["api_completion_tokens"] += _usage_number(usage, "completion_tokens")
        result["api_total_tokens"] += _usage_number(usage, "total_tokens")
        result["api_cost"] += _usage_number(usage, "cost", "total_cost")
        result["api_latency_ms"] += float(api.get("latency_ms") or 0.0)
        result["api_retry_count"] += int(api.get("retry_count") or 0)
        stats = decision.get("context_stats") or {}
        result["max_raw_prompt_tokens"] = max(
            result["max_raw_prompt_tokens"], int(stats.get("raw_prompt_tokens") or 0)
        )
        result["max_managed_prompt_tokens"] = max(
            result["max_managed_prompt_tokens"],
            int(stats.get("managed_prompt_tokens") or 0),
        )
        result["total_dropped_turns"] += int(stats.get("dropped_turns") or 0)
        result["total_clipped_observations"] += int(
            stats.get("clipped_observations") or 0
        )
        result["length_truncated_decisions"] += int(
            bool(decision.get("truncated_by_length"))
        )
    return result


class DecisionTraceRecorder:
    """API callback + AgentFlow observer for one private trajectory."""

    def __init__(
        self,
        *,
        teacher_llm: OpenAITeacherLLM,
        tokenizer: Any,
        response_token_limit: int,
        temperature: float,
        top_p: Optional[float],
        store_prompt_messages: bool,
    ):
        self.teacher_llm = teacher_llm
        self.tokenizer = tokenizer
        self.response_token_limit = int(response_token_limit)
        self.temperature = float(temperature)
        self.top_p = top_p
        self.store_prompt_messages = bool(store_prompt_messages)
        self.trace: List[Dict[str, Any]] = []
        self._inflight: Optional[Dict[str, Any]] = None

    def chat(self, messages: List[Dict[str, str]], **_: Any) -> Dict[str, Any]:
        if self._inflight is not None:
            raise RuntimeError("decision observer did not consume the previous API call")
        prompt_messages = copy.deepcopy(messages)
        prompt_ids = chat_template_ids(
            self.tokenizer, prompt_messages, add_generation_prompt=True
        )
        call_kwargs: Dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.response_token_limit,
        }
        if self.top_p is not None:
            call_kwargs["top_p"] = float(self.top_p)
        content, api_metadata = self.teacher_llm(prompt_messages, **call_kwargs)
        api_metadata = json_safe(api_metadata or {})
        completion_ids = list(
            self.tokenizer.encode(str(content), add_special_tokens=False)
        )
        provider_finish_reason = api_metadata.get("finish_reason")
        finish_reason = provider_finish_reason
        truncated_by_length = str(provider_finish_reason) == "length"
        length_source = "provider" if truncated_by_length else None
        if len(completion_ids) > self.response_token_limit:
            finish_reason = "length"
            truncated_by_length = True
            length_source = "qwen35_retokenization"
        output = {
            "role": "assistant",
            "content": str(content),
            "sampled_content": str(content),
            "finish_reason": finish_reason,
            "truncated_by_length": truncated_by_length,
            "stop_reason": api_metadata.get("native_finish_reason"),
            "_teacher_api": api_metadata,
        }
        self._inflight = {
            "prompt_messages": prompt_messages,
            "prompt_token_ids": prompt_ids,
            "prompt_messages_sha256": canonical_json_hash(prompt_messages),
            "completion_content": str(content),
            "completion_token_ids": completion_ids,
            "provider_finish_reason": provider_finish_reason,
            "finish_reason": finish_reason,
            "truncated_by_length": truncated_by_length,
            "length_source": length_source,
            "api": api_metadata,
        }
        return output

    def observe(
        self,
        *,
        step_index: int,
        prompt_messages: List[Dict[str, str]],
        context_result: Any,
        llm_output: Mapping[str, Any],
    ) -> None:
        inflight = self._inflight
        if inflight is None:
            raise RuntimeError("decision observer ran without a preceding API call")
        if prompt_messages != inflight["prompt_messages"]:
            raise RuntimeError("AgentFlow prompt changed across the teacher API call")
        if context_result is None:
            raise RuntimeError("structured context result is missing")
        pending_messages = json_safe(context_result.messages)
        pending_ids = list(context_result.prompt_token_ids)
        if pending_messages != prompt_messages:
            raise RuntimeError("API messages differ from StructuredContextPolicy output")
        if pending_ids != inflight["prompt_token_ids"]:
            raise RuntimeError("API prompt tokens differ from context-policy tokens")
        record = {
            "step_index": int(step_index),
            "prompt_message_count": len(prompt_messages),
            "prompt_messages_sha256": inflight["prompt_messages_sha256"],
            "prompt_token_count": len(pending_ids),
            "prompt_token_ids_sha256": sha256_bytes(
                ",".join(str(token_id) for token_id in pending_ids).encode("ascii")
            ),
            "raw_prompt_token_ids_sha256": str(context_result.raw_prompt_hash),
            "context_stats": json_safe(context_result.stats),
            "completion_content": inflight["completion_content"],
            "completion_token_count": len(inflight["completion_token_ids"]),
            "completion_token_ids_sha256": sha256_bytes(
                ",".join(
                    str(token_id) for token_id in inflight["completion_token_ids"]
                ).encode("ascii")
            ),
            "provider_finish_reason": inflight["provider_finish_reason"],
            "finish_reason": inflight["finish_reason"],
            "native_finish_reason": inflight["api"].get("native_finish_reason"),
            "truncated_by_length": inflight["truncated_by_length"],
            "length_source": inflight["length_source"],
            "api": inflight["api"],
        }
        if self.store_prompt_messages:
            record["prompt_messages"] = copy.deepcopy(prompt_messages)
        if str(llm_output.get("content", "")) != inflight["completion_content"]:
            raise RuntimeError("observer saw a different completion than the API callback")
        self.trace.append(record)
        self._inflight = None


def _valid_tagged_action(content: str) -> bool:
    post_think = str(content).split("</think>")[-1]
    match = re.search(
        r"<action>\s*(.*?)\s*</action>", post_think, flags=re.IGNORECASE | re.DOTALL
    )
    return bool(match and match.group(1).strip())


def validate_success_trace(trace: Sequence[Mapping[str, Any]], max_prompt: int) -> None:
    if not trace:
        raise RuntimeError("successful trajectory has no decisions")
    for decision in trace:
        if int(decision["prompt_token_count"]) > int(max_prompt):
            raise RuntimeError("managed teacher prompt exceeds the student prompt budget")
        if bool(decision.get("truncated_by_length")):
            raise RuntimeError("length-truncated decision cannot enter expert data")
        if not _valid_tagged_action(str(decision.get("completion_content", ""))):
            raise RuntimeError("teacher decision lacks one valid post-think action block")


@dataclass(frozen=True)
class WorkItem:
    task_id: str
    rollout_index: int
    rollout_id: str
    next_attempt: int = 0


@dataclass
class WorkResult:
    item: WorkItem
    success_record: Optional[Dict[str, Any]]
    attempts: int
    metrics: Dict[str, float] = field(default_factory=dict)
    final_error: Optional[str] = None


class TeacherCollector:
    def __init__(
        self,
        *,
        config: DictConfig,
        tokenizer: Any,
        teacher_llm: OpenAITeacherLLM,
        contract_sha256: str,
        task_manifest: Mapping[str, Any],
        resolved_config_sha256: str,
        tokenizer_manifest: Mapping[str, Any],
        teacher_model: str,
        temperature: float,
        top_p: Optional[float],
        max_attempts_per_rollout: int,
        store_prompt_messages: bool,
        attempt_journal: JsonlJournal,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.teacher_llm = teacher_llm
        self.contract_sha256 = contract_sha256
        self.task_manifest = dict(task_manifest)
        self.resolved_config_sha256 = resolved_config_sha256
        self.tokenizer_manifest = dict(tokenizer_manifest)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.top_p = top_p
        self.max_attempts_per_rollout = int(max_attempts_per_rollout)
        self.store_prompt_messages = bool(store_prompt_messages)
        self.attempt_journal = attempt_journal

    def _attempt_event(
        self,
        item: WorkItem,
        attempt_index: int,
        *,
        event: str,
        **fields: Any,
    ) -> Dict[str, Any]:
        return {
            "schema_version": ATTEMPT_SCHEMA_VERSION,
            "contract_sha256": self.contract_sha256,
            "event": event,
            "timestamp": utc_now(),
            "environment": str(self.config.env_service.env_type),
            "task_id": item.task_id,
            "rollout_index": item.rollout_index,
            "rollout_id": item.rollout_id,
            "attempt_index": attempt_index,
            **json_safe(fields),
        }

    def collect(self, item: WorkItem) -> WorkResult:
        attempts_made = 0
        aggregate_metrics: Dict[str, float] = {
            "api_prompt_tokens": 0.0,
            "api_completion_tokens": 0.0,
            "api_total_tokens": 0.0,
            "api_cost": 0.0,
            "api_latency_ms": 0.0,
            "api_retry_count": 0.0,
        }
        final_error: Optional[str] = None
        remaining_attempts = max(
            0, self.max_attempts_per_rollout - int(item.next_attempt)
        )
        if remaining_attempts == 0:
            return WorkResult(
                item=item,
                success_record=None,
                attempts=0,
                metrics=aggregate_metrics,
                final_error="attempt_budget_exhausted",
            )
        for offset in range(remaining_attempts):
            attempt_index = item.next_attempt + offset
            attempts_made += 1
            self.attempt_journal.append(
                self._attempt_event(item, attempt_index, event="attempt_started")
            )
            recorder = DecisionTraceRecorder(
                teacher_llm=self.teacher_llm,
                tokenizer=self.tokenizer,
                response_token_limit=int(
                    self.config.actor_rollout_ref.rollout.response_length
                ),
                temperature=self.temperature,
                top_p=self.top_p,
                store_prompt_messages=self.store_prompt_messages,
            )
            try:
                flow = AgentFlow(
                    llm_chat_fn=recorder.chat,
                    tokenizer=self.tokenizer,
                    config=self.config,
                    decision_observer=recorder.observe,
                )
                task = Task(
                    task_id=item.task_id,
                    env_type=str(self.config.env_service.env_type),
                    open_query=False,
                    evaluator="env",
                )
                worker = EnvWorker(
                    task=task,
                    thread_index=0,
                    tokenizer=self.tokenizer,
                    config=self.config,
                )
                trajectory = worker.execute(
                    data_id=item.task_id,
                    rollout_id=item.rollout_id,
                    traj_exp_config=TrajExpConfig(
                        add_exp=False,
                        train_mode="discard",
                        task_id=item.task_id,
                        data_id=item.task_id,
                        rollout_id=item.rollout_id,
                        query="",
                        mode="sample",
                    ),
                    agent_flow=flow,
                    tmux={"step": [0], "token": [0]},
                    stop=[False],
                )
                reward = trajectory.reward
                success = bool(
                    reward is not None
                    and float(getattr(reward, "success_rate", 0.0)) > 0.0
                )
                trace_metrics = aggregate_trace(recorder.trace)
                for key in aggregate_metrics:
                    aggregate_metrics[key] += float(trace_metrics.get(key, 0.0))
                episode_end_reason = str(
                    (trajectory.metadata or {}).get("episode_end_reason", "unknown")
                )
                self.attempt_journal.append(
                    self._attempt_event(
                        item,
                        attempt_index,
                        event="attempt_finished",
                        status="success" if success else "policy_failure",
                        reward=float(reward.outcome if reward else 0.0),
                        success=success,
                        episode_end_reason=episode_end_reason,
                        decision_count=len(recorder.trace),
                        metrics=trace_metrics,
                    )
                )
                if not success:
                    final_error = f"policy_failure:{episode_end_reason}"
                    continue
                validate_success_trace(
                    recorder.trace,
                    int(self.config.actor_rollout_ref.rollout.prompt_length),
                )
                messages_raw = trajectory.to_role_content_raw(
                    trajectory.full_context
                )
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "contract_sha256": self.contract_sha256,
                    "environment": str(self.config.env_service.env_type),
                    "task_id": item.task_id,
                    "data_id": item.task_id,
                    "rollout_index": item.rollout_index,
                    "rollout_id": item.rollout_id,
                    "messages": messages_raw,
                    "query": str(trajectory.query or ""),
                    "reward": float(reward.outcome),
                    "success_rate": float(reward.success_rate),
                    "success": True,
                    "is_terminated": bool(trajectory.is_terminated),
                    "teacher_model": self.teacher_model,
                    "decision_trace": recorder.trace,
                    "metadata": {
                        "is_teacher": True,
                        "has_log_prob": False,
                        "collected_at": utc_now(),
                        "attempt_index": attempt_index,
                        "attempt_count_for_slot": attempts_made,
                        "episode_end_reason": episode_end_reason,
                        "task_manifest_sha256": self.task_manifest[
                            "ordered_newline_sha256"
                        ],
                        "resolved_student_config_sha256": self.resolved_config_sha256,
                        "student_tokenizer": self.tokenizer_manifest,
                        "context_management": OmegaConf.to_container(
                            self.config.actor_rollout_ref.rollout.context_management,
                            resolve=True,
                        ),
                        "api_and_context_totals": trace_metrics,
                    },
                }
                return WorkResult(
                    item=item,
                    success_record=record,
                    attempts=attempts_made,
                    metrics=aggregate_metrics,
                )
            except Exception as error:
                final_error = safe_error_text(error)
                trace_metrics = aggregate_trace(recorder.trace)
                for key in aggregate_metrics:
                    aggregate_metrics[key] += float(trace_metrics.get(key, 0.0))
                self.attempt_journal.append(
                    self._attempt_event(
                        item,
                        attempt_index,
                        event="attempt_finished",
                        status="infrastructure_or_contract_error",
                        success=False,
                        error_type=type(error).__name__,
                        error=final_error,
                        decision_count=len(recorder.trace),
                        metrics=trace_metrics,
                    )
                )
                logger.warning(
                    f"teacher attempt failed task={item.task_id} "
                    f"slot={item.rollout_index} attempt={attempt_index}: {final_error}"
                )
        return WorkResult(
            item=item,
            success_record=None,
            attempts=attempts_made,
            metrics=aggregate_metrics,
            final_error=final_error,
        )


def implementation_hashes(env_name: Optional[str] = None) -> Dict[str, str]:
    relative_paths = [
        "scripts/collect_openrouter_teacher_trajectories.py",
        "agentevolver/module/agent_flow/agent_flow.py",
        "agentevolver/module/env_manager/env_worker.py",
        "agentevolver/module/context_manager/context_policy.py",
        "agentevolver/module/context_manager/cmt_linear.py",
        "agentevolver/module/teacher/openai_teacher_llm.py",
        "env_service/environments/alfworld/alfworld_env.py",
        "env_service/environments/webshop/webshop_env.py",
    ]
    # 新环境战役额外钉住各自的环境实现;AF/WS 的键集保持原样(契约结构不动,
    # 老战役 manifest 里的 implementation_sha256 键集可逐一对上)。
    extra_paths = {
        "sciworld": ["env_service/environments/sciworld/sciworld_env.py"],
        "deepsearch": [
            "env_service/environments/deepsearch/deepsearch_env.py",
            # 检索服务实现与索引共同决定观测,一并入契约。
            "env_service/launch_script/retrieval_server.py",
        ],
    }
    relative_paths = relative_paths + extra_paths.get(env_name or "", [])
    return {
        relative: sha256_file(PROJECT_ROOT / relative) for relative in relative_paths
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect fixed-curriculum DeepSeek teacher trajectories"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env-url", required=True)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-task-count", type=int, default=1600)
    parser.add_argument("--task-seed", type=int, default=2026)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--rollouts-per-task", type=int, default=10)
    parser.add_argument("--max-attempts-per-rollout", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument(
        "--api-key-source",
        type=Path,
        default=Path("/data/home/qisheng/test_openrouter.py"),
    )
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--api-timeout", type=float, default=1200.0)
    parser.add_argument("--api-max-retries", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--store-prompt-messages", action="store_true")
    parser.add_argument("--skip-live-profile-check", action="store_true")
    parser.add_argument("--wandb-project", default="agentevolver")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    # Teacher collection is a production experiment: silently falling back to
    # an untracked offline run is not an accepted execution mode.
    parser.add_argument("--wandb-mode", choices=["online"], default="online")
    parser.add_argument("--contract-only", action="store_true")
    return parser.parse_args(argv)


def _source_config_hash(config: DictConfig) -> str:
    return canonical_json_hash(OmegaConf.to_container(config, resolve=True))


def build_contract(
    *,
    args: argparse.Namespace,
    source_config: DictConfig,
    student_contract: Mapping[str, Any],
    task_manifest: Mapping[str, Any],
    tokenizer_manifest: Mapping[str, Any],
    selected_tasks: Sequence[str],
    temperature: float,
    top_p: Optional[float],
) -> Dict[str, Any]:
    selected_payload = ("\n".join(selected_tasks) + "\n").encode("utf-8")
    return {
        "schema_version": SCHEMA_VERSION,
        "teacher": {
            "model": args.model,
            "api_base": args.api_base.rstrip("/"),
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": int(source_config.actor_rollout_ref.rollout.response_length),
            "collect_log_prob": False,
        },
        "student_contract": json_safe(student_contract),
        "student_config_path": str(args.config.resolve()),
        "resolved_student_config_sha256": _source_config_hash(source_config),
        "collection_override": {
            "context_management.snapshot_training": False,
            "env_service.env_url": args.env_url.rstrip("/"),
        },
        "task_manifest": dict(task_manifest),
        "selected_task_count": len(selected_tasks),
        "selected_tasks_ordered_newline_sha256": sha256_bytes(selected_payload),
        "rollouts_per_task": int(args.rollouts_per_task),
        "collection_policy": {
            "max_attempts_per_rollout_total_across_resumes": int(
                args.max_attempts_per_rollout
            ),
            "max_workers": int(args.max_workers),
            "api_max_retries_per_decision": int(args.api_max_retries),
            "api_timeout_seconds": float(args.api_timeout),
            "wandb_mode": str(args.wandb_mode),
        },
        "student_tokenizer": dict(tokenizer_manifest),
        "store_prompt_messages": bool(args.store_prompt_messages),
        # 传入 env 以便为新环境附加各自的实现哈希(AF/WS 键集不变)。
        "implementation_sha256": implementation_hashes(
            str(student_contract["environment"])
        ),
    }


def _metric_accumulator() -> Dict[str, float]:
    return {
        "completed_work_items": 0.0,
        "successful_work_items": 0.0,
        "failed_work_items": 0.0,
        "trajectory_attempts": 0.0,
        "api_prompt_tokens": 0.0,
        "api_completion_tokens": 0.0,
        "api_total_tokens": 0.0,
        "api_cost": 0.0,
        "api_latency_ms": 0.0,
        "api_retry_count": 0.0,
    }


def run(args: argparse.Namespace) -> int:
    if args.rollouts_per_task <= 0:
        raise ValueError("--rollouts-per-task must be positive")
    if args.max_attempts_per_rollout <= 0:
        raise ValueError("--max-attempts-per-rollout must be positive")
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive")

    source_config = compose_student_config(args.config)
    student_contract = validate_student_contract(source_config)
    env_name = student_contract["environment"]
    task_ids, task_manifest = load_and_validate_task_file(
        args.task_file,
        env_name=env_name,
        task_seed=args.task_seed,
        expected_count=args.expected_task_count,
    )
    max_tasks = len(task_ids) if args.max_tasks is None else int(args.max_tasks)
    if max_tasks <= 0 or max_tasks > len(task_ids):
        raise ValueError(f"invalid --max-tasks={args.max_tasks}")
    selected_tasks = task_ids[:max_tasks]
    if not args.skip_live_profile_check:
        verify_live_task_profile(args.env_url, env_name, task_ids, args.task_seed)

    tokenizer, tokenizer_manifest = load_student_tokenizer(
        student_contract["student_model_path"]
    )
    rollout = source_config.actor_rollout_ref.rollout
    temperature = (
        float(args.temperature)
        if args.temperature is not None
        else float(getattr(rollout, "temperature", 0.9))
    )
    top_p = args.top_p
    if top_p is None:
        configured_top_p = getattr(rollout, "top_p", None)
        top_p = float(configured_top_p) if configured_top_p is not None else 1.0

    contract = build_contract(
        args=args,
        source_config=source_config,
        student_contract=student_contract,
        task_manifest=task_manifest,
        tokenizer_manifest=tokenizer_manifest,
        selected_tasks=selected_tasks,
        temperature=temperature,
        top_p=top_p,
    )
    contract_sha256 = canonical_json_hash(contract)
    resolved_config_sha256 = _source_config_hash(source_config)
    logger.info(
        f"contract verified env={env_name} tasks={len(selected_tasks)}/"
        f"{len(task_ids)} task_sha={task_manifest['ordered_newline_sha256']} "
        f"contract_sha={contract_sha256}"
    )
    if args.contract_only:
        print(
            json.dumps(
                {
                    "environment": env_name,
                    "contract_sha256": contract_sha256,
                    "task_manifest": task_manifest,
                    "student_contract": student_contract,
                    "student_tokenizer": tokenizer_manifest,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    output = args.output.resolve()
    attempts_path = Path(str(output) + ".attempts.jsonl")
    manifest_path = Path(str(output) + ".manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    with ExclusiveOutputLock(output):
        if output.exists() and output.stat().st_size > 0 and not args.resume:
            raise RuntimeError(
                f"refusing to overwrite non-empty output without --resume: {output}"
            )
        if (attempts_path.exists() or manifest_path.exists()) and not args.resume:
            raise RuntimeError(
                "refusing to mix with existing manifest/attempt ledger without --resume"
            )

        existing_manifest: Optional[Dict[str, Any]] = None
        if manifest_path.exists():
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing_manifest.get("contract_sha256") != contract_sha256:
                raise RuntimeError("resume manifest contract mismatch")
        elif args.resume and (output.exists() or attempts_path.exists()):
            raise RuntimeError("resume artifacts exist but the manifest is missing")

        allowed_tasks = set(selected_tasks)
        completed = scan_success_output(
            output,
            contract_sha256=contract_sha256,
            allowed_tasks=allowed_tasks,
        )
        next_attempt = scan_attempt_ledger(attempts_path, contract_sha256)

        try:
            import wandb
        except ImportError as error:  # pragma: no cover - production env has wandb
            raise RuntimeError("wandb is required for teacher collection") from error

        wandb_run_id = (
            existing_manifest.get("wandb_run_id")
            if existing_manifest
            else wandb.util.generate_id()
        )
        run_name = args.wandb_run_name or (
            f"{env_name}_deepseek_v4_flash_teacher_"
            f"t{len(selected_tasks)}_r{args.rollouts_per_task}"
        )
        manifest = {
            "manifest_version": 1,
            "created_at": (
                existing_manifest.get("created_at") if existing_manifest else utc_now()
            ),
            "updated_at": utc_now(),
            "contract_sha256": contract_sha256,
            "contract": contract,
            "wandb_run_id": wandb_run_id,
            "wandb_project": args.wandb_project,
            "wandb_run_name": run_name,
            "output": str(output),
            "attempts": str(attempts_path),
        }
        atomic_write_json(manifest_path, manifest)

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            id=wandb_run_id,
            resume="allow",
            mode=args.wandb_mode,
            config={
                "contract_sha256": contract_sha256,
                "environment": env_name,
                "teacher_model": args.model,
                "task_count": len(selected_tasks),
                "rollouts_per_task": args.rollouts_per_task,
                "max_attempts_per_rollout": args.max_attempts_per_rollout,
                "max_workers": args.max_workers,
                "student_config": str(args.config.resolve()),
                "student_config_sha256": resolved_config_sha256,
                "task_manifest_sha256": task_manifest["ordered_newline_sha256"],
                "prompt_length": student_contract["prompt_length"],
                "response_length": student_contract["response_length"],
                "max_model_len": student_contract["max_model_len"],
                "context_management": student_contract["context_management"],
            },
        )
        if wandb_run is None:
            raise RuntimeError("wandb.init returned no run")

        exit_code = 1
        try:
            api_key = load_api_key(args.api_key_source, args.api_key_env)
            teacher_llm = OpenAITeacherLLM(
                model_name=args.model,
                api_base=args.api_base,
                api_key=api_key,
                temperature=temperature,
                max_tokens=student_contract["response_length"],
                collect_log_prob=False,
                max_retries=args.api_max_retries,
                timeout=args.api_timeout,
            )
            del api_key

            runtime_config = collection_config(source_config, args.env_url)
            attempt_journal = JsonlJournal(attempts_path)
            success_journal = JsonlJournal(output)
            collector = TeacherCollector(
                config=runtime_config,
                tokenizer=tokenizer,
                teacher_llm=teacher_llm,
                contract_sha256=contract_sha256,
                task_manifest=task_manifest,
                resolved_config_sha256=resolved_config_sha256,
                tokenizer_manifest=tokenizer_manifest,
                teacher_model=args.model,
                temperature=temperature,
                top_p=top_p,
                max_attempts_per_rollout=args.max_attempts_per_rollout,
                store_prompt_messages=args.store_prompt_messages,
                attempt_journal=attempt_journal,
            )

            work_items: List[WorkItem] = []
            for task_id in selected_tasks:
                for rollout_index in range(args.rollouts_per_task):
                    rollout_id = (
                        f"{env_name}:{task_id}:deepseek-v4-flash:{rollout_index}"
                    )
                    if rollout_id in completed:
                        continue
                    work_items.append(
                        WorkItem(
                            task_id=task_id,
                            rollout_index=rollout_index,
                            rollout_id=rollout_id,
                            next_attempt=next_attempt.get(rollout_id, 0),
                        )
                    )

            total_target = len(selected_tasks) * args.rollouts_per_task
            logger.info(
                f"teacher collection start: completed={len(completed)} "
                f"remaining={len(work_items)} target={total_target} workers={args.max_workers}"
            )
            totals = _metric_accumulator()
            initial_success = len(completed)
            with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                future_to_item: Dict[Future, WorkItem] = {
                    pool.submit(collector.collect, item): item for item in work_items
                }
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    try:
                        result = future.result()
                    except Exception as error:  # defensive: collect() is fail-soft
                        result = WorkResult(
                            item=item,
                            success_record=None,
                            attempts=0,
                            final_error=safe_error_text(error),
                        )
                    totals["completed_work_items"] += 1
                    totals["trajectory_attempts"] += result.attempts
                    for key, value in result.metrics.items():
                        totals[key] = totals.get(key, 0.0) + float(value)
                    if result.success_record is not None:
                        success_journal.append(result.success_record)
                        completed[result.item.rollout_id] = result.success_record
                        totals["successful_work_items"] += 1
                    else:
                        totals["failed_work_items"] += 1

                    current_success = len(completed)
                    covered_tasks = len(
                        {str(record["task_id"]) for record in completed.values()}
                    )
                    wandb_run.log(
                        {
                            "collection/completed_work_items": totals[
                                "completed_work_items"
                            ],
                            "collection/successful_work_items_this_run": totals[
                                "successful_work_items"
                            ],
                            "collection/failed_work_items_this_run": totals[
                                "failed_work_items"
                            ],
                            "collection/successful_trajectories_total": current_success,
                            "collection/target_trajectories": total_target,
                            "collection/task_coverage": covered_tasks / len(selected_tasks),
                            "collection/trajectory_attempts": totals[
                                "trajectory_attempts"
                            ],
                            "api/prompt_tokens": totals["api_prompt_tokens"],
                            "api/completion_tokens": totals[
                                "api_completion_tokens"
                            ],
                            "api/total_tokens": totals["api_total_tokens"],
                            "api/cost": totals["api_cost"],
                            "api/latency_ms": totals["api_latency_ms"],
                            "api/retry_count": totals["api_retry_count"],
                        },
                        step=int(totals["completed_work_items"]),
                    )
                    if int(totals["completed_work_items"]) % 10 == 0:
                        logger.info(
                            f"progress {int(totals['completed_work_items'])}/"
                            f"{len(work_items)} success_total={current_success}/"
                            f"{total_target} coverage={covered_tasks}/{len(selected_tasks)}"
                        )

            audited = scan_success_output(
                output,
                contract_sha256=contract_sha256,
                allowed_tasks=allowed_tasks,
            )
            covered = {str(record["task_id"]) for record in audited.values()}
            missing_rollouts = total_target - len(audited)
            missing_tasks = len(selected_tasks) - len(covered)
            complete = missing_rollouts == 0 and missing_tasks == 0
            wandb_run.summary.update(
                {
                    "collection/final_trajectory_count": len(audited),
                    "collection/final_task_coverage_count": len(covered),
                    "collection/missing_rollouts": missing_rollouts,
                    "collection/missing_tasks": missing_tasks,
                    "collection/complete": complete,
                    "contract_sha256": contract_sha256,
                    "output": str(output),
                    "api/cost_this_run": totals["api_cost"],
                    "api/total_tokens_this_run": totals["api_total_tokens"],
                    "collection/new_successes_this_run": len(audited)
                    - initial_success,
                }
            )
            manifest["updated_at"] = utc_now()
            manifest["audit"] = {
                "trajectory_count": len(audited),
                "target_trajectory_count": total_target,
                "task_coverage_count": len(covered),
                "target_task_count": len(selected_tasks),
                "missing_rollouts": missing_rollouts,
                "missing_tasks": missing_tasks,
                "complete": complete,
            }
            atomic_write_json(manifest_path, manifest)
            if not complete:
                logger.error(
                    f"collection incomplete but safely resumable: "
                    f"missing_rollouts={missing_rollouts}, missing_tasks={missing_tasks}"
                )
                return 2
            exit_code = 0
            logger.info(
                f"collection complete and audited: trajectories={len(audited)} "
                f"tasks={len(covered)} output={output}"
            )
            return 0
        finally:
            wandb_run.finish(exit_code=exit_code)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    try:
        return run(args)
    except Exception as error:
        logger.exception(f"teacher collection failed closed: {safe_error_text(error)}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
