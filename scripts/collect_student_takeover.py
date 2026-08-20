#!/usr/bin/env python3
"""学生 rollout 采集器——entry-k 状态接管版(CATALYST P-B 试点,通路②)。

本脚本是 scripts/collect_student_rollouts.py 的薄 fork(同一 fork 谱系:
整文件复制后只改差异点,共享模块零改动)。母本的全部行为——成败都落盘、
军规级上下文渲染与 sha 审计、flock 账本、--resume——原样保留;行为差异是
**每任务先逐步重放教师成功轨迹的前 k 个 action 推进环境,学生从第 k+1 步
接管**(k = max(1, floor(entry_frac × n_teacher_decisions)))。

与母本 collect_student_rollouts.py 的全部差异(其余逐字节相同):
  1. 新增 CLI:--teacher-files(≥1 个 jsonl,schema =
     openrouter_teacher_trajectory_v2,含 messages 与 decision_trace)与
     --entry-frac(float ∈ (0,1),如 0.5)。每任务取教师文件中该 task_id
     的**第一条成功轨迹**(按文件顺序、文件内行序);无教师轨迹的选中
     任务启动即报错退出(fail-fast)。
  2. 接管机制:TakeoverAgentFlow(AgentFlow 薄子类)在 env create 后:
     (a) 逐步重放教师前 k 个 decision 的 action(从 completion_content
         post-</think> 段的最后一个 <action>...</action> 提取,与采集器/
         context policy 一致的解析习惯),把重放后的 env 观测与教师轨迹
         记录的对应 env 消息文本比对——不一致只计数不中止,计入输出
         记录字段 replay_divergence;
     (b) 以"教师 messages 的 init 前缀 + 前 k 对 (assistant, user) 消息"
         作为已发生历史 seed 进 CMT(历史 think 由 StructuredContextPolicy
         按契约自动剥除为 action-only,无需手工处理);
     (c) 学生从第 k+1 步接管,步数预算 = max_steps − k;成败按环境正常
         判定。渲染/sha 走母本正常军规链路,零特殊分支。
  3. 输出记录额外字段(顶层):entry_frac、k_steps、replay_divergence;
     metadata.takeover 记教师轨迹溯源与 init 前缀比对诊断。
     schema_version 因此改为 student_takeover_trajectory_v1(字段集合已
     与 student_rollout_trajectory_v1 不同,如实标注防误混)。
  4. manifest contract 新增 takeover 段(teacher 文件路径/sha256、
     entry_frac、教师 schema 版本)。
  5. 任务文件校验放宽为"seed-2026 全池 shuffle 的保序子序列"
     (CATALYST 试点任务集是 T_fail ∩ T_covered 的按 seed 序前 120,
     不是全课程前缀;母本的严格前缀校验会拒绝它)。子序列校验 +
     池 sha + 文件 sha 仍全部入 manifest,审计强度不降。

账本/manifest/resume/attempts 机制与母本一致。
"""

from __future__ import annotations

import argparse
import ast
import copy
import fcntl
import hashlib
import json
import math
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
# fork 差异:seed 教师历史需要直接构造 ExtendedMessage(与 save_init_input /
# save_env_output 的构造方式一致);tool→user 归一化沿用 AgentFlow 的工具。
from agentevolver.module.context_manager.cmt_linear import ExtendedMessage
from agentevolver.module.env_manager.env_worker import EnvWorker
from agentevolver.module.exp_manager.exp_manager import TrajExpConfig
from agentevolver.module.teacher.openai_teacher_llm import OpenAITeacherLLM
from agentevolver.schema.task import Task
from agentevolver.utils.utils import convert_tool_to_user_message


# fork 差异:接管版记录多出 entry_frac / k_steps / replay_divergence 顶层
# 字段,字段集合已与 student_rollout_trajectory_v1 不同,如实改版本号,
# 防止接管 rollout 误混入普通学生 rollout 分析池;其余字段结构一致。
SCHEMA_VERSION = "student_takeover_trajectory_v1"
ATTEMPT_SCHEMA_VERSION = "student_takeover_attempt_v1"
# 教师轨迹的期望 schema(flash 教师采集器 collect_openrouter_teacher_
# trajectories_dsv4.py 产物)。
TEACHER_SCHEMA_VERSION = "openrouter_teacher_trajectory_v2"
# fork 差异:默认指向本地 vLLM 服务的学生模型(served model name 通常就是
# 启动 vLLM 时的模型路径;若服务用了 --served-model-name 别名请用 --model 覆盖)。
DEFAULT_MODEL = "/projects_vol/gp_wangwy/models/Qwen3.5-4B"
DEFAULT_API_BASE = "http://127.0.0.1:8000/v1"
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
        # fork 差异:本地 vLLM 端点通常不做鉴权;env/文件都未提供时回退为
        # openai 客户端惯用的占位 key "EMPTY"(教师版在此 fail-fast)。
        return "EMPTY"
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
    # fork 差异:CATALYST 试点任务集(T_fail ∩ T_covered,按 seed-2026 序
    # 取前 N)不是全课程的前缀,母本的严格"前缀相等"校验会拒绝它。这里
    # 放宽为:文件必须是 canonical 池按 task_seed shuffle 后顺序的**保序
    # 子序列**(成员合法 + 相对顺序与 seed 序一致,仍然"不许挑题重排")。
    pool, source = canonical_task_pool(env_name)
    ordered_pool = list(pool)
    random.Random(task_seed).shuffle(ordered_pool)
    position = {tid: index for index, tid in enumerate(ordered_pool)}
    unknown = [tid for tid in task_ids if tid not in position]
    if unknown:
        raise RuntimeError(
            f"task file contains IDs outside the canonical {env_name} pool: "
            f"{unknown[:5]} (+{max(0, len(unknown) - 5)} more)"
        )
    ranks = [position[tid] for tid in task_ids]
    if ranks != sorted(ranks):
        raise RuntimeError(
            "task file is not an order-preserving subsequence of the "
            f"seed-{task_seed} shuffled canonical pool (re-ordering is not "
            "allowed)"
        )
    newline_payload = raw.encode("utf-8")
    sorted_payload = (
        "\n".join(sorted(task_ids, key=_membership_sort_key(env_name))) + "\n"
    ).encode("utf-8")
    manifest = {
        "environment": env_name,
        # fork 差异:算法名如实标注为子序列校验(供 manifest 审计辨识)
        "algorithm": (
            "python_random_seed_then_shuffle_then_ordered_subsequence"
        ),
        "task_seed": task_seed,
        "pool_count": len(pool),
        "pool_unique_count": len(set(pool)),
        "count": expected_count,
        "source_path": str(source.relative_to(PROJECT_ROOT)),
        "source_sha256": sha256_file(source),
        "ordered_newline_sha256": sha256_bytes(newline_payload),
        "ordered_json_sha256": canonical_json_hash(task_ids),
        "sorted_membership_sha256": sha256_bytes(sorted_payload),
        "task_file": str(path),
        "task_file_sha256": sha256_bytes(newline_payload),
    }
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
    # fork 差异:任务文件是 seed 序的保序子序列(见 load_and_validate_
    # task_file),这里同步改为校验"expected 是 live shuffle 序的保序
    # 子序列"——环境池成员或顺序漂移仍会被抓住。
    live_position = {tid: index for index, tid in enumerate(live)}
    missing = [tid for tid in expected if tid not in live_position]
    if missing:
        raise RuntimeError(
            "live environment profile is missing frozen-curriculum tasks: "
            f"{missing[:5]} (+{max(0, len(missing) - 5)} more)"
        )
    live_ranks = [live_position[tid] for tid in expected]
    if live_ranks != sorted(live_ranks):
        raise RuntimeError(
            "live environment profile ordering does not match the frozen "
            "curriculum subsequence"
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


def scan_rollout_output(
    path: Path,
    *,
    contract_sha256: str,
    allowed_tasks: set[str],
) -> Dict[str, Dict[str, Any]]:
    """扫描已落盘的学生 rollout(--resume 用)。

    fork 差异:教师版(scan_success_output)要求每条记录 success=True;
    学生 rollout 成败都落盘,这里只校验 schema/契约/任务归属/rollout_id
    唯一性,success 字段仅要求存在且为 bool。
    """
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
            if not isinstance(record.get("success"), bool):
                raise RuntimeError(
                    f"missing/non-bool success field at {path}:{line_number}"
                )
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


def trace_quality_metrics(
    trace: Sequence[Mapping[str, Any]], max_prompt: int
) -> Dict[str, int]:
    """fork 差异:替代教师版的 validate_success_trace(硬门禁,不合格即拒收)。

    学生 rollout 是判别器的"学生类"样本,截断/非法动作本身就是分布的一部分,
    不能拒收;这里只统计同一组质量诊断量,写进记录 metadata 供离线分析过滤。
    """
    quality = {
        "decision_count": len(trace),
        "over_prompt_budget_decisions": 0,
        "length_truncated_decisions": 0,
        "invalid_action_decisions": 0,
    }
    for decision in trace:
        if int(decision["prompt_token_count"]) > int(max_prompt):
            quality["over_prompt_budget_decisions"] += 1
        if bool(decision.get("truncated_by_length")):
            quality["length_truncated_decisions"] += 1
        if not _valid_tagged_action(str(decision.get("completion_content", ""))):
            quality["invalid_action_decisions"] += 1
    return quality


# ---------------------------------------------------------------------------
# fork 差异(核心):教师轨迹装载与 entry-k 接管计划
# ---------------------------------------------------------------------------


def _extract_tagged_action(completion_content: str) -> str:
    """从教师 completion 提取待重放 action(与采集器一致的解析习惯)。

    与 _valid_tagged_action / StructuredContextPolicy._action_only 同一习惯:
    只看 post-</think> 段,取**最后一个** <action>...</action> 的内容。
    成功教师轨迹的每个 decision 都必须有合法 action,否则 fail-fast。
    """
    post_think = str(completion_content).split("</think>")[-1]
    matches = list(
        re.finditer(
            r"<action>\s*(.*?)\s*</action>",
            post_think,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    if not matches or not matches[-1].group(1).strip():
        raise RuntimeError(
            "teacher decision has no well-formed <action>...</action> block; "
            "cannot replay"
        )
    return matches[-1].group(1).strip()


@dataclass(frozen=True)
class TakeoverPlan:
    """一个任务的接管计划(启动时从教师轨迹一次性构建,只读)。"""

    task_id: str
    entry_frac: float
    k_steps: int
    n_teacher_decisions: int
    teacher_rollout_id: str
    teacher_source_file: str
    # 教师 messages 的 init 前缀(到首条 user 任务说明为止,含之)
    init_messages: List[Dict[str, str]]
    # 前 k 个待重放 action(纯 action 文本,重放时包回 <action> 标签)
    replay_actions: List[str]
    # 教师轨迹记录的前 k 个 env 响应原文(重放观测的比对基准)
    expected_observations: List[str]
    # 前 k 对 (assistant, user) 原文——seed 进 CMT 的已发生历史
    seed_pairs: List[tuple]


def load_teacher_successes(
    files: Sequence[Path],
    *,
    env_name: str,
    selected_tasks: Sequence[str],
) -> tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """扫描教师 jsonl,每个选中任务取第一条成功轨迹(按文件序、行序)。

    返回 {task_id: {"record": 教师记录, "source_file": 路径}} 与
    per-file manifest 片段;任何选中任务无成功教师轨迹即 fail-fast。
    """
    wanted = set(selected_tasks)
    chosen: Dict[str, Dict[str, Any]] = {}
    file_manifests: List[Dict[str, Any]] = []
    for file_path in files:
        file_path = file_path.expanduser().resolve()
        if not file_path.is_file():
            raise FileNotFoundError(f"teacher file does not exist: {file_path}")
        file_manifests.append(
            {
                "path": str(file_path),
                "sha256": sha256_file(file_path),
            }
        )
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise RuntimeError(
                        f"blank line in teacher file {file_path}:{line_number}"
                    )
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"invalid teacher JSON at {file_path}:{line_number}: "
                        f"{error}"
                    ) from error
                if record.get("schema_version") != TEACHER_SCHEMA_VERSION:
                    raise RuntimeError(
                        f"teacher schema mismatch at {file_path}:{line_number}: "
                        f"{record.get('schema_version')!r} != "
                        f"{TEACHER_SCHEMA_VERSION!r}"
                    )
                if str(record.get("environment")) != env_name:
                    raise RuntimeError(
                        f"teacher environment mismatch at "
                        f"{file_path}:{line_number}: "
                        f"{record.get('environment')!r} != {env_name!r}"
                    )
                task_id = _canonical_task_id(env_name, record.get("task_id"))
                if task_id not in wanted or task_id in chosen:
                    continue
                if not bool(record.get("success")):
                    continue
                chosen[task_id] = {
                    "record": record,
                    "source_file": str(file_path),
                }
    missing = [tid for tid in selected_tasks if tid not in chosen]
    if missing:
        raise RuntimeError(
            f"no successful teacher trajectory for {len(missing)} selected "
            f"task(s): {missing[:10]}{' ...' if len(missing) > 10 else ''} "
            "(fail-fast, refusing to run without teacher state)"
        )
    return chosen, file_manifests


def build_takeover_plan(
    task_id: str,
    teacher: Mapping[str, Any],
    *,
    entry_frac: float,
    max_steps: int,
) -> TakeoverPlan:
    """从一条成功教师轨迹构建 entry-k 接管计划。

    教师 messages 布局(母采集器 to_role_content_raw + remove_last_context
    的产物):init 前缀(以首条 user 任务说明收尾)+ (assistant, user) 交替,
    最后一条为 assistant(末次 env 响应已被 remove_last_context 移除),即
    n 个 assistant、n−1 个 user 对应 n = len(decision_trace) 个 decision。
    """
    record = teacher["record"]
    messages = list(record["messages"])
    trace = list(record["decision_trace"])
    n_decisions = len(trace)
    if n_decisions <= 0:
        raise RuntimeError(f"teacher trajectory for task {task_id} is empty")
    first_user = next(
        (
            index
            for index, message in enumerate(messages)
            if str(message.get("role")) == "user"
        ),
        None,
    )
    if first_user is None:
        raise RuntimeError(
            f"teacher trajectory for task {task_id} has no user init message"
        )
    init_len = first_user + 1
    body = messages[init_len:]
    if len(body) != 2 * n_decisions - 1:
        raise RuntimeError(
            f"teacher trajectory for task {task_id} has unexpected layout: "
            f"{len(body)} post-init messages for {n_decisions} decisions "
            f"(expected {2 * n_decisions - 1})"
        )
    for offset, message in enumerate(body):
        expected_role = "assistant" if offset % 2 == 0 else "user"
        if str(message.get("role")) != expected_role:
            raise RuntimeError(
                f"teacher trajectory for task {task_id} breaks the "
                f"assistant/user alternation at post-init offset {offset}"
            )
    # k = max(1, floor(entry_frac × n_decisions));必须给学生留出至少
    # 一步接管空间(k ≤ n−1,单步教师轨迹在 entry_frac<1 下无法接管),
    # 且不能吃光学生步数预算(k < max_steps)。
    k_steps = max(1, math.floor(float(entry_frac) * n_decisions))
    if k_steps > n_decisions - 1:
        raise RuntimeError(
            f"task {task_id}: k={k_steps} leaves no teacher env feedback to "
            f"seed (teacher has only {n_decisions} decision(s)); trajectory "
            "too short for this --entry-frac"
        )
    if k_steps >= int(max_steps):
        raise RuntimeError(
            f"task {task_id}: k={k_steps} exhausts the student step budget "
            f"(max_steps={max_steps})"
        )
    replay_actions = [
        _extract_tagged_action(trace[index]["completion_content"])
        for index in range(k_steps)
    ]
    expected_observations = [
        str(body[2 * index + 1].get("content", "")) for index in range(k_steps)
    ]
    seed_pairs = [
        (
            str(body[2 * index].get("content", "")),
            str(body[2 * index + 1].get("content", "")),
        )
        for index in range(k_steps)
    ]
    init_messages = [
        {"role": str(message.get("role")), "content": str(message.get("content"))}
        for message in messages[:init_len]
    ]
    return TakeoverPlan(
        task_id=task_id,
        entry_frac=float(entry_frac),
        k_steps=k_steps,
        n_teacher_decisions=n_decisions,
        teacher_rollout_id=str(record.get("rollout_id", "")),
        teacher_source_file=str(teacher["source_file"]),
        init_messages=init_messages,
        replay_actions=replay_actions,
        expected_observations=expected_observations,
        seed_pairs=seed_pairs,
    )


class TakeoverAgentFlow(AgentFlow):
    """fork 差异(核心):entry-k 状态接管的 AgentFlow 薄子类。

    在母本 AgentFlow.execute 之前完成三件事:
      (1) 逐步重放教师前 k 个 action 推进 env 实例,观测与教师记录逐条
          文本比对(不一致只累加 replay_divergence,不中止);
      (2) 通过 save_init_input 挂钩把"教师 init 前缀 + 前 k 对
          (assistant, user)"seed 进 CMT 的 full_context——教师历史 think
          由 StructuredContextPolicy 按契约剥为 action-only,渲染/sha 走
          母本正常军规链路;
      (3) 步数预算收缩为 max_steps − k,学生从第 k+1 步接管。
    """

    def __init__(self, *, plan: TakeoverPlan, **kwargs: Any):
        super().__init__(**kwargs)
        self.plan = plan
        base_max_steps = int(self.max_steps)
        # 学生步数预算 = max_steps − k(重放的 k 步算教师的,不算学生的)
        self.max_steps = base_max_steps - int(plan.k_steps)
        if self.max_steps < 1:
            raise RuntimeError(
                f"takeover leaves no student steps: k={plan.k_steps} >= "
                f"max_steps={base_max_steps}"
            )
        self.replay_divergence = 0
        self.replay_steps_done = 0
        self.init_prefix_matches_live: Optional[bool] = None

    def execute(self, context_manager, init_messages, env, instance_id, **kwargs):  # type: ignore[override]
        plan = self.plan
        # 诊断:live env 的 init 与教师 init 前缀是否逐字一致(确定性环境
        # 应当一致;不一致仅记录,不计入 replay_divergence)。
        live_init = [
            {"role": str(m.get("role")), "content": str(m.get("content"))}
            for m in init_messages
        ]
        self.init_prefix_matches_live = live_init == plan.init_messages

        # (1) 逐步重放教师前 k 个 action。发送形态与 context policy 的
        # action-only 规范一致(<action>\n…\n</action>);env 端(react_tags)
        # 用同样的正则提取 action。传输层异常按母本习惯向上抛,走
        # infrastructure 重试路径。
        for step_index, action in enumerate(plan.replay_actions):
            env_output = env.step(
                instance_id,
                {"content": f"<action>\n{action}\n</action>", "role": "assistant"},
            )
            states = env_output["state"]
            assert len(states) == 1
            state = states[0]
            if state.get("role") == "tool":
                state = convert_tool_to_user_message(
                    state, self.tokenizer, format="qwen"
                )
            observed = str(state.get("content", ""))
            expected = str(plan.expected_observations[step_index])
            if observed != expected:
                # 不一致只计数不中止(试点预注册:确定性已验,divergence
                # 是数据质量观测量而非硬门禁)。
                self.replay_divergence += 1
                logger.warning(
                    f"takeover replay divergence task={plan.task_id} "
                    f"replay_step={step_index}"
                )
            self.replay_steps_done = step_index + 1
            if bool(env_output.get("is_terminated")):
                # 教师前缀不应终结 episode;终结说明重放假设被破坏,
                # 属基础设施/契约异常,fail-fast 交给上层重试记账。
                raise RuntimeError(
                    f"environment terminated during teacher replay at step "
                    f"{step_index + 1}/{plan.k_steps} (task {plan.task_id})"
                )

        # (2) seed 已发生历史:挂钩 save_init_input——母本 execute 里
        # save_init_input 断言 full_context 为空,故只能在其完成后立刻
        # 追加教师的 k 对 (assistant, user)。构造方式与 save_llm_output /
        # save_env_output 一致(author=llm/env、env 侧带 clip 上限);
        # 渲染时 StructuredContextPolicy 自动把历史 assistant 剥为
        # action-only。
        original_save_init = context_manager.save_init_input
        tokenizer = self.tokenizer

        def _seeded_save_init(init_input_arr: List[dict], add_nothink: bool = False):
            original_save_init(init_input_arr, add_nothink)
            for assistant_content, user_content in plan.seed_pairs:
                context_manager.full_context.append(
                    ExtendedMessage(
                        author="llm",
                        role="assistant",
                        content=str(assistant_content),
                        token_generator="auto",
                        tokenizer=tokenizer,
                    )
                )
                context_manager.full_context.append(
                    ExtendedMessage(
                        author="env",
                        role="user",
                        content=str(user_content),
                        clip=True,
                        clip_token_limit=context_manager.max_env_output_length,
                        token_generator="auto",
                        tokenizer=tokenizer,
                    )
                )

        context_manager.save_init_input = _seeded_save_init

        # (3) 以教师 init 前缀为初始消息进入母本 episode 循环,学生从
        # 第 k+1 步接管(预算已在 __init__ 收缩)。
        return super().execute(
            context_manager=context_manager,
            init_messages=copy.deepcopy(plan.init_messages),
            env=env,
            instance_id=instance_id,
            **kwargs,
        )


@dataclass(frozen=True)
class WorkItem:
    task_id: str
    rollout_index: int
    rollout_id: str
    next_attempt: int = 0


@dataclass
class WorkResult:
    item: WorkItem
    # fork 差异:字段名沿用教师版,但这里的含义是"已落盘的轨迹记录
    # (成功/失败皆可)";None 仅表示该尝试因基础设施异常没产出轨迹。
    success_record: Optional[Dict[str, Any]]
    attempts: int
    metrics: Dict[str, float] = field(default_factory=dict)
    final_error: Optional[str] = None


class StudentRolloutCollector:
    """fork 差异:由 TeacherCollector 改名。

    构造参数与教师版一致(teacher_llm/teacher_model 等字段名保持不动以
    最小化 diff)——但这里的 OpenAITeacherLLM 客户端连的是**学生模型**的
    本地 vLLM OpenAI 兼容端点,teacher_model 存的也是学生模型名。
    """

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
        plans: Mapping[str, TakeoverPlan],
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
        # fork 差异:每任务的 entry-k 接管计划(启动时已 fail-fast 校验全覆盖)
        self.plans = dict(plans)

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
        """采集一个 rollout 槽位;轨迹无论成败都落盘,无 stop-on-success 重试。

        下面的 for 循环只有在"尝试抛异常、没产出轨迹"时才会进入下一轮
        (消耗 --max-attempts-per-rollout 剩余额度;默认 1,即固定跑满
        1 次——每任务 x 每槽位恰好 1 条轨迹)。
        """
        # fork 差异(核心):任务无接管计划直接抛错(启动时已整体校验,
        # 这里是最后一道 fail-fast 防线;放在记账循环之前,不消耗额度)。
        plan = self.plans.get(item.task_id)
        if plan is None:
            raise RuntimeError(
                f"no takeover plan for task {item.task_id} "
                "(fail-fast, refusing to run without teacher state)"
            )
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
                # fork 差异(核心):AgentFlow → TakeoverAgentFlow,
                # 重放教师前 k 步 + seed 历史 + 收缩步数预算
                flow = TakeoverAgentFlow(
                    plan=plan,
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
                # fork 差异(核心):教师版此处 stop-on-success——失败 continue
                # 重试、成功才过质量门禁 + 落盘;本 fork 成败一视同仁:轨迹
                # 已产出即记账并落盘,该槽位随之完成(账本 status 统一记为
                # rollout_recorded,success 字段如实记录成败)。
                self.attempt_journal.append(
                    self._attempt_event(
                        item,
                        attempt_index,
                        event="attempt_finished",
                        status="rollout_recorded",
                        reward=float(reward.outcome if reward else 0.0),
                        success=success,
                        episode_end_reason=episode_end_reason,
                        decision_count=len(recorder.trace),
                        metrics=trace_metrics,
                        # fork 差异:接管诊断也入尝试账本
                        k_steps=plan.k_steps,
                        replay_divergence=int(flow.replay_divergence),
                    )
                )
                # fork 差异:教师版的 validate_success_trace 硬门禁改为
                # 质量诊断统计,随记录写入 metadata.trace_quality。
                quality = trace_quality_metrics(
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
                    # fork 差异:失败轨迹的 reward/success 也如实落盘
                    "reward": float(reward.outcome) if reward is not None else 0.0,
                    "success_rate": (
                        float(getattr(reward, "success_rate", 0.0))
                        if reward is not None
                        else 0.0
                    ),
                    "success": success,
                    "is_terminated": bool(trajectory.is_terminated),
                    # 字段名沿用教师 schema(读取端零改动);值为学生模型名
                    "teacher_model": self.teacher_model,
                    # fork 差异(核心):entry-k 接管的三个顶层字段
                    "entry_frac": float(plan.entry_frac),
                    "k_steps": int(plan.k_steps),
                    "replay_divergence": int(flow.replay_divergence),
                    "decision_trace": recorder.trace,
                    "metadata": {
                        "is_teacher": False,  # fork 差异:这是学生 rollout
                        "policy_role": "student_takeover",
                        # fork 差异:接管溯源与诊断
                        "takeover": {
                            "teacher_rollout_id": plan.teacher_rollout_id,
                            "teacher_source_file": plan.teacher_source_file,
                            "n_teacher_decisions": plan.n_teacher_decisions,
                            "replay_steps_done": int(flow.replay_steps_done),
                            "init_prefix_matches_live": (
                                flow.init_prefix_matches_live
                            ),
                            "student_step_budget": int(flow.max_steps),
                        },
                        "has_log_prob": False,
                        "collected_at": utc_now(),
                        "attempt_index": attempt_index,
                        "attempt_count_for_slot": attempts_made,
                        "episode_end_reason": episode_end_reason,
                        "trace_quality": quality,
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
                    f"student rollout attempt failed task={item.task_id} "
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
    # fork 差异:第一项指向本 fork 自身(其余共享模块与母本相同)
    relative_paths = [
        "scripts/collect_student_takeover.py",
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
        description=(
            "采集学生策略的 entry-k 状态接管 rollout:重放教师前 k 步推进 "
            "env,学生从第 k+1 步接管,成败都落盘"
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    # fork 差异:教师轨迹文件(≥1 个 jsonl,openrouter_teacher_trajectory_v2)
    # 与接管入口比例 entry_frac ∈ (0,1)
    parser.add_argument(
        "--teacher-files", type=Path, nargs="+", required=True
    )
    parser.add_argument("--entry-frac", type=float, required=True)
    parser.add_argument("--env-url", required=True)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-task-count", type=int, default=1600)
    parser.add_argument("--task-seed", type=int, default=2026)
    parser.add_argument("--max-tasks", type=int, default=None)
    # fork 差异:每任务默认只跑 1 条 rollout(判别器学生类逐任务 1 条即可)
    parser.add_argument("--rollouts-per-task", type=int, default=1)
    # fork 差异:语义改为"每槽位固定跑满 1 次"——只有尝试抛异常没产出轨迹
    # 时才可能消耗 >1 的额度;默认 1 即每任务恰好 1 条轨迹。
    parser.add_argument("--max-attempts-per-rollout", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=4)
    # fork 差异:默认指向本地 vLLM 学生模型(见文件头常量注释)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    # fork 差异:本地 vLLM 无鉴权,key 默认缺省(load_api_key 回退 "EMPTY")
    parser.add_argument("--api-key-source", type=Path, default=None)
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    # 默认 None → 取学生 config 的 rollout.temperature / rollout.top_p
    # (与训练 rollout 同分布,判别器"学生类"的正确采样分布)。
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
    # fork 差异:学生 rollout 是离线分析类采集,不强制在线 wandb;
    # 默认 disabled(完全不启动 wandb),需要追踪时可显式开 offline/online。
    parser.add_argument(
        "--wandb-mode",
        choices=["disabled", "offline", "online"],
        default="disabled",
    )
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
    takeover_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    selected_payload = ("\n".join(selected_tasks) + "\n").encode("utf-8")
    return {
        "schema_version": SCHEMA_VERSION,
        # fork 差异(核心):entry-k 接管契约(teacher 文件 sha /
        # entry_frac / 教师 schema)——进 contract_sha256,与普通学生
        # rollout 采集不可混淆。
        "takeover": dict(takeover_manifest),
        # fork 差异:契约里的采样端不再是 "teacher" 而是学生策略本身
        "policy": {
            "role": "student_rollout",
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
            # fork 差异:成败都落盘、无 stop-on-success(写进契约防误读)
            "record_all_attempts": True,
            "stop_on_success": False,
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


class _NullWandbRun:
    """fork 差异:--wandb-mode=disabled 时替代 wandb run 的空实现。

    只实现本脚本用到的接口(log / summary.update / finish),全部为空操作,
    使 run() 主体无需为"不开 wandb"分叉两套代码。
    """

    class _Summary(dict):
        def update(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
            dict.update(self, *args, **kwargs)

    def __init__(self) -> None:
        self.summary = self._Summary()

    def log(self, *args: Any, **kwargs: Any) -> None:
        return None

    def finish(self, exit_code: int = 0) -> None:
        return None


def _metric_accumulator() -> Dict[str, float]:
    return {
        "completed_work_items": 0.0,
        # fork 差异:successful_work_items 含义变为"已落盘轨迹的槽位数";
        # env_success_work_items 才是环境判定成功(reward>0)的条数。
        "env_success_work_items": 0.0,
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
    # fork 差异:entry_frac 必须在 (0,1) 开区间——0 无重放,1 无接管空间
    if not (0.0 < float(args.entry_frac) < 1.0):
        raise ValueError("--entry-frac must be in the open interval (0, 1)")

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
    # 默认温度/Top-p 直接继承学生 config 的 rollout 采样参数(alfworld 0.9 /
    # webshop 0.6)——与训练 rollout 同分布,即判别器"学生类"的正确采样分布。
    # 该逻辑与教师版相同,此处仅加注释强调其对判别器分析的意义。
    temperature = (
        float(args.temperature)
        if args.temperature is not None
        else float(getattr(rollout, "temperature", 0.9))
    )
    top_p = args.top_p
    if top_p is None:
        configured_top_p = getattr(rollout, "top_p", None)
        top_p = float(configured_top_p) if configured_top_p is not None else 1.0

    # fork 差异(核心):装载教师成功轨迹并为每个选中任务构建接管计划
    # (fail-fast:无成功教师轨迹 / 轨迹过短 / 布局异常都在启动期报错)。
    teacher_by_task, teacher_file_manifests = load_teacher_successes(
        list(args.teacher_files),
        env_name=env_name,
        selected_tasks=selected_tasks,
    )
    plans: Dict[str, TakeoverPlan] = {
        task_id: build_takeover_plan(
            task_id,
            teacher_by_task[task_id],
            entry_frac=float(args.entry_frac),
            max_steps=int(student_contract["max_steps"]),
        )
        for task_id in selected_tasks
    }
    takeover_manifest = {
        "teacher_schema_version": TEACHER_SCHEMA_VERSION,
        "teacher_files": teacher_file_manifests,
        "entry_frac": float(args.entry_frac),
        "k_steps_by_task": {
            task_id: plan.k_steps for task_id, plan in plans.items()
        },
    }
    contract = build_contract(
        args=args,
        source_config=source_config,
        student_contract=student_contract,
        task_manifest=task_manifest,
        tokenizer_manifest=tokenizer_manifest,
        selected_tasks=selected_tasks,
        temperature=temperature,
        top_p=top_p,
        takeover_manifest=takeover_manifest,
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
        completed = scan_rollout_output(
            output,
            contract_sha256=contract_sha256,
            allowed_tasks=allowed_tasks,
        )
        next_attempt = scan_attempt_ledger(attempts_path, contract_sha256)

        # fork 差异:wandb 可选(默认 disabled)。disabled 时以本地生成的
        # run id 占位(仅写入 manifest 供追溯),run 对象用空实现替代。
        use_wandb = args.wandb_mode != "disabled"
        if use_wandb:
            try:
                import wandb
            except ImportError as error:
                raise RuntimeError(
                    "--wandb-mode=offline/online 需要安装 wandb"
                ) from error
            generated_run_id = wandb.util.generate_id()
        else:
            generated_run_id = f"local-{uuid.uuid4().hex[:12]}"

        wandb_run_id = (
            existing_manifest.get("wandb_run_id")
            if existing_manifest
            else generated_run_id
        )
        # fork 差异:run 名标注 takeover 与 entry_frac
        run_name = args.wandb_run_name or (
            f"{env_name}_student_takeover_e{args.entry_frac:g}_"
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

        if use_wandb:
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
                    "student_model": args.model,  # fork 差异:采样端是学生
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
        else:
            wandb_run = _NullWandbRun()

        exit_code = 1
        try:
            api_key = load_api_key(args.api_key_source, args.api_key_env)
            # fork 差异:客户端类沿用教师模块(OpenAI 兼容通用封装),
            # 但端点/模型是本地 vLLM 上的学生模型。
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
            # fork 差异:输出账本收的是"全部已产出轨迹"(成败皆有),
            # 变量名沿用教师版以最小化 diff。
            success_journal = JsonlJournal(output)
            collector = StudentRolloutCollector(
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
                plans=plans,
            )

            work_items: List[WorkItem] = []
            for task_id in selected_tasks:
                for rollout_index in range(args.rollouts_per_task):
                    # fork 差异:rollout_id 标注为 takeover,防与普通学生
                    # rollout 的槽位 id 撞名(合并分析时可直接区分)
                    rollout_id = f"{env_name}:{task_id}:takeover:{rollout_index}"
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
                f"student rollout collection start: completed={len(completed)} "
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
                        # fork 差异:单独累计环境判定成功(success=True)的条数
                        totals["env_success_work_items"] += float(
                            bool(result.success_record.get("success"))
                        )
                    else:
                        totals["failed_work_items"] += 1

                    # fork 差异:completed 是"已落盘轨迹"(成败皆有);
                    # env_success_total 才是其中环境成功的条数。
                    current_recorded = len(completed)
                    env_success_total = sum(
                        1
                        for record in completed.values()
                        if bool(record.get("success"))
                    )
                    covered_tasks = len(
                        {str(record["task_id"]) for record in completed.values()}
                    )
                    wandb_run.log(
                        {
                            "collection/completed_work_items": totals[
                                "completed_work_items"
                            ],
                            "collection/recorded_work_items_this_run": totals[
                                "successful_work_items"
                            ],
                            "collection/infra_failed_work_items_this_run": totals[
                                "failed_work_items"
                            ],
                            "collection/recorded_trajectories_total": current_recorded,
                            "collection/env_success_total": env_success_total,
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
                            f"{len(work_items)} recorded={current_recorded}/"
                            f"{total_target} env_success={env_success_total} "
                            f"coverage={covered_tasks}/{len(selected_tasks)}"
                        )

            audited = scan_rollout_output(
                output,
                contract_sha256=contract_sha256,
                allowed_tasks=allowed_tasks,
            )
            covered = {str(record["task_id"]) for record in audited.values()}
            # fork 差异:审计对象是"已落盘轨迹"(成败皆有);另报环境成功率,
            # 供判别器分析时了解学生类的成功占比。
            env_success_count = sum(
                1 for record in audited.values() if bool(record.get("success"))
            )
            missing_rollouts = total_target - len(audited)
            missing_tasks = len(selected_tasks) - len(covered)
            complete = missing_rollouts == 0 and missing_tasks == 0
            wandb_run.summary.update(
                {
                    "collection/final_trajectory_count": len(audited),
                    "collection/final_env_success_count": env_success_count,
                    "collection/final_env_success_rate": (
                        env_success_count / len(audited) if audited else 0.0
                    ),
                    "collection/final_task_coverage_count": len(covered),
                    "collection/missing_rollouts": missing_rollouts,
                    "collection/missing_tasks": missing_tasks,
                    "collection/complete": complete,
                    "contract_sha256": contract_sha256,
                    "output": str(output),
                    "api/cost_this_run": totals["api_cost"],
                    "api/total_tokens_this_run": totals["api_total_tokens"],
                    "collection/new_records_this_run": len(audited)
                    - initial_success,
                }
            )
            manifest["updated_at"] = utc_now()
            manifest["audit"] = {
                "trajectory_count": len(audited),
                "env_success_count": env_success_count,
                "env_success_rate": (
                    env_success_count / len(audited) if audited else 0.0
                ),
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
                f"env_success={env_success_count} tasks={len(covered)} "
                f"output={output}"
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
        logger.exception(
            f"student rollout collection failed closed: {safe_error_text(error)}"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
