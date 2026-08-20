"""CATALYST M1 训练侧集成(提示臂 / 分臂基线 / 去提示重放 BC / 治理层)。

设计文档:docs/design/CATALYST_IMPL_SPEC.md(2026-08-08 审定稿)。
零模仿不变式:教师 token 永不进损失——本模块的教师影响仅有两条通道:
  1. 提示臂:教师 think 摘要注入 prompt(prompt 不进损失);
  2. 重放 BC:提示臂**学生自写、环境盖章**的成功轨迹,去提示重渲染后作辅助 BC。

纯增量纪律:本模块只在 ``exp_manager.catalyst.enable=true`` 时被构造;
默认关闭时训练路径逐字节等价(tests/test_catalyst_default_off.py 作证)。

模块内容:
  * HINT_TEMPLATE 单一事实源(ast 提取采集器常量 + sha pin,规格 F8);
  * 试点同款 hint 清洗管线(规格 F9,已对试点产物 230/230 字节验证);
  * strip_hint_messages:与 analysis/catalyst_purity_score.py 同语义的剥除+断言;
  * CatalystHintBook / CatalystGovernor / CatalystReplayPool / CatalystRuntime。

重依赖(torch / ExtendedMessage / StructuredContextPolicy / Sample)全部函数内
惰性 import——scripts/build_catalyst_hints.py 只用清洗函数,不拉起训练栈。
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# HINT_TEMPLATE:以试点采集器源码为单一事实源(规格 F8)。
# ---------------------------------------------------------------------------
HINTED_COLLECTOR_PATH = (
    PROJECT_ROOT / "scripts" / "collect_student_rollouts_hinted.py"
)
# 试点 A1 manifest /contract/hint_injection/hint_template_sha256 的 pin 值。
# 任何一端(采集器源码 / 本模块)漂移都会在 load_hint_template() 处 fail-fast。
HINT_TEMPLATE_SHA256_PIN = (
    "e72d043eb44793852cf2b342697ffe899f6c60aed72b409e25199f1f1074efb0"
)
HINT_MARKER = "[Reference approach"

# 内嵌副本(2026-08-09):Ray 的 working_dir 只打包 launcher backup 的代码树
# (agentevolver/config/cookbook/external/runtime_files),**不含 scripts/**,
# 训练 worker 里 ast 提取会 FileNotFoundError。此处保留逐字节副本作回退;
# 两条路径都必须过同一个 sha pin 断言,任何一端漂移仍然 fail-fast。
_HINT_TEMPLATE_EMBEDDED = (
    "\n\n[Reference approach from a colleague]\n{hint}\n"
    "[End of reference. The reference may be imperfect; "
    "solve the task yourself.]"
)

# 重放样本的 data_id 基址(A1 防御:与真实 group_ids 撞号时整体偏移)。
CATALYST_REPLAY_DATA_ID_BASE = 100000


def load_hint_template(path: Path = HINTED_COLLECTOR_PATH) -> str:
    """ast 提取采集器模块级 HINT_TEMPLATE 常量并断言 sha pin。

    与 analysis/catalyst_purity_score.py::load_hint_template 同法(避免 import
    采集器拉起 env_worker/exp_manager 重依赖);额外多一道 sha pin 断言,保证
    训练侧注入与试点采集逐字节同模板。
    """
    def _assert_pin(value: str, origin: str) -> str:
        if not isinstance(value, str) or value.count("{hint}") != 1:
            raise RuntimeError(f"HINT_TEMPLATE 形状异常({origin}): {value!r}")
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        if digest != HINT_TEMPLATE_SHA256_PIN:
            raise RuntimeError(
                f"HINT_TEMPLATE sha256 与试点 pin 不符({origin}):"
                f"{digest} != {HINT_TEMPLATE_SHA256_PIN};"
                "模板被改动过,训练侧注入拒绝启动"
            )
        return value

    if not path.is_file():
        # Ray worker 内 scripts/ 不存在 → 用内嵌副本(同样过 sha pin)
        return _assert_pin(_HINT_TEMPLATE_EMBEDDED, "embedded")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == "HINT_TEMPLATE"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if not isinstance(value, str) or value.count("{hint}") != 1:
                raise RuntimeError(f"HINT_TEMPLATE 形状异常: {value!r}")
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
            if digest != HINT_TEMPLATE_SHA256_PIN:
                raise RuntimeError(
                    "HINT_TEMPLATE sha256 与试点 pin 不符:"
                    f"{digest} != {HINT_TEMPLATE_SHA256_PIN};"
                    "采集器模板被改动过,训练侧注入拒绝启动"
                )
            return value
    raise RuntimeError(f"HINT_TEMPLATE not found in {path}")


def hint_template_parts() -> Tuple[str, str]:
    """返回 (prefix, suffix);剥除逻辑与 purity 脚本共用这一拆分。"""
    template = load_hint_template()
    prefix, suffix = template.split("{hint}")
    if HINT_MARKER not in prefix:
        raise RuntimeError("HINT_MARKER 必须是 prefix 的子串")
    return prefix, suffix


def hint_sha256(hint_text: str) -> str:
    return hashlib.sha256(str(hint_text).encode("utf-8")).hexdigest()


def inject_hint_into_init_messages(
    init_messages: List[dict], hint_text: str
) -> List[dict]:
    """把 hint 注入 init 消息副本的首条 user 消息末尾(试点逐字节同构)。

    对齐 scripts/collect_student_rollouts_hinted.py::HintedAgentFlow.execute
    (L1029-1050):deepcopy → 首条 user 消息 content += HINT_TEMPLATE.format;
    无 user 消息 fail-fast。
    """
    if not str(hint_text).strip():
        raise RuntimeError("catalyst hint injection requires a non-empty hint")
    template = load_hint_template()
    init_messages = copy.deepcopy(init_messages)
    user_indices = [
        index
        for index, message in enumerate(init_messages)
        if str(message.get("role")) == "user"
    ]
    if not user_indices:
        raise RuntimeError(
            "cannot inject catalyst hint: init messages contain no user message"
        )
    target = user_indices[0]
    init_messages[target]["content"] = str(
        init_messages[target]["content"]
    ) + template.format(hint=str(hint_text).strip())
    return init_messages


def strip_hint_messages(messages: List[Dict[str, Any]]) -> None:
    """就地剥除 messages 中的 hint 块;断言失败抛 ValueError(军规)。

    语义逐条对齐 analysis/catalyst_purity_score.py::strip_hint_messages
    (L132-158):
      * 剥除 = content 中 [prefix 起, suffix 止](含两端)的整段;
      * 恰好剥了 1 条消息;
      * 剥后全部 messages 不再含 HINT_MARKER。
    """
    prefix, suffix = hint_template_parts()
    stripped = 0
    for index, msg in enumerate(messages):
        content = str(msg["content"])
        if HINT_MARKER not in content:
            continue
        start = content.find(prefix)
        if start < 0:
            raise ValueError(
                f"message {index}: 含 {HINT_MARKER!r} 但找不到完整 HINT_PREFIX"
            )
        end = content.find(suffix, start + len(prefix))
        if end < 0:
            raise ValueError(f"message {index}: 有 HINT_PREFIX 但缺 HINT_SUFFIX")
        msg["content"] = content[:start] + content[end + len(suffix):]
        stripped += 1
    if stripped != 1:
        raise ValueError(f"应恰好剥除 1 条消息,实际剥了 {stripped} 条")
    for index, msg in enumerate(messages):
        if HINT_MARKER in str(msg["content"]):
            raise ValueError(f"剥除后 message {index} 仍含 {HINT_MARKER!r}")


def arm_uid_values(
    group_ids: Sequence[Any], extras_array: Optional[Sequence[Any]]
) -> List[str]:
    """分臂基线(D1 uid 后缀方案)的 uid 构造。

    hint 臂样本 uid = f"{group_id}|h";entry 臂(v2)uid =
    f"{group_id}|e{rung}"(同任务同 rung 的接管 rollout 共享基线——不同
    rung 起点不同,不可比);其余(裸臂/缺 extras)保持 str(int(group_id))。
    现有 GRPO 分组函数按 uid 分组即自动得到 (task, arm) 分组的均值/方差。
    trainer fit 的 uid 构造点是唯一调用方。
    """
    uid_values: List[str] = []
    for row_index, gid in enumerate(group_ids):
        uid = str(int(gid))
        extra = (
            extras_array[row_index]
            if extras_array is not None and row_index < len(extras_array)
            else None
        )
        if isinstance(extra, Mapping):
            arm = extra.get("catalyst_arm")
            if arm == "hint":
                uid = f"{uid}|h"
            elif arm == "entry":
                uid = f"{uid}|e{int(extra.get('catalyst_entry_rung', 0))}"
        uid_values.append(uid)
    return uid_values


def compute_replay_bc_terms(
    log_prob: Any,
    entropy: Any,
    *,
    w_cap: float = 1.0,
    phi_tau: float = 1.0,
) -> Tuple[Any, Any]:
    """重放 BC 的逐 token 权重与损失矩阵(单一事实源,actor 与测试共用)。

    w = stop-grad[min(w_cap, exp((logπ + H)/τ))](φ 零点自校准:对学生自身
    样本 E[φ]=0 → w≈1,审计只挡个别漂移点);返回 (w, w·(−logπ))。
    w 在 no_grad 下计算 → 反传梯度恰为 −w·∂logπ,无二阶项(规格 T3)。
    """
    import torch

    with torch.no_grad():
        phi = (log_prob + entropy) / float(phi_tau)
        w = torch.clamp(torch.exp(phi), max=float(w_cap))
    return w, w * (-log_prob)


# ---------------------------------------------------------------------------
# hint 清洗管线(规格 F9;对试点产物 AF 120/120 + WS 110/110 字节验证)。
# ---------------------------------------------------------------------------
HINT_CLEAN_VERSION = "catalyst_hint_clean/1.0.0"
HINT_MAX_CHARS = 5000
_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
# flash 家族草稿块(SIEVE M3 口癖):think 内混入的 "response<action>…</action>"
_FLASH_DRAFT_RE = re.compile(r"response<action>.*?</action>", re.DOTALL)
# 粘连句界(".The" 类):句号后直接跟大写字母 → 补空格
_GLUED_BOUNDARY_RE = re.compile(r"\.(?=[A-Z])")


def build_hint_from_v2_record(record: Mapping[str, Any]) -> Optional[str]:
    """从一条 openrouter_teacher_trajectory_v2 记录构建卫生清洗后的 hint。

    步骤(顺序即试点顺序,勿动;回归 fixture 见 tests/test_catalyst_hints.py):
      1. 按 decision 顺序取每条 completion_content 的 <think>…</think> 内文(strip);
      2. '\\n'.join;
      3. 去 flash 草稿块;4. 修粘连句界;5. 截断 HINT_MAX_CHARS。
    无 think 段 → 返回 None(该任务不产 hint,自然落 R0)。
    """
    thinks: List[str] = []
    for decision in record.get("decision_trace") or []:
        match = _THINK_RE.search(str(decision.get("completion_content", "")))
        if match:
            thinks.append(match.group(1).strip())
    if not thinks:
        return None
    text = "\n".join(thinks)
    text = _FLASH_DRAFT_RE.sub("", text)
    text = _GLUED_BOUNDARY_RE.sub(". ", text)
    text = text[:HINT_MAX_CHARS]
    return text if text.strip() else None


# ---------------------------------------------------------------------------
# 素材簿
# ---------------------------------------------------------------------------
class CatalystHintBook:
    """加载 data/catalyst_hints/{env}_{teacher}.json 并做 fail-fast 校验。

    文件格式 {task_id: {"raw": str}}(与试点 hints 文件同构;raw 键沿用)。
    manifest(旁挂 *.manifest.json)校验:clean_version 必须一致。
    """

    def __init__(self, path: str, *, require_manifest: bool = True):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"[CATALYST] hints file does not exist: {self.path} "
                "(catalyst.enable=true 时必须先跑 scripts/build_catalyst_hints.py)"
            )
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not payload:
            raise RuntimeError(f"[CATALYST] hints file 为空或非对象: {self.path}")
        self._hints: Dict[str, str] = {}
        for task_id, entry in payload.items():
            if not isinstance(entry, Mapping):
                raise RuntimeError(
                    f"[CATALYST] hint entry for {task_id!r} must be an object"
                )
            value = entry.get("raw")
            if isinstance(value, str) and value.strip():
                self._hints[str(task_id)] = value.strip()
        if not self._hints:
            raise RuntimeError(f"[CATALYST] hints file 无有效 raw 条目: {self.path}")

        manifest_path = Path(str(self.path) + ".manifest.json")
        self.manifest: Optional[Dict[str, Any]] = None
        if manifest_path.is_file():
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            recorded = str(self.manifest.get("clean_version", ""))
            if recorded and recorded != HINT_CLEAN_VERSION:
                raise RuntimeError(
                    "[CATALYST] hints manifest clean_version 不符:"
                    f"{recorded} != {HINT_CLEAN_VERSION};请重跑 build_catalyst_hints.py"
                )
        elif require_manifest:
            raise FileNotFoundError(
                f"[CATALYST] hints manifest missing: {manifest_path} "
                "(hints.require_manifest=true)"
            )
        logger.info(
            f"[CATALYST] hint book loaded: {len(self._hints)} tasks from {self.path}"
        )

    def get(self, task_id: str) -> Optional[str]:
        return self._hints.get(str(task_id))

    def __contains__(self, task_id: str) -> bool:
        return str(task_id) in self._hints

    def __len__(self) -> int:
        return len(self._hints)


# ---------------------------------------------------------------------------
# 治理层
# ---------------------------------------------------------------------------
@dataclass
class TaskGovState:
    sr_bare_ema: float = 0.0
    sr_hint_ema: float = 0.0
    n_bare_obs: int = 0        # 累计裸臂 rollout 数
    n_hint_obs: int = 0        # 累计提示臂 rollout 数
    u_low_streak: int = 0      # U<=delta 连续窗计数(仅 bootstrap 结束后累计)
    retired: bool = False
    retired_step: int = -1


class CatalystGovernor:
    """每任务双臂 EMA + ρ 控制器 + 退休(CATALYST 通路③因果计价的执行器)。

    公式(SIEVE §4.1 / 总设计 §2):
        U_task     = SR_hint_ema − SR_bare_ema
        ρ_task     = clip(1 − SR_bare_ema / s_hi, 0, ρ_max) · 1[U 门通过]
        k_hint     = quantize(round(ρ·n)) ∈ {0} ∪ [min_hint_rollouts, n−2]
        退休        : U ≤ δ 连续 retire_windows 个更新窗(仅当该任务提示臂累计
                     观测 ≥ u_bootstrap_min_obs 后才开始累计)→ 永久停用素材(M1)。

    冷启动语义:
      * 从未观测过的任务 sr_bare_ema=0(悲观:没证明会做 → 视为难)→ ρ=ρ_max;
      * 每臂的**首次**观测直接以该步 batch SR 播种(避免 EMA 从 0 混合的暖机偏差),
        之后按 ema_alpha 平滑;
      * 提示臂累计观测 < u_bootstrap_min_obs 时 U 门直接放行(否则提示臂永远拿
        不到第一批读数,鸡生蛋死锁)。

    与重放池的关系(A3,有意设计,勿当 bug 修):任务退休只停发**新的**提示臂
    rollout;该任务已在重放池中的既有条目继续由 TTL 自然代谢——它们是已被环境
    盖章的学生自写成功经验,其价值不因"教师素材停用"而消失。退休影响的是素材
    的未来使用,不追溯已验证的成果。
    """

    def __init__(self, cfg: Mapping[str, Any], hint_book: CatalystHintBook):
        self.hint_book = hint_book
        self.s_hi = float(cfg.get("s_hi", 0.8))
        self.rho_max = float(cfg.get("rho_max", 0.5))
        self.delta_u = float(cfg.get("delta_u", 0.0))
        self.ema_alpha = float(cfg.get("ema_alpha", 0.2))
        self.u_bootstrap_min_obs = int(cfg.get("u_bootstrap_min_obs", 8))
        self.min_hint_rollouts = int(cfg.get("min_hint_rollouts", 2))
        self.max_hint_rollouts = int(cfg.get("max_hint_rollouts", -1))
        self.retire_windows = int(cfg.get("retire_windows", 3))
        if not (0.0 < self.s_hi <= 1.0):
            raise ValueError("catalyst.governance.s_hi must be in (0, 1]")
        if not (0.0 <= self.rho_max <= 1.0):
            raise ValueError("catalyst.governance.rho_max must be in [0, 1]")
        if not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError("catalyst.governance.ema_alpha must be in (0, 1]")
        if self.min_hint_rollouts < 1:
            raise ValueError("catalyst.governance.min_hint_rollouts must be >= 1")
        if self.retire_windows < 1:
            raise ValueError("catalyst.governance.retire_windows must be >= 1")
        self._tasks: Dict[str, TaskGovState] = {}
        self._retired_total = 0

    # -- 状态访问 --------------------------------------------------------
    def state(self, task_id: str) -> TaskGovState:
        return self._tasks.setdefault(str(task_id), TaskGovState())

    # -- 难度自举(v2:消灭冷启动聋子期) ---------------------------------
    def bootstrap_from_stats(self, stats: Mapping[str, Mapping[str, Any]]) -> int:
        """用离线难度画像播种 sr_bare_ema(builder:
        scripts/build_catalyst_task_stats.py)。

        只播种**尚无在线观测**的任务(n_bare_obs == 0 且不在 _tasks 或刚建),
        resume 加载的在线状态永远优先。n_bare_obs 记为离线样本数(封顶 16),
        使首次在线观测走 EMA 平滑而非播种覆盖。返回播种任务数。
        """
        seeded = 0
        for task_id, row in stats.items():
            st = self._tasks.get(str(task_id))
            if st is not None and st.n_bare_obs > 0:
                continue
            n = int(row.get("n_bare", 0))
            if n <= 0:
                continue
            st = self.state(task_id)
            st.sr_bare_ema = float(row.get("sr_bare", 0.0))
            st.n_bare_obs = min(n, 16)
            seeded += 1
        logger.info(
            f"[CATALYST] governor bootstrapped {seeded} task(s) from offline "
            "difficulty stats"
        )
        return seeded

    @property
    def retired_total(self) -> int:
        return self._retired_total

    # -- ρ 控制器 --------------------------------------------------------
    def rho(self, task_id: str) -> float:
        st = self.state(task_id)
        return max(0.0, min(self.rho_max, 1.0 - st.sr_bare_ema / self.s_hi))

    def _u_gate_open(self, st: TaskGovState) -> bool:
        if st.n_hint_obs < self.u_bootstrap_min_obs:
            return True  # 冷启动自举
        return (st.sr_hint_ema - st.sr_bare_ema) > self.delta_u

    def quantize_k(self, k: int, n_rollout: int) -> int:
        """k_hint ∈ {0} ∪ [min_hint_rollouts, n−2](两臂各 ≥2,规格 F5 防单样本臂)。"""
        cap = n_rollout - 2
        if self.max_hint_rollouts >= 0:
            cap = min(cap, self.max_hint_rollouts)
        k = min(int(k), max(0, cap))
        if k < self.min_hint_rollouts:
            return 0
        return k

    def plan_k_hint(self, task_id: str, n_rollout: int) -> int:
        """返回该任务本步的提示臂 rollout 数(0 = R0/退休/门未开)。"""
        st = self.state(task_id)
        if self.hint_book.get(task_id) is None:
            return 0
        if st.retired:
            return 0
        if st.sr_bare_ema >= self.s_hi:
            return 0
        if not self._u_gate_open(st):
            return 0
        rho = self.rho(task_id)
        return self.quantize_k(round(rho * n_rollout), n_rollout)

    # -- 在线更新 --------------------------------------------------------
    def update_from_outcomes(
        self,
        arm_outcomes: Mapping[str, Mapping[str, Sequence[bool]]],
        global_step: int,
    ) -> Dict[str, float]:
        """按任务聚合的双臂成败更新 EMA / U / 退休。

        arm_outcomes: {task_id: {"bare": [bool...], "hint": [bool...]}}
        返回本步聚合遥测(catalyst/ 前缀在 trainer 侧统一加)。
        """
        u_values: List[float] = []
        sr_bare_batch: List[float] = []
        sr_hint_batch: List[float] = []
        newly_retired = 0
        for task_id, arms in arm_outcomes.items():
            st = self.state(task_id)
            bare = list(arms.get("bare") or [])
            hint = list(arms.get("hint") or [])
            if bare:
                batch_sr = sum(bare) / len(bare)
                sr_bare_batch.append(batch_sr)
                if st.n_bare_obs == 0:
                    st.sr_bare_ema = batch_sr  # 首次观测播种
                else:
                    st.sr_bare_ema = (
                        (1.0 - self.ema_alpha) * st.sr_bare_ema
                        + self.ema_alpha * batch_sr
                    )
                st.n_bare_obs += len(bare)
            if hint:
                batch_sr = sum(hint) / len(hint)
                sr_hint_batch.append(batch_sr)
                if st.n_hint_obs == 0:
                    st.sr_hint_ema = batch_sr
                else:
                    st.sr_hint_ema = (
                        (1.0 - self.ema_alpha) * st.sr_hint_ema
                        + self.ema_alpha * batch_sr
                    )
                st.n_hint_obs += len(hint)
            # U / 退休:一个"更新窗" = 该任务两臂在同一步都有读数
            if bare and hint:
                u = st.sr_hint_ema - st.sr_bare_ema
                u_values.append(u)
                if st.n_hint_obs >= self.u_bootstrap_min_obs and not st.retired:
                    if u <= self.delta_u:
                        st.u_low_streak += 1
                    else:
                        st.u_low_streak = 0
                    if st.u_low_streak >= self.retire_windows:
                        st.retired = True
                        st.retired_step = int(global_step)
                        self._retired_total += 1
                        newly_retired += 1
                        logger.info(
                            f"[CATALYST] task {task_id} retired at step "
                            f"{global_step} (U={u:.4f} <= {self.delta_u} for "
                            f"{st.u_low_streak} windows)"
                        )
        metrics: Dict[str, float] = {
            "tasks_retired_total": float(self._retired_total),
            "tasks_newly_retired": float(newly_retired),
        }
        if sr_bare_batch:
            metrics["sr_bare_batch"] = sum(sr_bare_batch) / len(sr_bare_batch)
        if sr_hint_batch:
            metrics["sr_hint_batch"] = sum(sr_hint_batch) / len(sr_hint_batch)
        if u_values:
            metrics["u_ema_mean"] = sum(u_values) / len(u_values)
            metrics["u_pos_frac"] = sum(
                1.0 for u in u_values if u > self.delta_u
            ) / len(u_values)
        seen = [self.state(t) for t in arm_outcomes]
        if seen:
            metrics["sr_bare_ema_mean"] = sum(
                s.sr_bare_ema for s in seen
            ) / len(seen)
            metrics["sr_hint_ema_mean"] = sum(
                s.sr_hint_ema for s in seen
            ) / len(seen)
        return metrics

    # -- 持久化(断点续训) ----------------------------------------------
    def save_state(self, path: str) -> None:
        payload = {
            "schema": "catalyst_governor_state_v1",
            "retired_total": self._retired_total,
            "tasks": {tid: asdict(st) for tid, st in self._tasks.items()},
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1),
            encoding="utf-8",
        )
        os.replace(tmp, path)

    def load_state(self, path: str) -> bool:
        path = Path(path)
        if not path.is_file():
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "catalyst_governor_state_v1":
            raise RuntimeError(f"[CATALYST] unknown governor state schema: {path}")
        self._retired_total = int(payload.get("retired_total", 0))
        self._tasks = {
            str(tid): TaskGovState(**st)
            for tid, st in (payload.get("tasks") or {}).items()
        }
        logger.info(
            f"[CATALYST] governor state loaded: {len(self._tasks)} tasks, "
            f"{self._retired_total} retired, from {path}"
        )
        return True

    def per_task_dump(self) -> Dict[str, Dict[str, Any]]:
        return {tid: asdict(st) for tid, st in sorted(self._tasks.items())}


# ---------------------------------------------------------------------------
# 去提示重放池
# ---------------------------------------------------------------------------
@dataclass
class ReplayEntry:
    task_id: str
    rollout_id: str
    n_init: int
    msgs: List[Dict[str, str]]          # [{"author","role","content"}] 原文(含 hint)
    decisions: List[Dict[str, Any]]     # [{"step_index","completion_token_ids","content"}]
    reward_scores: Dict[str, Any]
    inserted_step: int
    hint_sha: str = ""


class CatalystReplayPool:
    """提示臂成功轨迹 → 去提示重渲染 → 辅助 BC 样本(军规链路)。

    入池(insert_from_cmt):只收 snapshot 模式、成功、未 discard、未长度截断的
    提示臂 CMT;audit_on_insert=true 时带提示原样逐 decision 重渲染,断言 prompt
    token ids 与采样时快照逐一相等——证明"消息重构 → StructuredContextPolicy"
    这条链在本训练栈下 100% 保真,此后去提示渲染的唯一差异就是 hint 文本本身
    (比 analysis/catalyst_purity_score.py 只能验 completion sha 更强)。

    出池(build_replay_samples):仅取当前批任务(规格 F7:union 按 task_id 回拼,
    池外任务会 KeyError);每条目每步产 1 个单 decision 快照式 Sample(D2),
    decision 用 token 加权哈希选择(seed 掺 global_step,跨步覆盖不同 decision);
    response 直接用**原采样 token ids**(零重分词漂移)。

    退休任务的既有条目不做立即清除,由 TTL 自然代谢(A3;见 Governor docstring)。
    """

    def __init__(self, cfg: Mapping[str, Any]):
        self.per_task = int(cfg.get("per_task", 1))
        self.pool_max_per_task = int(cfg.get("pool_max_per_task", 4))
        self.ttl_steps = int(cfg.get("ttl_steps", 20))
        self.audit_on_insert = bool(cfg.get("audit_on_insert", True))
        if self.per_task < 1 or self.pool_max_per_task < 1 or self.ttl_steps < 1:
            raise ValueError("catalyst.replay knobs must be >= 1")
        self._pool: Dict[str, List[ReplayEntry]] = {}
        self._policy = None
        self._tokenizer = None
        self._snapshot_selection_seed = 0
        # 健康度计数(遥测;预期恒 0 的两项是军规探针)
        self.audit_failures_total = 0
        self.render_drops_total = 0
        self.insert_skips_total = 0

    # -- 渲染基建 --------------------------------------------------------
    def attach_renderer(self, tokenizer: Any, rollout_config: Any) -> None:
        from agentevolver.module.context_manager.context_policy import (
            StructuredContextPolicy,
        )

        self._tokenizer = tokenizer
        self._policy = StructuredContextPolicy(tokenizer, rollout_config)
        self._snapshot_selection_seed = int(self._policy.snapshot_selection_seed)

    def _require_renderer(self):
        if self._policy is None or self._tokenizer is None:
            raise RuntimeError(
                "[CATALYST] replay pool renderer not attached; call "
                "attach_renderer(tokenizer, rollout_config) first"
            )
        return self._policy, self._tokenizer

    def _ext_messages(self, msgs: Sequence[Mapping[str, str]]) -> List[Any]:
        from agentevolver.module.context_manager.cmt_base import ExtendedMessage

        _, tokenizer = self._require_renderer()
        # token_generator="manual" + token_arr=[]:policy.build 只读
        # author/role/content(对齐 cll_teacher_profile.reconstruct_ext_messages)。
        return [
            ExtendedMessage(
                author=str(m["author"]),
                role=str(m["role"]),
                content=str(m["content"]),
                token_arr=[],
                token_generator="manual",
                tokenizer=tokenizer,
            )
            for m in msgs
        ]

    # -- 入池 ------------------------------------------------------------
    def insert_from_cmt(self, cmt: Any, global_step: int) -> bool:
        """存一条提示臂成功 CMT;结构或军规审计不过 → 不入池并计数。"""
        snapshots = list(getattr(cmt, "decision_snapshots", []) or [])
        if not snapshots:
            # transcript(非 snapshot)模式 M1 不支持重放
            self.insert_skips_total += 1
            return False
        full_context = list(getattr(cmt, "full_context", []) or [])
        msgs = [
            {
                "author": str(m.author),
                "role": str(m.role),
                "content": str(m.content),
            }
            for m in full_context
        ]
        n_init = 0
        for m in msgs:
            if m["author"] == "initialization":
                n_init += 1
            else:
                break
        n_dec = len(snapshots)
        # 成功轨迹形状:init + (llm, env)*(T-1) + llm(remove_last_context 已弹尾)
        if len(msgs) != n_init + 2 * n_dec - 1 or n_init < 1:
            self.insert_skips_total += 1
            logger.warning(
                f"[CATALYST] replay insert skipped (shape): task={cmt.task_id} "
                f"msgs={len(msgs)} n_init={n_init} decisions={n_dec}"
            )
            return False
        for k in range(n_dec):
            if msgs[n_init + 2 * k]["author"] != "llm":
                self.insert_skips_total += 1
                return False

        if self.audit_on_insert:
            policy, _ = self._require_renderer()
            ext = self._ext_messages(msgs)
            for t, snapshot in enumerate(snapshots):
                result = policy.build(ext[: n_init + 2 * t])
                if list(result.prompt_token_ids) != list(
                    snapshot.prompt_token_ids
                ):
                    self.audit_failures_total += 1
                    logger.error(
                        "[CATALYST] replay insert audit FAILED: rebuilt prompt "
                        f"!= snapshot at task={cmt.task_id} step={t}; entry "
                        "rejected (military-rule probe, expected 0)"
                    )
                    return False

        entry = ReplayEntry(
            task_id=str(cmt.task_id),
            rollout_id=str(cmt.rollout_id),
            n_init=n_init,
            msgs=msgs,
            decisions=[
                {
                    "step_index": int(s.step_index),
                    "completion_token_ids": list(s.completion_token_ids),
                    "content": str(s.assistant_content),
                }
                for s in snapshots
            ],
            reward_scores=(
                cmt.reward.model_dump() if cmt.reward is not None else {}
            ),
            inserted_step=int(global_step),
            hint_sha=str(
                (getattr(cmt, "metadata", None) or {}).get(
                    "catalyst_hint_sha256", ""
                )
            ),
        )
        bucket = self._pool.setdefault(entry.task_id, [])
        if len(bucket) >= self.pool_max_per_task:
            bucket.pop(0)  # FIFO:新胜旧
        bucket.append(entry)
        return True

    def evict_stale(self, global_step: int) -> int:
        """TTL 淘汰;返回淘汰条数。"""
        evicted = 0
        for task_id in list(self._pool.keys()):
            kept = [
                e
                for e in self._pool[task_id]
                if global_step - e.inserted_step <= self.ttl_steps
            ]
            evicted += len(self._pool[task_id]) - len(kept)
            if kept:
                self._pool[task_id] = kept
            else:
                del self._pool[task_id]
        return evicted

    # -- 出池 ------------------------------------------------------------
    def _select_decision_index(
        self, entry: ReplayEntry, global_step: int
    ) -> int:
        """token 加权确定性选择(对齐 cmt_linear._select_decision_snapshot,
        digest 额外掺 global_step 使同一条目跨步训练不同 decision)。"""
        total = sum(len(d["completion_token_ids"]) for d in entry.decisions)
        if total <= 0:
            raise ValueError("replay entry has empty completions")
        digest = hashlib.sha256()
        digest.update(str(self._snapshot_selection_seed).encode("utf-8"))
        digest.update(str(entry.task_id).encode("utf-8"))
        digest.update(str(entry.rollout_id).encode("utf-8"))
        digest.update(str(int(global_step)).encode("utf-8"))
        target = int.from_bytes(digest.digest()[:8], "big") % total
        cursor = 0
        for index, decision in enumerate(entry.decisions):
            cursor += len(decision["completion_token_ids"])
            if target < cursor:
                return index
        return len(entry.decisions) - 1

    def build_replay_samples(
        self,
        tasks: Sequence[Any],
        *,
        global_step: int,
        max_prompt_len: int,
        max_response_len: int,
        existing_group_ids: Sequence[int],
    ) -> Tuple[List[Any], Dict[str, float]]:
        """为当前批任务构建去提示重放 BC 样本。

        返回 (samples, metrics)。metrics 不带 catalyst/ 前缀(trainer 统一加)。
        """
        policy, _ = self._require_renderer()
        self.evict_stale(global_step)

        # A1 防御:重放 data_id 不得与本批真实 group_ids 撞号。
        existing = {int(g) for g in existing_group_ids}
        base = CATALYST_REPLAY_DATA_ID_BASE
        budget = sum(
            min(self.per_task, len(self._pool.get(str(t.task_id), [])))
            for t in tasks
        )
        while existing & set(range(base, base + max(budget, 1))):
            base = max(existing) + CATALYST_REPLAY_DATA_ID_BASE
        assert not (existing & set(range(base, base + max(budget, 1)))), (
            "catalyst replay data_id range still collides with real group_ids"
        )

        samples: List[Any] = []
        drops = 0
        next_data_id = base
        for task in tasks:
            bucket = self._pool.get(str(task.task_id), [])
            for entry in list(reversed(bucket))[: self.per_task]:  # 最新优先
                try:
                    sample = self._render_entry(
                        entry,
                        policy=policy,
                        data_id=next_data_id,
                        global_step=global_step,
                        max_prompt_len=max_prompt_len,
                        max_response_len=max_response_len,
                    )
                except ValueError as error:
                    drops += 1
                    self.render_drops_total += 1
                    logger.warning(
                        f"[CATALYST] replay render dropped: task={entry.task_id} "
                        f"rollout={entry.rollout_id}: {error}"
                    )
                    continue
                samples.append(sample)
                next_data_id += 1

        ages = [
            float(global_step - e.inserted_step)
            for bucket in self._pool.values()
            for e in bucket
        ]
        metrics: Dict[str, float] = {
            "replay_pool_entries": float(len(ages)),
            "replay_pool_tasks": float(len(self._pool)),
            "replay_samples_in_batch": float(len(samples)),
            "replay_render_drops": float(drops),
            "replay_render_drops_total": float(self.render_drops_total),
            "replay_audit_failures_total": float(self.audit_failures_total),
            "replay_insert_skips_total": float(self.insert_skips_total),
        }
        if ages:
            metrics["replay_pool_age_mean"] = sum(ages) / len(ages)
            metrics["replay_pool_age_max"] = max(ages)
        return samples, metrics

    def _render_entry(
        self,
        entry: ReplayEntry,
        *,
        policy: Any,
        data_id: int,
        global_step: int,
        max_prompt_len: int,
        max_response_len: int,
    ) -> Any:
        from agentevolver.schema.trajectory import Sample

        msgs = copy.deepcopy(entry.msgs)
        strip_hint_messages(msgs)  # 军规剥除(恰 1 条 + 无 marker 残留),失败抛 ValueError

        t = self._select_decision_index(entry, global_step)
        decision = entry.decisions[t]
        ext = self._ext_messages(msgs)
        result = policy.build(ext[: entry.n_init + 2 * t])
        for message in result.messages:
            if HINT_MARKER in str(message.get("content", "")):
                raise ValueError("hint marker survived de-hinted rendering")

        prompt_ids = list(result.prompt_token_ids)
        response_ids = list(decision["completion_token_ids"])  # 原采样 ids,零漂移
        if not response_ids:
            raise ValueError("replay decision has empty completion")
        if len(prompt_ids) > max_prompt_len:
            raise ValueError(
                f"de-hinted prompt exceeds limit: {len(prompt_ids)} > {max_prompt_len}"
            )
        if len(response_ids) > max_response_len:
            raise ValueError(
                f"replay completion exceeds limit: {len(response_ids)} > {max_response_len}"
            )

        input_ids = prompt_ids + response_ids
        attention_mask = [1] * len(input_ids)
        position_ids = list(range(len(input_ids)))  # 全 1 attention 下与 verl 等价
        prompt_loss_mask = [0] * len(prompt_ids)
        response_loss_mask = [1] * len(response_ids)
        extras = {
            "is_experience_replay": True,   # → exp_mask=1(auxiliary;豁免恒等门,规格 F4/F6)
            "is_catalyst_replay": True,     # → catalyst_replay_mask=1(只 BC 无 PG)
            "is_teacher": False,            # 零模仿:重放样本永远不是教师样本
            "has_log_prob": False,
            "snapshot_training": False,     # 允许缺 rollout_log_probs(规格 F4)
            "rollout_log_probs": None,
            "rollout_mode": None,
            "old_log_probs": None,
            "catalyst_arm": "replay",
            "task_id": entry.task_id,
            "rollout_id": entry.rollout_id,
            "catalyst_replay_inserted_step": entry.inserted_step,
            "catalyst_replay_age": int(global_step) - entry.inserted_step,
            "catalyst_replay_decision_step": int(decision["step_index"]),
            "catalyst_hint_sha256": entry.hint_sha,
        }
        sample = Sample(
            data_id=str(int(data_id)),
            task_id=entry.task_id,
            rollout_id=f"cr{global_step}_{entry.rollout_id}",
            minor_index_id=int(decision["step_index"]),
            messages=copy.deepcopy(result.messages)
            + [{"role": "assistant", "content": decision["content"]}],
            messages_raw=[
                {"role": m["role"], "content": m["content"]} for m in msgs
            ],
            input_ids=input_ids,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            attention_mask=attention_mask,
            prompt_attention_mask=[1] * len(prompt_ids),
            response_attention_mask=[1] * len(response_ids),
            loss_mask=prompt_loss_mask + response_loss_mask,
            prompt_loss_mask=prompt_loss_mask,
            response_loss_mask=response_loss_mask,
            position_ids=position_ids,
            prompt_position_ids=position_ids[: len(prompt_ids)],
            response_position_ids=position_ids[len(prompt_ids):],
            reward_scores=dict(entry.reward_scores),
            max_prompt_len=int(max_prompt_len),
            max_response_len=int(max_response_len),
            max_model_len=int(max_prompt_len) + int(max_response_len),
        )
        sample.extras = extras
        return sample

    # -- 观测 ------------------------------------------------------------
    def size(self) -> int:
        return sum(len(bucket) for bucket in self._pool.values())


# ---------------------------------------------------------------------------
# 门面:trainer / exp_manager 的唯一接入点
# ---------------------------------------------------------------------------
class CatalystRuntime:
    """CATALYST M1 运行时(hint book + governor + replay pool 的门面)。

    由 ``ExperienceManager.__init__`` 在 ``exp_manager.catalyst.enable=true`` 时
    构造;trainer 在 fit 开头调 ``attach_renderer``(补 tokenizer)并做互斥断言。
    """

    def __init__(self, config: Any):
        exp_cfg = config.exp_manager
        cat = exp_cfg.get("catalyst", {}) or {}
        if not bool(cat.get("enable", False)):
            raise RuntimeError("CatalystRuntime constructed while disabled")
        self.config = config
        hints_cfg = cat.get("hints", {}) or {}
        hints_file = hints_cfg.get("file", None)
        if not hints_file:
            raise RuntimeError(
                "[CATALYST] exp_manager.catalyst.hints.file is required when "
                "catalyst.enable=true"
            )
        self.hint_book = CatalystHintBook(
            str(hints_file),
            require_manifest=bool(hints_cfg.get("require_manifest", True)),
        )
        self.governor = CatalystGovernor(
            cat.get("governance", {}) or {}, self.hint_book
        )
        # v3.1:hint 臂 critic 基线(默认 off,v1/v2 字节等价)
        self.hint_critic_baseline = bool(
            (cat.get("governance", {}) or {}).get(
                "hint_critic_baseline", False
            )
        )
        replay_cfg = cat.get("replay", {}) or {}
        self.replay_enabled = bool(replay_cfg.get("enable", False))
        self.replay_pool = (
            CatalystReplayPool(replay_cfg) if self.replay_enabled else None
        )
        # -- v2 entry-k 状态课程(与转写重放互斥:重放已被失败归因裁定) ----
        entry_cfg = cat.get("entry", {}) or {}
        self.entry_enabled = bool(entry_cfg.get("enable", False))
        self.entry_book = None
        self.entry_scheduler = None
        self.entry_state_pool = None
        self.entry_mode = "ladder"
        self.entry_corner_hint_slots = 0
        self.entry_adv_scale = 1.0
        self.entry_s_lo = float(entry_cfg.get("s_lo", 0.125))
        self.entry_replay_failures_total = 0
        self.entry_divergence_total = 0
        self._entry_pool_hits = 0
        self._entry_plans_total = 0
        if self.entry_enabled:
            if self.replay_enabled:
                raise RuntimeError(
                    "[CATALYST] entry-k and strip-replay are mutually "
                    "exclusive (v2 design: replay channel is retired; see "
                    "docs/research/CATALYST_失败归因_2026-08-13.md)"
                )
            from agentevolver.module.exp_manager.catalyst_entry import (
                CatalystEntryBook,
                CatalystEntryIntervalScheduler,
                CatalystEntryScheduler,
                CatalystStatePool,
            )

            book_file = entry_cfg.get("book_file", None)
            if not book_file:
                raise RuntimeError(
                    "[CATALYST] catalyst.entry.book_file is required when "
                    "entry.enable=true"
                )
            self.entry_book = CatalystEntryBook(
                str(book_file),
                require_manifest=bool(
                    entry_cfg.get("require_manifest", True)
                ),
            )
            # v3 分布课程(mode: interval)vs v2 阶梯(mode: ladder,默认,归档对照)
            self.entry_mode = str(entry_cfg.get("mode", "ladder"))
            if self.entry_mode == "interval":
                self.entry_scheduler = CatalystEntryIntervalScheduler(entry_cfg)
                self.entry_state_pool = (
                    CatalystStatePool(entry_cfg)
                    if bool(entry_cfg.get("student_pool", True))
                    else None
                )
            elif self.entry_mode == "ladder":
                self.entry_scheduler = CatalystEntryScheduler(entry_cfg)
                self.entry_state_pool = None
            else:
                raise RuntimeError(
                    f"[CATALYST] unknown entry.mode: {self.entry_mode}"
                )
            # v3:角点任务混臂的 hint 槽数(0 = v2 行为;>0 需要 hint 素材)
            self.entry_corner_hint_slots = int(
                entry_cfg.get("corner_hint_slots", 0)
            )
            self.entry_adv_scale = float(entry_cfg.get("adv_scale", 1.0))
            stats_file = entry_cfg.get("stats_bootstrap_file", None)
            if stats_file:
                stats_path = str(stats_file)
                if not os.path.isfile(stats_path):
                    raise FileNotFoundError(
                        f"[CATALYST] entry.stats_bootstrap_file missing: "
                        f"{stats_path}(跑 scripts/build_catalyst_task_stats.py)"
                    )
                stats_payload = json.loads(
                    Path(stats_path).read_text(encoding="utf-8")
                )
                self._entry_stats = {
                    str(k): v
                    for k, v in (stats_payload.get("tasks") or {}).items()
                }
            else:
                self._entry_stats = {}
        # ===== v4 统一分配法则(catalyst_v4.py;与 entry/replay 互斥)=====
        v4_cfg = cat.get("v4", {}) or {}
        self.v4_enabled = bool(v4_cfg.get("enable", False))
        self.v4_table = None
        self.v4_alloc = None
        self.v4_n0 = float(v4_cfg.get("n0", 2.0))
        if self.v4_enabled:
            if self.replay_enabled or self.entry_enabled:
                raise RuntimeError(
                    "[CATALYST] v4 is mutually exclusive with legacy replay/"
                    "entry modes (v1-v3 paths are archival)"
                )
            from agentevolver.module.exp_manager.catalyst_entry import (
                CatalystEntryBook,
                CatalystStatePool,
            )
            from agentevolver.module.exp_manager.catalyst_v4 import (
                V4Allocator,
                V4ValueTable,
            )
            book_file = entry_cfg.get("book_file", None)
            if not book_file:
                raise RuntimeError(
                    "[CATALYST] v4 requires catalyst.entry.book_file "
                    "(teacher fallback material)"
                )
            self.entry_book = CatalystEntryBook(
                str(book_file),
                require_manifest=bool(entry_cfg.get("require_manifest", True)),
            )
            self.entry_state_pool = CatalystStatePool(entry_cfg)
            self.v4_table = V4ValueTable(v4_cfg)
            self.v4_alloc = V4Allocator(v4_cfg)
            # v5:rescue 第四格(学生失败前缀 + 提示重试);默认 off = v4 原样
            self.v4_rescue = bool(v4_cfg.get("rescue", False))
            stats_file = entry_cfg.get("stats_bootstrap_file", None)
            if stats_file:
                stats_payload = json.loads(
                    Path(str(stats_file)).read_text(encoding="utf-8")
                )
                self._v4_stats = {
                    str(k): v
                    for k, v in (stats_payload.get("tasks") or {}).items()
                }
            else:
                self._v4_stats = {}
        # rollout.multi_turn.max_steps:entry 计划的 k < max_steps 防御要用。
        # 仅 entry 启用时强制存在(测试夹具/最小配置不带 actor_rollout_ref)。
        try:
            self._rollout_max_steps = int(
                config.actor_rollout_ref.rollout.multi_turn.max_steps
            )
        except Exception:  # noqa: BLE001 - 最小配置容忍
            if self.entry_enabled:
                raise RuntimeError(
                    "[CATALYST] entry.enable=true requires "
                    "actor_rollout_ref.rollout.multi_turn.max_steps"
                )
            self._rollout_max_steps = None
        arm_cfg = cat.get("arm_baseline", {}) or {}
        self.arm_baseline_enabled = bool(arm_cfg.get("enable", True))
        thermo_cfg = cat.get("thermostat", {}) or {}
        self.thermostat_enabled = bool(thermo_cfg.get("enable", False))
        self.thermo_h_ref = float(thermo_cfg.get("h_ref", 0.35))
        self.thermo_eta = float(thermo_cfg.get("eta", 0.01))
        self.thermo_lambda = float(thermo_cfg.get("lambda_init", 0.0))
        self.thermo_lambda_max = float(thermo_cfg.get("lambda_max", 1.0))
        state_dir = (cat.get("governance", {}) or {}).get("state_dir", None)
        self.state_dir = str(state_dir) if state_dir else None
        # 注入模板在构造期即校验(fail-fast,而不是首个 hint rollout 才炸)
        load_hint_template()
        self._last_plan: Dict[str, int] = {}

    # -- 基建 ------------------------------------------------------------
    def attach_renderer(self, tokenizer: Any, rollout_config: Any) -> None:
        if self.replay_pool is not None:
            self.replay_pool.attach_renderer(tokenizer, rollout_config)

    def state_path(self, default_local_dir: str) -> str:
        base = self.state_dir or os.path.join(
            str(default_local_dir), "catalyst_state"
        )
        return os.path.join(base, "governor_latest.json")

    def _entry_state_path(self, governor_state_path: str) -> str:
        return os.path.join(
            os.path.dirname(str(governor_state_path)),
            "entry_scheduler_latest.json",
        )

    def _v4_state_path(self, governor_state_path: str) -> str:
        return os.path.join(
            os.path.dirname(str(governor_state_path)), "catalyst_v4_state.json"
        )

    def load_persistent_state(self, governor_state_path: str) -> None:
        """resume 语义:在线状态(governor + entry 调度器 + 状态池)优先加载,
        难度自举只播种没有在线观测的任务(v1 教训:状态必须过重启)。"""
        if self.v4_enabled:
            path = self._v4_state_path(governor_state_path)
            p = Path(path)
            if p.is_file():
                payload = json.loads(p.read_text(encoding="utf-8"))
                self.v4_table.load_payload(payload["table"])
                if payload.get("pool") is not None:
                    self.entry_state_pool.load_payload(payload["pool"])
                logger.info(f"[CATALYST-v4] state loaded from {path}")
            if self._v4_stats:
                self.v4_table.bootstrap_bare(self._v4_stats)
            return
        self.governor.load_state(governor_state_path)
        if self.entry_scheduler is not None:
            entry_path = self._entry_state_path(governor_state_path)
            if self.entry_mode == "interval":
                p = Path(entry_path)
                if p.is_file():
                    payload = json.loads(p.read_text(encoding="utf-8"))
                    if payload.get("schema") != "catalyst_entry_v3_bundle_v1":
                        raise RuntimeError(
                            f"[CATALYST] entry state schema mismatch at "
                            f"{entry_path}(v2 阶梯状态不能喂 v3 区间调度器)"
                        )
                    self.entry_scheduler.load_payload(payload["scheduler"])
                    if (
                        self.entry_state_pool is not None
                        and payload.get("pool") is not None
                    ):
                        self.entry_state_pool.load_payload(payload["pool"])
                    logger.info(
                        f"[CATALYST] entry v3 state loaded from {entry_path}"
                    )
            else:
                self.entry_scheduler.load_state(entry_path)
        if self.entry_enabled and self._entry_stats:
            self.governor.bootstrap_from_stats(self._entry_stats)

    def save_persistent_state(self, governor_state_path: str) -> None:
        if self.v4_enabled:
            payload = {
                "schema": "catalyst_v4_state_v1",
                "table": self.v4_table.save_payload(),
                "pool": self.entry_state_pool.save_payload(),
            }
            p = Path(self._v4_state_path(governor_state_path))
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(tmp, p)
            return
        self.governor.save_state(governor_state_path)
        if self.entry_scheduler is not None:
            entry_path = self._entry_state_path(governor_state_path)
            if self.entry_mode == "interval":
                payload = {
                    "schema": "catalyst_entry_v3_bundle_v1",
                    "scheduler": self.entry_scheduler.save_payload(),
                    "pool": (
                        self.entry_state_pool.save_payload()
                        if self.entry_state_pool is not None
                        else None
                    ),
                }
                p = Path(entry_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
                tmp.write_text(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    encoding="utf-8",
                )
                os.replace(tmp, p)
            else:
                self.entry_scheduler.save_state(entry_path)

    # -- 臂规划(fit 每步,rollout 前) -----------------------------------
    def plan_arms(
        self,
        tasks: Sequence[Any],
        task_exp_configs: Sequence[Any],
        n_rollout: int,
        global_step: int,
    ) -> Dict[str, float]:
        assert len(tasks) == len(task_exp_configs)
        if self.v4_enabled:
            return self._plan_arms_v4(
                tasks, task_exp_configs, n_rollout, global_step
            )
        if n_rollout < 4:
            raise RuntimeError(
                "[CATALYST] arm split requires rollout.n >= 4 (both arms >= 2); "
                f"got n={n_rollout}"
            )
        n_r1 = 0
        hint_rollouts = 0
        rho_values: List[float] = []
        entry_tasks = 0
        entry_rollouts = 0
        entry_rungs: List[int] = []
        self._last_plan = {}
        for task, tec in zip(tasks, task_exp_configs):
            task_id = str(task.task_id)
            # -- 通道优先级:角点任务先走 entry(零机会成本论证:全败组
            # 的裸 rollout 在 GRPO 里梯度恒 0,挪槽给 entry 是复活不是税)。
            if self._entry_eligible(task_id):
                k_e = min(
                    self.entry_scheduler.slots_per_task, n_rollout - 2
                )
                if self.entry_mode == "interval":
                    payloads = self._build_interval_payloads(
                        task_id, global_step, k_e
                    )
                else:
                    payloads = self._build_ladder_payloads(task_id, k_e)
                if payloads is None:
                    self._last_plan[task_id] = 0
                    continue
                tec.catalyst_entry_slots = (
                    payloads + [None] * (n_rollout - len(payloads))
                )
                # v3 角点混臂:entry 槽之后紧跟 corner_hint_slots 个 hint 槽
                # (为该任务制造学生成功轨迹 → 喂状态池;v2 配置该值为 0)。
                n_h = 0
                if self.entry_corner_hint_slots > 0:
                    hint = self.hint_book.get(task_id)
                    st = self.governor.state(task_id)
                    if hint and not st.retired:
                        n_h = min(
                            self.entry_corner_hint_slots,
                            n_rollout - len(payloads) - 2,  # 裸臂 >= 2
                        )
                        if n_h > 0:
                            tec.catalyst_hint_slots = (
                                [None] * len(payloads)
                                + [hint] * n_h
                                + [None] * (n_rollout - len(payloads) - n_h)
                            )
                            if self.hint_critic_baseline:
                                tec.catalyst_hint_vhat = float(
                                    st.sr_hint_ema
                                )
                            n_r1 += 1
                            hint_rollouts += n_h
                entry_tasks += 1
                entry_rollouts += len(payloads)
                if self.entry_mode != "interval":
                    entry_rungs.append(
                        self.entry_scheduler.current_frac(task_id)[1]
                    )
                self._last_plan[task_id] = -len(payloads)
                continue
            k = self.governor.plan_k_hint(task.task_id, n_rollout)
            self._last_plan[task_id] = k
            if k <= 0:
                continue
            hint = self.hint_book.get(task.task_id)
            assert hint, "plan_k_hint returned k>0 without hint material"
            # 槽位约定:前 k 个 rollout_id 为提示臂(确定性,便于测试/审计)
            tec.catalyst_hint_slots = [hint] * k + [None] * (n_rollout - k)
            if self.hint_critic_baseline:
                # v3.1:hint 臂 critic 基线(plan 时冻结)。SR_hint≈0.83 的
                # 小组大概率全成 → 分臂组基线优势恒 0 —— 最强通道发零梯度
                # (v1/v2 同病;账本见 v2 复盘追录)。V̂ = sr_hint_ema。
                tec.catalyst_hint_vhat = float(
                    self.governor.state(task_id).sr_hint_ema
                )
            n_r1 += 1
            hint_rollouts += k
            rho_values.append(self.governor.rho(task.task_id))
        metrics = {
            "tasks_r1": float(n_r1),
            "tasks_r0": float(len(tasks) - n_r1 - entry_tasks),
            "hint_rollouts": float(hint_rollouts),
            "hint_rollout_frac": (
                float(hint_rollouts) / float(len(tasks) * n_rollout)
                if tasks
                else 0.0
            ),
        }
        if rho_values:
            metrics["rho_mean"] = sum(rho_values) / len(rho_values)
            metrics["rho_max_task"] = max(rho_values)
        if self.entry_enabled:
            metrics["entry_tasks"] = float(entry_tasks)
            metrics["entry_rollouts"] = float(entry_rollouts)
            metrics["entry_rollout_frac"] = (
                float(entry_rollouts) / float(len(tasks) * n_rollout)
                if tasks
                else 0.0
            )
            if entry_rungs:
                metrics["entry_rung_mean"] = sum(entry_rungs) / len(
                    entry_rungs
                )
            metrics["entry_retired_total"] = float(
                self.entry_scheduler.retired_total
            )
            metrics["entry_graduated_total"] = float(
                self.entry_scheduler.graduated_total
            )
            if self.entry_mode == "interval":
                metrics["entry_pool_hit_frac"] = (
                    float(self._entry_pool_hits)
                    / float(max(self._entry_plans_total, 1))
                )
                if self.entry_state_pool is not None:
                    metrics["entry_pool_size"] = float(
                        self.entry_state_pool.size()
                    )
                    metrics["entry_pool_tasks"] = float(
                        self.entry_state_pool.tasks()
                    )
        return metrics

    def _plan_arms_v4(
        self,
        tasks: Sequence[Any],
        task_exp_configs: Sequence[Any],
        n_rollout: int,
        global_step: int,
    ) -> Dict[str, float]:
        """v4:统一分配法则排产。每槽 SlotPlan → 既有 hint/entry 执行链;
        m(plan 时冻结)经 catalyst_v4_m_slots → metadata → extras 透传给
        trainer 的统一优势覆写与在线校准探针。"""
        aux_slots = 0
        total_slots = 0
        mid_mass = 0
        entry_built = 0
        entry_fallback_bare = 0
        hint_slots_n = 0
        self._last_plan = {}
        for task, tec in zip(tasks, task_exp_configs):
            task_id = str(task.task_id)
            has_hint = self.hint_book.get(task_id) is not None
            has_entry = (
                task_id in self.entry_book or task_id in self.entry_state_pool
            )
            has_rescue = (
                self.v4_rescue
                and has_hint
                and self.entry_state_pool.has_failure(task_id)
            )
            plans = self.v4_alloc.allocate(
                task_id,
                global_step,
                n_rollout,
                self.v4_table,
                has_hint=has_hint,
                has_entry=has_entry,
                has_rescue=has_rescue,
            )
            hint_list: List[Any] = []
            entry_list: List[Any] = []
            m_list: List[float] = []
            for slot, sp in enumerate(plans):
                total_slots += 1
                if 0.2 <= sp.m <= 0.8:
                    mid_mass += 1
                if sp.arm == "rescue":
                    plan = self.entry_state_pool.build_rescue_plan(
                        task_id, frac=sp.frac, max_steps=self._rollout_max_steps
                    )
                    if plan is None:
                        entry_fallback_bare += 1
                        hint_list.append(None)
                        entry_list.append(None)
                        m_list.append(
                            self.v4_table.prior(task_id, ("bare", None))
                        )
                        continue
                    aux_slots += 1
                    entry_list.append(plan.to_payload())
                    hint_list.append(self.hint_book.get(task_id))  # 第四格:状态+指导
                    m_list.append(sp.m)
                    continue
                if sp.arm == "entry":
                    plan = self.entry_state_pool.build_plan(
                        task_id, frac=sp.frac, max_steps=self._rollout_max_steps
                    )
                    if plan is None and task_id in self.entry_book:
                        try:
                            plan = self.entry_book.build_plan(
                                task_id,
                                frac=sp.frac,
                                rung=0,
                                max_steps=self._rollout_max_steps,
                            )
                        except Exception:  # noqa: BLE001 - 单槽降级为裸
                            plan = None
                    if plan is None:
                        entry_fallback_bare += 1
                        hint_list.append(None)
                        entry_list.append(None)
                        m_list.append(
                            self.v4_table.prior(task_id, ("bare", None))
                        )
                        continue
                    entry_built += 1
                    aux_slots += 1
                    entry_list.append(plan.to_payload())
                    hint_list.append(None)
                    m_list.append(sp.m)
                elif sp.arm == "hint":
                    aux_slots += 1
                    hint_slots_n += 1
                    hint_list.append(self.hint_book.get(task_id))
                    entry_list.append(None)
                    m_list.append(sp.m)
                else:
                    hint_list.append(None)
                    entry_list.append(None)
                    m_list.append(sp.m)
            if any(entry_list):
                tec.catalyst_entry_slots = entry_list
            if any(hint_list):
                tec.catalyst_hint_slots = hint_list
            tec.catalyst_v4_m_slots = m_list
            self._last_plan[task_id] = -sum(1 for p in plans if p.arm != "bare")
        metrics: Dict[str, float] = {
            "v4_aux_share": aux_slots / max(total_slots, 1),
            "v4_midband_share": mid_mass / max(total_slots, 1),
            "v4_entry_slots": float(entry_built),
            "v4_hint_slots": float(hint_slots_n),
            "v4_entry_fallback_bare": float(entry_fallback_bare),
            "entry_pool_size": float(self.entry_state_pool.size()),
            "entry_pool_tasks": float(self.entry_state_pool.tasks()),
        }
        return metrics

    def _entry_eligible(self, task_id: str) -> bool:
        """角点判定:有素材(教师册或学生池)+ 调度器活跃 + **有观测**且
        裸 EMA < s_lo。

        难度画像来自离线自举 + 在线更新的合并 EMA。要求 n_bare_obs > 0:
        没见过的任务先跑裸臂拿读数,而不是按悲观缺省 0 直接吃 entry 槽——
        否则自举未覆盖的任务会在首个 epoch 挤爆 entry 预算,且其中不乏
        裸臂本来就会做的题。"""
        if not self.entry_enabled:
            return False
        has_material = task_id in self.entry_book or (
            self.entry_state_pool is not None
            and task_id in self.entry_state_pool
        )
        if not has_material:
            return False
        if not self.entry_scheduler.active(task_id):
            return False
        st = self.governor.state(task_id)
        return st.n_bare_obs > 0 and st.sr_bare_ema < self.entry_s_lo

    def _build_interval_payloads(
        self, task_id: str, global_step: int, k_e: int
    ) -> Optional[List[dict]]:
        """v3:逐槽独立采样 frac,学生池优先、教师册兜底;payload 附
        vhat(课程 critic 基线)与 source(池命中遥测)。"""
        fracs = self.entry_scheduler.plan_fracs(task_id, global_step, k_e)
        payloads: List[dict] = []
        for frac in fracs:
            plan = None
            source = "student"
            if self.entry_state_pool is not None:
                plan = self.entry_state_pool.build_plan(
                    task_id, frac=frac, max_steps=self._rollout_max_steps
                )
            if plan is None:
                source = "teacher"
                if task_id not in self.entry_book:
                    continue
                try:
                    plan = self.entry_book.build_plan(
                        task_id,
                        frac=frac,
                        rung=0,
                        max_steps=self._rollout_max_steps,
                    )
                except Exception as error:  # noqa: BLE001 - 单槽失败丢槽
                    logger.warning(
                        f"[CATALYST] entry plan build failed for task "
                        f"{task_id} frac={frac:.2f}: {error}"
                    )
                    continue
            payload = plan.to_payload()
            payload["vhat"] = float(
                self.entry_scheduler.vhat(task_id, frac)
            )
            payload["source"] = source
            payloads.append(payload)
            self._entry_pool_hits += int(source == "student")
            self._entry_plans_total += 1
        return payloads or None

    def _build_ladder_payloads(
        self, task_id: str, k_e: int
    ) -> Optional[List[dict]]:
        """v2 阶梯路径(归档对照,行为与 v2 逐字一致:同 payload × k_e)。"""
        frac, rung = self.entry_scheduler.current_frac(task_id)
        try:
            plan = self.entry_book.build_plan(
                task_id,
                frac=frac,
                rung=rung,
                max_steps=self._rollout_max_steps,
            )
        except Exception as error:  # noqa: BLE001 - 单任务失败降级为裸
            logger.warning(
                f"[CATALYST] entry plan build failed for task "
                f"{task_id}: {error}; falling back to bare"
            )
            return None
        return [plan.to_payload()] * k_e

    # -- rollout 后更新(治理 + 入池) ------------------------------------
    def _update_after_rollout_v4(
        self, trajectories: Sequence[Any], global_step: int
    ) -> Dict[str, float]:
        """v4:值表更新 + 状态池入池 + 在线校准探针(Brier)。"""
        from agentevolver.module.exp_manager.catalyst_v4 import ctx_key

        brier = []
        brier_const = []
        pool_inserted = 0
        entry_divergence = 0
        entry_degraded = 0
        sr_batch: Dict[str, List[bool]] = {"bare": [], "hint": [], "entry": []}
        for cmt in trajectories:
            if bool(getattr(cmt, "discarded", False)):
                continue
            reward = getattr(cmt, "reward", None)
            if reward is None:
                continue
            metadata = getattr(cmt, "metadata", None) or {}
            task_id = str(getattr(cmt, "task_id", ""))
            success = float(getattr(reward, "success_rate", 0.0)) > 0.0
            arm = metadata.get("catalyst_arm") or "bare"
            if bool(metadata.get("catalyst_entry_degraded")):
                entry_degraded += 1
            if arm in ("entry", "rescue"):
                entry_divergence += int(
                    metadata.get("catalyst_entry_divergence", 0) or 0
                )
            ctx = ctx_key(
                arm,
                metadata.get("catalyst_entry_frac"),
                self.v4_alloc.n_fbins,
            )
            m = metadata.get("catalyst_v4_m")
            if m is not None:
                brier.append((float(m) - (1.0 if success else 0.0)) ** 2)
                brier_const.append((0.5 - (1.0 if success else 0.0)) ** 2)
            self.v4_table.update(task_id, ctx, success)
            sr_batch.setdefault(arm, []).append(success)
            if (
                arm in ("bare", "hint")
                and success
                and not bool(metadata.get("length_truncation_terminated"))
            ):
                pool_inserted += int(
                    self.entry_state_pool.insert_from_cmt(cmt, global_step)
                )
            elif (
                getattr(self, "v4_rescue", False)
                and arm == "bare"
                and not success
            ):
                # v5:裸失败 = 救场素材("学生独立尝试且失败的现场")
                self.entry_state_pool.insert_failure_from_cmt(cmt, global_step)
        self.entry_divergence_total += entry_divergence
        self.entry_replay_failures_total += entry_degraded
        metrics: Dict[str, float] = {
            "entry_pool_inserted": float(pool_inserted),
            "entry_divergence": float(entry_divergence),
            "entry_divergence_total": float(self.entry_divergence_total),
            "entry_degraded": float(entry_degraded),
            "entry_degraded_total": float(self.entry_replay_failures_total),
        }
        if brier:
            metrics["v4_brier"] = sum(brier) / len(brier)
            metrics["v4_brier_const"] = sum(brier_const) / len(brier_const)
        for arm, outs in sr_batch.items():
            if outs:
                metrics[f"sr_{arm}_batch"] = sum(outs) / len(outs)
        return metrics

    def update_after_rollout(
        self, trajectories: Sequence[Any], global_step: int
    ) -> Dict[str, float]:
        if self.v4_enabled:
            return self._update_after_rollout_v4(trajectories, global_step)
        arm_outcomes: Dict[str, Dict[str, List[bool]]] = {}
        entry_outcomes: Dict[str, List[Any]] = {}
        inserted = 0
        pool_inserted = 0
        hint_ctx_overflow = 0
        entry_divergence = 0
        entry_degraded = 0
        entry_sr_batch: List[bool] = []
        entry_group_live: Dict[str, set] = {}
        for cmt in trajectories:
            if bool(getattr(cmt, "discarded", False)):
                continue
            reward = getattr(cmt, "reward", None)
            if reward is None:
                continue
            metadata = getattr(cmt, "metadata", None) or {}
            raw_arm = metadata.get("catalyst_arm")
            success = float(getattr(reward, "success_rate", 0.0)) > 0.0
            task_id = str(getattr(cmt, "task_id", ""))
            if bool(metadata.get("catalyst_entry_degraded")):
                # 重放失败降级为裸臂:计数示警(军规探针,预期≈0),
                # 成败照常记入裸臂 EMA(它确实是裸 rollout)。
                entry_degraded += 1
            if raw_arm == "entry":
                # entry 臂成败只喂课程调度器,**不进** governor 的
                # 裸/hint EMA(接管起点不同,混入即污染难度画像)。
                frac = float(metadata.get("catalyst_entry_frac", 0.0) or 0.0)
                entry_outcomes.setdefault(task_id, []).append((frac, success))
                entry_sr_batch.append(success)
                entry_group_live.setdefault(task_id, set()).add(success)
                entry_divergence += int(
                    metadata.get("catalyst_entry_divergence", 0) or 0
                )
                continue
            arm = "hint" if raw_arm == "hint" else "bare"
            arm_outcomes.setdefault(task_id, {"bare": [], "hint": []})[
                arm
            ].append(success)
            # ⭐ v3 学生状态池:任一非 entry 臂成功轨迹入池(纯 action/obs,
            # 无 token/think/hint;质量门在池内)。
            if (
                success
                and self.entry_state_pool is not None
                and not bool(metadata.get("length_truncation_terminated"))
            ):
                pool_inserted += int(
                    self.entry_state_pool.insert_from_cmt(cmt, global_step)
                )
            if arm == "hint":
                if metadata.get("episode_end_reason") == "context_overflow":
                    hint_ctx_overflow += 1
                if (
                    success
                    and self.replay_pool is not None
                    and not bool(metadata.get("length_truncation_terminated"))
                ):
                    inserted += int(
                        self.replay_pool.insert_from_cmt(cmt, global_step)
                    )
        metrics = self.governor.update_from_outcomes(arm_outcomes, global_step)
        metrics["replay_inserted"] = float(inserted)
        metrics["hint_ctx_overflow"] = float(hint_ctx_overflow)
        if self.replay_pool is not None:
            metrics["replay_pool_size"] = float(self.replay_pool.size())
        if self.entry_state_pool is not None:
            metrics["entry_pool_inserted"] = float(pool_inserted)
        if entry_group_live:
            metrics["entry_group_live_frac"] = sum(
                1.0 for s in entry_group_live.values() if len(s) > 1
            ) / len(entry_group_live)
        if self.entry_scheduler is not None:
            for task_id, outs in entry_outcomes.items():
                if self.entry_mode == "interval":
                    self.entry_scheduler.update(task_id, outs, global_step)
                else:
                    self.entry_scheduler.update(
                        task_id, [s for _, s in outs], global_step
                    )
            self.entry_divergence_total += entry_divergence
            self.entry_replay_failures_total += entry_degraded
            metrics["entry_divergence"] = float(entry_divergence)
            metrics["entry_divergence_total"] = float(
                self.entry_divergence_total
            )
            metrics["entry_degraded"] = float(entry_degraded)
            metrics["entry_degraded_total"] = float(
                self.entry_replay_failures_total
            )
            if entry_sr_batch:
                metrics["sr_entry_batch"] = sum(entry_sr_batch) / len(
                    entry_sr_batch
                )
        return metrics

    # -- 重放样本 ---------------------------------------------------------
    def build_replay_samples(
        self,
        tasks: Sequence[Any],
        *,
        global_step: int,
        max_prompt_len: int,
        max_response_len: int,
    ) -> Tuple[Optional[List[Any]], Dict[str, float]]:
        if self.replay_pool is None:
            return None, {}
        samples, metrics = self.replay_pool.build_replay_samples(
            tasks,
            global_step=global_step,
            max_prompt_len=max_prompt_len,
            max_response_len=max_response_len,
            # 真实 data_id 是 enumerate(tasks)(env_manager.rollout L539)
            existing_group_ids=list(range(len(tasks))),
        )
        return (samples or None), metrics

    # -- 遥测/持久化 -------------------------------------------------------
    def per_task_dump(self) -> Dict[str, Any]:
        dump = self.governor.per_task_dump()
        entry_dump = (
            self.entry_scheduler.per_task_dump()
            if self.entry_scheduler is not None
            else {}
        )
        for task_id, st in dump.items():
            planned = self._last_plan.get(task_id, 0)
            # 约定:正数 = hint 臂 k;负数 = entry 臂 −k_e
            st["k_hint_planned"] = max(planned, 0)
            st["k_entry_planned"] = max(-planned, 0)
            if task_id in entry_dump:
                st["entry"] = entry_dump[task_id]
        return dump
