"""CATALYST v2 entry-k 状态课程(通路②主力化,设计:CATALYST_v2_设计_2026-08-13.md)。

角色分工:
  * CatalystEntryBook —— task_id → 接管素材(离线构建,manifest+sha 校验;
    builder: scripts/build_catalyst_entry_book.py)。素材只存教师 action 的
    规范形 ``<action>\\n…\\n</action>`` 与逐步观察,**教师 think 从不进册**
    (泄漏防御在数据层完成,而不是渲染层)。
  * EntryPlan —— 单 rollout 的接管计划(k 步重放 + seed 素材),经
    TrajExpConfig 以 payload dict 跨线程传输(worker 不碰 EntryBook)。
  * CatalystEntryScheduler —— walk-back rung 课程(fracs 梯子)+ 退休,
    带持久化;时间尺度依赖难度自举(governor.bootstrap_from_stats)消灭
    v1 的冷启动聋子期。

不变式(v2 §0):教师信息只经「初始状态选择」进入本通道;可训练 token 仅来自
本 episode 的 decision snapshot,seed 消息结构性拿不到 loss(cmt_linear
snapshot 机制),故零模仿不变式在 token 层与信息层同时成立。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from loguru import logger

ENTRY_BOOK_VERSION = "catalyst_entry_book/1.0.0"
ENTRY_SCHED_STATE_SCHEMA = "catalyst_entry_scheduler_state_v1"

# 与试点 collect_student_takeover._extract_tagged_action 同习惯:
# 只看 post-</think> 段,取最后一个 <action> 块。
_ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)


def extract_tagged_action(completion_content: str) -> str:
    """从教师 completion 提取待重放 action;无合法块 fail-fast(军规)。"""
    post_think = str(completion_content).split("</think>")[-1]
    matches = list(_ACTION_RE.finditer(post_think))
    if not matches or not matches[-1].group(1).strip():
        raise RuntimeError(
            "teacher decision has no well-formed <action>...</action> block; "
            "cannot replay"
        )
    return matches[-1].group(1).strip()


def canonical_action_message(action: str) -> str:
    """重放/seed 用的规范 action 形(与试点 env.step 发送形态逐字一致)。"""
    return f"<action>\n{action}\n</action>"


@dataclass(frozen=True)
class EntryPlan:
    """单 rollout 接管计划;payload 形态跨线程传输(纯 str/int/float/list)。"""

    task_id: str
    frac: float
    rung: int
    k_steps: int
    n_teacher_decisions: int
    teacher_rollout_id: str
    init_messages: List[Dict[str, str]]
    replay_actions: List[str]
    expected_observations: List[str]

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_payload(payload: Mapping[str, Any]) -> "EntryPlan":
        return EntryPlan(
            task_id=str(payload["task_id"]),
            frac=float(payload["frac"]),
            rung=int(payload["rung"]),
            k_steps=int(payload["k_steps"]),
            n_teacher_decisions=int(payload["n_teacher_decisions"]),
            teacher_rollout_id=str(payload["teacher_rollout_id"]),
            init_messages=[dict(m) for m in payload["init_messages"]],
            replay_actions=[str(a) for a in payload["replay_actions"]],
            expected_observations=[
                str(o) for o in payload["expected_observations"]
            ],
        )


class CatalystEntryBook:
    """接管素材册。文件 schema(builder 落盘,manifest 同目录):

    {"version": ENTRY_BOOK_VERSION, "environment": "alfworld",
     "tasks": {task_id: {"teacher_rollout_id": str,
                          "init_messages": [{"role","content"}...],
                          "steps": [{"action": str, "observation": str}...]}}}
    """

    def __init__(self, path: str, *, require_manifest: bool = True):
        book_path = Path(path)
        if not book_path.is_file():
            raise FileNotFoundError(f"[CATALYST] entry book missing: {path}")
        payload = json.loads(book_path.read_text(encoding="utf-8"))
        if payload.get("version") != ENTRY_BOOK_VERSION:
            raise RuntimeError(
                f"[CATALYST] entry book version mismatch: "
                f"{payload.get('version')} != {ENTRY_BOOK_VERSION}; "
                "rerun scripts/build_catalyst_entry_book.py"
            )
        manifest_path = book_path.with_name(book_path.name + ".manifest.json")
        if require_manifest:
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"[CATALYST] entry book manifest missing: {manifest_path}"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            digest = hashlib.sha256(
                json.dumps(
                    payload, ensure_ascii=False, sort_keys=True
                ).encode("utf-8")
            ).hexdigest()
            recorded = str(manifest.get("book_sha256", ""))
            if recorded and recorded != digest:
                raise RuntimeError(
                    f"[CATALYST] entry book sha mismatch: {digest} != "
                    f"{recorded}(册与 manifest 不同批,重跑 builder)"
                )
        self.environment = str(payload.get("environment", ""))
        self._tasks: Dict[str, Dict[str, Any]] = {
            str(k): v for k, v in (payload.get("tasks") or {}).items()
        }
        for task_id, entry in self._tasks.items():
            steps = entry.get("steps") or []
            if len(steps) < 2:
                raise RuntimeError(
                    f"[CATALYST] entry book task {task_id} has "
                    f"{len(steps)} step(s); need >= 2 for any frac < 1"
                )
        logger.info(
            f"[CATALYST] entry book loaded: {len(self._tasks)} tasks "
            f"(env={self.environment}) from {path}"
        )

    def __contains__(self, task_id: str) -> bool:
        return str(task_id) in self._tasks

    def __len__(self) -> int:
        return len(self._tasks)

    def n_decisions(self, task_id: str) -> int:
        return len(self._tasks[str(task_id)]["steps"])

    def build_plan(
        self,
        task_id: str,
        *,
        frac: float,
        rung: int,
        max_steps: int,
    ) -> EntryPlan:
        """k 边界与试点 build_takeover_plan 逐字同构:
        k = max(1, floor(frac·n)),且 k ≤ n−1(留教师 env 反馈可 seed)、
        k < max_steps(不吃光学生预算)。

        frac 的基数 n 是教师轨迹的**真决策数**(册条目记录的
        n_teacher_decisions);steps 只存前 n−1 步,天然充当 k ≤ n−1 的上限。"""
        entry = self._tasks[str(task_id)]
        steps = entry["steps"]
        n_decisions = int(entry.get("n_teacher_decisions", len(steps) + 1))
        k_steps = max(1, math.floor(float(frac) * n_decisions))
        if k_steps > len(steps):
            k_steps = len(steps)
        if k_steps >= int(max_steps):
            raise RuntimeError(
                f"[CATALYST] task {task_id}: k={k_steps} exhausts student "
                f"budget (max_steps={max_steps})"
            )
        return EntryPlan(
            task_id=str(task_id),
            frac=float(frac),
            rung=int(rung),
            k_steps=k_steps,
            n_teacher_decisions=n_decisions,
            teacher_rollout_id=str(entry.get("teacher_rollout_id", "")),
            init_messages=[dict(m) for m in entry["init_messages"]],
            replay_actions=[str(s["action"]) for s in steps[:k_steps]],
            expected_observations=[
                str(s["observation"]) for s in steps[:k_steps]
            ],
        )


class CatalystEntryReplayError(RuntimeError):
    """教师前缀重放失败(env 中途终止/传输异常)。

    rollout_env_worker 捕获后把该槽**降级为裸臂**重试(换新 env 实例)——
    半推进的 env 状态绝不能当裸 rollout 用(任务做了一半,成功率虚高,
    会污染裸臂 EMA 与 GRPO 组基线)。"""


def replay_teacher_prefix(
    env: Any,
    instance_id: str,
    plan: EntryPlan,
    tokenizer: Any,
) -> Tuple[List[Tuple[str, str]], int]:
    """逐步重放教师前 k 个 action 推进 env;返回 (seed_pairs, divergence)。

    seed_pairs = [(assistant_content, user_content)],其中 assistant 侧是
    action-only 规范形(教师 think 从不出现),user 侧用 **live 观测**。
    与试点的有意差异:试点 seed 教师记录的观测(为字节一致性),这里 seed
    live——live 是当前 env 实例的真值,万一 divergence>0,上下文不说谎;
    divergence 仍逐步与教师记录比对并计数(军规探针,试点验证 ≈0)。

    env.step 契约与 AgentFlow 主循环逐字一致(state 列表长 1、tool→user
    转换)。任何异常/中途终止 → CatalystEntryReplayError(worker 降级重试)。
    """
    from agentevolver.utils.utils import convert_tool_to_user_message

    seed_pairs: List[Tuple[str, str]] = []
    divergence = 0
    for step_index, action in enumerate(plan.replay_actions):
        try:
            env_output = env.step(
                instance_id,
                {
                    "content": canonical_action_message(action),
                    "role": "assistant",
                },
            )
        except Exception as error:  # noqa: BLE001 - 统一降级语义
            raise CatalystEntryReplayError(
                f"env.step failed during teacher replay at step "
                f"{step_index + 1}/{plan.k_steps} (task {plan.task_id}): "
                f"{error}"
            ) from error
        states = env_output.get("state")
        if not isinstance(states, list) or len(states) != 1:
            raise CatalystEntryReplayError(
                f"unexpected env state shape during replay (task "
                f"{plan.task_id} step {step_index + 1})"
            )
        state = states[0]
        if state.get("role") == "tool":
            state = convert_tool_to_user_message(
                state, tokenizer, format="qwen"
            )
        observed = str(state.get("content", ""))
        if observed != str(plan.expected_observations[step_index]):
            divergence += 1
            logger.warning(
                f"[CATALYST] entry replay divergence task={plan.task_id} "
                f"replay_step={step_index}"
            )
        if bool(env_output.get("is_terminated")):
            # 教师前缀不应终结 episode(k ≤ n−1 保证):终结 = 契约被破坏
            raise CatalystEntryReplayError(
                f"environment terminated during teacher replay at step "
                f"{step_index + 1}/{plan.k_steps} (task {plan.task_id})"
            )
        seed_pairs.append(
            (canonical_action_message(action), observed)
        )
    return seed_pairs, divergence


@dataclass
class EntryTaskState:
    rung: int = 0                 # fracs 梯子下标(0 = 最易,即最大 frac)
    sr_ema: float = 0.0           # 当前 rung 的 entry 臂成功率 EMA
    n_obs: int = 0                # 当前 rung 的累计观测(rung 变更即清零)
    low_streak: int = 0           # rung 0 上连续低 SR 更新窗计数
    retired: bool = False         # 教师前缀无用,永久停用(去其糟粕)
    graduated: bool = False       # 走完梯子,毕业回 hint/bare 策略
    retired_step: int = -1
    graduated_step: int = -1


class CatalystEntryScheduler:
    """walk-back rung 课程。

    促升:sr_ema ≥ promote_hi 且 n_obs ≥ min_obs → rung+1(更早接管,更难);
          越过末档 → graduated。
    退休:rung 0(最易档)上连续 retire_windows 个更新窗 sr_ema ≤ demote_lo
          (n_obs ≥ min_obs 后才开始累计)→ retired。
    降档:非 0 rung 上连续 retire_windows 窗 ≤ demote_lo → rung−1(回撤,
          不退休——上过档说明素材有用,只是走快了)。
    """

    def __init__(self, cfg: Mapping[str, Any]):
        fracs = list(cfg.get("fracs", [0.75, 0.5, 0.25]))
        if not fracs or any(not (0.0 < float(f) < 1.0) for f in fracs):
            raise ValueError("catalyst.entry.fracs must be in (0, 1)")
        if sorted(fracs, reverse=True) != [float(f) for f in fracs]:
            raise ValueError(
                "catalyst.entry.fracs must be strictly descending "
                "(walk-back: earlier rung = later entry = easier)"
            )
        self.fracs = [float(f) for f in fracs]
        self.slots_per_task = int(cfg.get("slots_per_task", 4))
        self.promote_hi = float(cfg.get("promote_hi", 0.5))
        self.demote_lo = float(cfg.get("demote_lo", 0.125))
        self.min_obs = int(cfg.get("min_obs", 4))
        self.ema_alpha = float(cfg.get("ema_alpha", 0.5))
        self.retire_windows = int(cfg.get("retire_windows", 3))
        if self.slots_per_task < 2:
            # 单样本臂无组内基线(与 hint 臂 min 2 同理)
            raise ValueError("catalyst.entry.slots_per_task must be >= 2")
        if not (0.0 <= self.demote_lo < self.promote_hi <= 1.0):
            raise ValueError(
                "catalyst.entry: need 0 <= demote_lo < promote_hi <= 1"
            )
        self._tasks: Dict[str, EntryTaskState] = {}
        self.retired_total = 0
        self.graduated_total = 0

    # -- 状态 ------------------------------------------------------------
    def state(self, task_id: str) -> EntryTaskState:
        return self._tasks.setdefault(str(task_id), EntryTaskState())

    def active(self, task_id: str) -> bool:
        st = self.state(task_id)
        return not (st.retired or st.graduated)

    def current_frac(self, task_id: str) -> Tuple[float, int]:
        st = self.state(task_id)
        rung = min(st.rung, len(self.fracs) - 1)
        return self.fracs[rung], rung

    # -- 更新(每步,rollout 后) -----------------------------------------
    def update(
        self, task_id: str, successes: Sequence[bool], global_step: int
    ) -> None:
        if not successes:
            return
        st = self.state(task_id)
        if st.retired or st.graduated:
            return
        batch_sr = sum(bool(s) for s in successes) / len(successes)
        if st.n_obs == 0:
            st.sr_ema = batch_sr  # 首次观测播种(与 governor 同约定)
        else:
            st.sr_ema = (
                (1.0 - self.ema_alpha) * st.sr_ema + self.ema_alpha * batch_sr
            )
        st.n_obs += len(successes)
        if st.n_obs < self.min_obs:
            return
        if st.sr_ema >= self.promote_hi:
            st.rung += 1
            st.sr_ema = 0.0
            st.n_obs = 0
            st.low_streak = 0
            if st.rung >= len(self.fracs):
                st.graduated = True
                st.graduated_step = int(global_step)
                self.graduated_total += 1
                logger.info(
                    f"[CATALYST] entry task {task_id} GRADUATED at step "
                    f"{global_step}"
                )
            return
        if st.sr_ema <= self.demote_lo:
            st.low_streak += 1
            if st.low_streak >= self.retire_windows:
                if st.rung == 0:
                    st.retired = True
                    st.retired_step = int(global_step)
                    self.retired_total += 1
                    logger.info(
                        f"[CATALYST] entry task {task_id} RETIRED at step "
                        f"{global_step} (frac={self.fracs[0]} useless)"
                    )
                else:
                    st.rung -= 1
                    st.sr_ema = 0.0
                    st.n_obs = 0
                    st.low_streak = 0
        else:
            st.low_streak = 0

    # -- 持久化 ----------------------------------------------------------
    def save_state(self, path: str) -> None:
        payload = {
            "schema": ENTRY_SCHED_STATE_SCHEMA,
            "retired_total": self.retired_total,
            "graduated_total": self.graduated_total,
            "tasks": {tid: asdict(st) for tid, st in self._tasks.items()},
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1),
            encoding="utf-8",
        )
        os.replace(tmp, p)

    def load_state(self, path: str) -> bool:
        p = Path(path)
        if not p.is_file():
            return False
        payload = json.loads(p.read_text(encoding="utf-8"))
        if payload.get("schema") != ENTRY_SCHED_STATE_SCHEMA:
            raise RuntimeError(
                f"[CATALYST] unknown entry scheduler state schema: {path}"
            )
        self.retired_total = int(payload.get("retired_total", 0))
        self.graduated_total = int(payload.get("graduated_total", 0))
        self._tasks = {
            str(tid): EntryTaskState(**st)
            for tid, st in (payload.get("tasks") or {}).items()
        }
        logger.info(
            f"[CATALYST] entry scheduler state loaded: {len(self._tasks)} "
            f"tasks, {self.retired_total} retired, "
            f"{self.graduated_total} graduated, from {path}"
        )
        return True

    def per_task_dump(self) -> Dict[str, Dict[str, Any]]:
        return {tid: asdict(st) for tid, st in sorted(self._tasks.items())}


# ===========================================================================
# v3:分布课程 + 课程 critic + 学生状态池
# (设计:CATALYST_v3_设计_2026-08-15.md;v2 阶梯调度器保留供归档/对照)
# ===========================================================================
ENTRY_V3_STATE_SCHEMA = "catalyst_entry_v3_state_v1"


def deterministic_frac(
    task_id: str, global_step: int, slot: int, f_lo: float, f_hi: float
) -> float:
    """逐槽确定性 frac 采样(可复现,无进程 RNG 状态;同任务同步不同槽不同)。"""
    digest = hashlib.sha256(
        f"catalyst_entry_frac|{task_id}|{global_step}|{slot}".encode("utf-8")
    ).digest()
    u = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return f_lo + u * (f_hi - f_lo)


@dataclass
class EntryIntervalState:
    f_lo: float = 0.5
    f_hi: float = 0.9
    all_fail_streak: int = 0     # f_lo 顶到上限后连续全败计数(退休判据)
    retired: bool = False
    graduated: bool = False
    retired_step: int = -1
    graduated_step: int = -1
    # 课程 critic:五段 [0,.2)...[.8,1) 的逐任务 EMA 与观测数
    v_bins: List[float] = field(default_factory=lambda: [0.0] * 5)
    n_bins: List[int] = field(default_factory=lambda: [0] * 5)


def frac_bin(frac: float) -> int:
    return min(4, max(0, int(float(frac) * 5.0)))


class CatalystEntryIntervalScheduler:
    """v3 分布课程 + 课程 critic。

    区间语义:frac 越高 = 起点越靠近终点 = 越易。全败 → 区间上移(更易),
    全成 → 下移(更难),混合 → 不动(已在前沿带)。毕业/退休见设计 §1.1。
    critic:V̂(task, bin) = 逐任务分段 EMA;观测 < min_task_obs 时退化到
    全局分段 EMA(每步 ~12 观测,时间尺度靠全局段解决)。
    """

    def __init__(self, cfg: Mapping[str, Any]):
        f_init = list(cfg.get("f_init", [0.5, 0.9]))
        self.f_init_lo, self.f_init_hi = float(f_init[0]), float(f_init[1])
        self.f_delta = float(cfg.get("f_delta", 0.05))
        self.f_min = float(cfg.get("f_min", 0.05))
        self.f_max = float(cfg.get("f_max", 0.95))
        self.graduate_f_hi = float(cfg.get("graduate_f_hi", 0.15))
        self.retire_f_lo = float(cfg.get("retire_f_lo", 0.85))
        self.retire_windows = int(cfg.get("retire_windows", 3))
        self.slots_per_task = int(cfg.get("slots_per_task", 4))
        self.critic_alpha = float(cfg.get("critic_alpha", 0.5))
        self.critic_min_task_obs = int(cfg.get("critic_min_task_obs", 2))
        if not (
            self.f_min < self.f_init_lo < self.f_init_hi <= self.f_max <= 1.0
        ):
            raise ValueError("catalyst.entry: f_min < f_init_lo < f_init_hi <= f_max")
        if self.slots_per_task < 2:
            raise ValueError("catalyst.entry.slots_per_task must be >= 2")
        self._tasks: Dict[str, EntryIntervalState] = {}
        # 全局分段 critic(跨任务;逐任务观测不足时的退化基线)
        self.global_v = [0.0] * 5
        self.global_n = [0] * 5
        self.retired_total = 0
        self.graduated_total = 0

    # -- 状态 ------------------------------------------------------------
    def state(self, task_id: str) -> EntryIntervalState:
        return self._tasks.setdefault(
            str(task_id),
            EntryIntervalState(f_lo=self.f_init_lo, f_hi=self.f_init_hi),
        )

    def active(self, task_id: str) -> bool:
        st = self.state(task_id)
        return not (st.retired or st.graduated)

    def plan_fracs(
        self, task_id: str, global_step: int, n_slots: int
    ) -> List[float]:
        st = self.state(task_id)
        return [
            deterministic_frac(task_id, global_step, s, st.f_lo, st.f_hi)
            for s in range(n_slots)
        ]

    # -- critic ----------------------------------------------------------
    def vhat(self, task_id: str, frac: float) -> float:
        st = self.state(task_id)
        b = frac_bin(frac)
        if st.n_bins[b] >= self.critic_min_task_obs:
            return st.v_bins[b]
        if self.global_n[b] > 0:
            return self.global_v[b]
        return 0.0  # 双冷启动:悲观 0(全成得满优势,恰是想要的开局信号)

    def _ema(self, old: float, n: int, x: float) -> float:
        if n == 0:
            return x
        return (1.0 - self.critic_alpha) * old + self.critic_alpha * x

    # -- 更新(每步,rollout 后;outcomes = [(frac, success), ...]) -------
    def update(
        self,
        task_id: str,
        outcomes: Sequence[Tuple[float, bool]],
        global_step: int,
    ) -> None:
        if not outcomes:
            return
        st = self.state(task_id)
        if st.retired or st.graduated:
            return
        for frac, success in outcomes:
            b = frac_bin(frac)
            x = 1.0 if success else 0.0
            st.v_bins[b] = self._ema(st.v_bins[b], st.n_bins[b], x)
            st.n_bins[b] += 1
            self.global_v[b] = self._ema(self.global_v[b], self.global_n[b], x)
            self.global_n[b] += 1
        succ = [s for _, s in outcomes]
        width = st.f_hi - st.f_lo
        if all(succ):
            # 全成 → 下移(更难);触底且成功 → 毕业
            st.f_hi = max(self.graduate_f_hi, st.f_hi - self.f_delta)
            st.f_lo = max(self.f_min, min(st.f_lo - self.f_delta, st.f_hi - width))
            st.all_fail_streak = 0
            if st.f_hi <= self.graduate_f_hi:
                st.graduated = True
                st.graduated_step = int(global_step)
                self.graduated_total += 1
                logger.info(
                    f"[CATALYST] entry task {task_id} GRADUATED (interval) "
                    f"at step {global_step}"
                )
        elif not any(succ):
            # 全败 → 上移(更易);顶到上限仍连败 → 退休
            at_ceiling = st.f_lo >= self.retire_f_lo
            st.f_lo = min(self.retire_f_lo, st.f_lo + self.f_delta)
            st.f_hi = min(self.f_max, max(st.f_hi + self.f_delta, st.f_lo + width))
            if at_ceiling:
                st.all_fail_streak += 1
                if st.all_fail_streak >= self.retire_windows:
                    st.retired = True
                    st.retired_step = int(global_step)
                    self.retired_total += 1
                    logger.info(
                        f"[CATALYST] entry task {task_id} RETIRED (interval) "
                        f"at step {global_step}"
                    )
            else:
                st.all_fail_streak = 0
        else:
            st.all_fail_streak = 0  # 混合:前沿带,不动

    # -- 持久化 ----------------------------------------------------------
    def save_payload(self) -> Dict[str, Any]:
        return {
            "schema": ENTRY_V3_STATE_SCHEMA,
            "retired_total": self.retired_total,
            "graduated_total": self.graduated_total,
            "global_v": list(self.global_v),
            "global_n": list(self.global_n),
            "tasks": {tid: asdict(st) for tid, st in self._tasks.items()},
        }

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        if payload.get("schema") != ENTRY_V3_STATE_SCHEMA:
            raise RuntimeError(
                f"[CATALYST] unknown entry v3 state schema: {payload.get('schema')}"
            )
        self.retired_total = int(payload.get("retired_total", 0))
        self.graduated_total = int(payload.get("graduated_total", 0))
        self.global_v = [float(x) for x in payload.get("global_v", [0.0] * 5)]
        self.global_n = [int(x) for x in payload.get("global_n", [0] * 5)]
        self._tasks = {
            str(tid): EntryIntervalState(**st)
            for tid, st in (payload.get("tasks") or {}).items()
        }

    def per_task_dump(self) -> Dict[str, Dict[str, Any]]:
        return {tid: asdict(st) for tid, st in sorted(self._tasks.items())}


class CatalystStatePool:
    """学生自产状态池(v3 通路③的燃料仓)。

    入池:任一**非 entry 臂**成功轨迹(裸/hint)的 (action, observation) 序列。
    质量门:每个 decision 有合法 <action>、决策数 ∈ [2, max_steps]、无长度截断。
    每任务上限 pool_max FIFO(新胜旧)。出池:取最新路径构建 EntryPlan
    (与教师册同 payload 契约,executor 零改动);池空由教师册兜底。

    存的是 action 文本 + 环境观察 —— 纯环境事实,无 token、无 think、无 hint,
    上下文错配(v1 之毒)与教师状态分布错配(v2 断层③)双双结构性不存在。
    """

    def __init__(self, cfg: Mapping[str, Any]):
        self.pool_max = int(cfg.get("pool_max_per_task", 4))
        self.max_steps_cap = int(cfg.get("pool_max_decisions", 30))
        self._pool: Dict[str, List[Dict[str, Any]]] = {}
        # v5 rescue:失败路径桶(2×2 第四格的燃料;只收裸臂失败,语义 =
        # "学生独立尝试且失败的现场")。垃圾滤网:无效动作占比 ≤ 0.5。
        self._fail: Dict[str, List[Dict[str, Any]]] = {}
        self.inserted_total = 0
        self.fail_inserted_total = 0
        self.rejected_total = 0

    def __contains__(self, task_id: str) -> bool:
        return bool(self._pool.get(str(task_id)))

    def has_failure(self, task_id: str) -> bool:
        return bool(self._fail.get(str(task_id)))

    def size(self) -> int:
        return sum(len(v) for v in self._pool.values())

    def tasks(self) -> int:
        return len(self._pool)

    def _extract_pairs(self, cmt: Any):
        """从 CMT 提取 (action, observation) 对序列(成功/失败通用)。
        成功轨迹形状 init+(llm,env)*(n−1)+llm;失败轨迹可能以 env 收尾——
        通用解析:按交替配对收集,尾部不完整对丢弃。"""
        full = list(getattr(cmt, "full_context", []) or [])
        msgs = [(str(m.author), str(m.content)) for m in full]
        n_init = 0
        for author, _ in msgs:
            if author == "initialization":
                n_init += 1
            else:
                break
        body = msgs[n_init:]
        pairs = []
        i = 0
        while i + 1 < len(body):
            (a_author, a_content), (e_author, e_content) = body[i], body[i + 1]
            if a_author != "llm" or e_author != "env":
                raise ValueError("alternation broken")
            action = extract_tagged_action(a_content)
            pairs.append({"action": action, "observation": e_content})
            i += 2
        return pairs

    def insert_failure_from_cmt(self, cmt: Any, global_step: int) -> bool:
        """v5 rescue:失败轨迹前缀入桶(仅裸臂失败,调用侧过滤)。"""
        try:
            pairs = self._extract_pairs(cmt)
            if not (3 <= len(pairs) <= self.max_steps_cap):
                return False
            n_noop = sum(
                1 for p in pairs if "Nothing happened" in p["observation"]
            )
            if n_noop > 0.5 * len(pairs):
                return False   # 垃圾滤网:大半无效动作的轨迹没有救场价值
            entry = {
                "source": "student_failure",
                "rollout_id": str(getattr(cmt, "rollout_id", "")),
                "inserted_step": int(global_step),
                "n_decisions": len(pairs) + 1,
                "steps": pairs,
            }
        except Exception:  # noqa: BLE001
            return False
        bucket = self._fail.setdefault(str(cmt.task_id), [])
        if len(bucket) >= self.pool_max:
            bucket.pop(0)
        bucket.append(entry)
        self.fail_inserted_total += 1
        return True

    def build_rescue_plan(
        self, task_id: str, *, frac: float, max_steps: int
    ) -> Optional[EntryPlan]:
        """取最新失败路径构建救场计划(与 entry 同 payload 契约)。"""
        bucket = self._fail.get(str(task_id))
        if not bucket:
            return None
        entry = bucket[-1]
        steps = entry["steps"]
        n_decisions = int(entry["n_decisions"])
        k_steps = max(1, math.floor(float(frac) * n_decisions))
        k_steps = min(k_steps, len(steps))
        if k_steps >= int(max_steps):
            return None
        return EntryPlan(
            task_id=str(task_id),
            frac=float(frac),
            rung=0,
            k_steps=k_steps,
            n_teacher_decisions=n_decisions,
            teacher_rollout_id=f"failure:{entry['rollout_id']}",
            init_messages=[],
            replay_actions=[str(s["action"]) for s in steps[:k_steps]],
            expected_observations=[
                str(s["observation"]) for s in steps[:k_steps]
            ],
        )

    def insert_from_cmt(self, cmt: Any, global_step: int) -> bool:
        """从成功 CMT 提取 action/obs 路径;不合格静默拒收并计数。"""
        try:
            full = list(getattr(cmt, "full_context", []) or [])
            msgs = [
                (str(m.author), str(m.content))
                for m in full
            ]
            n_init = 0
            for author, _ in msgs:
                if author == "initialization":
                    n_init += 1
                else:
                    break
            body = msgs[n_init:]
            # 成功轨迹形状:(llm, env)*(n−1) + llm(末次 env 已被弹除)
            if n_init < 1 or len(body) % 2 != 1:
                self.rejected_total += 1
                return False
            n_dec = (len(body) + 1) // 2
            if not (2 <= n_dec <= self.max_steps_cap):
                self.rejected_total += 1
                return False
            steps = []
            for i in range(n_dec - 1):
                author_a, content_a = body[2 * i]
                author_e, content_e = body[2 * i + 1]
                if author_a != "llm" or author_e != "env":
                    self.rejected_total += 1
                    return False
                action = extract_tagged_action(content_a)  # 无合法 action 抛
                steps.append({"action": action, "observation": content_e})
            entry = {
                "source": "student",
                "rollout_id": str(getattr(cmt, "rollout_id", "")),
                "inserted_step": int(global_step),
                "n_decisions": n_dec,
                "steps": steps,
            }
        except Exception:  # noqa: BLE001 - 质量门:不合格即拒收
            self.rejected_total += 1
            return False
        bucket = self._pool.setdefault(str(cmt.task_id), [])
        if len(bucket) >= self.pool_max:
            bucket.pop(0)
        bucket.append(entry)
        self.inserted_total += 1
        return True

    def build_plan(
        self, task_id: str, *, frac: float, max_steps: int
    ) -> Optional[EntryPlan]:
        """取最新学生路径构建接管计划;池空返回 None(上层走教师册兜底)。"""
        bucket = self._pool.get(str(task_id))
        if not bucket:
            return None
        entry = bucket[-1]
        steps = entry["steps"]
        n_decisions = int(entry["n_decisions"])
        k_steps = max(1, math.floor(float(frac) * n_decisions))
        k_steps = min(k_steps, len(steps))
        if k_steps >= int(max_steps):
            return None
        return EntryPlan(
            task_id=str(task_id),
            frac=float(frac),
            rung=0,
            k_steps=k_steps,
            n_teacher_decisions=n_decisions,
            teacher_rollout_id=f"student:{entry['rollout_id']}",
            init_messages=[],
            replay_actions=[str(s["action"]) for s in steps[:k_steps]],
            expected_observations=[
                str(s["observation"]) for s in steps[:k_steps]
            ],
        )

    def save_payload(self) -> Dict[str, Any]:
        return {
            "inserted_total": self.inserted_total,
            "fail_inserted_total": self.fail_inserted_total,
            "rejected_total": self.rejected_total,
            "pool": self._pool,
            "fail": self._fail,
        }

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        self.inserted_total = int(payload.get("inserted_total", 0))
        self.fail_inserted_total = int(payload.get("fail_inserted_total", 0))
        self.rejected_total = int(payload.get("rejected_total", 0))
        self._pool = {
            str(k): list(v) for k, v in (payload.get("pool") or {}).items()
        }
        self._fail = {
            str(k): list(v) for k, v in (payload.get("fail") or {}).items()
        }
