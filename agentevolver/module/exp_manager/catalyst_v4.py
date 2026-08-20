"""CATALYST v4:统一分配法则(设计:CATALYST_进展汇报_2026-08-17.md §六)。

一个任务的一族练法 context c ∈ {bare, hint, entry@fbin(5 段)}:
  估计  V̂(task, c) = 三层收缩(全局分段 EMA ⊕ 逐任务 EMA,精度加权);
  排产  裸:m(1−m);辅助:m(1−m)·(1−m_bare)(目标锚定);每任务保底 2 裸槽;
  计账  A_i = r_i − (n₀·m + Σ_{j≠i} r_j) / (n₀ + k − 1)(LOO+先验统一基线,
        所有 context 一视同仁;trainer 侧统一覆写)。

性质:context 集合退化为 {bare} 且 n₀→0 时逐字还原 RLOO;课程的出现、聚焦、
退场全部是该法则的推论。参数共 4 个:n₀、α、f 段数、裸保底槽。

历史数据因果回放验证(analysis/catalyst_v4_replay_validation.py):
  携梯度行 77.2%(组基线 43.0%);锚定退火 39.8→33.9%;优势均值无系统正偏。
本模块的估计器与该脚本**逐式一致**,由回放一致性测试钉死(tests/test_catalyst_v4.py)。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from loguru import logger

V4_STATE_SCHEMA = "catalyst_v4_state_v1"
N_FBINS_DEFAULT = 5


def ctx_key(arm: str, frac: Optional[float] = None, n_fbins: int = N_FBINS_DEFAULT):
    """context 键:("bare",None)/("hint",None)/("entry",fbin)/("rescue",fbin)。
    rescue(v5)= 学生失败前缀重放 + 提示,2×2 练习空间的第四格。"""
    if arm in ("entry", "rescue"):
        b = min(n_fbins - 1, max(0, int(float(frac) * n_fbins)))
        return (str(arm), b)
    return (str(arm), None)


class V4ValueTable:
    """三层收缩 V̂。与回放验证脚本逐式一致(ALPHA/N0/n_eff 封顶 3/冷启动 0.5)。"""

    def __init__(self, cfg: Mapping[str, Any]):
        self.alpha = float(cfg.get("alpha", 0.5))
        self.n0 = float(cfg.get("n0", 2.0))
        self.n_eff_cap = int(cfg.get("n_eff_cap", 3))
        if not (0.0 < self.alpha <= 1.0) or self.n0 <= 0:
            raise ValueError("catalyst.v4: alpha in (0,1], n0 > 0")
        self._g: Dict[Any, List[float]] = {}      # ctx -> [ema, n]
        self._t: Dict[Any, List[float]] = {}      # (task, ctx) -> [ema, n]

    def _ema(self, table: Dict, key: Any, x: float) -> None:
        cell = table.setdefault(key, [0.0, 0])
        cell[0] = x if cell[1] == 0 else (1 - self.alpha) * cell[0] + self.alpha * x
        cell[1] += 1

    def prior(self, task_id: str, ctx) -> float:
        gv, gn = self._g.get(ctx, (0.5, 0))
        g = gv if gn else 0.5
        tv, tn = self._t.get((str(task_id), ctx), (0.0, 0))
        if not tn:
            return g
        n_eff = min(tn, self.n_eff_cap)
        return (n_eff * tv + self.n0 * g) / (n_eff + self.n0)

    def update(self, task_id: str, ctx, success: bool) -> None:
        x = 1.0 if success else 0.0
        self._ema(self._g, ctx, x)
        self._ema(self._t, (str(task_id), ctx), x)

    def bootstrap_bare(self, stats: Mapping[str, Mapping[str, Any]]) -> int:
        """难度自举:只播种无在线观测的 (task, bare)。全局层不播种(数步自愈)。"""
        seeded = 0
        for task_id, row in stats.items():
            key = (str(task_id), ("bare", None))
            if key in self._t:
                continue
            n = int(row.get("n_bare", 0))
            if n <= 0:
                continue
            self._t[key] = [float(row.get("sr_bare", 0.0)), min(n, self.n_eff_cap)]
            seeded += 1
        logger.info(f"[CATALYST-v4] value table bootstrapped {seeded} bare cells")
        return seeded

    # -- 持久化 ----------------------------------------------------------
    def save_payload(self) -> Dict[str, Any]:
        enc = lambda k: json.dumps(k, ensure_ascii=False)
        return {
            "g": {enc(k): v for k, v in self._g.items()},
            "t": {enc(list(k)): v for k, v in self._t.items()},
        }

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        dec = lambda s: json.loads(s)
        def key_g(s):
            k = dec(s)
            return (k[0], None if k[1] is None else int(k[1]))
        self._g = {key_g(k): [float(v[0]), int(v[1])] for k, v in (payload.get("g") or {}).items()}
        def key_t(s):
            task, ctx = dec(s)
            return (str(task), (ctx[0], None if ctx[1] is None else int(ctx[1])))
        self._t = {key_t(k): [float(v[0]), int(v[1])] for k, v in (payload.get("t") or {}).items()}


@dataclass(frozen=True)
class SlotPlan:
    """单槽排产结果(payload 交给既有 hint/entry 执行链)。"""
    arm: str                     # "bare" | "hint" | "entry"
    frac: Optional[float]        # entry 专用(段内确定性采样)
    m: float                     # plan 时冻结的 V̂(基线先验,经 extras 透传)


class V4Allocator:
    """目标锚定分配:6 个自由槽按分数比例(最大余数法取整,哈希决平局),
    2 个裸保底槽。分数:裸 m(1−m);辅助 m(1−m)·(1−m_bare)。"""

    def __init__(self, cfg: Mapping[str, Any]):
        self.n_fbins = int(cfg.get("fbins", N_FBINS_DEFAULT))
        self.bare_floor = int(cfg.get("bare_floor", 2))
        if self.bare_floor < 1:
            raise ValueError("catalyst.v4.bare_floor must be >= 1(目标分布锚定)")

    @staticmethod
    def _hash_u(task_id: str, step: int, tag: str) -> float:
        d = hashlib.sha256(f"v4|{task_id}|{step}|{tag}".encode()).digest()
        return int.from_bytes(d[:8], "big") / float(1 << 64)

    def frac_in_bin(self, task_id: str, step: int, slot: int, b: int) -> float:
        lo, hi = b / self.n_fbins, (b + 1) / self.n_fbins
        u = self._hash_u(task_id, step, f"frac{slot}")
        return min(0.95, max(0.05, lo + u * (hi - lo)))

    def allocate(
        self,
        task_id: str,
        step: int,
        n_rollout: int,
        table: V4ValueTable,
        *,
        has_hint: bool,
        has_entry: bool,
        has_rescue: bool = False,
    ) -> List[SlotPlan]:
        ctxs: List[Any] = [("bare", None)]
        if has_hint:
            ctxs.append(("hint", None))
        if has_entry:
            ctxs += [("entry", b) for b in range(self.n_fbins)]
        if has_rescue:
            ctxs += [("rescue", b) for b in range(self.n_fbins)]
        m_bare = table.prior(task_id, ("bare", None))
        scores: Dict[Any, float] = {}
        for c in ctxs:
            m = table.prior(task_id, c)
            s = m * (1 - m)
            if c[0] != "bare":
                s *= (1 - m_bare)          # 目标锚定(回放验证的退火来源)
            scores[c] = s
        free = n_rollout - self.bare_floor
        counts = {c: 0 for c in ctxs}
        counts[("bare", None)] = self.bare_floor
        tot = sum(scores.values())
        if tot <= 1e-12:
            counts[("bare", None)] += free      # 全无信号 → 全裸(还原 GRPO)
        else:
            quota = {c: free * scores[c] / tot for c in ctxs}
            base = {c: int(math.floor(q)) for c, q in quota.items()}
            rem = free - sum(base.values())
            order = sorted(
                ctxs,
                key=lambda c: (quota[c] - base[c], self._hash_u(task_id, step, str(c))),
                reverse=True,
            )
            for c in order[:rem]:
                base[c] += 1
            for c, n in base.items():
                counts[c] += n
        plans: List[SlotPlan] = []
        slot = 0
        # 槽位布局:entry/rescue 在前、hint 次之、bare 收尾(确定性,便于审计)
        order = {"entry": 0, "rescue": 1, "hint": 2, "bare": 3}
        for c in sorted(ctxs, key=lambda c: (order[c[0]], c[1] or 0)):
            for _ in range(counts.get(c, 0)):
                if c[0] in ("entry", "rescue"):
                    frac = self.frac_in_bin(task_id, step, slot, c[1])
                    plans.append(SlotPlan(c[0], frac, table.prior(task_id, c)))
                else:
                    plans.append(SlotPlan(c[0], None, table.prior(task_id, c)))
                slot += 1
        assert len(plans) == n_rollout, f"allocation produced {len(plans)} != {n_rollout}"
        return plans


def v4_loo_prior_advantage(
    rewards: Sequence[float], priors: Sequence[float], n0: float
) -> List[float]:
    """LOO+先验统一优势(闭式,与回放脚本逐式一致;trainer 与测试共用)。"""
    k = len(rewards)
    out = []
    s = sum(rewards)
    for i, r in enumerate(rewards):
        b = (n0 * priors[i] + (s - r)) / (n0 + k - 1)
        out.append(r - b)
    return out


class CatalystV4State:
    """v4 持久化包(值表;分配器无状态)。"""

    @staticmethod
    def save(path: str, table: V4ValueTable) -> None:
        payload = {"schema": V4_STATE_SCHEMA, "table": table.save_payload()}
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        os.replace(tmp, p)

    @staticmethod
    def load(path: str, table: V4ValueTable) -> bool:
        p = Path(path)
        if not p.is_file():
            return False
        payload = json.loads(p.read_text(encoding="utf-8"))
        if payload.get("schema") != V4_STATE_SCHEMA:
            raise RuntimeError(f"[CATALYST-v4] unknown state schema at {path}")
        table.load_payload(payload["table"])
        logger.info(f"[CATALYST-v4] value table loaded from {path}")
        return True
