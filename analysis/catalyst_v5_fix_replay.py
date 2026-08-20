#!/usr/bin/env python
"""v5(rescue 第四格)失败诊断与"两级分配"修法的闭环模拟验证。

背景(2026-08-20):v5 = v4 + rescue(重放学生失败前缀+hint,5 个 fbin),
val 从 38.3@10 单调跌到 30.5@60,同种子零自举 s3(无 rescue)val@30=67.2。
诊断:rescue 使辅助 context 从 6 个变 11 个,比例分配的多重性偏置复发——
辅助恒占 40-60% 预算 → 裸槽饿死 → V̂_bare 不涨 → 锚定 (1−V̂_bare) 不关闸。

离线重放(catalyst_v4_replay_validation.py)是开环的,无法表达"分配影响
学习、学习影响分配"的反馈环,故本验证用闭环模拟,三段式:

  C-1 标定:在 P6(比例分配、6 辅助 context = s3 真实配置)上拟合两个自由
      参数(λ 全局学习率、τ 辅助行迁移系数),目标 = s3 实测轨迹
      (sr_bare 0.29→0.69@28、aux_share 40 步归零)。
  C-2 样本外证伪检验:同参数跑 P11(+rescue 5 bin,v5 真实配置)。
      若不能复现 v5 的失败形态(sr_bare 停滞≤0.45、aux 长期 0.4-0.6),
      则多重性诊断被证伪,修法作废。
  C-3 修法预测:同参数跑 T11(两级分配)与 T6(两级、无 rescue,回归检查)。
      通过标准:T11 恢复 aux 退火与裸增长,且 T6 不劣于 P6。

两级分配(拟议 v5.1):先在类型 {bare, hint, entry, rescue} 间按类型分竞价
(类型分 = 类型内最优 bin 的 m(1−m),辅助类型乘锚定 (1−m_bare)),再在类型内
按 bin 分分配——竞争单元从 12 个 context 降为 4 个类型,多重性不随 bin 数增长,
且保持 GRPO 还原性(仅剩 bare 类型时逐字还原)。

诚实边界:结局模型(hint/entry/rescue 的条件成功率形状)是假设,标定只调
λ、τ 两个标量;模拟不建模熵坍缩与分布漂移,因此只能表达"停滞",不解释
v5 val 的绝对下降。C-2 通过 = 诊断必要条件成立,非充分。

【结局,2026-08-20】C-2 不成立:P11 与 P6 动力学几乎无差 → "多重性挤占"
诊断被本脚本证伪(这正是它的价值)。随后的轨迹验尸(think 切分 + 素材对账)
找到真凶:失败桶 90% 是超时徘徊轨迹,epoch 2 开闸后 22% 份额的垃圾上下文
续写训练拖垮全线。终版结论见 docs/research/CATALYST_v5_负结果验尸_2026-08-20.md;
两级分配不再立项。本脚本保留作证伪链一环。
"""
import json, math, sys, collections
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from agentevolver.module.exp_manager.catalyst_v4 import (
    V4ValueTable, V4Allocator, v4_loo_prior_advantage, ctx_key,
)

N_ROLLOUT = 8
TASKS_PER_STEP = 16
N_STEPS = 70
N_FBINS = 5
GRAD_THR = 0.05
V4CFG = {"alpha": 0.5, "n0": 2.0, "fbins": N_FBINS, "bare_floor": 2}

# ---------------- 结局模型(形状假设,标定不触碰) ----------------
def p_success(arm, frac, a):
    """条件成功率:a = 当前裸能力。辅助 context 抬升幅度按既往实测锚:
    hint SR(v3)~0.55-0.65 当裸~0.35;entry 越接近终点越易;rescue 实测
    从 0.07 爬到 0.55(能力相关)。"""
    if arm == "bare":
        return a
    if arm == "hint":
        return a + (1 - a) * 0.30
    if arm == "entry":
        return a + (1 - a) * 0.75 * frac
    if arm == "rescue":                     # 自身失败前缀 + hint,双重脚手架
        return a + (1 - a) * (0.20 + 0.40 * frac)
    raise ValueError(arm)


class TwoLevelAllocator:
    """拟议 v5.1:类型间竞价 → 类型内选 bin。接口对齐 V4Allocator.allocate。"""

    def __init__(self, cfg):
        self.n_fbins = int(cfg.get("fbins", N_FBINS))
        self.bare_floor = int(cfg.get("bare_floor", 2))

    @staticmethod
    def _lr(quota_map):
        """最大余数法取整(确定性平局:键序)。"""
        base = {k: int(math.floor(q)) for k, q in quota_map.items()}
        rem = round(sum(quota_map.values())) - sum(base.values())
        order = sorted(quota_map, key=lambda k: (quota_map[k] - base[k], str(k)), reverse=True)
        for k in order[:rem]:
            base[k] += 1
        return base

    def allocate(self, task_id, step, n_rollout, table, *, has_hint, has_entry, has_rescue=False):
        types = {"bare": [("bare", None)]}
        if has_hint:
            types["hint"] = [("hint", None)]
        if has_entry:
            types["entry"] = [("entry", b) for b in range(self.n_fbins)]
        if has_rescue:
            types["rescue"] = [("rescue", b) for b in range(self.n_fbins)]
        m_bare = table.prior(task_id, ("bare", None))
        bin_score = {c: (lambda m: m * (1 - m))(table.prior(task_id, c))
                     for cs in types.values() for c in cs}
        type_score = {}
        for t, cs in types.items():
            s = max(bin_score[c] for c in cs)
            if t != "bare":
                s *= (1 - m_bare)               # 锚定在类型层生效
            type_score[t] = s
        free = n_rollout - self.bare_floor
        counts = collections.defaultdict(int)
        counts[("bare", None)] = self.bare_floor
        tot = sum(type_score.values())
        if tot <= 1e-12:
            counts[("bare", None)] += free      # 还原 GRPO
        else:
            t_slots = self._lr({t: free * s / tot for t, s in type_score.items()})
            for t, n in t_slots.items():
                if n <= 0:
                    continue
                cs = types[t]
                sub = sum(bin_score[c] for c in cs)
                if sub <= 1e-12:
                    counts[cs[0]] += n
                else:
                    for c, k in self._lr({c: n * bin_score[c] / sub for c in cs}).items():
                        counts[c] += k
        plans = []
        for c, n in sorted(counts.items(), key=lambda kv: str(kv[0])):
            for _ in range(n):
                frac = None
                if c[0] in ("entry", "rescue"):
                    frac = (c[1] + 0.5) / self.n_fbins
                plans.append((c[0], frac, c, table.prior(task_id, c)))
        assert sum(counts.values()) == n_rollout
        return plans


def plans_from_v4(alloc, task_id, step, table, **kw):
    """把生产 V4Allocator 的 SlotPlan 归一成 (arm, frac, ctx, prior)。"""
    out = []
    for p in alloc.allocate(task_id, step, N_ROLLOUT, table, **kw):
        ctx = ctx_key(p.arm, p.frac) if p.arm in ("entry", "rescue") else (p.arm, None)
        out.append((p.arm, p.frac, ctx, p.m))
    return out


def simulate(alloc_kind, n_aux, lam, tau, a0, seed):
    """closed-loop:分配 → 采样结局 → LOO+先验优势 → 全局能力更新 → 值表更新。
    alloc_kind ∈ {P, T};n_aux ∈ {6, 11}(11 = 含 rescue)。"""
    rng = np.random.default_rng(seed)
    n_tasks = len(a0)
    table = V4ValueTable(V4CFG)
    alloc = V4Allocator(V4CFG) if alloc_kind == "P" else TwoLevelAllocator(V4CFG)
    has_rescue_cfg = (n_aux == 11)
    G = 0.0
    failed_once = np.zeros(n_tasks, bool)
    hist = []
    order = rng.permutation(n_tasks)
    for step in range(1, N_STEPS + 1):
        lo = ((step - 1) * TASKS_PER_STEP) % n_tasks
        if lo + TASKS_PER_STEP > n_tasks:
            order = rng.permutation(n_tasks)
            lo = 0
        batch = order[lo:lo + TASKS_PER_STEP]
        n_bare_rows = n_aux_rows = 0
        bare_succ = bare_tot = 0
        sig_bare = sig_aux = 0.0
        for tid in batch:
            a = a0[tid] + (1 - a0[tid]) * (1 - math.exp(-G))
            kw = dict(has_hint=True, has_entry=True,
                      has_rescue=has_rescue_cfg and bool(failed_once[tid]))
            plans = (plans_from_v4(alloc, str(tid), step, table, **kw)
                     if alloc_kind == "P" else
                     alloc.allocate(str(tid), step, N_ROLLOUT, table, **kw))
            by_ctx = collections.defaultdict(list)
            for arm, frac, ctx, prior in plans:
                succ = bool(rng.random() < p_success(arm, frac or 0.0, a))
                by_ctx[ctx].append((succ, prior))
                if arm == "bare":
                    n_bare_rows += 1; bare_tot += 1; bare_succ += succ
                    if not succ:
                        failed_once[tid] = True
                else:
                    n_aux_rows += 1
            for ctx, rows in by_ctx.items():
                rewards = [1.0 if s else 0.0 for s, _ in rows]
                priors = [p for _, p in rows]
                advs = v4_loo_prior_advantage(rewards, priors, V4CFG["n0"])
                for (succ, _), adv in zip(rows, advs):
                    if abs(adv) > GRAD_THR:
                        if ctx[0] == "bare":
                            sig_bare += 1
                        else:
                            sig_aux += 1
                    table.update(str(tid), ctx, succ)
        total_rows = n_bare_rows + n_aux_rows
        G += lam * (sig_bare + tau * sig_aux) / total_rows
        ability = float(np.mean(a0 + (1 - a0) * (1 - math.exp(-G))))
        hist.append(dict(step=step, aux=n_aux_rows / total_rows,
                         sr_bare=bare_succ / max(bare_tot, 1), ability=ability))
    return hist


def summarize(hist, at=(10, 28, 40, 55, 70)):
    d = {h["step"]: h for h in hist}
    return {s: d[s] for s in at if s in d}


def run_setting(alloc_kind, n_aux, lam, tau, a0, seeds=(0, 1, 2)):
    per = [simulate(alloc_kind, n_aux, lam, tau, a0, s) for s in seeds]
    out = []
    for i in range(len(per[0])):
        out.append({k: float(np.mean([h[i][k] for h in per])) for k in per[0][0]})
    return out


def main():
    stats = json.load(open(REPO / "data/catalyst_entry/alfworld_task_stats.json"))["tasks"]
    p0 = np.array([float(v.get("sr_bare", 0.0)) for v in stats.values()])
    a0 = np.clip(p0 * 0.58, 0.02, 0.95)       # 新策略起点:批裸 SR ≈ 0.30(v5 实测 0.29@3)
    print(f"tasks={len(a0)}  初始裸能力均值={a0.mean():.3f}")

    # ---- C-1 标定(P6 拟合 s3 实测:sr_bare≈0.69@28,aux→0@40) ----
    best, best_loss = None, 1e9
    for lam in (0.05, 0.10, 0.15, 0.22, 0.30):
        for tau in (0.0, 0.15, 0.30, 0.50):
            h = summarize(run_setting("P", 6, lam, tau, a0, seeds=(0, 1)))
            loss = (h[28]["sr_bare"] - 0.69) ** 2 + (h[40]["aux"] - 0.0) ** 2 * 0.5
            if loss < best_loss:
                best, best_loss = (lam, tau), loss
    lam, tau = best
    print(f"\nC-1 标定:λ={lam} τ={tau}(loss={best_loss:.4f})")

    settings = [("P", 6, "P6  =v4/s3 生产配置"), ("P", 11, "P11 =v5 生产配置"),
                ("T", 6, "T6  =两级·无rescue"), ("T", 11, "T11 =两级·含rescue(拟议 v5.1)")]
    results = {}
    for kind, n_aux, label in settings:
        results[(kind, n_aux)] = run_setting(kind, n_aux, lam, tau, a0)
        s = summarize(results[(kind, n_aux)])
        row = "  ".join(f"@{k}: aux={v['aux']:.2f} srb={v['sr_bare']:.2f} abl={v['ability']:.2f}"
                        for k, v in s.items())
        print(f"\n{label}\n  {row}")

    # ---- 裁决 ----
    s_p6, s_p11 = summarize(results[("P", 6)]), summarize(results[("P", 11)])
    s_t6, s_t11 = summarize(results[("T", 6)]), summarize(results[("T", 11)])
    c2 = (s_p11[70]["sr_bare"] < s_p6[40]["sr_bare"] - 0.10) and (s_p11[55]["aux"] > 0.25)
    c3 = (s_t11[70]["ability"] > s_p11[70]["ability"] + 0.05) and (s_t11[55]["aux"] < s_p11[55]["aux"] - 0.10)
    reg = s_t6[70]["ability"] > s_p6[70]["ability"] - 0.03
    print("\n== 裁决 ==")
    print(f"C-2 P11 复现 v5 失败形态(裸停滞+aux不关闸): {'成立' if c2 else '不成立 → 诊断证伪'}")
    print(f"C-3 T11 修复(能力 +{s_t11[70]['ability']-s_p11[70]['ability']:+.3f}, aux@55 {s_p11[55]['aux']:.2f}→{s_t11[55]['aux']:.2f}): {'通过' if c3 else '未通过'}")
    print(f"回归 T6 不劣于 P6(Δ能力@70 {s_t6[70]['ability']-s_p6[70]['ability']:+.3f}): {'通过' if reg else '未通过'}")


if __name__ == "__main__":
    main()
