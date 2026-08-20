#!/usr/bin/env python
"""CATALYST v4(统一法则)设计的历史数据因果回放验证。

按步序重放 v3 训练 rollout(arm 由 hint 标记 + 素材册前缀对账分类),
估计器只见"当步之前"的数据,验证四个可离线检验的主张:

  V-A 信号恢复:LOO+先验基线 vs 组基线 vs v3 frozen-critic 的携梯度行占比;
  V-B 校准:因果 V̂ 对下一步结局的 Brier 分数(对照:常数 0.5 / 全局单值);
  V-C 分配回放:m(1−m) 分配的预算落点(中带命中率)+ 辅助臂份额是否自发退火;
  V-D 均值归零:各方案逐步优势均值(正质量 = 熵消耗源)。

诚实边界:离线回放改变不了"策略会因新分配而不同演化"这一事实——本验证是
必要非充分;能证伪(若连回放都不成立则设计作废),不能完全证实。
"""
import json, glob, os, re, sys, collections, math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BOOK = json.load(open(REPO/"data/catalyst_entry/alfworld_dsv4flash_entry.json"))["tasks"]
HINTS = set(json.load(open(REPO/"data/catalyst_hints/alfworld_dsv4flash.json")).keys())
LOG = REPO/"experiments/alfworld/p0_catalystv3_af_s0/rollout_log"

GOAL = re.compile(r"Your task is to: ([^\n]+)")
ACT = re.compile(r"<action>\s*(.*?)\s*</action>", re.S)

def turns(text):
    out, cur, buf = [], None, []
    for line in text.split("\n"):
        if line in ("system", "user", "assistant"):
            if cur: out.append((cur, "\n".join(buf)))
            cur, buf = line, []
        else:
            buf.append(line)
    if cur: out.append((cur, "\n".join(buf)))
    return out

def classify(r, goal2tid):
    m = GOAL.search(r["input"]); tid = goal2tid.get(m.group(1).strip()) if m else None
    if tid is None: return None
    if "[Reference approach" in r["input"]:
        return tid, ("hint", None), float(r["score"]) >= 1.0
    ts = turns(r["input"] + "\nassistant\n" + r["output"])
    a = [c for role, c in ts if role == "assistant" and "I'll follow your instructions" not in c]
    acts = [(ACT.search(c).group(1).strip() if ACT.search(c) else None) for c in a]
    k = 0
    if tid in BOOK:
        sb = [s["action"] for s in BOOK[tid]["steps"]]
        while k < len(a) and k < len(sb) and acts[k] == sb[k] and "<think>" not in a[k]:
            k += 1
    # 注:学生池路径前缀无法离线复原,晚期(pool_hit~0.9)entry 行会漏判为
    # bare(占比 ~10%,方向是把辅助份额低估),如实注明。
    if k >= 3:
        frac = k / max(k + (len(a) - k), 1)
        return tid, ("entry", min(4, int(frac * 5))), float(r["score"]) >= 1.0
    return tid, ("bare", None), float(r["score"]) >= 1.0

# ---------------- 估计器(拟议三层 Beta 收缩) ----------------
ALPHA = 0.5; N0 = 2.0
class Est:
    def __init__(self):
        self.g = collections.defaultdict(lambda: [0.0, 0])   # 全局 (ctx,bin)
        self.t = collections.defaultdict(lambda: [0.0, 0])   # (task,ctx,bin)
    @staticmethod
    def _ema(cell, x):
        v, n = cell
        cell[0] = x if n == 0 else (1-ALPHA)*v + ALPHA*x
        cell[1] = n + 1
    def prior(self, tid, ctx):
        gv, gn = self.g[ctx]; tv, tn = self.t[(tid,)+ctx]
        g = gv if gn else 0.5
        n_eff = min(tn, 3)
        return (n_eff*tv + N0*g) / (n_eff + N0) if tn else g
    def update(self, tid, ctx, succ):
        self._ema(self.g[ctx], 1.0 if succ else 0.0)
        self._ema(self.t[(tid,)+ctx], 1.0 if succ else 0.0)

def main():
    files = sorted(glob.glob(str(LOG/"[0-9]*.jsonl")), key=lambda f: int(os.path.basename(f)[:-6]))
    est = Est()
    grad_thr = 0.05
    schemes = ["group", "v3critic", "proposed"]
    live = {s: [0,0] for s in schemes}          # 携梯度行 / 总行
    meanA = {s: collections.defaultdict(list) for s in schemes}  # era -> A list
    brier = {"proposed": [], "const": [], "globalonly": []}
    alloc_stats = collections.defaultdict(lambda: [0.0,0.0,0])  # era -> [aux_share_sum, midband_mass, batches]
    v3_alloc = collections.defaultdict(lambda: [0.0,0])
    era_of = lambda s: "s01-30" if s<=30 else ("s31-60" if s<=60 else "s61-100")
    for f in files:
        step = int(os.path.basename(f)[:-6])
        tf = LOG/f"task_{step}.jsonl"
        if not tf.exists(): continue
        goal2tid = {}
        for l in open(tf):
            t = json.loads(l); m = GOAL.search(t.get("query") or "")
            if m: goal2tid[m.group(1).strip()] = str(t["task_id"])
        rows = []
        for l in open(f):
            c = classify(json.loads(l), goal2tid)
            if c: rows.append(c)
        era = era_of(step)
        # ---- V-B 校准(预测发生在 update 之前) ----
        for tid, ctx, succ in rows:
            p = est.prior(tid, ctx)
            brier["proposed"].append((p - succ)**2)
            brier["const"].append((0.5 - succ)**2)
            gv, gn = est.g[ctx]
            brier["globalonly"].append(((gv if gn else 0.5) - succ)**2)
        # ---- V-A / V-D 优势方案对比 ----
        by_g = collections.defaultdict(list)
        for tid, ctx, succ in rows: by_g[(tid, ctx)].append(succ)
        for (tid, ctx), outs in by_g.items():
            k = len(outs); m0 = est.prior(tid, ctx)
            for i, r in enumerate(outs):
                r = 1.0 if r else 0.0
                # 组基线(GRPO/分臂):组均值;全同组 → 0
                gmean = sum(outs)/k
                Ag = (r - gmean) if k > 1 else 0.0
                if len(set(outs)) == 1: Ag = 0.0
                # v3 critic:frozen 先验(entry/hint),bare 用组
                Av = (r - m0) if ctx[0] != "bare" else Ag
                # 拟议:LOO + 先验(所有 context 统一)
                loo = (sum(outs) - r)
                Ap = r - (N0*m0 + loo) / (N0 + k - 1)
                for s, A in (("group", Ag), ("v3critic", Av), ("proposed", Ap)):
                    live[s][1] += 1
                    if abs(A) > grad_thr: live[s][0] += 1
                    meanA[s][era].append(A)
        # ---- V-C 分配回放(用先验,不看本步结局) ----
        batch_tasks = sorted({tid for tid, _, _ in rows})
        aux_share = 0.0; mid_mass = 0.0; n_slots = 0
        for tid in batch_tasks:
            ctxs = [("bare", None)]
            if tid in HINTS: ctxs.append(("hint", None))
            if tid in BOOK: ctxs += [("entry", b) for b in range(5)]
            pb = est.prior(tid, ("bare", None))
            scores = {}
            for c in ctxs:
                p = est.prior(tid, c)
                s = p*(1-p)
                if c[0] != "bare":
                    s *= (1 - pb)   # 目标锚定:裸已解决 → 脚手架价值归零
                scores[c] = s
            # 8 槽:2 裸保底 + 6 按分数比例
            tot = sum(scores.values()) or 1.0
            slots = {c: 6*scores[c]/tot for c in ctxs}
            slots[("bare", None)] = slots.get(("bare", None), 0) + 2
            aux = sum(v for c, v in slots.items() if c[0] != "bare")
            aux_share += aux/8; n_slots += 1
            for c, v in slots.items():
                p = est.prior(tid, c)
                if 0.2 <= p <= 0.8: mid_mass += v
        if n_slots:
            st = alloc_stats[era]
            st[0] += aux_share/n_slots; st[1] += mid_mass/(8*n_slots); st[2] += 1
        # v3 实际辅助份额
        n_aux_actual = sum(1 for _, ctx, _ in rows if ctx[0] != "bare")
        va = v3_alloc[era]; va[0] += n_aux_actual/max(len(rows),1); va[1] += 1
        # ---- 更新估计器(本步结局最后入账) ----
        for tid, ctx, succ in rows: est.update(tid, ctx, succ)

    print("== V-A 携梯度行占比(|A|>0.05) ==")
    for s in schemes:
        print(f"  {s:>9}: {live[s][0]/max(live[s][1],1):.1%}  (n={live[s][1]})")
    print("\n== V-D 优势均值(正质量 = 熵消耗源;越接近 0 越好) ==")
    for s in schemes:
        for era in ("s01-30","s31-60","s61-100"):
            v = meanA[s][era]
            print(f"  {s:>9} {era}: mean A = {sum(v)/max(len(v),1):+.4f}")
    print("\n== V-B 校准(Brier,越低越好) ==")
    for k2, v in brier.items():
        print(f"  {k2:>10}: {sum(v)/len(v):.4f}")
    print("\n== V-C 分配回放:辅助臂份额(拟议 vs v3 实际)与中带命中 ==")
    print(f"  {'era':>8} {'拟议aux份额':>10} {'v3实际aux':>9} {'拟议中带命中':>10}")
    for era in ("s01-30","s31-60","s61-100"):
        a = alloc_stats[era]; va = v3_alloc[era]
        print(f"  {era:>8} {a[0]/max(a[2],1):>10.1%} {va[0]/max(va[1],1):>9.1%} {a[1]/max(a[2],1):>10.1%}")

if __name__ == "__main__":
    main()
