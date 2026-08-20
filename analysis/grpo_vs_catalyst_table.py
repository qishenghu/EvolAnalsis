#!/usr/bin/env python3
"""Build analysis_outputs/ckpt_sweep/grpo_vs_catalyst.md.

Reuses analysis/sweep_table.py's estimators (imported, not re-implemented) so
the two methods are scored by literally the same code.
"""
from __future__ import annotations

import collections
import glob
import json
import pathlib
import re
import sys

sys.path.insert(0, "/home/qisheng001/DUET_H200/EvolAnalsis/analysis")
from sweep_table import pass_at_k  # noqa: E402

REPO = pathlib.Path("/home/qisheng001/DUET_H200/EvolAnalsis")
SCRATCH = pathlib.Path("/projects_vol/gp_wangwy/qisheng/duet_h200")

RUNS = {
    "CATALYST": dict(sweep=SCRATCH / "ckpt_sweep", steps=[10, 20, 30, 40, 50, 60, 70],
                     exp="p0_catalyst_af_s0"),
    "GRPO": dict(sweep=SCRATCH / "ckpt_sweep_grpo", steps=[50, 100],
                 exp="p0_grpo_af_s0"),
}


def load(sweep_dir: pathlib.Path, step: int, mode: str):
    """Shards are the source of truth; a resumed collector can re-emit a slot."""
    recs = {}
    for f in sorted(glob.glob(f"{sweep_dir}/{step}_{mode}_shard[0-9].jsonl")):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                recs[r["rollout_id"]] = r
    return list(recs.values())


def metrics(sweep_dir: pathlib.Path, step: int):
    g = load(sweep_dir, step, "greedy")
    s = load(sweep_dir, step, "sampled")
    if not g and not s:
        return None
    out = {"n_greedy": len(g), "n_sampled": len(s)}
    out["greedy"] = sum(1 for r in g if r.get("success")) / len(g) if g else float("nan")
    per = collections.defaultdict(list)
    for r in s:
        per[str(r.get("task_id"))].append(bool(r.get("success")))
    if per:
        out["pass1"] = sum(sum(v) / len(v) for v in per.values()) / len(per)
        out["pass3"] = sum(pass_at_k(len(v), sum(v), 3) for v in per.values()
                           if len(v) >= 3) / len(per)
    else:
        out["pass1"] = out["pass3"] = float("nan")
    out["tasks"] = len(per)
    return out


def entropy_curve(exp: str):
    d = REPO / "checkpoints/agentevolver" / exp / "Trajectory"
    out = {}
    for f in d.glob("batch_diag_step_*.json"):
        st = int(re.search(r"step_(\d+)", f.name).group(1))
        try:
            e = json.loads(f.read_text()).get("diag/entropy_onpolicy_token_mean")
        except Exception:
            e = None
        if e is not None:
            out[st] = e
    return out


def ent_win(cv, step, k=3):
    v = [cv[x] for x in range(step - k, step + k + 1) if x in cv]
    return sum(v) / len(v) if v else None


def val_curve(exp: str):
    d = REPO / "experiments/alfworld" / exp / "validation_log"
    pts = []
    for f in sorted(d.glob("*.jsonl"), key=lambda p: int(p.stem) if p.stem.isdigit() else -1):
        if not f.stem.isdigit():
            continue
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        if rows:
            pts.append((int(f.stem),
                        sum(1 for r in rows if float(r.get("reward", 0)) > 0) / len(rows)))
    return pts


def trunc_rate(sweep_dir: pathlib.Path, step: int, mode: str):
    """Episodes stopped by the length cap — the mechanism behind the greedy gap."""
    recs = load(sweep_dir, step, mode)
    if not recs:
        return None
    n = 0
    for r in recs:
        meta = r.get("metadata") or {}
        q = meta.get("trace_quality") or {}
        t = meta.get("api_and_context_totals") or {}
        if int(q.get("length_truncated_decisions",
                     t.get("length_truncated_decisions", 0))) > 0:
            n += 1
    return n / len(recs)


def pct(x):
    return "n/a" if x is None or x != x else f"{x:.1%}"


def main():
    data = {}
    for name, cfg in RUNS.items():
        data[name] = {st: metrics(cfg["sweep"], st) for st in cfg["steps"]}
        data[name] = {k: v for k, v in data[name].items() if v}

    A = []
    w = A.append
    w("# GRPO 基线 vs CATALYST:checkpoint 扫评对照(同协议)")
    w("")
    w("> 由 `run_ckpt_sweep.pbs` 两次运行产出,解码协议逐字相同:同一份冻结的 128 题 ALFWorld")
    w("> held-out 集(`data/alfworld/val_task_ids_128_seed2025.txt`,")
    w("> sha256 `d90efe60…0187`,两次作业均在 preflight 校验通过)、")
    w("> greedy(T=0,top_p=1.0,n=1)+ sampled(T=0.9,top_p=1.0,n=4)、")
    w("> MNS=16 / GPU_MEM_UTIL=0.35 / PKD=1、每个 checkpoint 之间重启 AlfWorld 栈。")
    w("> pass@3 用 Chen et al. 无偏估计量 `1 - C(n-c,3)/C(n,3)`(n=4)。")
    w("")
    w("**回答的问题**:CATALYST 的 val 下滑已判定主要是贪心解码假象;GRPO 基线的下滑是不是真实的?")
    w("")

    for name in ("GRPO", "CATALYST"):
        cfg = RUNS[name]
        w(f"## {name}({cfg['exp']})")
        w("")
        w("| step | greedy | sampled pass@1 | pass@3 | greedy−pass@1 缺口 | 训练熵(±3步均值) "
          "| greedy 截断率 | sampled 截断率 |")
        w("|---|---|---|---|---|---|---|---|")
        ent = entropy_curve(cfg["exp"])
        if not data[name]:
            w("| — | (扫评尚未产出数据) | | | | | | |")
        for st in sorted(data[name]):
            m = data[name][st]
            e = ent_win(ent, st)
            gap = 100 * (m["pass1"] - m["greedy"])
            w(f"| {st} | {pct(m['greedy'])} | {pct(m['pass1'])} | {pct(m['pass3'])} | "
              f"{gap:+.1f}pp | {'n/a' if e is None else f'{e:.3f}'} | "
              f"{pct(trunc_rate(cfg['sweep'], st, 'greedy'))} | "
              f"{pct(trunc_rate(cfg['sweep'], st, 'sampled'))} |")
        w("")

    w("## 峰→末跌幅(checkpoint 级;正数 = 下滑)")
    w("")
    w("判据是**同一段 step 区间**上比较两种解码:以 greedy 的峰值步为起点、末点为终点,")
    w("三个指标走同一个窗口。这才是「贪心把跌幅放大了多少」的苹果对苹果比较。")
    w("")
    w("| 方法 | 窗口 | greedy 跌幅 | pass@1 跌幅 | pass@3 跌幅 | greedy/pass@1 放大倍数 |")
    w("|---|---|---|---|---|---|")
    drops = {}
    for name in ("GRPO", "CATALYST"):
        steps = sorted(data[name])
        if not steps:
            w(f"| {name} | (无数据) | | | | |")
            drops[name] = {}
            continue
        gpts = [(s, data[name][s]["greedy"]) for s in steps]
        pk_step = max(gpts, key=lambda kv: kv[1])[0]
        fin_step = steps[-1]
        d = {k: 100 * (data[name][pk_step][k] - data[name][fin_step][k])
             for k in ("greedy", "pass1", "pass3")}
        drops[name] = dict(d, peak_step=pk_step, final_step=fin_step)
        amp = ("n/a" if abs(d["pass1"]) < 1e-9 else f"{d['greedy']/d['pass1']:.2f}x")
        win = (f"{pk_step}→{fin_step}" if pk_step != fin_step
               else f"{pk_step}→{fin_step}(峰值即末点,无下滑区间)")
        w(f"| {name} | {win} | {d['greedy']:.1f}pp | {d['pass1']:.1f}pp | "
          f"{d['pass3']:.1f}pp | {amp} |")
    w("")
    w("各指标自身峰值(峰值步可能不同,仅供参考):")
    w("")
    for name in ("GRPO", "CATALYST"):
        if not data[name]:
            continue
        parts = []
        for key, label in (("greedy", "greedy"), ("pass1", "pass@1"), ("pass3", "pass@3")):
            pts = [(s, data[name][s][key]) for s in sorted(data[name])]
            pk = max(pts, key=lambda kv: kv[1])
            parts.append(f"{label} 峰 {pk[1]:.1%}@{pk[0]} → 末 {pts[-1][1]:.1%}@{pts[-1][0]} "
                         f"({100*(pk[1]-pts[-1][1]):.1f}pp)")
        w(f"- **{name}**: " + ";".join(parts))
    w("")

    w("## 在训 greedy val 曲线(trainer 自带协议,交叉验证用)")
    w("")
    for name in ("GRPO", "CATALYST"):
        pts = val_curve(RUNS[name]["exp"])
        pk = max(pts, key=lambda kv: kv[1])
        w(f"- **{name}**: " + " → ".join(f"{s}:{sr:.1%}" for s, sr in pts))
        w(f"  峰值 {pk[1]:.1%}@{pk[0]},末点 {pts[-1][1]:.1%}@{pts[-1][0]},"
          f"跌幅 {100*(pk[1]-pts[-1][1]):.1f}pp")
    w("")
    w("## 同一 step 直接对照(step 50,唯一严格对齐的点)")
    w("")
    w("| 指标 | GRPO@50 | CATALYST@50 | 差(GRPO−CATALYST) |")
    w("|---|---|---|---|")
    if 50 in data["GRPO"] and 50 in data["CATALYST"]:
        for key, label in (("greedy", "greedy"), ("pass1", "sampled pass@1"), ("pass3", "pass@3")):
            a, b = data["GRPO"][50][key], data["CATALYST"][50][key]
            w(f"| {label} | {pct(a)} | {pct(b)} | {100*(a-b):+.1f}pp |")
    w("")
    w("## 结论")
    w("")
    g = drops.get("GRPO") or {}
    if g:
        gs = sorted(data["GRPO"])
        up = all(data["GRPO"][gs[-1]][k] >= data["GRPO"][gs[0]][k]
                 for k in ("greedy", "pass1", "pass3"))
        w(f"**一句话:GRPO 基线在采样协议下没有下滑 —— 在它仅有的两个 checkpoint "
          f"(step {gs[0]} → {gs[-1]}) 上,greedy / pass@1 / pass@3 "
          f"{'三个指标全部上升' if up else '并非全部上升'},"
          f"因此「GRPO 的退化是真实的」这一假设在现有 checkpoint 上得不到支持。**")
        w("")
        w("三条必须如实标注的限定:")
        w("")
        w("1. **峰值 checkpoint 根本没存下来。** GRPO 的在训 greedy val 峰值在 step 60"
          "(55.5%),但该 run `save_freq=50`,只有 step 50 / 100 两个 checkpoint。"
          "在训曲线的 16.4pp「峰→末」跌幅(55.5%@60 → 39.1%@100)横跨的正是没有存档的区间,"
          "**本次扫评在物理上无法复现它**。扫评能确认的只是 50→100 这一段,"
          "而这一段在训曲线本身就是上升的(35.2% → 39.1%),扫评 greedy 复现为 33.6% → 39.1%。")
        w("2. **「熵低 ⇒ 贪心≈采样」这个前提被数据否定了。** GRPO step 100 的训练熵只有 "
          "0.069(全程最低),但 greedy 39.1% 与 sampled pass@1 53.5% 仍差 14.4pp。"
          "低熵并没有让贪心逼近采样。机制在截断率上:step 100 贪心有 35.2% 的 episode "
          "至少有一次决策的生成撞上长度上限(`trace_quality.length_truncated_decisions>0`),"
          "采样只有 10.0%;同一指标在 CATALYST 的 step 30–70 上恒为 0.0%。"
          "也就是说 GRPO 的贪心解码会退化成「一次决策里啰嗦到写爆预算」,"
          "而这是分布**众数**的形状问题,与 token 级平均熵高低是两回事 —— "
          "熵低只说明分布尖锐,不保证那个尖峰对应一条能走完的轨迹。")
        w("3. **CATALYST 的熵并没有「保持 0.36–0.50」。** 它同样在衰减,只是比 GRPO 晚约 25 步:"
          "step 60 已降到 0.311,step 70 降到 0.232,step 76(该 run 最后一步)0.182。"
          "两条熵曲线是同一个形状的平移,因此「治理层防住了基线熵坍缩」在这批证据上**不成立**。")
        w("")
        w("附带的反向证据:在唯一严格对齐的 step 50 上,两者 greedy 几乎相同"
          "(33.6% vs 34.4%),但 GRPO 的 sampled pass@1 高 9.6pp、pass@3 高 12.9pp。"
          "GRPO 基线在采样协议下比 CATALYST 更强,而不是更弱。")
    w("")
    w("## 与 CATALYST 的 step 不对齐说明")
    w("")
    w("两次扫评的 checkpoint 网格不同,**不能按 step 直接连成一条曲线比较**:")
    w("")
    w("- CATALYST(`p0_catalyst_af_s0`)每 10 步存一次,扫了 10–70,**没有 step 100**;"
      "其末点是 step 70。")
    w("- GRPO(`p0_grpo_af_s0`)`save_freq=50`,只有 step 50 / 100 两个 checkpoint。")
    w("- 因此「末点」含义不同:CATALYST 末点 = step 70,GRPO 末点 = step 100;"
      "唯一严格对齐的比较点是 step 50。")
    w("")
    (REPO / "analysis_outputs/ckpt_sweep").mkdir(parents=True, exist_ok=True)
    out = REPO / "analysis_outputs/ckpt_sweep/grpo_vs_catalyst.md"
    out.write_text("\n".join(A) + "\n", encoding="utf-8")
    print("\n".join(A))
    print(f"\n[written] {out}")
    print("[drops]", json.dumps(drops, indent=2))


if __name__ == "__main__":
    main()
