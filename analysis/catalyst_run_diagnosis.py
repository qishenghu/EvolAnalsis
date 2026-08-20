"""CATALYST 主跑退化诊断(2026-08-10)。

回答一个问题:**为什么转写通路把前 30 步拉高了,却守不住后期?**
输入全部来自已落盘产物,零 GPU:
  * 训练日志(逐步 metrics:val / 熵 / 长度 / catalyst 遥测)
  * rollout_log/catalyst_gov_step_*.json(逐任务 SR_bare/SR_hint/U/退休/k_hint)
  * validation_log/*.jsonl(评测轨迹,用于长度截断与格式失败归因)

用法:
  python analysis/catalyst_run_diagnosis.py \
      --train-log $SCRATCH/logs/p0_q35af.train.log \
      --exp-dir experiments/alfworld/p0_catalyst_af_s0 \
      --run-name p0_catalyst_af_s0 --out analysis_outputs/catalyst_diag
"""

import argparse
import collections
import json
import re
from pathlib import Path

STEP_RE = re.compile(r"step:(\d+) - ")
METRIC_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_/@.\-]*):(-?[0-9.]+(?:[eE][+-]?\d+)?)")
WANTED = (
    "val-core/alfworld/reward/mean@1",
    "critic/success_onpolicy/mean",
    "actor/entropy_loss",
    "response_length/mean",
    "catalyst/u_ema_mean",
    "catalyst/rho_mean",
    "catalyst/hint_rollout_frac",
    "catalyst/sr_bare_ema_mean",
    "catalyst/sr_hint_ema_mean",
    "catalyst/tasks_r0",
    "catalyst/tasks_r1",
    "catalyst/tasks_retired_total",
    "catalyst/w_mean",
    "catalyst/w_p10",
    "catalyst/replay_samples_in_batch",
    "catalyst/replay_pool_age_mean",
    "catalyst/bc_loss",
    "catalyst/replay_audit_failures_total",
    "catalyst/replay_dp_dropped",
    "diag/entropy_onpolicy_token_mean",
)


def parse_train_log(path, run_marker=None):
    """逐步收集 metrics。日志被多次运行追加(resume),同一 step 取最后一次出现
    的值即可——续跑写在后面。键名含 '-'/'@'(如 val-core/...@1)故用宽正则。"""
    merged = {}
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        step = int(m.group(1))
        row = merged.setdefault(step, {})
        for key, val in METRIC_RE.findall(line):
            if key in WANTED or key.startswith("catalyst/"):
                try:
                    row[key] = float(val)
                except ValueError:
                    pass
    return merged


def load_gov(exp_dir: Path):
    out = {}
    for f in sorted((exp_dir / "rollout_log").glob("catalyst_gov_step_*.json")):
        step = int(f.stem.rsplit("_", 1)[1])
        try:
            out[step] = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-log", required=True)
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    steps = parse_train_log(Path(a.train_log), a.run_name)
    gov = load_gov(Path(a.exp_dir))
    lines = [f"# CATALYST 运行诊断:{a.run_name}", ""]

    # ---- 1) 主曲线 ----
    lines.append("## 1. 逐步主指标")
    lines.append("| step | val | on-policy SR | entropy | resp len | U | rho | hint% | w_p10 | replay/batch |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    def g(row, k, fmt="{:.3f}"):
        v = row.get(k)
        return fmt.format(v) if isinstance(v, float) else "-"
    for step in sorted(steps):
        r = steps[step]
        if step % 5 and "val-core/alfworld/reward/mean@1" not in r:
            continue
        lines.append(
            f"| {step} | {g(r,'val-core/alfworld/reward/mean@1')} | "
            f"{g(r,'critic/success_onpolicy/mean')} | {g(r,'actor/entropy_loss')} | "
            f"{g(r,'response_length/mean','{:.0f}')} | {g(r,'catalyst/u_ema_mean')} | "
            f"{g(r,'catalyst/rho_mean')} | {g(r,'catalyst/hint_rollout_frac')} | "
            f"{g(r,'catalyst/w_p10')} | {g(r,'catalyst/replay_samples_in_batch','{:.0f}')} |"
        )

    # ---- 2) 军规与健康度(必须恒零的量)----
    lines += ["", "## 2. 军规/健康度(应恒为 0)"]
    for key in ("catalyst/replay_audit_failures_total", "catalyst/replay_render_drops_total",
                "catalyst/replay_insert_skips_total", "catalyst/hint_ctx_overflow"):
        vals = [r.get(key) for r in steps.values() if key in r]
        lines.append(f"- `{key}`: max={max(vals) if vals else '-'} (n={len(vals)})")

    # ---- 3) 治理动力学:任务级迁移与退休 ----
    lines += ["", "## 3. 治理动力学(逐任务 dump)"]
    if gov:
        lines.append("| step | tasks | retired | k_hint>0 | mean U | U<=0 占比 | SR_bare 均值 |")
        lines.append("|---|---|---|---|---|---|---|")
        for step in sorted(gov)[:: max(1, len(gov) // 12)]:
            d = gov[step]
            n = len(d)
            retired = sum(1 for v in d.values() if v.get("retired"))
            khint = sum(1 for v in d.values() if (v.get("k_hint_planned") or 0) > 0)
            us = [v.get("sr_hint_ema", 0) - v.get("sr_bare_ema", 0) for v in d.values()]
            bare = [v.get("sr_bare_ema", 0) for v in d.values()]
            lines.append(
                f"| {step} | {n} | {retired} | {khint} | {sum(us)/max(n,1):.3f} | "
                f"{sum(1 for u in us if u <= 0)/max(n,1):.1%} | {sum(bare)/max(n,1):.3f} |"
            )
        last = gov[max(gov)]
        lines += [
            "",
            f"末步任务数 {len(last)};退休 {sum(1 for v in last.values() if v.get('retired'))};"
            f"提示臂在编 {sum(1 for v in last.values() if (v.get('k_hint_planned') or 0) > 0)}",
        ]
    else:
        lines.append("(无 gov dump)")

    # ---- 4) 退化归因线索 ----
    lines += ["", "## 4. 退化归因线索"]
    vals = [(s, r["val-core/alfworld/reward/mean@1"]) for s, r in sorted(steps.items())
            if "val-core/alfworld/reward/mean@1" in r]
    if vals:
        peak_step, peak = max(vals, key=lambda x: x[1])
        last_step, last = vals[-1]
        lines.append(f"- val 峰值 {peak:.3f} @step{peak_step};末值 {last:.3f} @step{last_step}"
                     f"(落差 {peak-last:+.3f})")
        def window(lo, hi, key):
            xs = [r[key] for s, r in steps.items() if lo <= s <= hi and key in r]
            return sum(xs)/len(xs) if xs else None
        for key in ("actor/entropy_loss", "response_length/mean", "catalyst/u_ema_mean",
                    "catalyst/rho_mean", "catalyst/w_p10", "catalyst/bc_loss",
                    "critic/success_onpolicy/mean"):
            pre = window(1, peak_step, key)
            post = window(peak_step + 1, last_step, key)
            if pre is not None and post is not None:
                lines.append(f"- `{key}`:峰前均值 {pre:.3f} → 峰后 {post:.3f}({post-pre:+.3f})")
    (outdir / f"{a.run_name}_diagnosis.md").write_text("\n".join(lines), encoding="utf-8")
    json.dump({"steps": steps}, open(outdir / f"{a.run_name}_steps.json", "w"))
    print("\n".join(lines[:60]))
    print(f"\n[written] {outdir}/{a.run_name}_diagnosis.md")


if __name__ == "__main__":
    main()
