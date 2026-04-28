#!/usr/bin/env python3
"""Parse val@100 from 1.5B sweep logs and emit a markdown table.

Sources:
  - logs/webshop_qwen1.5b_duet_swA_*.log  (9 new WS sweep cells)
  - logs/alfworld_qwen1.5b_duet_swA_*.log (9 new AF sweep cells)
  - logs/{webshop,alfworld}_qwen1.5b_duet_v39{,b,c}_postfix.log  (3 cells already done)

Output: analysis_reports/1.5b_master_experiment_table.md
"""
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path("/data/home/qisheng/EvolAnalsis")
LOGS = ROOT / "logs"
OUT = ROOT / "analysis_reports/1.5b_master_experiment_table.md"

# (cell_id, tag, peak, valley, d_floor, d_ema_alpha)
SWEEP = [
    ("01", "v39b_default",     0.3, 0.05, 0.5, 0.5),
    ("02", "peak02",           0.2, 0.05, 0.5, 0.5),
    ("03", "peak04",           0.4, 0.05, 0.5, 0.5),
    ("04", "peak05",           0.5, 0.05, 0.5, 0.5),
    ("05", "peak06",           0.6, 0.05, 0.5, 0.5),
    ("06", "peak07",           0.7, 0.05, 0.5, 0.5),
    ("07", "ema02",            0.3, 0.05, 0.5, 0.2),
    ("08", "ema08",            0.3, 0.05, 0.5, 0.8),
    ("09", "floor04",          0.3, 0.05, 0.4, 0.5),
    ("10", "pk05_ema02",       0.5, 0.05, 0.5, 0.2),
    ("11", "pk05_v10",         0.5, 0.10, 0.5, 0.5),
    ("12", "pk05_ema02_v10",   0.5, 0.10, 0.5, 0.2),
]

# Cells already covered by old runs (mapped tag -> existing log filename stem)
PRE_EXISTING = {
    "01": "v39b_postfix",   # v39b = peak=0.3, valley=0.05, d_floor=0.5, d_ema=0.5
    "07": "v39_postfix",    # v39  = peak=0.3, valley=0.05, d_floor=0.5, d_ema=0.2
    "09": "v39c_postfix",   # v39c = peak=0.3, valley=0.05, d_floor=0.4, d_ema=0.5
}

# Baselines (from raw logs we've previously verified)
BASELINES_WS = [
    ("OnPolicy", 0.152, 0.005),
    ("LUFFY",    0.573, 0.055),
    ("CHORD",    0.603, 0.115),
    ("SFT+RL",   0.641, 0.185),
    ("DUET v1",  0.549, None),
    ("DUET v24 (prev SOTA)", 0.678, 0.220),
]

BASELINES_AF = [
    ("OnPolicy", 0.010, 0.010),
    ("LUFFY",    0.055, 0.055),
    ("CHORD",    0.270, 0.270),
    ("SFT+RL",   0.300, 0.300),
    ("DUET v1",  0.325, 0.325),
    ("DUET v24", 0.305, 0.305),
]


VAL_KEY = re.compile(
    r"'val-summary/(?:webshop|alfworld)/(reward_mean_all|success_rate_mean_all)':\s*([0-9.]+)"
)
STEP_KEY = re.compile(r"Training Progress:\s*(\d+)%\|[^|]*\|\s*(\d+)/(\d+)")


def find_log(env: str, name: str) -> Path | None:
    """Try multiple naming patterns to locate the log."""
    candidates = [
        LOGS / f"{env}_qwen1.5b_duet_{name}.log",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def parse_log(log_path: Path) -> tuple[float | None, float | None, int | None]:
    """Return (reward_mean_all, success_rate_mean_all, last_train_step)."""
    if log_path is None or not log_path.exists():
        return None, None, None
    reward = None
    success = None
    last_step = 0
    try:
        with log_path.open("r", errors="ignore") as f:
            for line in f:
                m = VAL_KEY.search(line)
                if m:
                    key, val = m.group(1), float(m.group(2))
                    if key == "reward_mean_all":
                        reward = val
                    elif key == "success_rate_mean_all":
                        success = val
                m = STEP_KEY.search(line)
                if m:
                    last_step = max(last_step, int(m.group(2)))
    except OSError:
        pass
    return reward, success, last_step


def fmt(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "—"
    return f"{x:.{digits}f}"


def fmt_pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x*100:.1f}%"


def render_env_table(env: str, log_prefix: str) -> str:
    lines = []
    if env == "webshop":
        lines.append("| # | Tag | peak | valley | d_floor | d_ema_α | reward_mean | success | step |")
    else:
        lines.append("| # | Tag | peak | valley | d_floor | d_ema_α | success | step |")
    lines.append("|" + "|".join(["---"] * (len(lines[0].split("|")) - 2)) + "|")
    for cell_id, tag, peak, valley, d_floor, d_ema in SWEEP:
        # Decide which log to read
        if cell_id in PRE_EXISTING:
            stem = PRE_EXISTING[cell_id]
            log_path = LOGS / f"{log_prefix}_qwen1.5b_duet_{stem}.log"
        else:
            stem = f"swA_{cell_id}_{tag}"
            log_path = LOGS / f"{log_prefix}_qwen1.5b_duet_{stem}.log"
        reward, success, step = parse_log(log_path)
        status = (
            f"{step}/100" if step and step < 100
            else "✓" if step and step >= 100
            else "—"
        )
        if env == "webshop":
            lines.append(
                f"| {cell_id} | {tag} | {peak} | {valley} | {d_floor} | {d_ema} | "
                f"{fmt(reward)} | {fmt_pct(success)} | {status} |"
            )
        else:
            lines.append(
                f"| {cell_id} | {tag} | {peak} | {valley} | {d_floor} | {d_ema} | "
                f"{fmt_pct(success)} | {status} |"
            )
    return "\n".join(lines)


def render_baseline_table(env: str) -> str:
    lines = []
    if env == "webshop":
        lines.append("| Method | reward_mean | success |")
        lines.append("|---|---|---|")
        for name, reward, success in BASELINES_WS:
            lines.append(f"| {name} | {fmt(reward)} | {fmt_pct(success)} |")
    else:
        lines.append("| Method | success |")
        lines.append("|---|---|")
        for name, _, success in BASELINES_AF:
            lines.append(f"| {name} | {fmt_pct(success)} |")
    return "\n".join(lines)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = []
    text.append("# 1.5B Master Experiment Table — DUET\\* v39b Sweep")
    text.append("")
    text.append("Auto-generated by `scripts/parse_1.5b_sweep_results.py`. Re-run anytime to refresh.")
    text.append("")
    text.append("**Hardware**: Qwen2.5-1.5B-Instruct, 4×A100-80G, 100 train steps, val@100 over 200 tasks.")
    text.append("")
    text.append("**Locked infrastructure (NOT swept)**: same as 1.5B v24 winner config (= '1.5B v39b cfg').")
    text.append("`ppo_micro_batch=2, log_prob_micro_batch=2, offload=false, gpu_mem=0.75 (WS) / 0.70 (AF), env_worker=32 (WS) / 64 (AF), n=8, n_teacher_per_task=1, T=0.6, kl=0.001 (WS) / 0.005 (AF), disc_temp=1.5, clip_max=2.0, SC β=0.2, step_level=off, teacher_baseline_separation=on`")
    text.append("")
    text.append("**Sweep dimensions (handoff §5)**: only `(chord_mu_peak, chord_mu_valley, chord_mu_d_floor, chord_mu_d_ema_alpha)`.")
    text.append("")
    text.append("---")
    text.append("")
    text.append("## WebShop")
    text.append("")
    text.append("### 1.5B Baselines (Val@100)")
    text.append("")
    text.append(render_baseline_table("webshop"))
    text.append("")
    text.append("### Sweep Results")
    text.append("")
    text.append(render_env_table("webshop", "webshop"))
    text.append("")
    text.append("---")
    text.append("")
    text.append("## ALFWorld")
    text.append("")
    text.append("### 1.5B Baselines (Val@100)")
    text.append("")
    text.append(render_baseline_table("alfworld"))
    text.append("")
    text.append("### Sweep Results")
    text.append("")
    text.append(render_env_table("alfworld", "alfworld"))
    text.append("")
    text.append("---")
    text.append("")
    text.append("## Targets")
    text.append("")
    text.append("- **WebShop SOTA**: reward_mean ≥ 0.678 (beat DUET v24)")
    text.append("- **WebShop must-beat**: reward_mean ≥ 0.641 (beat strongest baseline SFT+RL)")
    text.append("- **ALFWorld SOTA**: success ≥ 32.5% (beat DUET v1)")
    text.append("- **ALFWorld must-beat**: success ≥ 30.0% (beat SFT+RL)")
    text.append("")

    OUT.write_text("\n".join(text))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
