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

# (phase, cell_id, tag, peak, valley, d_floor, d_ema_alpha)
SWEEP = [
    # Phase A — handoff §5 grid
    ("swA", "01", "v39b_default",     0.3, 0.05, 0.5, 0.5),
    ("swA", "02", "peak02",           0.2, 0.05, 0.5, 0.5),
    ("swA", "03", "peak04",           0.4, 0.05, 0.5, 0.5),
    ("swA", "04", "peak05",           0.5, 0.05, 0.5, 0.5),
    ("swA", "05", "peak06",           0.6, 0.05, 0.5, 0.5),
    ("swA", "06", "peak07",           0.7, 0.05, 0.5, 0.5),
    ("swA", "07", "ema02",            0.3, 0.05, 0.5, 0.2),
    ("swA", "08", "ema08",            0.3, 0.05, 0.5, 0.8),
    ("swA", "09", "floor04",          0.3, 0.05, 0.4, 0.5),
    ("swA", "10", "pk05_ema02",       0.5, 0.05, 0.5, 0.2),
    ("swA", "11", "pk05_v10",         0.5, 0.10, 0.5, 0.5),
    ("swA", "12", "pk05_ema02_v10",   0.5, 0.10, 0.5, 0.2),
    # Phase B — Plan B (raise valley)
    ("swB", "01", "pk03_v10_ema02",   0.3, 0.10, 0.5, 0.2),
    ("swB", "02", "pk03_v15_ema02",   0.3, 0.15, 0.5, 0.2),
    ("swB", "03", "pk03_v10_ema01",   0.3, 0.10, 0.5, 0.1),
    # Phase C — Plan C (precision search around swB_01 = 21.5%)
    ("swC", "01", "pk03_v10_floor04", 0.3, 0.10, 0.4, 0.2),
    ("swC", "02", "pk03_v10_floor06", 0.3, 0.10, 0.6, 0.2),
    ("swC", "03", "pk03_v12_ema02",   0.3, 0.12, 0.5, 0.2),
]

# Cells already covered by old runs (mapped (phase, cell_id) -> existing log filename stem)
PRE_EXISTING = {
    ("swA", "01"): "v39b_postfix",   # peak=0.3, valley=0.05, d_floor=0.5, d_ema=0.5
    ("swA", "07"): "v39_postfix",    # peak=0.3, valley=0.05, d_floor=0.5, d_ema=0.2
    ("swA", "09"): "v39c_postfix",   # peak=0.3, valley=0.05, d_floor=0.4, d_ema=0.5
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


VAL_KEY_QUOTED = re.compile(
    r"'val-summary/(?:webshop|alfworld)/(reward_mean_all|success_rate_mean_all)':\s*([0-9.]+)"
)
# Some runs only emit the inline " - val-summary/<env>/<key>:<float>" format
# in the per-step metric block. Fall back to that when the quoted dict form
# is missing (cut off, malformed, etc).
VAL_KEY_INLINE = re.compile(
    r"val-summary/(?:webshop|alfworld)/(reward_mean_all|success_rate_mean_all):([0-9.]+)"
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
                # Prefer quoted dict form (more precise, full-precision float)
                for m in VAL_KEY_QUOTED.finditer(line):
                    key, val = m.group(1), float(m.group(2))
                    if key == "reward_mean_all":
                        reward = val
                    elif key == "success_rate_mean_all":
                        success = val
                # Fall back to inline metric-block form
                if reward is None or success is None:
                    for m in VAL_KEY_INLINE.finditer(line):
                        key, val = m.group(1), float(m.group(2))
                        if key == "reward_mean_all" and reward is None:
                            reward = val
                        elif key == "success_rate_mean_all" and success is None:
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


def render_env_table(env: str, log_prefix: str, only_phase: str | None = None) -> str:
    lines = []
    if env == "webshop":
        lines.append("| Phase | # | Tag | peak | valley | d_floor | d_ema_α | reward_mean | success | step |")
    else:
        lines.append("| Phase | # | Tag | peak | valley | d_floor | d_ema_α | success | step |")
    lines.append("|" + "|".join(["---"] * (len(lines[0].split("|")) - 2)) + "|")
    for phase, cell_id, tag, peak, valley, d_floor, d_ema in SWEEP:
        if only_phase and phase != only_phase:
            continue
        # AF only has swA cells
        if env == "alfworld" and phase != "swA":
            continue
        # Decide which log to read
        key = (phase, cell_id)
        if key in PRE_EXISTING:
            stem = PRE_EXISTING[key]
            log_path = LOGS / f"{log_prefix}_qwen1.5b_duet_{stem}.log"
        else:
            stem = f"{phase}_{cell_id}_{tag}"
            log_path = LOGS / f"{log_prefix}_qwen1.5b_duet_{stem}.log"
        reward, success, step = parse_log(log_path)
        status = (
            f"{step}/100" if step and step < 100
            else "✓" if step and step >= 100
            else "—"
        )
        if env == "webshop":
            lines.append(
                f"| {phase} | {cell_id} | {tag} | {peak} | {valley} | {d_floor} | {d_ema} | "
                f"{fmt(reward)} | {fmt_pct(success)} | {status} |"
            )
        else:
            lines.append(
                f"| {phase} | {cell_id} | {tag} | {peak} | {valley} | {d_floor} | {d_ema} | "
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
