"""Parse per-step metrics from ALFWorld 1.5B DUET training logs (v1 and v24).

Mirrors _parse_logs.py but for ALFWorld.
"""
import json
import os
import re
import sys
from pathlib import Path

LOG_DIR = Path("/data/home/qisheng/EvolAnalsis/logs")
OUT_DIR = Path("/data/home/qisheng/EvolAnalsis/analysis_reports/_parsed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Variant name -> log file basename
VARIANTS = {
    "alfworld_duet_v1": "alfworld_qwen1.5b_duet.log",
    "alfworld_duet_v24": "alfworld_qwen1.5b_duet_v24.log",
    "alfworld_chord": "alfworld_qwen1.5b_chord.log",
    "alfworld_luffy": "alfworld_qwen1.5b_luffy.log",
    "alfworld_onpolicy": "alfworld_qwen1.5b_onpolicy.log",
}

# Keys we care about across all figures.
KEYS_OF_INTEREST = [
    # Performance
    "critic/rewards_onpolicy/mean",
    "critic/rewards/mean",
    "critic/score/mean",
    "critic/success_onpolicy/mean",
    # Policy dynamics
    "actor/kl_loss",
    "actor/entropy_loss",
    "actor/grad_norm",
    "actor/pg_loss",
    # Length
    "response_length/mean",
    # DUET channels
    "dr3/disc_acc",
    "dr3/w_off_mean",
    "dr3/ess_ratio",
    "duet/teacher_gradient_share",
    "duet/adv_teacher_effective_mean",
    "duet/adv_teacher_effective_abs_mean",
    "duet/adv_onpolicy_effective_mean",
    "duet/adv_onpolicy_effective_abs_mean",
    "duet/adv_onpolicy_effective_std",
    "duet/group_mixed_ratio",
    "state_channel/progress_onpolicy_mean",
    "state_channel/progress_teacher_mean",
    "state_channel/progress_mean",
    "state_channel/bonus_vs_reward_ratio",
    "state_channel/bonus_total_mean",
    "state_channel/beta_effective",
    "chord/mu",
    "chord/sft_loss",
    "chord/weighted_sft_loss",
    "chord/phi_mean",
    "chord/weighted_sft_nonzero_ratio",
    # Diag
    "diag/teacher_sample_ratio",
    "diag/reward_onpolicy_mean",
    "diag/reward_teacher_mean",
    # Training step
    "training/global_step",
]

# regex: find `key:number` where key has slashes/underscores; value is float/int
PAIR_RE = re.compile(r"([A-Za-z][A-Za-z0-9_/@\-]*)[:]([\-0-9][\-0-9.eE+]*)")
STEP_RE = re.compile(r"\bstep:(\d+)\b")


def parse_log(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if "step:" not in line:
                continue
            # must be a training metric line (not validation summary or noise)
            if "critic/" not in line and "actor/" not in line:
                continue
            step_m = STEP_RE.search(line)
            if not step_m:
                continue
            step = int(step_m.group(1))
            if step <= 0 or step > 200:
                continue
            # Collect pairs
            row = {"step": step}
            for k, v in PAIR_RE.findall(line):
                if k in KEYS_OF_INTEREST:
                    try:
                        row[k] = float(v)
                    except ValueError:
                        continue
            # Only keep lines that carry training metrics
            if "critic/score/mean" in row or "critic/rewards_onpolicy/mean" in row or "actor/grad_norm" in row:
                rows.append(row)
    # Merge rows with same step (keep most populated version, overlay)
    merged: dict[int, dict] = {}
    for r in rows:
        s = r["step"]
        if s not in merged:
            merged[s] = r
        else:
            # overlay: keep latest non-None values
            for k, v in r.items():
                merged[s][k] = v
    return [merged[s] for s in sorted(merged)]


def main():
    all_data = {}
    for variant, fname in VARIANTS.items():
        rows = parse_log(LOG_DIR / fname)
        all_data[variant] = rows
        print(f"{variant:24s} from {fname}: {len(rows)} rows"
              + (f" (first step {rows[0]['step']}, last {rows[-1]['step']})" if rows else ""))
    out_path = OUT_DIR / "alfworld_metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_data, f)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
