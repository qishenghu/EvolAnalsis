"""Parse per-step metrics from WebShop 1.5B DUET training logs.

Each step writes a single long line:
  (TaskRunner pid=...) step:N - key:val - key:val - ...

Some keys include slashes (e.g. critic/rewards_onpolicy/mean).
We match on key_name:value pairs.
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
    "chord":  "webshop_qwen1.5b_chord.log",
    "duet_v1": "webshop_qwen1.5b_duet.log",
    "duet_v10": "webshop_qwen1.5b_duet_v10.log",
    "duet_v11": "webshop_qwen1.5b_duet_v11.log",
    "duet_v12": "webshop_qwen1.5b_duet_v12.log",
    "duet_v22": "webshop_qwen1.5b_duet_v22.log",
    "duet_v23": "webshop_qwen1.5b_duet_v23.log",
    "duet_v24": "webshop_qwen1.5b_duet_v24.log",
    "duet_v25": "webshop_qwen1.5b_duet_v25.log",
    "duet_v26": "webshop_qwen1.5b_duet_v26.log",
    "duet_v28": "webshop_qwen1.5b_duet_v28.log",
    "duet_v29": "webshop_qwen1.5b_duet_v29.log",
    "duet_v30": "webshop_qwen1.5b_duet_v30.log",
    "duet_v33": "webshop_qwen1.5b_duet_v33.log",
    "duet_v36": "webshop_qwen1.5b_duet_v36.log",
    "luffy":   "webshop_qwen1.5b_luffy.log",
    "onpolicy": "webshop_qwen1.5b_onpolicy.log",
    "sft_rl": "webshop_qwen1.5b_sft_rl.log",
    "sft":    "webshop_qwen1.5b_sft.log",
}

# Keys we care about across all figures.
KEYS_OF_INTEREST = [
    # Performance
    "val-summary/webshop/reward_mean_all",
    "critic/rewards_onpolicy/mean",
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
    "duet/teacher_gradient_share",
    "state_channel/progress_onpolicy_mean",
    "state_channel/bonus_vs_reward_ratio",
    "chord/mu",
    "chord/sft_loss",
    "chord/weighted_sft_loss",
    # Training step
    "training/global_step",
]

# regex: find `key:number` where key has slashes/underscores; value is float/int
# Use lookbehind to ensure separator
PAIR_RE = re.compile(r"([A-Za-z][A-Za-z0-9_/@\-]*)[:]([\-0-9][\-0-9.eE+]*)")
STEP_RE = re.compile(r"\bstep:(\d+)\b")


def parse_log(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", errors="ignore") as f:
        for line in f:
            # Require the presence of step:N and one of our key markers
            if "step:" not in line:
                continue
            if "critic/rewards/mean" not in line and "val-summary" not in line:
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
            # Only keep lines that carry training metrics (have critic/)
            if "critic/score/mean" in row or "critic/rewards_onpolicy/mean" in row:
                rows.append(row)
    # Deduplicate by step (keep last)
    seen: dict[int, dict] = {}
    for r in rows:
        seen[r["step"]] = r
    return [seen[s] for s in sorted(seen)]


def main():
    all_data = {}
    for variant, fname in VARIANTS.items():
        rows = parse_log(LOG_DIR / fname)
        all_data[variant] = rows
        print(f"{variant:12s} from {fname}: {len(rows)} rows"
              + (f" (first step {rows[0]['step']}, last {rows[-1]['step']})" if rows else ""))
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(all_data, f)
    print(f"\nWrote {OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
