"""Parse ALL candidate adaptive-mu signals from per-step log lines.

Writes a single JSON keyed by variant -> list of per-step rows.
Includes WebShop v1/v12/v24/v36/v38 and ALFWorld v24.
"""
import json
import re
from pathlib import Path

LOG_DIR = Path("/data/home/qisheng/EvolAnalsis/logs")
OUT_DIR = Path("/data/home/qisheng/EvolAnalsis/analysis_reports/_parsed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = {
    "ws_v1":  "webshop_qwen1.5b_duet.log",
    "ws_v12": "webshop_qwen1.5b_duet_v12.log",
    "ws_v24": "webshop_qwen1.5b_duet_v24.log",
    "ws_v36": "webshop_qwen1.5b_duet_v36.log",
    "ws_v37": "webshop_qwen1.5b_duet_v37.log",
    "ws_v38": "webshop_qwen1.5b_duet_v38.log",
    "ws_v39": "webshop_qwen1.5b_duet_v39.log",
    "alf_v24": "alfworld_qwen1.5b_duet_v24.log",
    "alf_v1":  "alfworld_qwen1.5b_duet.log",
    "alf_v39": "alfworld_qwen1.5b_duet_v39.log",
}

KEYS = [
    "chord/mu",
    "chord/mu_adaptive_gated",
    "chord/mu_adaptive",
    "chord/sft_loss",
    "chord/sft_loss_unweighted_mean",
    "chord/weighted_sft_loss",
    "chord/log_prob_mean",
    "chord/log_prob_std",
    "chord/phi_mean",
    "chord/phi_std",
    "chord/grpo_loss",
    "chord/disc_acc_current",
    "chord/disc_acc_ema",
    # DR3
    "dr3/disc_acc",
    "dr3/w_mean",
    "dr3/w_std",
    "dr3/w_off_mean",
    "dr3/w_off_std",
    "dr3/ess_off_window",
    "dr3/ess_window_len",
    "dr3/disc_loss",
    # DUET
    "duet/teacher_gradient_share",
    "duet/group_reward_variance_mean",
    "duet/group_mixed_ratio",
    "duet/adv_onpolicy_effective_std",
    "duet/adv_onpolicy_effective_abs_mean",
    "duet/adv_teacher_effective_abs_mean",
    # Policy
    "actor/kl_loss",
    "actor/entropy_loss",
    "actor/grad_norm",
    "actor/pg_loss",
    # Length & perf
    "response_length/mean",
    # Core training rewards
    "critic/rewards_onpolicy/mean",
    "critic/score/mean",
    "critic/success_onpolicy/mean",
    "critic/rewards/mean",
    # State channel
    "state_channel/progress_onpolicy_mean",
    "state_channel/bonus_vs_reward_ratio",
    # Validation
    "val-summary/webshop/reward_mean_all",
    "val-summary/webshop/success_rate_mean_all",
    "val-core/webshop/reward/mean@200",
    "val-core/alfworld/reward/mean@134",
    "val-summary/alfworld/reward_mean_all",
    "val-summary/alfworld/success_rate_mean_all",
    # Step
    "training/global_step",
]

KEY_SET = set(KEYS)
PAIR_RE = re.compile(r"([A-Za-z][A-Za-z0-9_/@\-]*)[:]([\-0-9][\-0-9.eE+]*)")
STEP_RE = re.compile(r"\bstep:(\d+)\b")


def parse_log(path: Path, max_step: int = 200) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if "step:" not in line:
                continue
            if "critic/" not in line and "val-summary" not in line and "val-core" not in line:
                continue
            m = STEP_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            if step <= 0 or step > max_step:
                continue
            row = {"step": step}
            for k, v in PAIR_RE.findall(line):
                if k in KEY_SET:
                    try:
                        row[k] = float(v)
                    except ValueError:
                        pass
            if "critic/score/mean" in row or "critic/rewards_onpolicy/mean" in row or "critic/rewards/mean" in row:
                rows.append(row)
    # Dedup by step keep last
    seen = {}
    for r in rows:
        seen[r["step"]] = r
    return [seen[s] for s in sorted(seen)]


def main():
    all_data = {}
    for variant, fname in VARIANTS.items():
        rows = parse_log(LOG_DIR / fname)
        all_data[variant] = rows
        if rows:
            print(f"{variant:10s} {fname:45s} {len(rows):4d} rows  steps {rows[0]['step']}..{rows[-1]['step']}")
        else:
            print(f"{variant:10s} {fname:45s} EMPTY")

    out = OUT_DIR / "adaptive_signals.json"
    with open(out, "w") as f:
        json.dump(all_data, f)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
