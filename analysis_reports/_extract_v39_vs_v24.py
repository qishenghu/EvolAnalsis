#!/usr/bin/env python3
"""Extract per-step metrics from WebShop v24 vs v39-family logs.

Format of step lines:
  [36m(TaskRunner pid=...)[0m step:N - key:val - key:val - ...
Tokens are separated by " - " (space-dash-space). Each token is "key:value".
"""
import os, re, json, glob
from collections import defaultdict

LOG_DIR = "/data/home/qisheng/EvolAnalsis/logs"
OUT_DIR = "/data/home/qisheng/EvolAnalsis/analysis_reports/_parsed/v39_vs_v24"
os.makedirs(OUT_DIR, exist_ok=True)

RUNS = {
    "v24": "webshop_qwen1.5b_duet_v24.log",
    "v39_postfix": "webshop_qwen1.5b_duet_v39_postfix.log",
    "v39b_postfix": "webshop_qwen1.5b_duet_v39b_postfix.log",
    "swA_04_peak05": "webshop_qwen1.5b_duet_swA_04_peak05.log",
    "swA_11_pk05_v10": "webshop_qwen1.5b_duet_swA_11_pk05_v10.log",
}

# Metrics we care about (subset; we'll pull anything we hit anyway).
KEYS_OF_INTEREST = {
    # CHORD/BC
    "chord/mu", "chord/mu_adaptive_gated", "chord/mu_mode",
    "chord/disc_acc_ema", "chord/disc_acc_current", "chord/disc_acc_raw",
    "chord/d_floor", "chord/sft_loss", "chord/weighted_sft_loss",
    "chord/n_expert_tokens", "chord/sft_loss_unweighted_mean",
    "chord/grpo_loss", "chord/log_prob_mean",
    # DR3
    "dr3/disc_acc", "dr3/disc_loss", "dr3/w_mean", "dr3/w_std",
    "dr3/teacher_gradient_share", "dr3/ess_off_window",
    "dr3/alpha", "dr3/alpha_ema",
    # Critic / rewards / success
    "critic/rewards/mean", "critic/rewards_onpolicy/mean",
    "critic/rewards_teacher/mean",
    "critic/success_onpolicy/mean", "critic/rewards_sum/mean",
    # Validation summaries
    "val-summary/webshop/reward_mean_all",
    "val-summary/webshop/success_rate_mean_all",
    "val-summary/webshop/reward_std_all",
    "val-summary/webshop/n_outputs",
    # State channel
    "state_channel/progress_mean", "state_channel/progress_onpolicy_mean",
    "state_channel/progress_teacher_mean",
    "state_channel/bonus_total_mean", "state_channel/bonus_vs_reward_ratio",
    "state_channel/shaped_ratio", "state_channel/reward_pre_shaping_mean",
    # Entropy / actor
    "actor/entropy_loss", "actor/kl_loss",
    "exp_replay/entropy_llm_mean", "exp_replay/entropy_llm_onpolicy_mean",
    "exp_replay/entropy_llm_offpolicy_mean",
    # Lengths
    "response_length/mean", "response_length/max", "response_length/clip_ratio",
    # luffy / teacher mix
    "luffy/total_teacher_rollouts", "luffy/total_onpolicy_kept",
    "diag/teacher_sample_ratio",
    # duet share
    "duet/teacher_gradient_share",
}

STEP_RE = re.compile(r"step:(\d+)\s+-\s+(.*)$")
KV_RE = re.compile(r"([A-Za-z0-9_\-/\.]+):(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


def parse_log(path):
    """Return dict[step] -> dict[metric] -> float (last value seen for that step)."""
    by_step = defaultdict(dict)
    fmt_warns = 0
    length_warns = 0
    with open(path, "r", errors="replace") as f:
        for line in f:
            # Strip ANSI
            ln = re.sub(r"\x1b\[[0-9;]*m", "", line)
            # Track auxiliary signals
            if "simple_completion_callback" in ln and ("length" in ln.lower() or "max" in ln.lower()):
                length_warns += 1
            if "format" in ln.lower() and "error" in ln.lower():
                fmt_warns += 1
            m = STEP_RE.search(ln)
            if not m:
                continue
            step = int(m.group(1))
            tail = m.group(2)
            for km in KV_RE.finditer(tail):
                k = km.group(1)
                try:
                    v = float(km.group(2))
                except ValueError:
                    continue
                by_step[step][k] = v
    return by_step, {"length_warns": length_warns, "fmt_warns": fmt_warns}


def integrate_mu(steps_dict, key="chord/mu"):
    """Trapezoidal integral of mu over steps 1..max."""
    pts = sorted((s, v[key]) for s, v in steps_dict.items() if key in v)
    if len(pts) < 2:
        return None
    total = 0.0
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        total += 0.5 * (y0 + y1) * (x1 - x0)
    return total


def main():
    summary = {}
    for run, fn in RUNS.items():
        path = os.path.join(LOG_DIR, fn)
        if not os.path.isfile(path):
            print(f"MISSING: {path}")
            continue
        print(f"Parsing {run} ...")
        bs, aux = parse_log(path)
        # Filter only KEYS_OF_INTEREST to keep the JSON small.
        compact = {}
        for step, kv in bs.items():
            sub = {k: v for k, v in kv.items() if k in KEYS_OF_INTEREST}
            if sub:
                compact[step] = sub
        out_path = os.path.join(OUT_DIR, f"{run}.json")
        with open(out_path, "w") as f:
            json.dump(compact, f, indent=1, sort_keys=True)
        steps_seen = sorted(compact.keys())
        last_step = steps_seen[-1] if steps_seen else None
        # Integrate mu
        mu_int = integrate_mu(compact, "chord/mu")
        mu_eff_int = integrate_mu(compact, "chord/mu_adaptive_gated")
        summary[run] = {
            "n_steps_logged": len(steps_seen),
            "first_step": steps_seen[0] if steps_seen else None,
            "last_step": last_step,
            "mu_AUC_steps_1_to_last": mu_int,
            "mu_adaptive_gated_AUC": mu_eff_int,
            "log_aux": aux,
        }
        print(f"  steps={len(steps_seen)} last={last_step} mu_AUC={mu_int} mu_gated_AUC={mu_eff_int}")
    with open(os.path.join(OUT_DIR, "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nWrote", OUT_DIR)


if __name__ == "__main__":
    main()
