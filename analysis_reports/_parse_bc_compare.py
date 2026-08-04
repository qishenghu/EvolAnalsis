#!/usr/bin/env python3
"""Parse training logs and extract metrics for DUET* BC comparison (1.5B WS vs 3B WS vs 3B AF)."""
import os, re, json, sys
from collections import defaultdict

LOGS_DIR = "/data/home/qisheng/EvolAnalsis/logs"

# Metrics we care about
METRICS = [
    "critic/success_onpolicy/mean",
    "critic/rewards_onpolicy/mean",
    "critic/rewards_teacher/mean",
    "chord/mu",
    "chord/mu_adaptive_gated",
    "chord/disc_acc_current",
    "chord/disc_acc_ema",
    "chord/rs_latched",
    "chord/d_floor",
    "chord/sft_loss",
    "chord/grpo_loss",
    "chord/weighted_sft_loss",
    "chord/n_expert_tokens",
    "dr3/disc_acc",
    "dr3/w_off_mean",
    "dr3/w_mean",
    "dr3/ess_off_window",
    "dr3/logw_applied_abs_mean",
    "duet/teacher_gradient_share",
    "duet/adv_teacher_effective_abs_mean",
    "duet/adv_onpolicy_effective_abs_mean",
    "state_channel/bonus_vs_reward_ratio",
    "state_channel/bonus_total_mean",
    "state_channel/progress_mean",
    "state_channel/progress_onpolicy_mean",
    "state_channel/progress_teacher_mean",
    "state_channel/shaped_ratio",
    "actor/kl_loss",
    "actor/ppo_kl",
    "actor/grad_norm",
    "actor/entropy_loss",
    "actor/pg_loss",
    "actor/on_pg_loss",
    "actor/off_pg_loss",
    "actor/teacher_off_pg_loss",
    "diag/response_len_teacher_mean",
    "diag/response_len_onpolicy_mean",
    "diag/response_len_ratio_teacher_vs_on",
    "diag/teacher_token_ratio",
    "diag/teacher_llm_token_ratio",
    "diag/llm_token_ratio_in_response",
    "diag/teacher_sample_ratio",
    "diag/entropy_teacher_token_mean",
    "diag/entropy_onpolicy_token_mean",
    "diag/reward_teacher_mean",
    "diag/reward_onpolicy_mean",
    "exp_replay/entropy_llm_mean",
    "exp_replay/entropy_llm_onpolicy_mean",
    "exp_replay/entropy_llm_offpolicy_mean",
    "val-summary/webshop/reward_mean_all",
    "val-summary/webshop/success_rate_mean_all",
    "val-summary/alfworld/reward_mean_all",
    "val-summary/alfworld/success_rate_mean_all",
    "val-core/webshop/reward/mean@200",
    "val-core/webshop/reward/best@200/mean",
    "val-core/webshop/reward/best@8/mean",
    "val-core/alfworld/reward/mean@200",
    "response_length/mean",
]

STEP_RE = re.compile(r"step:(\d+)\s*-\s*(.*)")
KV_RE = re.compile(r"([a-zA-Z0-9/_\-@]+):([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)")
VAL_LINE_RE = re.compile(r"val-summary/[a-zA-Z]+/([a-z_]+):([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)")
VAL_STEP_RE = re.compile(r"val.*step.*?(\d+)", re.IGNORECASE)


def strip_prefix(line):
    # Remove ANSI / Ray prefixes like "[36m(TaskRunner pid=4111698)[0m "
    return re.sub(r"\x1b\[[0-9;]*m", "", line).strip()


def parse_log(path, sample_steps=(10, 25, 50, 75, 100, 125, 150)):
    """Returns dict: {metric: {step: value}} plus 'val' list of {step, sr, reward}."""
    by_step = defaultdict(dict)  # step -> metric -> value
    val_records = []  # list of (step, key, value)
    last_step = 0
    max_step = 0
    last_step_with_metrics = 0

    try:
        with open(path, "r", errors="replace") as f:
            for raw in f:
                line = strip_prefix(raw)
                # Step metric line
                m = STEP_RE.search(line)
                if m:
                    step = int(m.group(1))
                    rest = m.group(2)
                    last_step = step
                    if step > max_step:
                        max_step = step
                    kvs = KV_RE.findall(rest)
                    if kvs:
                        last_step_with_metrics = step
                    for k, v in kvs:
                        if k in METRICS:
                            try:
                                by_step[step][k] = float(v)
                            except ValueError:
                                pass
                # val-summary lines
                vm = VAL_LINE_RE.search(line)
                if vm:
                    key = vm.group(1)
                    try:
                        val = float(vm.group(2))
                        val_records.append((last_step, key, val))
                    except ValueError:
                        pass
    except Exception as e:
        return {"error": str(e)}

    # Sample at requested steps (nearest available <= step)
    sampled = {}
    available_steps = sorted(by_step.keys())
    for tgt in sample_steps:
        # find largest step <= tgt
        cand = [s for s in available_steps if s <= tgt]
        if not cand:
            continue
        s = cand[-1]
        sampled[tgt] = {"actual_step": s, **by_step[s]}

    # Aggregate val records: organize by step
    val_by_step = defaultdict(dict)
    for step, key, val in val_records:
        val_by_step[step][key] = val

    # Best success rate from inline val-summary metrics (more reliable)
    best_sr = 0.0
    best_step = 0
    final_sr = 0.0
    final_reward = 0.0
    best_reward = 0.0
    val_steps_inline = []
    for step in sorted(by_step.keys()):
        kv = by_step[step]
        # Try webshop first, then alfworld
        sr = kv.get("val-summary/webshop/success_rate_mean_all")
        if sr is None:
            sr = kv.get("val-summary/alfworld/success_rate_mean_all")
        rew = kv.get("val-summary/webshop/reward_mean_all") or kv.get("val-summary/alfworld/reward_mean_all")
        if sr is not None:
            val_steps_inline.append(step)
            if sr > best_sr:
                best_sr = sr
                best_step = step
            final_sr = sr
        if rew is not None:
            if rew > best_reward:
                best_reward = rew
            final_reward = rew

    # Compute trajectory means (avg over all logged steps) for some metrics
    trajectory_means = {}
    for metric in METRICS:
        vals = [by_step[s].get(metric) for s in by_step if metric in by_step[s]]
        if vals:
            trajectory_means[metric] = {
                "mean": sum(vals) / len(vals),
                "min": min(vals),
                "max": max(vals),
                "n": len(vals),
                "first": vals[0],
                "last": vals[-1],
            }

    return {
        "path": path,
        "max_step": max_step,
        "last_step_with_metrics": last_step_with_metrics,
        "n_steps_logged": len(by_step),
        "sampled": sampled,
        "trajectory": trajectory_means,
        "val_by_step": dict(val_by_step),
        "best_sr": best_sr,
        "best_step": best_step,
        "final_sr": final_sr,
        "best_reward": best_reward,
        "final_reward": final_reward,
        "val_steps_inline": val_steps_inline,
        "n_val_evals": len(val_steps_inline),
    }


def main():
    targets = {
        # 1.5B WS — winners
        "1.5b_swC_02_SOTA": "webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06.log",
        "1.5b_swA_02_peak02": "webshop_qwen1.5b_duet_swA_02_peak02.log",
        "1.5b_swA_03_peak04": "webshop_qwen1.5b_duet_swA_03_peak04.log",
        "1.5b_swA_05_peak06": "webshop_qwen1.5b_duet_swA_05_peak06.log",
        "1.5b_swA_11_pk05_v10": "webshop_qwen1.5b_duet_swA_11_pk05_v10.log",
        "1.5b_swB_01_pk03_v10_ema02": "webshop_qwen1.5b_duet_swB_01_pk03_v10_ema02.log",
        "1.5b_swB_02_pk03_v15_ema02": "webshop_qwen1.5b_duet_swB_02_pk03_v15_ema02.log",
        "1.5b_swC_01_pk03_v10_floor04": "webshop_qwen1.5b_duet_swC_01_pk03_v10_floor04.log",
        "1.5b_swC_03_pk03_v12_ema02": "webshop_qwen1.5b_duet_swC_03_pk03_v12_ema02.log",
        # 3B WS — losers
        "3b_ws_swC_pk03_v00_v2latch": "ws_swC_v_pk03_v00.log",
        "3b_ws_swC_pk04_v00_v2latch": "ws_swC_v_pk04_v00.log",
        "3b_ws_swC_pk03_v00_v1latch": "ws_swC_v_pk03_v00.LATCH_V1_2234.log",
        "3b_ws_swC_pk04_v00_v1latch": "ws_swC_v_pk04_v00.LATCH_V1_2234.log",
        "3b_ws_swC_pk03_v00_buggy": "ws_swC_v_pk03_v00.BUGGY.log",
        "3b_ws_swD_01_pk03_v10_floor06": "webshop_qwen3b_duet_swD_01_pk03_v10_floor06.log",
        "3b_ws_swD_02_pk03_v10_floor07": "webshop_qwen3b_duet_swD_02_pk03_v10_floor07.log",
        "3b_ws_swE_01_pk03_v05_ema01": "webshop_qwen3b_duet_swE_01_pk03_v05_ema01.log",
        "3b_ws_swE_02_pk02_v10": "webshop_qwen3b_duet_swE_02_pk02_v10.log",
        # 3B AF — DUET works on AF
        "3b_af_chord": "alfworld_qwen3b_chord.log",
        "1.5b_af_duet": "alfworld_qwen1.5b_duet.log",
    }

    results = {}
    for name, fname in targets.items():
        path = os.path.join(LOGS_DIR, fname)
        if not os.path.exists(path):
            print(f"SKIP {name}: missing {path}", file=sys.stderr)
            continue
        print(f"Parsing {name}...", file=sys.stderr)
        r = parse_log(path)
        if "error" in r:
            print(f"  ERROR: {r['error']}", file=sys.stderr)
            continue
        results[name] = r
        print(f"  steps_logged={r['n_steps_logged']} max_step={r['max_step']} "
              f"best_sr={r['best_sr']:.3f}@step{r['best_step']} val_evals={r['n_val_evals']}",
              file=sys.stderr)

    out_path = "/data/home/qisheng/EvolAnalsis/analysis_reports/_parsed/bc_compare_2026-05-03.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
