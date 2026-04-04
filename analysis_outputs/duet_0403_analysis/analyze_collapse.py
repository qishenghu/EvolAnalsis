"""
DUET 0403 Collapse Analysis
Comprehensive investigation of performance collapse at steps 80-100.
"""
import json
import os
import numpy as np
from collections import defaultdict

DUET_DIR = "checkpoints/agentevolver/webshop_3b_duet_0403/Trajectory"
LUFFY_DIR = "checkpoints/agentevolver/webshop_3b_luffy/Trajectory"
OUTPUT_DIR = "analysis_outputs/duet_0403_analysis"

def load_batch_diag(base_dir, step):
    path = os.path.join(base_dir, f"batch_diag_step_{step}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def load_trajectories(base_dir, step):
    path = os.path.join(base_dir, f"trajectories_step_{step}.jsonl")
    if not os.path.exists(path):
        return []
    trajs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                trajs.append(json.loads(line))
    return trajs

def analyze_batch_diag_timeseries(base_dir, label, steps):
    """Extract key metrics from batch_diag files across steps."""
    metrics = defaultdict(list)
    available_steps = []

    for step in steps:
        d = load_batch_diag(base_dir, step)
        if d is None:
            continue
        available_steps.append(step)
        for key in [
            "diag/reward_onpolicy_mean",
            "diag/reward_teacher_mean",
            "diag/adv_teacher_sample_mean",
            "diag/adv_onpolicy_sample_mean",
            "diag/adv_teacher_token_mean",
            "diag/adv_onpolicy_token_mean",
            "diag/teacher_adv_pos_ratio",
            "diag/onpolicy_adv_pos_ratio",
            "diag/entropy_onpolicy_token_mean",
            "diag/entropy_teacher_token_mean",
            "diag/entropy_onpolicy_token_std",
            "diag/teacher_sample_ratio",
            "diag/group_teacher_minus_on_reward_mean",
            "diag/group_all_reward_mean",
            "diag/group_non_teacher_reward_mean",
            "actor/on_pg_clipfrac",
            "actor/on_pg_clipfrac_lower",
        ]:
            metrics[key].append(d.get(key, float('nan')))

    return available_steps, metrics

def analyze_trajectories_detailed(base_dir, step):
    """Analyze per-trajectory data for a specific step."""
    trajs = load_trajectories(base_dir, step)
    if not trajs:
        return None

    result = {
        "total_samples": len(trajs),
        "teacher_samples": 0,
        "onpolicy_samples": 0,
        # Rewards
        "rewards_onpolicy": [],
        "rewards_teacher": [],
        "rewards_all": [],
        # Advantages
        "adv_mean_onpolicy": [],
        "adv_mean_teacher": [],
        # Response lengths
        "response_len_onpolicy": [],
        "response_len_teacher": [],
        # Entropy
        "entropy_mean_onpolicy": [],
        "entropy_mean_teacher": [],
        # Old log probs (proxy for KL)
        "old_logp_onpolicy": [],
        "old_logp_teacher": [],
        # SC metrics
        "sc_progress_onpolicy": [],
        "sc_bonus_onpolicy": [],
        "sc_coverage_onpolicy": [],
        "sc_step_deltas_onpolicy": [],
        # Success
        "success_onpolicy": [],
        "success_teacher": [],
    }

    for t in trajs:
        is_teacher = t.get("diag", {}).get("is_teacher", False)
        reward_outcome = t.get("reward", {}).get("outcome", 0)
        success = t.get("success", False)
        diag = t.get("diag", {})
        entropy = t.get("entropy", {})

        result["rewards_all"].append(reward_outcome)

        if is_teacher:
            result["teacher_samples"] += 1
            result["rewards_teacher"].append(reward_outcome)
            result["adv_mean_teacher"].append(diag.get("adv_mean", 0))
            result["response_len_teacher"].append(diag.get("response_valid_tokens", 0))
            result["entropy_mean_teacher"].append(entropy.get("mean", 0))
            if diag.get("old_log_prob_mean") is not None:
                result["old_logp_teacher"].append(diag["old_log_prob_mean"])
            result["success_teacher"].append(1 if success else 0)
        else:
            result["onpolicy_samples"] += 1
            result["rewards_onpolicy"].append(reward_outcome)
            result["adv_mean_onpolicy"].append(diag.get("adv_mean", 0))
            result["response_len_onpolicy"].append(diag.get("response_valid_tokens", 0))
            result["entropy_mean_onpolicy"].append(entropy.get("mean", 0))
            if diag.get("old_log_prob_mean") is not None:
                result["old_logp_onpolicy"].append(diag["old_log_prob_mean"])
            result["success_onpolicy"].append(1 if success else 0)

            # SC metrics (only on-policy)
            if diag.get("sc_progress") is not None:
                result["sc_progress_onpolicy"].append(diag["sc_progress"])
            if diag.get("sc_bonus") is not None:
                result["sc_bonus_onpolicy"].append(diag["sc_bonus"])
            if diag.get("sc_coverage") is not None:
                result["sc_coverage_onpolicy"].append(diag["sc_coverage"])
            if diag.get("sc_step_deltas"):
                result["sc_step_deltas_onpolicy"].extend(diag["sc_step_deltas"])

    return result

def safe_mean(arr):
    if not arr:
        return float('nan')
    return np.mean(arr)

def safe_std(arr):
    if not arr:
        return float('nan')
    return np.std(arr)

def safe_pct(arr, threshold, direction="above"):
    if not arr:
        return float('nan')
    a = np.array(arr)
    if direction == "above":
        return (a > threshold).mean() * 100
    elif direction == "below":
        return (a < threshold).mean() * 100
    elif direction == "abs_above":
        return (np.abs(a) > threshold).mean() * 100

def main():
    all_steps = list(range(1, 101))
    focus_steps = list(range(40, 101))  # Broader window to see onset

    report = []
    report.append("# DUET 0403 Collapse Analysis Report")
    report.append(f"Generated: 2026-04-02\n")

    # ================================================================
    # 1. BATCH DIAG TIMESERIES
    # ================================================================
    report.append("## 1. Batch Diagnostics Timeseries (Steps 1-100)")
    report.append("")

    duet_steps, duet_metrics = analyze_batch_diag_timeseries(DUET_DIR, "DUET", all_steps)
    luffy_steps, luffy_metrics = analyze_batch_diag_timeseries(LUFFY_DIR, "LUFFY", all_steps)

    # Find collapse onset: when reward_onpolicy_mean starts dropping
    duet_rewards = duet_metrics["diag/reward_onpolicy_mean"]
    report.append("### 1a. Reward Trajectory (reward_onpolicy_mean)")
    report.append("```")
    report.append(f"{'Step':>5} | {'DUET':>10} | {'LUFFY':>10} | {'Delta':>10}")
    report.append("-" * 45)
    for i, step in enumerate(duet_steps):
        duet_r = duet_rewards[i] if i < len(duet_rewards) else float('nan')
        luffy_r = float('nan')
        if step in luffy_steps:
            li = luffy_steps.index(step)
            luffy_r = luffy_metrics["diag/reward_onpolicy_mean"][li]
        delta = duet_r - luffy_r if not (np.isnan(duet_r) or np.isnan(luffy_r)) else float('nan')
        report.append(f"{step:>5} | {duet_r:>10.4f} | {luffy_r:>10.4f} | {delta:>+10.4f}")
    report.append("```")
    report.append("")

    # Find peak and collapse
    if duet_rewards:
        peak_idx = np.argmax(duet_rewards)
        peak_step = duet_steps[peak_idx]
        peak_val = duet_rewards[peak_idx]
        # Find first step where reward drops >10% from peak (after peak)
        collapse_step = None
        for i in range(peak_idx + 1, len(duet_rewards)):
            if duet_rewards[i] < peak_val * 0.90:
                collapse_step = duet_steps[i]
                break
        report.append(f"**Peak reward**: {peak_val:.4f} at step {peak_step}")
        if collapse_step:
            report.append(f"**Collapse onset** (>10% drop from peak): step {collapse_step}")
        report.append("")

    # Advantages
    report.append("### 1b. Advantage Trajectories")
    report.append("```")
    report.append(f"{'Step':>5} | {'DUET T.Adv':>12} | {'DUET On.Adv':>12} | {'LUFFY T.Adv':>12} | {'LUFFY On.Adv':>12}")
    report.append("-" * 65)
    for i, step in enumerate(duet_steps):
        if step < 40:
            continue
        dt = duet_metrics["diag/adv_teacher_sample_mean"][i]
        do = duet_metrics["diag/adv_onpolicy_sample_mean"][i]
        lt = float('nan')
        lo = float('nan')
        if step in luffy_steps:
            li = luffy_steps.index(step)
            lt = luffy_metrics["diag/adv_teacher_sample_mean"][li]
            lo = luffy_metrics["diag/adv_onpolicy_sample_mean"][li]
        report.append(f"{step:>5} | {dt:>12.6f} | {do:>12.6f} | {lt:>12.6f} | {lo:>12.6f}")
    report.append("```")
    report.append("")

    # Entropy
    report.append("### 1c. Entropy Trajectories")
    report.append("```")
    report.append(f"{'Step':>5} | {'DUET On.Ent':>12} | {'DUET On.Std':>12} | {'LUFFY On.Ent':>12}")
    report.append("-" * 55)
    for i, step in enumerate(duet_steps):
        if step < 40:
            continue
        de = duet_metrics["diag/entropy_onpolicy_token_mean"][i]
        ds = duet_metrics["diag/entropy_onpolicy_token_std"][i]
        le = float('nan')
        if step in luffy_steps:
            li = luffy_steps.index(step)
            le = luffy_metrics["diag/entropy_onpolicy_token_mean"][li]
        report.append(f"{step:>5} | {de:>12.4f} | {ds:>12.4f} | {le:>12.4f}")
    report.append("```")
    report.append("")

    # PG clip fractions
    report.append("### 1d. Policy Gradient Clip Fractions")
    report.append("```")
    report.append(f"{'Step':>5} | {'DUET clip':>10} | {'DUET clip_lo':>12} | {'LUFFY clip':>10}")
    report.append("-" * 45)
    for i, step in enumerate(duet_steps):
        if step < 40:
            continue
        dc = duet_metrics["actor/on_pg_clipfrac"][i]
        dl = duet_metrics["actor/on_pg_clipfrac_lower"][i]
        lc = float('nan')
        if step in luffy_steps:
            li = luffy_steps.index(step)
            lc = luffy_metrics["actor/on_pg_clipfrac"][li]
        report.append(f"{step:>5} | {dc:>10.4f} | {dl:>12.4f} | {lc:>10.4f}")
    report.append("```")
    report.append("")

    # ================================================================
    # 2. TEACHER ADVANTAGE CLIP ANALYSIS
    # ================================================================
    report.append("## 2. Teacher Advantage Clip Analysis (±5)")
    report.append("")

    # We need per-sample advantages from trajectory files
    clip_analysis_steps = list(range(40, 101))
    report.append("### 2a. Per-step teacher advantage distribution")
    report.append("```")
    report.append(f"{'Step':>5} | {'N_teach':>7} | {'Adv Mean':>10} | {'Adv Std':>10} | {'|Adv|>5%':>10} | {'>+5%':>8} | {'<-5%':>8} | {'Max':>10} | {'Min':>10}")
    report.append("-" * 100)

    for step in clip_analysis_steps:
        trajs = load_trajectories(DUET_DIR, step)
        teacher_advs = [t["diag"]["adv_mean"] for t in trajs if t.get("diag", {}).get("is_teacher", False)]
        if not teacher_advs:
            continue
        a = np.array(teacher_advs)
        report.append(
            f"{step:>5} | {len(teacher_advs):>7} | {a.mean():>10.4f} | {a.std():>10.4f} | "
            f"{(np.abs(a) > 5).mean()*100:>9.1f}% | {(a > 5).mean()*100:>7.1f}% | {(a < -5).mean()*100:>7.1f}% | "
            f"{a.max():>10.4f} | {a.min():>10.4f}"
        )
    report.append("```")
    report.append("")

    # Check token-level advantage clipping from batch_diag
    report.append("### 2b. Token-level teacher advantage (from batch_diag)")
    report.append("```")
    report.append(f"{'Step':>5} | {'Token Adv Mean':>15} | {'Sample Adv Mean':>15} | {'Teacher Adv Pos%':>15}")
    report.append("-" * 60)
    for i, step in enumerate(duet_steps):
        if step < 40:
            continue
        tam = duet_metrics["diag/adv_teacher_token_mean"][i]
        sam = duet_metrics["diag/adv_teacher_sample_mean"][i]
        tap = duet_metrics["diag/teacher_adv_pos_ratio"][i]
        report.append(f"{step:>5} | {tam:>15.6f} | {sam:>15.6f} | {tap:>15.4f}")
    report.append("```")
    report.append("")

    # ================================================================
    # 3. KL DIVERGENCE ANALYSIS (via old_log_prob proxy)
    # ================================================================
    report.append("## 3. KL Divergence Analysis (old_log_prob as proxy)")
    report.append("")
    report.append("Lower old_log_prob_mean → policy drifting further from data distribution")
    report.append("")

    report.append("### 3a. Old Log Prob Trajectory")
    report.append("```")
    report.append(f"{'Step':>5} | {'DUET On.LogP':>14} | {'DUET T.LogP':>14} | {'LUFFY On.LogP':>14}")
    report.append("-" * 55)

    for step in range(40, 101):
        duet_traj = analyze_trajectories_detailed(DUET_DIR, step)
        luffy_traj = analyze_trajectories_detailed(LUFFY_DIR, step)

        d_on_lp = safe_mean(duet_traj["old_logp_onpolicy"]) if duet_traj else float('nan')
        d_t_lp = safe_mean(duet_traj["old_logp_teacher"]) if duet_traj else float('nan')
        l_on_lp = safe_mean(luffy_traj["old_logp_onpolicy"]) if luffy_traj else float('nan')

        report.append(f"{step:>5} | {d_on_lp:>14.4f} | {d_t_lp:>14.4f} | {l_on_lp:>14.4f}")
    report.append("```")
    report.append("")

    # ================================================================
    # 4. SC-GRPO DECOUPLING VERIFICATION
    # ================================================================
    report.append("## 4. State Channel Decoupling Verification")
    report.append("")

    report.append("### 4a. SC Metrics Over Time (on-policy only)")
    report.append("```")
    report.append(f"{'Step':>5} | {'SC Bonus':>10} | {'SC Prog':>10} | {'SC Cov':>10} | {'Orig Rew':>10} | {'Total Rew':>10} | {'Bonus/Rew':>10}")
    report.append("-" * 75)

    for step in range(40, 101):
        duet_traj = analyze_trajectories_detailed(DUET_DIR, step)
        if not duet_traj:
            continue

        sc_bonus = safe_mean(duet_traj["sc_bonus_onpolicy"])
        sc_prog = safe_mean(duet_traj["sc_progress_onpolicy"])
        sc_cov = safe_mean(duet_traj["sc_coverage_onpolicy"])
        orig_rew = safe_mean(duet_traj["rewards_onpolicy"])
        total_rew = orig_rew + sc_bonus if not np.isnan(sc_bonus) else orig_rew
        bonus_ratio = sc_bonus / max(total_rew, 0.01) if not np.isnan(sc_bonus) else float('nan')

        report.append(
            f"{step:>5} | {sc_bonus:>10.4f} | {sc_prog:>10.4f} | {sc_cov:>10.4f} | "
            f"{orig_rew:>10.4f} | {total_rew:>10.4f} | {bonus_ratio:>10.4f}"
        )
    report.append("```")
    report.append("")

    # Check if teacher samples have SC bonus (they shouldn't)
    report.append("### 4b. Teacher SC Contamination Check")
    report.append("```")
    for step in [60, 70, 80, 90, 100]:
        trajs = load_trajectories(DUET_DIR, step)
        teacher_sc = [t["diag"].get("sc_bonus", 0) for t in trajs if t.get("diag", {}).get("is_teacher", False)]
        if teacher_sc:
            nonzero = sum(1 for x in teacher_sc if x and x != 0)
            report.append(f"Step {step}: {len(teacher_sc)} teacher samples, {nonzero} with nonzero SC bonus")
        else:
            report.append(f"Step {step}: no teacher samples found")
    report.append("```")
    report.append("")

    # ================================================================
    # 5. RESPONSE LENGTH AND DEGENERACY
    # ================================================================
    report.append("## 5. Response Length and Degeneracy Analysis")
    report.append("")

    report.append("### 5a. Response Length Over Time")
    report.append("```")
    report.append(f"{'Step':>5} | {'DUET On.Len':>12} | {'DUET On.Std':>12} | {'DUET T.Len':>12} | {'LUFFY On.Len':>12}")
    report.append("-" * 60)

    for step in range(40, 101):
        duet_traj = analyze_trajectories_detailed(DUET_DIR, step)
        luffy_traj = analyze_trajectories_detailed(LUFFY_DIR, step)

        d_on_len = safe_mean(duet_traj["response_len_onpolicy"]) if duet_traj else float('nan')
        d_on_std = safe_std(duet_traj["response_len_onpolicy"]) if duet_traj else float('nan')
        d_t_len = safe_mean(duet_traj["response_len_teacher"]) if duet_traj else float('nan')
        l_on_len = safe_mean(luffy_traj["response_len_onpolicy"]) if luffy_traj else float('nan')

        report.append(f"{step:>5} | {d_on_len:>12.1f} | {d_on_std:>12.1f} | {d_t_len:>12.1f} | {l_on_len:>12.1f}")
    report.append("```")
    report.append("")

    # Success rate
    report.append("### 5b. Success Rate Over Time")
    report.append("```")
    report.append(f"{'Step':>5} | {'DUET On.Succ':>12} | {'DUET Total':>12} | {'LUFFY On.Succ':>13}")
    report.append("-" * 50)

    for step in range(40, 101):
        duet_traj = analyze_trajectories_detailed(DUET_DIR, step)
        luffy_traj = analyze_trajectories_detailed(LUFFY_DIR, step)

        d_on_s = safe_mean(duet_traj["success_onpolicy"]) if duet_traj else float('nan')
        d_total = safe_mean(duet_traj["success_onpolicy"] + duet_traj["success_teacher"]) if duet_traj else float('nan')
        l_on_s = safe_mean(luffy_traj["success_onpolicy"]) if luffy_traj else float('nan')

        report.append(f"{step:>5} | {d_on_s:>12.4f} | {d_total:>12.4f} | {l_on_s:>13.4f}")
    report.append("```")
    report.append("")

    # ================================================================
    # 6. COMPARATIVE ANALYSIS: DUET vs LUFFY STABILITY
    # ================================================================
    report.append("## 6. DUET vs LUFFY Stability Comparison")
    report.append("")

    # Volatility analysis
    report.append("### 6a. Reward Volatility (rolling 5-step std)")
    report.append("```")

    duet_r_series = []
    luffy_r_series = []
    common_steps = []
    for i, step in enumerate(duet_steps):
        if step < 40:
            continue
        duet_r_series.append(duet_metrics["diag/reward_onpolicy_mean"][i])
        if step in luffy_steps:
            li = luffy_steps.index(step)
            luffy_r_series.append(luffy_metrics["diag/reward_onpolicy_mean"][li])
        else:
            luffy_r_series.append(float('nan'))
        common_steps.append(step)

    window = 5
    report.append(f"{'Step':>5} | {'DUET Vol':>10} | {'LUFFY Vol':>10} | {'DUET Rew':>10} | {'LUFFY Rew':>10}")
    report.append("-" * 55)
    for i in range(window, len(common_steps)):
        step = common_steps[i]
        d_vol = np.std(duet_r_series[i-window:i])
        l_vol = np.std(luffy_r_series[i-window:i]) if not any(np.isnan(luffy_r_series[i-window:i])) else float('nan')
        report.append(f"{step:>5} | {d_vol:>10.4f} | {l_vol:>10.4f} | {duet_r_series[i]:>10.4f} | {luffy_r_series[i]:>10.4f}")
    report.append("```")
    report.append("")

    # ================================================================
    # 7. DIAGNOSIS SUMMARY
    # ================================================================
    report.append("## 7. Diagnosis Summary")
    report.append("")

    # Compute key diagnostics
    # Check if teacher advantage hits ±5 clip frequently
    clip_hits_late = []
    for step in range(70, 101):
        trajs = load_trajectories(DUET_DIR, step)
        teacher_advs = [t["diag"]["adv_mean"] for t in trajs if t.get("diag", {}).get("is_teacher", False)]
        if teacher_advs:
            a = np.array(teacher_advs)
            clip_hits_late.append((np.abs(a) > 4.5).mean())  # near-clip

    avg_clip_rate = np.mean(clip_hits_late) * 100 if clip_hits_late else 0

    # Check entropy collapse
    entropy_60 = duet_metrics["diag/entropy_onpolicy_token_mean"][duet_steps.index(60)] if 60 in duet_steps else float('nan')
    entropy_90 = duet_metrics["diag/entropy_onpolicy_token_mean"][duet_steps.index(90)] if 90 in duet_steps else float('nan')
    entropy_drop = (entropy_60 - entropy_90) / entropy_60 * 100 if not np.isnan(entropy_60) and entropy_60 > 0 else float('nan')

    # Check reward drop
    reward_60 = duet_metrics["diag/reward_onpolicy_mean"][duet_steps.index(60)] if 60 in duet_steps else float('nan')
    reward_80 = duet_metrics["diag/reward_onpolicy_mean"][duet_steps.index(80)] if 80 in duet_steps else float('nan')
    reward_100 = duet_metrics["diag/reward_onpolicy_mean"][duet_steps.index(100)] if 100 in duet_steps else float('nan')

    # Compare old_log_prob drift
    duet_traj_60 = analyze_trajectories_detailed(DUET_DIR, 60)
    duet_traj_80 = analyze_trajectories_detailed(DUET_DIR, 80)
    duet_traj_100 = analyze_trajectories_detailed(DUET_DIR, 100)
    luffy_traj_60 = analyze_trajectories_detailed(LUFFY_DIR, 60)
    luffy_traj_80 = analyze_trajectories_detailed(LUFFY_DIR, 80)
    luffy_traj_100 = analyze_trajectories_detailed(LUFFY_DIR, 100)

    d_lp_60 = safe_mean(duet_traj_60["old_logp_onpolicy"]) if duet_traj_60 else float('nan')
    d_lp_80 = safe_mean(duet_traj_80["old_logp_onpolicy"]) if duet_traj_80 else float('nan')
    d_lp_100 = safe_mean(duet_traj_100["old_logp_onpolicy"]) if duet_traj_100 else float('nan')
    l_lp_60 = safe_mean(luffy_traj_60["old_logp_onpolicy"]) if luffy_traj_60 else float('nan')
    l_lp_80 = safe_mean(luffy_traj_80["old_logp_onpolicy"]) if luffy_traj_80 else float('nan')
    l_lp_100 = safe_mean(luffy_traj_100["old_logp_onpolicy"]) if luffy_traj_100 else float('nan')

    report.append("### Key Findings")
    report.append("")
    report.append(f"1. **Reward trajectory**: step60={reward_60:.4f}, step80={reward_80:.4f}, step100={reward_100:.4f}")
    report.append(f"   - Drop 60→100: {(reward_60-reward_100)/reward_60*100:.1f}%")
    report.append("")
    report.append(f"2. **Teacher adv clip activation** (steps 70-100): {avg_clip_rate:.1f}% of teacher samples near ±5 boundary")
    report.append("")
    report.append(f"3. **Entropy**: step60={entropy_60:.4f}, step90={entropy_90:.4f} (drop: {entropy_drop:.1f}%)")
    report.append("")
    report.append(f"4. **KL drift** (old_log_prob_mean on-policy):")
    report.append(f"   - DUET:  step60={d_lp_60:.4f}, step80={d_lp_80:.4f}, step100={d_lp_100:.4f}")
    report.append(f"   - LUFFY: step60={l_lp_60:.4f}, step80={l_lp_80:.4f}, step100={l_lp_100:.4f}")
    report.append(f"   - DUET drift 60→100: {abs(d_lp_60-d_lp_100):.4f}")
    report.append(f"   - LUFFY drift 60→100: {abs(l_lp_60-l_lp_100):.4f}")
    report.append("")

    # Compute correlation between reward drop and other metrics
    # Check if advantages become degenerate
    report.append("### Potential Collapse Mechanisms")
    report.append("")
    report.append("Check each hypothesis:")
    report.append("")

    # H1: Teacher adv clip causing over-imitation
    late_teacher_advs = []
    for step in range(80, 101):
        trajs = load_trajectories(DUET_DIR, step)
        for t in trajs:
            if t.get("diag", {}).get("is_teacher", False):
                late_teacher_advs.append(t["diag"]["adv_mean"])
    if late_teacher_advs:
        a = np.array(late_teacher_advs)
        frac_pos5 = (a >= 4.9).mean() * 100
        frac_neg5 = (a <= -4.9).mean() * 100
        report.append(f"**H1: Teacher adv clip over-imitation** (steps 80-100)")
        report.append(f"  - {frac_pos5:.1f}% teacher samples clipped to ~+5")
        report.append(f"  - {frac_neg5:.1f}% teacher samples clipped to ~-5")
        report.append(f"  - Mean teacher adv: {a.mean():.4f}, Std: {a.std():.4f}")
        report.append(f"  - Verdict: {'LIKELY' if frac_pos5 > 30 else 'UNLIKELY'} cause")
        report.append("")

    # H2: KL explosion
    if not np.isnan(d_lp_60) and not np.isnan(d_lp_100):
        kl_drift = abs(d_lp_60 - d_lp_100)
        luffy_kl_drift = abs(l_lp_60 - l_lp_100) if not np.isnan(l_lp_60) and not np.isnan(l_lp_100) else float('nan')
        report.append(f"**H2: KL explosion**")
        report.append(f"  - DUET log_prob drift: {kl_drift:.4f}")
        report.append(f"  - LUFFY log_prob drift: {luffy_kl_drift:.4f}")
        if not np.isnan(luffy_kl_drift):
            report.append(f"  - DUET/LUFFY drift ratio: {kl_drift/max(luffy_kl_drift,0.0001):.2f}x")
        report.append(f"  - Verdict: {'LIKELY' if kl_drift > 2 * luffy_kl_drift else 'UNLIKELY'} cause")
        report.append("")

    # H3: Entropy collapse
    report.append(f"**H3: Entropy collapse / mode collapse**")
    report.append(f"  - Entropy drop 60→90: {entropy_drop:.1f}%")
    # Check for very low entropy at collapse
    entropy_100 = duet_metrics["diag/entropy_onpolicy_token_mean"][duet_steps.index(100)] if 100 in duet_steps else float('nan')
    report.append(f"  - Step 100 entropy: {entropy_100:.4f}")
    report.append(f"  - Verdict: {'LIKELY' if entropy_drop > 50 else 'POSSIBLE' if entropy_drop > 25 else 'UNLIKELY'} cause")
    report.append("")

    # H4: On-policy advantage degeneracy
    report.append(f"**H4: On-policy advantage degeneracy**")
    for step in [60, 70, 80, 90, 100]:
        if step in duet_steps:
            idx = duet_steps.index(step)
            oam = duet_metrics["diag/adv_onpolicy_sample_mean"][idx]
            opr = duet_metrics["diag/onpolicy_adv_pos_ratio"][idx]
            report.append(f"  - Step {step}: on_adv_mean={oam:.6f}, pos_ratio={opr:.4f}")
    report.append("")

    # H5: Response length explosion/collapse
    report.append(f"**H5: Response length degeneracy**")
    for step in [60, 70, 80, 90, 100]:
        duet_traj = analyze_trajectories_detailed(DUET_DIR, step)
        if duet_traj:
            m = safe_mean(duet_traj["response_len_onpolicy"])
            s = safe_std(duet_traj["response_len_onpolicy"])
            report.append(f"  - Step {step}: mean={m:.1f}, std={s:.1f}")
    report.append("")

    # Write report
    report_text = "\n".join(report)
    output_path = os.path.join(OUTPUT_DIR, "collapse_analysis.md")
    with open(output_path, "w") as f:
        f.write(report_text)

    print(f"Report written to {output_path}")
    print("\n" + "="*60)
    print("QUICK SUMMARY")
    print("="*60)
    print(f"Reward: step60={reward_60:.4f} → step80={reward_80:.4f} → step100={reward_100:.4f}")
    print(f"Entropy: step60={entropy_60:.4f} → step90={entropy_90:.4f} (drop {entropy_drop:.1f}%)")
    print(f"Teacher adv clip rate (steps 70-100): {avg_clip_rate:.1f}%")
    print(f"KL drift DUET: {abs(d_lp_60-d_lp_100):.4f} vs LUFFY: {abs(l_lp_60-l_lp_100):.4f}")

if __name__ == "__main__":
    main()
