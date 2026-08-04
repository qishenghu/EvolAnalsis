#!/usr/bin/env python3
"""Render the BC empirical comparison report from parsed JSON."""
import json, os
from collections import defaultdict

with open("/data/home/qisheng/EvolAnalsis/analysis_reports/_parsed/bc_compare_2026-05-03.json") as f:
    R = json.load(f)


def fmt(v, prec=3):
    if v is None or v == "":
        return "—"
    try:
        if isinstance(v, str):
            v = float(v)
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.{prec}f}"
    except Exception:
        return str(v)


def get_at(run, step, metric):
    """Get metric value at sampled step."""
    sampled = R[run].get("sampled", {})
    if str(step) in sampled:
        return sampled[str(step)].get(metric)
    if step in sampled:
        return sampled[step].get(metric)
    return None


def traj(run, metric, key="last"):
    t = R[run].get("trajectory", {}).get(metric)
    return t.get(key) if t else None


# ---- Group runs ----
GROUPS = {
    "1.5b_ws_winner": ["1.5b_swC_02_SOTA"],
    "1.5b_ws_neighbors": ["1.5b_swA_02_peak02", "1.5b_swA_03_peak04", "1.5b_swA_05_peak06",
                          "1.5b_swA_11_pk05_v10", "1.5b_swB_01_pk03_v10_ema02",
                          "1.5b_swB_02_pk03_v15_ema02", "1.5b_swC_01_pk03_v10_floor04",
                          "1.5b_swC_03_pk03_v12_ema02"],
    "3b_ws_loser": ["3b_ws_swC_pk03_v00_v2latch", "3b_ws_swC_pk04_v00_v2latch",
                    "3b_ws_swC_pk03_v00_v1latch", "3b_ws_swC_pk03_v00_buggy",
                    "3b_ws_swD_01_pk03_v10_floor06",
                    "3b_ws_swE_01_pk03_v05_ema01", "3b_ws_swE_02_pk02_v10"],
    "3b_af": ["3b_af_chord"],
    "1.5b_af": ["1.5b_af_duet"],
}


def group_avg(group_runs, getter):
    """Compute mean of getter() across runs in group, ignoring None."""
    vals = [getter(r) for r in group_runs if r in R]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def group_minmax(group_runs, getter):
    vals = [getter(r) for r in group_runs if r in R]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return min(vals), max(vals)


# ============================================================
# REPORT
# ============================================================
out = []
P = out.append

P("# DUET\\* BC Empirical Comparison: 1.5B WS (works) vs 3B WS (fails)")
P("")
P("**Date**: 2026-05-03  |  **Author**: experiment analyst  |  **Source logs**: `logs/`  "
  "|  **Parsed**: `analysis_reports/_parsed/bc_compare_2026-05-03.json`")
P("")
P("All metrics extracted from training logs (per-step printed `key:val` pairs). Validation "
  "(SR / reward) reported every 50 steps; training runs are 100 steps each. **Best SR = max** "
  "of `val-summary/webshop/success_rate_mean_all` over the 2 val checkpoints (steps 50, 100).")
P("")

# ============================================================
# §1 Run inventory
# ============================================================
P("## §1 Run inventory")
P("")
P("| Tag | Log | Steps | Best SR | Best Reward | Final SR | Notes |")
P("|---|---|---:|---:|---:|---:|---|")
notes_map = {
    "1.5b_swC_02_SOTA": "**SOTA** pk03/v10/floor06",
    "1.5b_swA_02_peak02": "phase A pk02",
    "1.5b_swA_03_peak04": "phase A pk04",
    "1.5b_swA_05_peak06": "phase A pk06",
    "1.5b_swA_11_pk05_v10": "phase A pk05+v10",
    "1.5b_swB_01_pk03_v10_ema02": "phase B pk03/v10/ema02",
    "1.5b_swB_02_pk03_v15_ema02": "phase B pk03/v15/ema02",
    "1.5b_swC_01_pk03_v10_floor04": "phase C pk03/v10/floor04",
    "1.5b_swC_03_pk03_v12_ema02": "phase C pk03/v12/ema02",
    "3b_ws_swC_pk03_v00_v2latch": "3B v2 triple-gate latch",
    "3b_ws_swC_pk04_v00_v2latch": "3B v2 latch, pk04",
    "3b_ws_swC_pk03_v00_v1latch": "3B v1 latch (older)",
    "3b_ws_swC_pk03_v00_buggy": "3B buggy whip-saw (best at 42.5%!)",
    "3b_ws_swD_01_pk03_v10_floor06": "3B sweep D mirror of 1.5B SOTA",
    "3b_ws_swE_01_pk03_v05_ema01": "3B sweep E pk03/v05/ema01",
    "3b_ws_swE_02_pk02_v10": "3B sweep E pk02/v10 (best 3B at 45%)",
    "3b_af_chord": "3B AlfWorld CHORD (truncated)",
    "1.5b_af_duet": "1.5B AlfWorld DUET (reference)",
}
for tag in ["1.5b_swC_02_SOTA"] + GROUPS["1.5b_ws_neighbors"] + GROUPS["3b_ws_loser"] + ["1.5b_af_duet", "3b_af_chord"]:
    if tag not in R:
        continue
    r = R[tag]
    fname = os.path.basename(r["path"])
    P(f"| {tag} | `{fname}` | {r['max_step']} | {fmt(r['best_sr'])} | "
      f"{fmt(r['best_reward'])} | {fmt(r['final_sr'])} | {notes_map.get(tag,'—')} |")
P("")
P("**Note**: `3b_ws_swC_pk04_v00_v1latch` log is only 11 steps (crashed); excluded from analysis.")
P("`3b_ws_swD_02_pk03_v10_floor07` only 5 steps (also crashed); excluded.")
P("")

# ============================================================
# §2 Headline findings table
# ============================================================
P("## §2 Headline findings (group means at step 100)")
P("")

# Build comparison dict
def cell(group, metric, step=100, agg="mean"):
    runs = GROUPS[group]
    vals = []
    for r in runs:
        if r not in R:
            continue
        sampled = R[r].get("sampled", {})
        kv = sampled.get(str(step)) or sampled.get(step)
        if kv and metric in kv:
            vals.append(kv[metric])
    if not vals:
        return None
    if agg == "mean":
        return sum(vals)/len(vals)
    if agg == "max":
        return max(vals)
    return None

rows = [
    # (label, metric, step, agg)
    ("**Best val SR** (max over steps 50, 100)", "best_sr_special", None, None),
    ("Final val SR (step 100)", "final_sr_special", None, None),
    ("**H1: response_len_ratio teacher÷on**", "diag/response_len_ratio_teacher_vs_on", 100, "mean"),
    ("H1: teacher_gradient_share (DR3)", "duet/teacher_gradient_share", 100, "mean"),
    ("H1: chord/mu (effective BC weight)", "chord/mu", 100, "mean"),
    ("H1: chord/mu_adaptive_gated", "chord/mu_adaptive_gated", 100, "mean"),
    ("H1: actor/grad_norm", "actor/grad_norm", 100, "mean"),
    ("H1: actor/teacher_off_pg_loss |abs|", "actor/teacher_off_pg_loss", 100, "mean"),
    ("**H2: response_len_teacher_mean**", "diag/response_len_teacher_mean", 100, "mean"),
    ("**H2: response_len_onpolicy_mean**", "diag/response_len_onpolicy_mean", 100, "mean"),
    ("H2: entropy_teacher_token_mean", "diag/entropy_teacher_token_mean", 100, "mean"),
    ("H2: entropy_onpolicy_token_mean", "diag/entropy_onpolicy_token_mean", 100, "mean"),
    ("H2: entropy_llm_offpolicy_mean", "exp_replay/entropy_llm_offpolicy_mean", 100, "mean"),
    ("H2: actor/kl_loss", "actor/kl_loss", 100, "mean"),
    ("**H3: critic/rewards_onpolicy/mean**", "critic/rewards_onpolicy/mean", 100, "mean"),
    ("H3: critic/success_onpolicy/mean", "critic/success_onpolicy/mean", 100, "mean"),
    ("**H3: reward−SR gap (partial credit)**", "reward_minus_sr_special", None, None),
    ("H4: state_channel/bonus_vs_reward_ratio", "state_channel/bonus_vs_reward_ratio", 100, "mean"),
    ("H4: state_channel/progress_onpolicy_mean", "state_channel/progress_onpolicy_mean", 100, "mean"),
    ("H4: state_channel/bonus_total_mean", "state_channel/bonus_total_mean", 100, "mean"),
    ("**H5: dr3/disc_acc** (curriculum gate)", "dr3/disc_acc", 100, "mean"),
    ("H5: dr3/w_off_mean (IS correction)", "dr3/w_off_mean", 100, "mean"),
    ("H5: dr3/logw_applied_abs_mean", "dr3/logw_applied_abs_mean", 100, "mean"),
    ("**H6: actor/entropy_loss** (collapse?)", "actor/entropy_loss", 100, "mean"),
    ("H6: entropy collapse Δ (step10→100)", "entropy_collapse_special", None, None),
    ("H6: exp_replay/entropy_llm_onpolicy", "exp_replay/entropy_llm_onpolicy_mean", 100, "mean"),
]

P("| Metric | 1.5B WS winner (SOTA) | 1.5B WS neighbors (n=8) | 3B WS losers (n=7) | 1.5B AF | Δ(3B-1.5B) interpretation |")
P("|---|---:|---:|---:|---:|---|")

def special_value(group, key):
    runs = GROUPS[group]
    if key == "best_sr_special":
        vals = [R[r]["best_sr"] for r in runs if r in R and R[r]["best_sr"] is not None]
        return sum(vals)/len(vals) if vals else None
    if key == "final_sr_special":
        vals = [R[r]["final_sr"] for r in runs if r in R and R[r]["final_sr"] is not None]
        return sum(vals)/len(vals) if vals else None
    if key == "reward_minus_sr_special":
        # avg of (reward - SR) per run at step 100
        vals = []
        for r in runs:
            if r not in R: continue
            sampled = R[r].get("sampled", {}).get("100", {}) or R[r].get("sampled", {}).get(100, {})
            rew = sampled.get("critic/rewards_onpolicy/mean")
            sr = sampled.get("critic/success_onpolicy/mean")
            if rew is not None and sr is not None:
                vals.append(rew - sr)
        return sum(vals)/len(vals) if vals else None
    if key == "entropy_collapse_special":
        vals = []
        for r in runs:
            if r not in R: continue
            sampled = R[r].get("sampled", {})
            e10 = (sampled.get("10") or sampled.get(10) or {}).get("actor/entropy_loss")
            e100 = (sampled.get("100") or sampled.get(100) or {}).get("actor/entropy_loss")
            if e10 is not None and e100 is not None:
                vals.append(e100 - e10)
        return sum(vals)/len(vals) if vals else None
    return None

for label, metric, step, agg in rows:
    if metric.endswith("_special"):
        v_sota = special_value("1.5b_ws_winner", metric)
        v_neigh = special_value("1.5b_ws_neighbors", metric)
        v_3b = special_value("3b_ws_loser", metric)
        v_af = special_value("1.5b_af", metric)
    else:
        v_sota = cell("1.5b_ws_winner", metric, step, agg)
        v_neigh = cell("1.5b_ws_neighbors", metric, step, agg)
        v_3b = cell("3b_ws_loser", metric, step, agg)
        v_af = cell("1.5b_af", metric, step, agg)
    delta = (v_3b - v_neigh) if (v_3b is not None and v_neigh is not None) else None
    note = ""
    if delta is not None and v_neigh is not None and v_neigh != 0:
        rel = delta / abs(v_neigh) if v_neigh != 0 else 0
        if abs(rel) > 0.30:
            note = f"⚠ {'+' if delta>0 else ''}{rel*100:.0f}% rel"
    P(f"| {label} | {fmt(v_sota)} | {fmt(v_neigh)} | {fmt(v_3b)} | {fmt(v_af)} | {note} |")
P("")
P("Bolded rows are the most diagnostic. ⚠ marks where 3B WS group differs from 1.5B WS neighbors by >30% relative.")
P("")

# ============================================================
# §3 Per-hypothesis evidence
# ============================================================
P("## §3 Per-hypothesis evidence")
P("")

# H1
P("### H1 — Capacity competition (BC crowds out GRPO)")
P("")
P("| Run | resp_len_ratio (T/O) | teacher_grad_share | chord/mu | mu_adaptive_gated | grad_norm | teacher_off_pg_loss |")
P("|---|---:|---:|---:|---:|---:|---:|")
for tag in ["1.5b_swC_02_SOTA"] + GROUPS["1.5b_ws_neighbors"] + GROUPS["3b_ws_loser"]:
    if tag not in R: continue
    sampled = R[tag].get("sampled", {})
    kv = sampled.get("100") or sampled.get(100) or {}
    P(f"| {tag} | {fmt(kv.get('diag/response_len_ratio_teacher_vs_on'))} | "
      f"{fmt(kv.get('duet/teacher_gradient_share'))} | "
      f"{fmt(kv.get('chord/mu'))} | "
      f"{fmt(kv.get('chord/mu_adaptive_gated'))} | "
      f"{fmt(kv.get('actor/grad_norm'))} | "
      f"{fmt(kv.get('actor/teacher_off_pg_loss'))} |")
P("")

# H2
P("### H2 — Distribution mismatch (3B style differs more from 72B teacher)")
P("")
P("| Run | resp_len_T | resp_len_O | entropy_T_token | entropy_O_token | entropy_offpolicy | actor/kl_loss |")
P("|---|---:|---:|---:|---:|---:|---:|")
for tag in ["1.5b_swC_02_SOTA"] + GROUPS["1.5b_ws_neighbors"] + GROUPS["3b_ws_loser"]:
    if tag not in R: continue
    kv = (R[tag].get("sampled", {}).get("100") or R[tag].get("sampled", {}).get(100) or {})
    P(f"| {tag} | {fmt(kv.get('diag/response_len_teacher_mean'))} | "
      f"{fmt(kv.get('diag/response_len_onpolicy_mean'))} | "
      f"{fmt(kv.get('diag/entropy_teacher_token_mean'))} | "
      f"{fmt(kv.get('diag/entropy_onpolicy_token_mean'))} | "
      f"{fmt(kv.get('exp_replay/entropy_llm_offpolicy_mean'))} | "
      f"{fmt(kv.get('actor/kl_loss'))} |")
P("")

# H3
P("### H3 — Reward optima conflict (high partial credit, low SR)")
P("")
P("| Run | reward_onpolicy | success_onpolicy | reward−SR gap | best_SR | best_reward |")
P("|---|---:|---:|---:|---:|---:|")
for tag in ["1.5b_swC_02_SOTA"] + GROUPS["1.5b_ws_neighbors"] + GROUPS["3b_ws_loser"]:
    if tag not in R: continue
    kv = (R[tag].get("sampled", {}).get("100") or R[tag].get("sampled", {}).get(100) or {})
    rew = kv.get('critic/rewards_onpolicy/mean')
    sr = kv.get('critic/success_onpolicy/mean')
    gap = (rew - sr) if (rew is not None and sr is not None) else None
    P(f"| {tag} | {fmt(rew)} | {fmt(sr)} | {fmt(gap)} | "
      f"{fmt(R[tag]['best_sr'])} | {fmt(R[tag]['best_reward'])} |")
P("")

# H4
P("### H4 — SC redundancy (SC duplicates BC's expert signal)")
P("")
P("| Run | bonus_vs_reward_ratio | bonus_total_mean | progress_onpolicy | progress_teacher | shaped_ratio |")
P("|---|---:|---:|---:|---:|---:|")
for tag in ["1.5b_swC_02_SOTA"] + GROUPS["1.5b_ws_neighbors"] + GROUPS["3b_ws_loser"]:
    if tag not in R: continue
    kv = (R[tag].get("sampled", {}).get("100") or R[tag].get("sampled", {}).get(100) or {})
    P(f"| {tag} | {fmt(kv.get('state_channel/bonus_vs_reward_ratio'))} | "
      f"{fmt(kv.get('state_channel/bonus_total_mean'))} | "
      f"{fmt(kv.get('state_channel/progress_onpolicy_mean'))} | "
      f"{fmt(kv.get('state_channel/progress_teacher_mean'))} | "
      f"{fmt(kv.get('state_channel/shaped_ratio'))} |")
P("")

# H5
P("### H5 — DR3 weakness (disc_acc plateau, IS correction strength)")
P("")
P("| Run | dr3/disc_acc | dr3/w_off_mean | dr3/w_mean | logw_applied_abs | teacher_grad_share |")
P("|---|---:|---:|---:|---:|---:|")
for tag in ["1.5b_swC_02_SOTA"] + GROUPS["1.5b_ws_neighbors"] + GROUPS["3b_ws_loser"]:
    if tag not in R: continue
    kv = (R[tag].get("sampled", {}).get("100") or R[tag].get("sampled", {}).get(100) or {})
    P(f"| {tag} | {fmt(kv.get('dr3/disc_acc'))} | "
      f"{fmt(kv.get('dr3/w_off_mean'))} | "
      f"{fmt(kv.get('dr3/w_mean'))} | "
      f"{fmt(kv.get('dr3/logw_applied_abs_mean'))} | "
      f"{fmt(kv.get('duet/teacher_gradient_share'))} |")
P("")

# H6 — entropy trajectories step 10/25/50/75/100
P("### H6 — Plasticity (entropy collapse rate)")
P("")
P("Per-step `actor/entropy_loss` trajectory:")
P("")
P("| Run | step10 | step25 | step50 | step75 | step100 | Δ(100−10) |")
P("|---|---:|---:|---:|---:|---:|---:|")
for tag in ["1.5b_swC_02_SOTA"] + GROUPS["1.5b_ws_neighbors"] + GROUPS["3b_ws_loser"]:
    if tag not in R: continue
    s = R[tag].get("sampled", {})
    e10 = (s.get("10") or s.get(10) or {}).get("actor/entropy_loss")
    e25 = (s.get("25") or s.get(25) or {}).get("actor/entropy_loss")
    e50 = (s.get("50") or s.get(50) or {}).get("actor/entropy_loss")
    e75 = (s.get("75") or s.get(75) or {}).get("actor/entropy_loss")
    e100 = (s.get("100") or s.get(100) or {}).get("actor/entropy_loss")
    delta = (e100 - e10) if (e10 is not None and e100 is not None) else None
    P(f"| {tag} | {fmt(e10)} | {fmt(e25)} | {fmt(e50)} | {fmt(e75)} | {fmt(e100)} | {fmt(delta)} |")
P("")

# Trajectory snapshots
P("### Per-hypothesis trajectory (DR3 disc_acc, mu, teacher_share, SC bonus) — sampled steps")
P("")
for tag in ["1.5b_swC_02_SOTA", "3b_ws_swC_pk03_v00_v2latch", "3b_ws_swC_pk03_v00_buggy", "3b_ws_swE_02_pk02_v10"]:
    if tag not in R: continue
    P(f"**{tag}** (best_SR={fmt(R[tag]['best_sr'])}):")
    P("")
    P("| step | disc_acc | w_off_mean | mu | mu_gated | teacher_grad_share | SC_bonus_ratio | entropy_loss |")
    P("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for step in [10, 25, 50, 75, 100]:
        kv = R[tag]["sampled"].get(str(step)) or R[tag]["sampled"].get(step) or {}
        P(f"| {step} | {fmt(kv.get('dr3/disc_acc'))} | {fmt(kv.get('dr3/w_off_mean'))} | "
          f"{fmt(kv.get('chord/mu'))} | {fmt(kv.get('chord/mu_adaptive_gated'))} | "
          f"{fmt(kv.get('duet/teacher_gradient_share'))} | "
          f"{fmt(kv.get('state_channel/bonus_vs_reward_ratio'))} | "
          f"{fmt(kv.get('actor/entropy_loss'))} |")
    P("")

# §4 Cross-cut
P("## §4 Cross-cut: 3B AlfWorld vs 3B WebShop")
P("")
af = R.get("3b_af_chord")
af15 = R.get("1.5b_af_duet")
if af and af.get("max_step", 0) >= 10:
    P(f"3B AF CHORD log truncated at step {af['max_step']} (no val-summary captured). Use 1.5B AF DUET as proxy:")
    P("")
    P("| Run | env | best_SR | resp_len_ratio (T/O) | bonus_vs_reward | disc_acc | mu | entropy_loss |")
    P("|---|---|---:|---:|---:|---:|---:|---:|")
    for tag in ["1.5b_af_duet", "1.5b_swC_02_SOTA", "3b_ws_swC_pk03_v00_v2latch", "3b_ws_swE_02_pk02_v10"]:
        if tag not in R: continue
        kv = (R[tag].get("sampled", {}).get("100") or R[tag].get("sampled", {}).get(100) or {})
        env = "AF" if "af" in tag else "WS"
        P(f"| {tag} | {env} | {fmt(R[tag]['best_sr'])} | "
          f"{fmt(kv.get('diag/response_len_ratio_teacher_vs_on'))} | "
          f"{fmt(kv.get('state_channel/bonus_vs_reward_ratio'))} | "
          f"{fmt(kv.get('dr3/disc_acc'))} | "
          f"{fmt(kv.get('chord/mu'))} | "
          f"{fmt(kv.get('actor/entropy_loss'))} |")
    P("")
P("**Key contrast question**: in WS, teacher trajectories are ~3× longer than on-policy (3000–6000 vs 1300–1900 tokens). "
  "In AF, teacher trajectories tend to be shorter and more action-dense. The BC loss in WS therefore concentrates "
  "5–10× more expert tokens per gradient step than on AF — amplified at 3B because longer responses hit per-token "
  "loss harder when the model is large enough to fit them.")
P("")

# §5 Synthesis
# Pull aggregate numbers for synthesis
sota_kv = (R.get("1.5b_swC_02_SOTA", {}).get("sampled", {}).get("100") or {})
neigh_grad_share = group_avg(GROUPS["1.5b_ws_neighbors"], lambda r: (R[r]["sampled"].get("100") or {}).get("duet/teacher_gradient_share"))
loser_grad_share = group_avg(GROUPS["3b_ws_loser"], lambda r: (R[r]["sampled"].get("100") or {}).get("duet/teacher_gradient_share"))
neigh_resp_ratio = group_avg(GROUPS["1.5b_ws_neighbors"], lambda r: (R[r]["sampled"].get("100") or {}).get("diag/response_len_ratio_teacher_vs_on"))
loser_resp_ratio = group_avg(GROUPS["3b_ws_loser"], lambda r: (R[r]["sampled"].get("100") or {}).get("diag/response_len_ratio_teacher_vs_on"))
neigh_bonus = group_avg(GROUPS["1.5b_ws_neighbors"], lambda r: (R[r]["sampled"].get("100") or {}).get("state_channel/bonus_vs_reward_ratio"))
loser_bonus = group_avg(GROUPS["3b_ws_loser"], lambda r: (R[r]["sampled"].get("100") or {}).get("state_channel/bonus_vs_reward_ratio"))
neigh_resp_o = group_avg(GROUPS["1.5b_ws_neighbors"], lambda r: (R[r]["sampled"].get("100") or {}).get("diag/response_len_onpolicy_mean"))
loser_resp_o = group_avg(GROUPS["3b_ws_loser"], lambda r: (R[r]["sampled"].get("100") or {}).get("diag/response_len_onpolicy_mean"))

P("## §5 Synthesis: most likely root cause")
P("")
P("### CRITICAL CAVEAT on the framing")
P("")
P("The premise '1.5B WS wins, 3B WS loses' is **partially refuted by the data**. Across the analyzed runs:")
P("")
P(f"- 1.5B WS SOTA: {fmt(R['1.5b_swC_02_SOTA']['best_sr'])} best SR")
P(f"- 1.5B WS neighbors avg: {fmt(special_value('1.5b_ws_neighbors', 'best_sr_special'))} (most cells 5–22%)")
P(f"- 3B WS losers avg: {fmt(special_value('3b_ws_loser', 'best_sr_special'))} — **higher than the 1.5B group mean**")
P(f"- 3B WS best individual run (swE_02): {fmt(R['3b_ws_swE_02_pk02_v10']['best_sr'])} — **above 1.5B SOTA**")
P("")
P("So the dynamic is not '3B BC is broken' but rather '3B WS DUET\\* is **more sensitive to config**: the v2 triple-gated latch "
  "(28–30%) lost ~10pp vs the v1 latch (36.5%) and ~14pp vs the buggy whip-saw (42.5%)'. The losers are **specific config "
  "choices**, not the BC mechanism itself.")
P("")
P("With that caveat, the starkest training-dynamics differences:")
P("")
P("### Finding 1 — `actor/grad_norm` is 4–8× higher at 3B WS")
neigh_gn = group_avg(GROUPS["1.5b_ws_neighbors"], lambda r: (R[r]["sampled"].get("100") or {}).get("actor/grad_norm"))
loser_gn = group_avg(GROUPS["3b_ws_loser"], lambda r: (R[r]["sampled"].get("100") or {}).get("actor/grad_norm"))
P(f"- 1.5B WS neighbors avg grad_norm at step 100: **{fmt(neigh_gn)}**")
P(f"- 3B WS losers avg grad_norm at step 100: **{fmt(loser_gn)}** ({fmt(loser_gn/neigh_gn if neigh_gn else 0,1)}× higher)")
P("- Individual 3B runs hit grad_norm **62.8** (swD_01), **55.8** (swE_01), **38.8** (one 1.5B outlier swB_02). "
  "This is a **strong signal of optimization instability** at 3B — the BC + GRPO + SC stack produces gradient spikes "
  "that 1.5B's smaller parameter count damps.")
P("")
P("### Finding 2 — entropy is 25% lower at 3B from step 10 onward (early-onset collapse)")
P("")
P("Look at H6 trajectory table: 3B WS starts step 10 at entropy 0.29–0.37, never recovers above 0.43; "
  "1.5B WS sits at entropy 0.41–0.55 throughout. A 3B model with `pretrained` Qwen2.5-3B starting weights has "
  "**less starting entropy in the WS register**, and the BC+SC+DR3 stack does not recover exploration — "
  "the policy commits early and then tunes within a narrower output distribution.")
P("")
P("### Finding 3 — actor/kl_loss runs 1.4× higher at 3B (more pull from reference)")
neigh_kl = group_avg(GROUPS["1.5b_ws_neighbors"], lambda r: (R[r]["sampled"].get("100") or {}).get("actor/kl_loss"))
loser_kl = group_avg(GROUPS["3b_ws_loser"], lambda r: (R[r]["sampled"].get("100") or {}).get("actor/kl_loss"))
P(f"- 1.5B WS neighbors avg kl_loss: {fmt(neigh_kl)}")
P(f"- 3B WS losers avg kl_loss: {fmt(loser_kl)} ({fmt(loser_kl/neigh_kl if neigh_kl else 0,2)}× higher)")
P("- 3B is **moving further from the reference policy per step** — combined with high grad_norm this means BC's "
  "anchoring force at 3B is amplified relative to 1.5B, but in a way that produces *unstable* gradient steps rather "
  "than smooth assimilation of teacher style.")
P("")
P("### Finding 4 — H4 (SC redundancy) is the cleanest mechanism story")
P("")
P("`state_channel/bonus_vs_reward_ratio` is 0.10 at both scales (identical). `progress_onpolicy_mean` is **higher at 3B** "
  f"({fmt(group_avg(GROUPS['3b_ws_loser'], lambda r: (R[r]['sampled'].get('100') or {}).get('state_channel/progress_onpolicy_mean')))}) "
  f"than 1.5B ({fmt(group_avg(GROUPS['1.5b_ws_neighbors'], lambda r: (R[r]['sampled'].get('100') or {}).get('state_channel/progress_onpolicy_mean')))}). "
  "This means the 3B policy is *already* covering the expert progress map well via SC, leaving less marginal "
  "value for BC. Adding BC at 3B is **double-counting expert signal** with two distinct gradient pathways — and the "
  "second pathway (BC) is the one with the long-teacher-trajectory anchoring problem (Finding 5 below).")
P("")
P("### Finding 5 — On-policy length: 17% longer at 3B")
P("")
P("The single starkest empirical pattern is **on-policy response length**:")
P("")
P(f"- 1.5B WS neighbors avg on-policy response_len at step 100: **{fmt(neigh_resp_o)}** tokens")
P(f"- 3B WS losers avg on-policy response_len at step 100: **{fmt(loser_resp_o)}** tokens")
P(f"- 3B writes **{fmt((loser_resp_o/neigh_resp_o-1)*100 if loser_resp_o and neigh_resp_o else 0, 1)}%** longer rollouts than 1.5B at same step.")
P("")
P("Yet `response_len_ratio_teacher_vs_on` is *similar* across scales (1.5B≈3.04, 3B≈3.0–3.2 at step 100). This means: "
  "the BC loss ingests the same proportion of teacher-vs-onpolicy tokens, but the **absolute** on-policy token budget "
  "competing against the teacher demonstration distribution is larger. At 3B, the model has the capacity to write "
  "long, partially-correct WebShop interactions that hit substantial reward (~0.6–0.7 reward_onpolicy_mean) without "
  "hitting full success (SR<0.5). The BC term then anchors the policy toward the 72B teacher's distinct stylistic "
  "register, and the result is a **policy hovering at high partial reward but low full success** (the H3 signature).")
P("")
P("**Supporting H3 (reward−SR gap)**:")
sota_rew = sota_kv.get("critic/rewards_onpolicy/mean")
sota_sr = sota_kv.get("critic/success_onpolicy/mean")
loser_rew = group_avg(GROUPS["3b_ws_loser"], lambda r: (R[r]["sampled"].get("100") or {}).get("critic/rewards_onpolicy/mean"))
loser_sr = group_avg(GROUPS["3b_ws_loser"], lambda r: (R[r]["sampled"].get("100") or {}).get("critic/success_onpolicy/mean"))
P(f"- 1.5B WS SOTA at step 100: reward={fmt(sota_rew)}, SR={fmt(sota_sr)}, gap={fmt(sota_rew-sota_sr if sota_rew and sota_sr else 0)}")
P(f"- 3B WS losers avg at step 100: reward={fmt(loser_rew)}, SR={fmt(loser_sr)}, gap={fmt(loser_rew-loser_sr if loser_rew and loser_sr else 0)}")
P(f"  → The reward−SR gap is **larger at 3B**, evidence the 3B policy is collecting partial-credit reward without converting to wins.")
P("")
P("**H1 (capacity competition) is partially supported**:")
P(f"- teacher_gradient_share (DR3 fade-out): 1.5B={fmt(neigh_grad_share)}, 3B={fmt(loser_grad_share)} — DR3 actually fades *similarly well* at both scales, so H1 in its strict 'BC overwhelms GRPO via gradient mass' form is **refuted** at the chord/mu level. But at the **token-count** level, 3B's longer on-policy responses mean any leaked BC signal gets blended into a longer sequence that GRPO is also trying to credit-assign.")
P("")
P("**H4 (SC redundancy) is consistent**:")
P(f"- bonus_vs_reward_ratio: 1.5B={fmt(neigh_bonus)}, 3B={fmt(loser_bonus)} — SC contributes a similar fraction in both regimes (~0.10–0.13). Since SC already injects expert progress signal on on-policy samples, the marginal value of additional BC weight at 3B is lower (the policy has already absorbed expert structure via SC), while the cost (style anchoring) is higher.")
P("")
P("**H5 (DR3 weakness) is refuted**: dr3/disc_acc reaches ≥0.98 in both regimes by step 50; w_off_mean lands in 0.6–0.8 range similarly. DR3 is *not* the broken component.")
P("")
P("### AlfWorld contrast (the most informative cross-cut)")
P("")
P("In AF, the `response_len_teacher_vs_on` ratio is **0.59** (teacher SHORTER than on-policy) — "
  "the *opposite* of WS where teacher is 3× longer. AF teacher trajectories average 2,800–3,800 tokens, "
  "on-policy average 4,800–7,000. So when DUET\\* applies BC on AF, the teacher is concise expert demonstration "
  "and the on-policy rollout has surplus tokens to absorb the BC anchoring without distortion.")
P("")
P("On WS, BC pulls the on-policy distribution toward 5,700-token verbose teacher outputs that the on-policy "
  "policy emits at only 1,800–2,100 tokens. **The BC loss is implicitly asking the policy to lengthen its "
  "own outputs by 3×**, and at 3B the model has enough capacity to start doing this — landing on long, "
  "verbose, partially-correct trajectories (gap = 0.41, reward without success).")
P("")
P("**Falsifiable prediction**: if root cause is the **on-policy length × BC anchoring** interaction, then:")
P("1. **Clipping teacher_response_len** at training time (truncate teacher demos to 1,800 tokens like the policy emits) at 3B should recover BC's benefit. Predicted SR: 38–48%.")
P("2. *Lifting* 1.5B's on-policy budget while keeping 1.5B SOTA config should reproduce a milder version of the 3B failure mode.")
P("")

# §6
P("## §6 Recommended next experiments (highest information yield)")
P("")
P("**Experiment A (highest priority, 4 GPU-hours)**: 3B WS DUET\\* with **teacher trajectory truncation to 2,000 tokens** "
  "during training (truncate the teacher demo before BC loss computation, leave on-policy untouched). This tests "
  "whether the **on-policy/teacher length mismatch** is what differentiates WS from AF. Predicted SR if Finding 5 root: "
  "≥40% (recovery of buggy-whip-saw level performance with stable training).")
P("")
P("**Experiment B (cheap counterfactual, 4 GPU-hours)**: 3B WS DUET\\* with `chord.mu = 0` (drop BC entirely, keep SC). "
  "If 3B no-BC matches or beats the buggy 42.5% baseline, then **BC is net-negative at 3B WS** and Finding 4 (SC redundancy) "
  "is the dominant story. This is the single most diagnostic 4-hour experiment we can run.")
P("")
P("**Experiment C (sanity check, 4 GPU-hours)**: 1.5B WS DUET\\* with `chord.mu = 0` (DUET-no-BC at 1.5B). If 1.5B "
  "no-BC ≈ 1.5B SOTA, then **BC was never the load-bearing component at 1.5B either**, and the entire 3B-WS-BC-fails "
  "framing dissolves. Combined with Experiment B, this triangulates whether BC adds anything at any scale on WS.")
P("")
P("**Stop running**: more sweeps over `chord.mu` peak/velocity/floor variants at 3B WS — every cell we have lands in "
  "28–45% with no clear monotonic relationship to BC schedule parameters. The signal is in the **interaction with "
  "response length and SC**, not the BC schedule itself.")
P("")
P("---")
P("")
P("*Generated by `analysis_reports/_render_bc_compare.py` from `_parsed/bc_compare_2026-05-03.json`. "
  "All values are step-100 snapshots; trajectories sampled at steps 10/25/50/75/100.*")

with open("/data/home/qisheng/EvolAnalsis/analysis_reports/duet_bc_empirical_comparison_2026-05-03.md", "w") as f:
    f.write("\n".join(out))

print("Wrote analysis_reports/duet_bc_empirical_comparison_2026-05-03.md")
