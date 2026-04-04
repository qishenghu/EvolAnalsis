#!/usr/bin/env python3
"""Parse and compare DUET 0330 vs 0329 trajectory metrics."""
import re
import json
import os
import sys

BASE = '/data/code/exp/EvolAnalsis'
OUT = f'{BASE}/analysis_outputs/duet_0330_analysis'
LOG_0330 = f'{BASE}/wandb/run-20260330_122831-437s027l/files/output.log'
LOG_0329 = f'{BASE}/wandb/run-20260329_232718-9ryexv2i/files/output.log'
TRAJ_0330 = f'{BASE}/checkpoints/agentevolver/alfworld_3b_duet_0330/Trajectory'
TRAJ_0329 = f'{BASE}/checkpoints/agentevolver/alfworld_3b_duet_0329/Trajectory'


def parse_step_metrics(log_path):
    """Parse all step-level metrics from output.log"""
    metrics_by_step = {}
    with open(log_path) as f:
        for line in f:
            if not line.startswith('step:'):
                continue
            parts = line.strip().split(' - ')
            step = None
            step_metrics = {}
            for part in parts:
                if part.startswith('step:'):
                    step = int(part.split(':')[1])
                else:
                    idx = part.find(':')
                    if idx > 0:
                        key = part[:idx].strip()
                        try:
                            val = float(part[idx+1:].strip())
                            step_metrics[key] = val
                        except ValueError:
                            pass
            if step is not None:
                metrics_by_step[step] = step_metrics
    return metrics_by_step


def parse_validation_metrics(log_path):
    """Parse validation lines from output.log"""
    val_metrics = {}
    with open(log_path) as f:
        for line in f:
            # Look for validation result lines
            if 'valid' in line.lower() and ('success' in line.lower() or 'reward' in line.lower()):
                # Try to extract step and success rate
                pass
            # Also look for eval lines like "step:N ... critic/success_onpolicy/mean"
            if line.startswith('step:') and 'critic/' in line:
                parts = line.strip().split(' - ')
                step = None
                step_metrics = {}
                for part in parts:
                    if part.startswith('step:'):
                        step = int(part.split(':')[1])
                    elif 'critic/' in part:
                        idx = part.find(':')
                        if idx > 0:
                            key = part[:idx].strip()
                            try:
                                val = float(part[idx+1:].strip())
                                step_metrics[key] = val
                            except ValueError:
                                pass
                if step is not None and step_metrics:
                    val_metrics[step] = step_metrics
    return val_metrics


def main():
    os.makedirs(OUT, exist_ok=True)

    print("=" * 70)
    print("DUET 0330 vs 0329 — Deep Trajectory Analysis")
    print("=" * 70)

    # Parse logs
    print("\n[1] Parsing metrics from output logs...")
    m0330 = parse_step_metrics(LOG_0330)
    m0329 = parse_step_metrics(LOG_0329)

    steps_0330 = sorted(m0330.keys())
    steps_0329 = sorted(m0329.keys())
    print(f"  0330: {len(steps_0330)} steps ({steps_0330[0]}-{steps_0330[-1]})")
    print(f"  0329: {len(steps_0329)} steps ({steps_0329[0]}-{steps_0329[-1]})")

    # Show available key categories
    sample_keys = sorted(m0330[steps_0330[0]].keys())
    sc_keys = [k for k in sample_keys if 'state_channel' in k]
    duet_keys = [k for k in sample_keys if 'duet' in k]
    dr3_keys = [k for k in sample_keys if 'dr3' in k]

    print(f"\n  SC keys ({len(sc_keys)}): {sc_keys[:10]}...")
    print(f"  DUET keys ({len(duet_keys)}): {duet_keys[:10]}...")
    print(f"  DR3 keys ({len(dr3_keys)}): {dr3_keys[:5]}...")

    # ================================================================
    # ANALYSIS 1: SC Bonus Evolution (MOST IMPORTANT)
    # ================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: SC Bonus Evolution — beta_decay Impact")
    print("=" * 70)

    sc_metrics_to_track = [
        'state_channel/beta_effective',
        'state_channel/bonus_total_mean',
        'state_channel/bonus_vs_reward_ratio',
        'state_channel/progress_mean',
        'state_channel/progress_onpolicy_mean',
        'state_channel/shaped_ratio',
        'state_channel/coverage_mean',
    ]

    # Sample at key steps
    sample_steps = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    print(f"\n{'Step':>5} | {'beta_eff(0330)':>14} {'beta_eff(0329)':>14} | {'bonus(0330)':>12} {'bonus(0329)':>12} | {'bonus_ratio(0330)':>18} {'bonus_ratio(0329)':>18}")
    print("-" * 110)

    sc_data = {'steps': [], '0330': {}, '0329': {}}
    for metric in sc_metrics_to_track:
        sc_data['0330'][metric] = []
        sc_data['0329'][metric] = []

    for s in sample_steps:
        if s in m0330 and s in m0329:
            sc_data['steps'].append(s)
            for metric in sc_metrics_to_track:
                sc_data['0330'][metric].append(m0330[s].get(metric, float('nan')))
                sc_data['0329'][metric].append(m0329[s].get(metric, float('nan')))

            beta_30 = m0330[s].get('state_channel/beta_effective', float('nan'))
            beta_29 = m0329[s].get('state_channel/beta_effective', float('nan'))
            bonus_30 = m0330[s].get('state_channel/bonus_total_mean', float('nan'))
            bonus_29 = m0329[s].get('state_channel/bonus_total_mean', float('nan'))
            ratio_30 = m0330[s].get('state_channel/bonus_vs_reward_ratio', float('nan'))
            ratio_29 = m0329[s].get('state_channel/bonus_vs_reward_ratio', float('nan'))
            print(f"{s:>5} | {beta_30:>14.4f} {beta_29:>14.4f} | {bonus_30:>12.4f} {bonus_29:>12.4f} | {ratio_30:>18.4f} {ratio_29:>18.4f}")

    print("\n  KEY FINDING — beta_effective:")
    betas_30 = [m0330[s].get('state_channel/beta_effective', 0) for s in steps_0330]
    betas_29 = [m0329[s].get('state_channel/beta_effective', 0) for s in steps_0329]
    print(f"    0330: min={min(betas_30):.4f}, max={max(betas_30):.4f}, mean={sum(betas_30)/len(betas_30):.4f}")
    print(f"    0329: min={min(betas_29):.4f}, max={max(betas_29):.4f}, mean={sum(betas_29)/len(betas_29):.4f}")
    print(f"    0330 early (1-10): {sum(betas_30[:10])/10:.4f}")
    print(f"    0330 mid (40-60): {sum([m0330[s].get('state_channel/beta_effective', 0) for s in range(40,61) if s in m0330])/max(1,len([s for s in range(40,61) if s in m0330])):.4f}")
    print(f"    0330 late (80-100): {sum([m0330[s].get('state_channel/beta_effective', 0) for s in range(80,101) if s in m0330])/max(1,len([s for s in range(80,101) if s in m0330])):.4f}")

    # ================================================================
    # ANALYSIS 2: Teacher Advantage Trajectories
    # ================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Teacher Advantage Trajectories")
    print("=" * 70)

    print(f"\n{'Step':>5} | {'teacher_adv(0330)':>18} {'teacher_adv(0329)':>18} | {'onpol_adv(0330)':>16} {'onpol_adv(0329)':>16} | {'teacher_pos%(0330)':>18} {'teacher_pos%(0329)':>18}")
    print("-" * 130)

    for s in sample_steps:
        if s in m0330 and s in m0329:
            ta30 = m0330[s].get('duet/adv_teacher_effective_mean', m0330[s].get('diag/adv_teacher_sample_mean', float('nan')))
            ta29 = m0329[s].get('duet/adv_teacher_effective_mean', m0329[s].get('diag/adv_teacher_sample_mean', float('nan')))
            oa30 = m0330[s].get('duet/adv_onpolicy_effective_mean', m0330[s].get('diag/adv_onpolicy_sample_mean', float('nan')))
            oa29 = m0329[s].get('duet/adv_onpolicy_effective_mean', m0329[s].get('diag/adv_onpolicy_sample_mean', float('nan')))
            tp30 = m0330[s].get('duet/teacher_adv_positive_ratio', m0330[s].get('diag/teacher_adv_pos_ratio', float('nan')))
            tp29 = m0329[s].get('duet/teacher_adv_positive_ratio', m0329[s].get('diag/teacher_adv_pos_ratio', float('nan')))
            print(f"{s:>5} | {ta30:>18.4f} {ta29:>18.4f} | {oa30:>16.4f} {oa29:>16.4f} | {tp30:>18.4f} {tp29:>18.4f}")

    # When does teacher advantage go negative?
    print("\n  Teacher advantage sign change:")
    for label, metrics in [("0330", m0330), ("0329", m0329)]:
        steps = sorted(metrics.keys())
        first_neg = None
        for s in steps:
            ta = metrics[s].get('duet/adv_teacher_effective_mean', metrics[s].get('diag/adv_teacher_sample_mean', 1))
            if ta < 0 and first_neg is None:
                first_neg = s
        if first_neg:
            print(f"    {label}: teacher advantage first goes negative at step {first_neg}")
        else:
            print(f"    {label}: teacher advantage stays positive throughout")

    # ================================================================
    # ANALYSIS 3: On-Policy Success Rate (Reward)
    # ================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 3: On-Policy Success Rate & Teacher Gap")
    print("=" * 70)

    print(f"\n{'Step':>5} | {'reward_onpol(0330)':>18} {'reward_onpol(0329)':>18} | {'reward_teach(0330)':>18} {'reward_teach(0329)':>18} | {'gap(0330)':>10} {'gap(0329)':>10}")
    print("-" * 120)

    for s in sample_steps:
        if s in m0330 and s in m0329:
            ro30 = m0330[s].get('diag/reward_onpolicy_mean', float('nan'))
            ro29 = m0329[s].get('diag/reward_onpolicy_mean', float('nan'))
            rt30 = m0330[s].get('diag/reward_teacher_mean', float('nan'))
            rt29 = m0329[s].get('diag/reward_teacher_mean', float('nan'))
            gap30 = rt30 - ro30 if not (ro30 != ro30 or rt30 != rt30) else float('nan')
            gap29 = rt29 - ro29 if not (ro29 != ro29 or rt29 != rt29) else float('nan')
            print(f"{s:>5} | {ro30:>18.4f} {ro29:>18.4f} | {rt30:>18.4f} {rt29:>18.4f} | {gap30:>10.4f} {gap29:>10.4f}")

    # ================================================================
    # ANALYSIS 4: Response Length Distribution
    # ================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Response Length / Token Metrics")
    print("=" * 70)

    # Check what length-related metrics exist
    len_keys = [k for k in sample_keys if 'seqlen' in k or 'response_len' in k or 'token_ratio' in k]
    print(f"  Available length keys: {len_keys}")

    print(f"\n{'Step':>5} | {'seqlen_mean(0330)':>18} {'seqlen_mean(0329)':>18} | {'llm_ratio(0330)':>16} {'llm_ratio(0329)':>16}")
    print("-" * 90)

    for s in sample_steps:
        if s in m0330 and s in m0329:
            sl30 = m0330[s].get('global_seqlen/mean', float('nan'))
            sl29 = m0329[s].get('global_seqlen/mean', float('nan'))
            lr30 = m0330[s].get('diag/llm_token_ratio_in_response', float('nan'))
            lr29 = m0329[s].get('diag/llm_token_ratio_in_response', float('nan'))
            print(f"{s:>5} | {sl30:>18.1f} {sl29:>18.1f} | {lr30:>16.4f} {lr29:>16.4f}")

    # ================================================================
    # ANALYSIS 5: DR3 & Teacher Gradient Share
    # ================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 5: DR3 & Teacher Gradient Share")
    print("=" * 70)

    # Check DR3 keys
    dr3_all = [k for k in sample_keys if 'dr3' in k.lower() or 'teacher_gradient' in k or 'disc' in k]
    duet_all = [k for k in sample_keys if 'duet' in k]
    print(f"  DR3 keys: {dr3_all}")
    print(f"  DUET keys: {duet_all}")

    teacher_grad_key = None
    for candidate in ['duet/teacher_gradient_share', 'duet/teacher_grad_share']:
        if candidate in sample_keys:
            teacher_grad_key = candidate
            break

    if teacher_grad_key:
        print(f"\n{'Step':>5} | {'teach_grad%(0330)':>16} {'teach_grad%(0329)':>16}")
        print("-" * 45)
        for s in sample_steps:
            if s in m0330 and s in m0329:
                tg30 = m0330[s].get(teacher_grad_key, float('nan'))
                tg29 = m0329[s].get(teacher_grad_key, float('nan'))
                print(f"{s:>5} | {tg30:>16.4f} {tg29:>16.4f}")
    else:
        print("  teacher_gradient_share key not found. Checking DUET keys at step 50:")
        for k in duet_all:
            v30 = m0330.get(50, m0330.get(steps_0330[-1], {})).get(k, 'N/A')
            v29 = m0329.get(50, m0329.get(steps_0329[-1], {})).get(k, 'N/A')
            print(f"    {k}: 0330={v30}, 0329={v29}")

    # ================================================================
    # ANALYSIS 6: KL & Entropy (Impact of KL coef reduction)
    # ================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 6: KL & Entropy (KL coef 0.005 vs 0.001)")
    print("=" * 70)

    kl_keys_all = [k for k in sample_keys if 'kl' in k.lower() or 'entropy' in k.lower()]
    print(f"  KL/entropy keys: {kl_keys_all}")

    kl_key = None
    for candidate in ['actor/kl_loss', 'actor/kl', 'kl_loss']:
        if candidate in sample_keys:
            kl_key = candidate
            break

    ent_key = 'actor/entropy_loss' if 'actor/entropy_loss' in sample_keys else None

    print(f"\n{'Step':>5} | {'kl(0330)':>10} {'kl(0329)':>10} | {'entropy(0330)':>14} {'entropy(0329)':>14} | {'onpol_ent(0330)':>16} {'onpol_ent(0329)':>16}")
    print("-" * 110)

    for s in sample_steps:
        if s in m0330 and s in m0329:
            kl30 = m0330[s].get(kl_key, float('nan')) if kl_key else float('nan')
            kl29 = m0329[s].get(kl_key, float('nan')) if kl_key else float('nan')
            ent30 = m0330[s].get(ent_key, float('nan')) if ent_key else float('nan')
            ent29 = m0329[s].get(ent_key, float('nan')) if ent_key else float('nan')
            oent30 = m0330[s].get('exp_replay/entropy_llm_onpolicy_mean', m0330[s].get('diag/entropy_onpolicy_token_mean', float('nan')))
            oent29 = m0329[s].get('exp_replay/entropy_llm_onpolicy_mean', m0329[s].get('diag/entropy_onpolicy_token_mean', float('nan')))
            print(f"{s:>5} | {kl30:>10.4f} {kl29:>10.4f} | {ent30:>14.4f} {ent29:>14.4f} | {oent30:>16.4f} {oent29:>16.4f}")

    # ================================================================
    # ANALYSIS 7: batch_diag trajectory data comparison
    # ================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 7: Batch Diag Trajectory Comparison")
    print("=" * 70)

    print(f"\n{'Step':>5} | {'adv_teach(0330)':>16} {'adv_teach(0329)':>16} | {'adv_onpol(0330)':>16} {'adv_onpol(0329)':>16} | {'reward_on(0330)':>16} {'reward_on(0329)':>16}")
    print("-" * 120)

    for s in sample_steps:
        f30 = f'{TRAJ_0330}/batch_diag_step_{s}.json'
        f29 = f'{TRAJ_0329}/batch_diag_step_{s}.json'
        if os.path.exists(f30) and os.path.exists(f29):
            d30 = json.load(open(f30))
            d29 = json.load(open(f29))
            at30 = d30.get('diag/adv_teacher_sample_mean', float('nan'))
            at29 = d29.get('diag/adv_teacher_sample_mean', float('nan'))
            ao30 = d30.get('diag/adv_onpolicy_sample_mean', float('nan'))
            ao29 = d29.get('diag/adv_onpolicy_sample_mean', float('nan'))
            ro30 = d30.get('diag/reward_onpolicy_mean', float('nan'))
            ro29 = d29.get('diag/reward_onpolicy_mean', float('nan'))
            print(f"{s:>5} | {at30:>16.4f} {at29:>16.4f} | {ao30:>16.4f} {ao29:>16.4f} | {ro30:>16.4f} {ro29:>16.4f}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Key Behavioral Differences")
    print("=" * 70)

    # Compute summary stats
    final_reward_30 = m0330[steps_0330[-1]].get('diag/reward_onpolicy_mean', 0)
    final_reward_29 = m0329[steps_0329[-1]].get('diag/reward_onpolicy_mean', 0)

    beta_early_30 = sum(m0330[s].get('state_channel/beta_effective', 0) for s in steps_0330[:10]) / 10
    beta_late_30 = sum(m0330[s].get('state_channel/beta_effective', 0) for s in steps_0330[-10:]) / 10
    beta_early_29 = sum(m0329[s].get('state_channel/beta_effective', 0) for s in steps_0329[:10]) / 10
    beta_late_29 = sum(m0329[s].get('state_channel/beta_effective', 0) for s in steps_0329[-10:]) / 10

    bonus_early_30 = sum(m0330[s].get('state_channel/bonus_total_mean', 0) for s in steps_0330[:10]) / 10
    bonus_late_30 = sum(m0330[s].get('state_channel/bonus_total_mean', 0) for s in steps_0330[-10:]) / 10
    bonus_early_29 = sum(m0329[s].get('state_channel/bonus_total_mean', 0) for s in steps_0329[:10]) / 10
    bonus_late_29 = sum(m0329[s].get('state_channel/bonus_total_mean', 0) for s in steps_0329[-10:]) / 10

    print(f"\n  1. Beta Decay Impact:")
    print(f"     0330 beta: {beta_early_30:.4f} (early) → {beta_late_30:.4f} (late)  [delta: {beta_late_30 - beta_early_30:+.4f}]")
    print(f"     0329 beta: {beta_early_29:.4f} (early) → {beta_late_29:.4f} (late)  [delta: {beta_late_29 - beta_early_29:+.4f}]")
    print(f"     0330 bonus: {bonus_early_30:.4f} (early) → {bonus_late_30:.4f} (late)  [delta: {bonus_late_30 - bonus_early_30:+.4f}]")
    print(f"     0329 bonus: {bonus_early_29:.4f} (early) → {bonus_late_29:.4f} (late)  [delta: {bonus_late_29 - bonus_early_29:+.4f}]")

    print(f"\n  2. Final On-Policy Reward:")
    print(f"     0330: {final_reward_30:.4f}")
    print(f"     0329: {final_reward_29:.4f}")

    # Teacher adv going negative
    for label, metrics in [("0330", m0330), ("0329", m0329)]:
        steps = sorted(metrics.keys())
        neg_steps = [s for s in steps if metrics[s].get('duet/adv_teacher_effective_mean', metrics[s].get('diag/adv_teacher_sample_mean', 1)) < 0]
        if neg_steps:
            print(f"\n  3. Teacher adv negative ({label}): first at step {neg_steps[0]}, {len(neg_steps)}/{len(steps)} steps total")
        else:
            print(f"\n  3. Teacher adv negative ({label}): never goes negative")

    return m0330, m0329


if __name__ == '__main__':
    m0330, m0329 = main()
