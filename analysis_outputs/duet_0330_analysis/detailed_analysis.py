#!/usr/bin/env python3
"""Detailed DR3, response length, SC progress, teacher gradient share analysis."""
import json
import os
import sys

BASE = '/data/code/exp/EvolAnalsis'

def parse_step_metrics(log_path):
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

m0330 = parse_step_metrics(f'{BASE}/wandb/run-20260330_122831-437s027l/files/output.log')
m0329 = parse_step_metrics(f'{BASE}/wandb/run-20260329_232718-9ryexv2i/files/output.log')

steps = list(range(1, 101))
sample_steps = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

def moving_avg(vals, window=10):
    result = []
    for i in range(len(vals)):
        start = max(0, i - window + 1)
        result.append(sum(vals[start:i+1]) / (i + 1 - start))
    return result

# DR3
print("=" * 70)
print("DR3 Discriminator & Weight Analysis")
print("=" * 70)

print(f"\n{'Step':>5} | {'disc_acc(0330)':>14} {'disc_acc(0329)':>14} | {'w_mean(0330)':>12} {'w_mean(0329)':>12} | {'w_max(0330)':>12} {'w_max(0329)':>12}")
print("-" * 100)

for s in sample_steps:
    da30 = m0330[s].get('dr3/disc_acc', float('nan'))
    da29 = m0329[s].get('dr3/disc_acc', float('nan'))
    wm30 = m0330[s].get('dr3/w_mean', float('nan'))
    wm29 = m0329[s].get('dr3/w_mean', float('nan'))
    wx30 = m0330[s].get('dr3/w_max', float('nan'))
    wx29 = m0329[s].get('dr3/w_max', float('nan'))
    print(f"{s:>5} | {da30:>14.4f} {da29:>14.4f} | {wm30:>12.4f} {wm29:>12.4f} | {wx30:>12.4f} {wx29:>12.4f}")

# Response length details
print(f"\n\n{'='*70}")
print("Response Length Details")
print(f"{'='*70}")

print(f"\n{'Step':>5} | {'resp_on(0330)':>14} {'resp_on(0329)':>14} | {'resp_teach(0330)':>16} {'resp_teach(0329)':>16} | {'ratio_t/o(30)':>14} {'ratio_t/o(29)':>14}")
print("-" * 100)

for s in sample_steps:
    rom30 = m0330[s].get('diag/response_len_onpolicy_mean', float('nan'))
    rom29 = m0329[s].get('diag/response_len_onpolicy_mean', float('nan'))
    rtm30 = m0330[s].get('diag/response_len_teacher_mean', float('nan'))
    rtm29 = m0329[s].get('diag/response_len_teacher_mean', float('nan'))
    rat30 = m0330[s].get('diag/response_len_ratio_teacher_vs_on', float('nan'))
    rat29 = m0329[s].get('diag/response_len_ratio_teacher_vs_on', float('nan'))
    print(f"{s:>5} | {rom30:>14.1f} {rom29:>14.1f} | {rtm30:>16.1f} {rtm29:>16.1f} | {rat30:>14.4f} {rat29:>14.4f}")

on_lens_30 = [m0330[s].get('diag/response_len_onpolicy_mean', 0) for s in steps if s in m0330]
on_lens_29 = [m0329[s].get('diag/response_len_onpolicy_mean', 0) for s in steps if s in m0329]
print(f"\n  0330 avg on-policy response length: {sum(on_lens_30)/len(on_lens_30):.1f}")
print(f"  0329 avg on-policy response length: {sum(on_lens_29)/len(on_lens_29):.1f}")
print(f"  0330 last 20 avg: {sum(on_lens_30[-20:])/20:.1f}")
print(f"  0329 last 20 avg: {sum(on_lens_29[-20:])/20:.1f}")

# SC Progress
print(f"\n\n{'='*70}")
print("SC Progress Onpolicy & Coverage Evolution")
print(f"{'='*70}")

print(f"\n{'Step':>5} | {'prog_on(30)':>12} {'prog_on(29)':>12} | {'prog_teach(30)':>14} {'prog_teach(29)':>14} | {'cover(30)':>10} {'cover(29)':>10}")
print("-" * 100)

for s in sample_steps:
    po30 = m0330[s].get('state_channel/progress_onpolicy_mean', float('nan'))
    po29 = m0329[s].get('state_channel/progress_onpolicy_mean', float('nan'))
    pt30 = m0330[s].get('state_channel/progress_teacher_mean', float('nan'))
    pt29 = m0329[s].get('state_channel/progress_teacher_mean', float('nan'))
    c30 = m0330[s].get('state_channel/coverage_mean', float('nan'))
    c29 = m0329[s].get('state_channel/coverage_mean', float('nan'))
    print(f"{s:>5} | {po30:>12.4f} {po29:>12.4f} | {pt30:>14.4f} {pt29:>14.4f} | {c30:>10.4f} {c29:>10.4f}")

# Step-level delta analysis
print(f"\n\n{'='*70}")
print("Step-Level Delta Analysis")
print(f"{'='*70}")

print(f"\n{'Step':>5} | {'delta_mean(30)':>14} {'delta_mean(29)':>14} | {'delta_abs(30)':>14} {'delta_abs(29)':>14} | {'pos_ratio(30)':>14} {'pos_ratio(29)':>14}")
print("-" * 100)

for s in sample_steps:
    dm30 = m0330[s].get('state_channel/step_delta_mean', float('nan'))
    dm29 = m0329[s].get('state_channel/step_delta_mean', float('nan'))
    da30 = m0330[s].get('state_channel/step_delta_abs_mean', float('nan'))
    da29 = m0329[s].get('state_channel/step_delta_abs_mean', float('nan'))
    pr30 = m0330[s].get('state_channel/step_delta_positive_ratio', float('nan'))
    pr29 = m0329[s].get('state_channel/step_delta_positive_ratio', float('nan'))
    print(f"{s:>5} | {dm30:>14.4f} {dm29:>14.4f} | {da30:>14.4f} {da29:>14.4f} | {pr30:>14.4f} {pr29:>14.4f}")

# Teacher gradient share
print(f"\n\n{'='*70}")
print("Teacher Gradient Share — Full Trajectory")
print(f"{'='*70}")

tgs_30 = [m0330[s].get('duet/teacher_gradient_share', 0) for s in steps if s in m0330]
tgs_29 = [m0329[s].get('duet/teacher_gradient_share', 0) for s in steps if s in m0329]
tgs_30_ma = moving_avg(tgs_30)
tgs_29_ma = moving_avg(tgs_29)

for s in sample_steps:
    idx = s - 1
    print(f"  Step {s:>3}: 0330={tgs_30[idx]:.4f} (MA10={tgs_30_ma[idx]:.4f})  0329={tgs_29[idx]:.4f} (MA10={tgs_29_ma[idx]:.4f})")

print(f"\n  0330 overall mean teacher_gradient_share: {sum(tgs_30)/len(tgs_30):.4f}")
print(f"  0329 overall mean teacher_gradient_share: {sum(tgs_29)/len(tgs_29):.4f}")
print(f"  0330 first 20: {sum(tgs_30[:20])/20:.4f}, last 20: {sum(tgs_30[-20:])/20:.4f}")
print(f"  0329 first 20: {sum(tgs_29[:20])/20:.4f}, last 20: {sum(tgs_29[-20:])/20:.4f}")

# Full reward with moving average
print(f"\n\n{'='*70}")
print("Reward Trajectory — 10-step Moving Average")
print(f"{'='*70}")

rew_30 = [m0330[s].get('diag/reward_onpolicy_mean', 0) for s in steps if s in m0330]
rew_29 = [m0329[s].get('diag/reward_onpolicy_mean', 0) for s in steps if s in m0329]
rew_30_ma = moving_avg(rew_30)
rew_29_ma = moving_avg(rew_29)

for s in range(1, 101, 5):
    idx = s - 1
    print(f"  Step {s:>3}: 0330={rew_30[idx]:.4f} (MA={rew_30_ma[idx]:.4f})  0329={rew_29[idx]:.4f} (MA={rew_29_ma[idx]:.4f})")

print(f"\n  0330 mean reward (all 100): {sum(rew_30)/len(rew_30):.4f}")
print(f"  0329 mean reward (all 100): {sum(rew_29)/len(rew_29):.4f}")
print(f"  0330 mean reward (last 20): {sum(rew_30[-20:])/20:.4f}")
print(f"  0329 mean reward (last 20): {sum(rew_29[-20:])/20:.4f}")
print(f"  0330 max reward: {max(rew_30):.4f} at step {rew_30.index(max(rew_30))+1}")
print(f"  0329 max reward: {max(rew_29):.4f} at step {rew_29.index(max(rew_29))+1}")

# Clip/PPO metrics
print(f"\n\n{'='*70}")
print("PPO Clip Metrics")
print(f"{'='*70}")

print(f"\n{'Step':>5} | {'on_clip(0330)':>14} {'on_clip(0329)':>14} | {'off_clip(0330)':>14} {'off_clip(0329)':>14}")
print("-" * 70)

for s in sample_steps:
    oc30 = m0330[s].get('actor/on_pg_clipfrac', float('nan'))
    oc29 = m0329[s].get('actor/on_pg_clipfrac', float('nan'))
    ofc30 = m0330[s].get('actor/off_pg_cliphit_rate', float('nan'))
    ofc29 = m0329[s].get('actor/off_pg_cliphit_rate', float('nan'))
    print(f"{s:>5} | {oc30:>14.4f} {oc29:>14.4f} | {ofc30:>14.4f} {ofc29:>14.4f}")
