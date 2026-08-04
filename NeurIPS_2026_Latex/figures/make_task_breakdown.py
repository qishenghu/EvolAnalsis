#!/usr/bin/env python3
"""
Generate per-task-type breakdown bar chart for ALFWorld experiments.
Two panels: 1.5B (left) and 3B (right), showing success rate by task type for each method.

Task types are classified from the strongest model's (DUET) think blocks and applied
consistently to all methods in the same validation set.

Output: fig_per_task_breakdown.pdf
"""

import json
import re
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

BASE = '/data/home/qisheng/EvolAnalsis/experiments/alfworld'
OUT_DIR = '/data/home/qisheng/EvolAnalsis/NeurIPS_2026_Latex/figures'

# =============================================================================
# Task classification
# =============================================================================

def get_all_thinks(output):
    return re.findall(r'<think>(.*?)</think>', output, re.DOTALL)

def get_all_agent_actions(output):
    return re.findall(r'<action>\s*(.*?)\s*</action>', output, re.DOTALL)

def classify_task(output):
    """
    Classify ALFWorld task type from trajectory output.
    Uses agent's think blocks and action patterns.
    """
    thinks = get_all_thinks(output)
    first_think = thinks[0].lower().strip() if thinks else ''
    all_thinks = ' '.join(t.lower() for t in thinks)
    actions = get_all_agent_actions(output)
    all_actions = ' '.join(a.lower() for a in actions)

    # pick_two: explicitly mentions TWO items
    if re.search(r'\btwo\b', first_think) and (
        'find' in first_think or 'need' in first_think or 'put' in first_think or 'get' in first_think
    ):
        return 'pick_two'
    if 'the second' in all_thinks[:800]:
        if not re.search(r'second\s+(place|location|spot|cabinet|drawer|shelf)', all_thinks[:800]):
            return 'pick_two'

    # heat
    if re.search(r'\bheat\b', first_think):
        return 'heat'
    if re.search(r'\bheat\b', all_thinks[:500]):
        return 'heat'

    # cool
    if re.search(r'\bcool\b', first_think):
        return 'cool'
    if re.search(r'\bcool\b', all_thinks[:500]):
        return 'cool'

    # clean
    if re.search(r'\bclean\b', first_think):
        return 'clean'
    if re.search(r'\bclean\b', all_thinks[:500]):
        return 'clean'

    # examine
    if re.search(r'\b(examine|look at|desklamp|desk lamp)\b', first_think):
        return 'examine'
    if 'use desklamp' in all_actions:
        return 'examine'
    if re.search(r'\b(examine|look at|desklamp|desk lamp|lamp)\b', all_thinks[:800]):
        return 'examine'

    # pick_and_place (default)
    return 'pick_and_place'


def load_and_classify(filepath):
    """Load a validation log and return list of (task_type, reward) tuples."""
    with open(filepath) as f:
        lines = f.readlines()
    results = []
    for line in lines:
        d = json.loads(line)
        tt = classify_task(d['output'])
        results.append((tt, d['reward']))
    return results


def compute_per_type_rates(task_types, rewards_by_method, method_names):
    """
    Given task_types (list of types for each task index) and
    rewards_by_method (dict: method -> list of rewards),
    compute per-type success rates.

    Returns: dict of method -> dict of task_type -> rate
    """
    result = {}
    for method in method_names:
        rewards = rewards_by_method[method]
        type_rates = {}
        for tt in set(task_types):
            indices = [i for i, t in enumerate(task_types) if t == tt]
            if indices:
                type_rates[tt] = sum(rewards[i] for i in indices) / len(indices) * 100
            else:
                type_rates[tt] = 0.0
        result[method] = type_rates
    return result


# =============================================================================
# 1.5B analysis
# =============================================================================

# All 1.5B methods share the same validation set (verified: 0 input mismatches)
methods_1_5b = {
    'DUET':  f'{BASE}/alfworld_qwen1.5b_duet_v39c_postfix/validation_log/100.jsonl',
    'CHORD': f'{BASE}/alfworld_qwen1.5b_chord/validation_log/100.jsonl',
    'LUFFY': f'{BASE}/alfworld_qwen1.5b_luffy/validation_log/100.jsonl',
    'GRPO':  f'{BASE}/alfworld_qwen1.5b_onpolicy/validation_log/100.jsonl',
}

# Classify tasks using DUET (strongest model)
duet_results_1_5b = load_and_classify(methods_1_5b['DUET'])
task_types_1_5b = [r[0] for r in duet_results_1_5b]

# Load rewards for all methods
rewards_1_5b = {}
for method, path in methods_1_5b.items():
    with open(path) as f:
        lines = f.readlines()
    rewards_1_5b[method] = [json.loads(l)['reward'] for l in lines]

rates_1_5b = compute_per_type_rates(task_types_1_5b, rewards_1_5b, list(methods_1_5b.keys()))

# =============================================================================
# 3B analysis
# =============================================================================

# DUET 3B, LUFFY 3B, GRPO 3B share one validation set (Set A)
# CHORD 3B uses a different validation set (Set B, same as 1.5B) and only has step 50
# For the figure, we classify each set independently.

methods_3b_setA = {
    'DUET':  f'{BASE}/alfworld_3b_duet_0329/validation_log/100.jsonl',
    'LUFFY': f'{BASE}/alfworld_3b_luffy/validation_log/100.jsonl',
    'GRPO':  f'{BASE}/alfworld_3b_grpo_react_tags/validation_log/100.jsonl',
}

# Classify Set A tasks using DUET 3B
duet_results_3b = load_and_classify(methods_3b_setA['DUET'])
task_types_3b_setA = [r[0] for r in duet_results_3b]

rewards_3b_setA = {}
for method, path in methods_3b_setA.items():
    with open(path) as f:
        lines = f.readlines()
    rewards_3b_setA[method] = [json.loads(l)['reward'] for l in lines]

# CHORD 3B: classify from its own validation set
chord_3b_path = f'{BASE}/alfworld_qwen3b_chord/validation_log/50.jsonl'
chord_3b_results = load_and_classify(chord_3b_path)
task_types_3b_chord = [r[0] for r in chord_3b_results]
rewards_3b_chord = [r[1] for r in chord_3b_results]

# Compute rates for Set A methods
rates_3b = compute_per_type_rates(task_types_3b_setA, rewards_3b_setA,
                                   list(methods_3b_setA.keys()))

# Compute rates for CHORD 3B separately
chord_3b_type_rates = {}
for tt in set(task_types_3b_chord):
    indices = [i for i, t in enumerate(task_types_3b_chord) if t == tt]
    if indices:
        chord_3b_type_rates[tt] = sum(rewards_3b_chord[i] for i in indices) / len(indices) * 100
    else:
        chord_3b_type_rates[tt] = 0.0
rates_3b['CHORD*'] = chord_3b_type_rates  # asterisk = different val set / step

# =============================================================================
# Print results
# =============================================================================

TASK_ORDER = ['pick_and_place', 'pick_two', 'examine', 'heat', 'cool', 'clean']
TASK_LABELS = ['Pick&Place', 'Pick Two', 'Examine', 'Heat', 'Cool', 'Clean']

print("=== 1.5B Per-Task Success Rates (%) ===")
print(f"{'Task Type':<16} {'N':>4}", end='')
for m in ['DUET', 'CHORD', 'LUFFY', 'GRPO']:
    print(f" {m:>8}", end='')
print()
for tt, label in zip(TASK_ORDER, TASK_LABELS):
    n = task_types_1_5b.count(tt)
    print(f"{label:<16} {n:>4}", end='')
    for m in ['DUET', 'CHORD', 'LUFFY', 'GRPO']:
        print(f" {rates_1_5b[m].get(tt, 0):>7.1f}%", end='')
    print()

# Overall
print(f"{'OVERALL':<16} {len(task_types_1_5b):>4}", end='')
for m in ['DUET', 'CHORD', 'LUFFY', 'GRPO']:
    total_r = sum(rewards_1_5b[m])
    print(f" {total_r/len(task_types_1_5b)*100:>7.1f}%", end='')
print()

print()
print("=== 3B Per-Task Success Rates (%) ===")
print("Note: CHORD* uses different validation set at step 50 (others at step 100)")
print(f"{'Task Type':<16} {'N_A':>4} {'N_C':>4}", end='')
for m in ['DUET', 'CHORD*', 'LUFFY', 'GRPO']:
    print(f" {m:>8}", end='')
print()
for tt, label in zip(TASK_ORDER, TASK_LABELS):
    n_a = task_types_3b_setA.count(tt)
    n_c = task_types_3b_chord.count(tt)
    print(f"{label:<16} {n_a:>4} {n_c:>4}", end='')
    for m in ['DUET', 'CHORD*', 'LUFFY', 'GRPO']:
        print(f" {rates_3b[m].get(tt, 0):>7.1f}%", end='')
    print()

print(f"{'OVERALL':<16} {len(task_types_3b_setA):>4} {len(task_types_3b_chord):>4}", end='')
for m in ['DUET', 'CHORD*', 'LUFFY', 'GRPO']:
    if m == 'CHORD*':
        total_r = sum(rewards_3b_chord)
        print(f" {total_r/len(task_types_3b_chord)*100:>7.1f}%", end='')
    else:
        total_r = sum(rewards_3b_setA[m])
        print(f" {total_r/len(task_types_3b_setA)*100:>7.1f}%", end='')
print()

# =============================================================================
# Figure
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

METHODS_PLOT = ['DUET', 'CHORD', 'LUFFY', 'GRPO']
COLORS = {
    'DUET':   '#2166ac',  # dark blue
    'CHORD':  '#d6604d',  # coral red
    'CHORD*': '#d6604d',  # same color
    'LUFFY':  '#4daf4a',  # green
    'GRPO':   '#984ea3',  # purple
}

bar_width = 0.18
x = np.arange(len(TASK_ORDER))

# --- Panel A: 1.5B ---
ax = axes[0]
for j, method in enumerate(METHODS_PLOT):
    vals = [rates_1_5b[method].get(tt, 0) for tt in TASK_ORDER]
    offset = (j - 1.5) * bar_width
    bars = ax.bar(x + offset, vals, bar_width, label=method,
                  color=COLORS[method], edgecolor='white', linewidth=0.5)
    # Add value labels on bars
    for bar, v in zip(bars, vals):
        if v > 3:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{v:.0f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')

ax.set_xlabel('Task Type', fontsize=12)
ax.set_ylabel('Success Rate (%)', fontsize=12)
ax.set_title('ALFWorld 1.5B (val@100)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(TASK_LABELS, rotation=25, ha='right', fontsize=9.5)
ax.set_ylim(0, 105)
ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add task count annotations
for i, tt in enumerate(TASK_ORDER):
    n = task_types_1_5b.count(tt)
    ax.text(i, -7, f'n={n}', ha='center', va='top', fontsize=7.5, color='gray')

# --- Panel B: 3B ---
ax = axes[1]
METHODS_3B_PLOT = ['DUET', 'CHORD*', 'LUFFY', 'GRPO']
for j, method in enumerate(METHODS_3B_PLOT):
    vals = [rates_3b[method].get(tt, 0) for tt in TASK_ORDER]
    offset = (j - 1.5) * bar_width
    label = method
    bars = ax.bar(x + offset, vals, bar_width, label=label,
                  color=COLORS[method], edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals):
        if v > 3:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{v:.0f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')

ax.set_xlabel('Task Type', fontsize=12)
ax.set_ylabel('Success Rate (%)', fontsize=12)
ax.set_title('ALFWorld 3B (val@100, CHORD* @50)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(TASK_LABELS, rotation=25, ha='right', fontsize=9.5)
ax.set_ylim(0, 105)
ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add task count annotations (from Set A)
for i, tt in enumerate(TASK_ORDER):
    n = task_types_3b_setA.count(tt)
    ax.text(i, -7, f'n={n}', ha='center', va='top', fontsize=7.5, color='gray')

plt.tight_layout()
outpath = os.path.join(OUT_DIR, 'fig_per_task_breakdown.pdf')
plt.savefig(outpath, bbox_inches='tight', dpi=150)
print(f"\nSaved figure to: {outpath}")

# Also save as PNG for quick preview
plt.savefig(outpath.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
print(f"Saved PNG preview to: {outpath.replace('.pdf', '.png')}")
