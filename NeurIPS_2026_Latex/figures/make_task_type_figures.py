#!/usr/bin/env python3
"""
Generate per-task-type breakdown grouped bar chart for ALFWorld (DUET NeurIPS 2026 Appendix).

Two subplots: 1.5B (left) and 3B (right).
6 task types: pick_and_place, examine_in_light, heat, cool, clean, pick_two.
4 methods:    DUET (blue), LUFFY (orange), CHORD (green), GRPO (gray).

Task classification uses the 3B reference models (GRPO/LUFFY/DUET all agree 100%)
and is applied positionally to 1.5B runs (same 200 validation tasks, same order).

NOTE: 3B CHORD only has step-50 validation data (all others at step 100).

Output:
  fig_task_type_breakdown.pdf
  fig_task_type_breakdown.png
"""

import json
import re
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

BASE = '/data/home/qisheng/EvolAnalsis/experiments/alfworld'
OUT_DIR = '/data/home/qisheng/EvolAnalsis/NeurIPS_2026_Latex/figures'

RUNS_1_5B = OrderedDict([
    ('DUET',  'alfworld_qwen1.5b_duet_v39c_postfix'),
    ('LUFFY', 'alfworld_qwen1.5b_luffy'),
    ('CHORD', 'alfworld_qwen1.5b_chord'),
    ('GRPO',  'alfworld_qwen1.5b_onpolicy'),
])

RUNS_3B = OrderedDict([
    ('DUET',  'alfworld_3b_duet_0329'),
    ('LUFFY', 'alfworld_3b_luffy'),
    ('CHORD', 'alfworld_qwen3b_chord'),          # step 50 only
    ('GRPO',  'alfworld_3b_grpo_react_tags'),
])

STEP_OVERRIDE_3B = {'CHORD': '50'}   # others default to '100'

TASK_ORDER = [
    'pick_and_place',
    'examine_in_light',
    'heat',
    'cool',
    'clean',
    'pick_two',
]

TASK_LABELS = ['Pick & Place', 'Examine', 'Heat', 'Cool', 'Clean', 'Pick Two']

METHOD_ORDER = ['DUET', 'LUFFY', 'CHORD', 'GRPO']

COLORS = {
    'DUET':  '#1f77b4',   # blue
    'LUFFY': '#ff7f0e',   # orange
    'CHORD': '#2ca02c',   # green
    'GRPO':  '#999999',   # gray
}


# ─────────────────────────────────────────────────────────────────────────────
# Task classification (from first think/thought block)
# ─────────────────────────────────────────────────────────────────────────────

def classify_task_from_think(output):
    """Classify ALFWorld task type from the first think/thought block."""
    m = re.search(r'<(?:think|thought)>(.*?)</(?:think|thought)>', output, re.DOTALL)
    if not m:
        return 'pick_and_place'  # fallback
    desc = m.group(1).strip().lower()

    if 'two' in desc:
        return 'pick_two'
    if re.search(r'\bheat\b', desc) or re.search(r'\bhot\b', desc):
        return 'heat'
    if re.search(r'\bcool\b', desc) or re.search(r'\bcold\b', desc):
        return 'cool'
    if re.search(r'\bclean\b', desc):
        return 'clean'
    if re.search(r'\bexamine\b', desc):
        return 'examine_in_light'
    if re.search(r'look at', desc) and re.search(r'lamp|light', desc):
        return 'examine_in_light'
    if re.search(r'\blamp\b|\bdesklamp\b|\bfloorlamp\b', desc):
        return 'examine_in_light'
    return 'pick_and_place'


def build_reference_task_types(filepath):
    """Build task-type list from a reliable reference model's validation log."""
    with open(filepath) as f:
        entries = [json.loads(line) for line in f]
    return [classify_task_from_think(e['output']) for e in entries]


# ─────────────────────────────────────────────────────────────────────────────
# Load scores
# ─────────────────────────────────────────────────────────────────────────────

def load_scores(filepath):
    """Return list of binary success indicators (score >= 1.0)."""
    with open(filepath) as f:
        entries = [json.loads(line) for line in f]
    return [1.0 if e['score'] >= 1.0 else 0.0 for e in entries]


def per_type_success_rate(task_types, scores, task_order):
    """Compute success rate (%) per task type."""
    rates = {}
    counts = {}
    for tt in task_order:
        indices = [i for i, t in enumerate(task_types) if t == tt]
        n = len(indices)
        counts[tt] = n
        if n > 0:
            rates[tt] = sum(scores[i] for i in indices) / n * 100
        else:
            rates[tt] = 0.0
    return rates, counts


# ─────────────────────────────────────────────────────────────────────────────
# Build reference task types from 3B GRPO (all three 3B methods agree 100%)
# ─────────────────────────────────────────────────────────────────────────────

ref_path = os.path.join(BASE, 'alfworld_3b_grpo_react_tags', 'validation_log', '100.jsonl')
TASK_TYPES_REF = build_reference_task_types(ref_path)
print(f"Reference task distribution: {Counter(TASK_TYPES_REF)}")
print(f"Total validation tasks: {len(TASK_TYPES_REF)}")

# ─────────────────────────────────────────────────────────────────────────────
# Compute per-method, per-task-type success rates
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_rates(runs, step_overrides, task_types):
    """Returns {method: {task_type: rate}}, {task_type: count}."""
    all_rates = {}
    all_counts = None
    notes = {}
    for method, run_name in runs.items():
        step = step_overrides.get(method, '100')
        fpath = os.path.join(BASE, run_name, 'validation_log', f'{step}.jsonl')
        if not os.path.exists(fpath):
            print(f"WARNING: {fpath} not found, skipping {method}")
            continue
        scores = load_scores(fpath)
        assert len(scores) == len(task_types), \
            f"{method}: got {len(scores)} entries, expected {len(task_types)}"
        rates, counts = per_type_success_rate(task_types, scores, TASK_ORDER)
        all_rates[method] = rates
        if all_counts is None:
            all_counts = counts
        if step != '100':
            notes[method] = f'step {step}'
    return all_rates, all_counts, notes


rates_1_5b, counts_1_5b, notes_1_5b = compute_all_rates(
    RUNS_1_5B, {}, TASK_TYPES_REF
)
rates_3b, counts_3b, notes_3b = compute_all_rates(
    RUNS_3B, STEP_OVERRIDE_3B, TASK_TYPES_REF
)

# ─────────────────────────────────────────────────────────────────────────────
# Print markdown tables
# ─────────────────────────────────────────────────────────────────────────────

def print_table(title, rates, counts, notes):
    print(f"\n{title}")
    header = f"| {'Task Type':<18} | {'N':>3} |"
    for m in METHOD_ORDER:
        suffix = f" ({notes[m]})" if m in notes else ""
        header += f" {m + suffix:>14} |"
    print(header)
    print("|" + "-" * 20 + "|" + "----:|" + "---------------:|" * len(METHOD_ORDER))

    for tt, label in zip(TASK_ORDER, TASK_LABELS):
        row = f"| {label:<18} | {counts.get(tt, 0):>3} |"
        for m in METHOD_ORDER:
            if m in rates:
                row += f" {rates[m].get(tt, 0):>13.1f}% |"
            else:
                row += f" {'N/A':>13} |"
        print(row)

    # Overall
    row = f"| {'**Overall**':<18} | {sum(counts.values()):>3} |"
    for m in METHOD_ORDER:
        if m in rates:
            total_rate = sum(rates[m].get(tt, 0) * counts.get(tt, 0)
                            for tt in TASK_ORDER) / sum(counts.values())
            row += f" {total_rate:>13.1f}% |"
        else:
            row += f" {'N/A':>13} |"
    print(row)


print_table("### ALFWorld 1.5B — Per-Task Success Rate (val@step 100)",
            rates_1_5b, counts_1_5b, notes_1_5b)
print_table("### ALFWorld 3B — Per-Task Success Rate (val@step 100 except noted)",
            rates_3b, counts_3b, notes_3b)


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bar_width = 0.18
x = np.arange(len(TASK_ORDER))
n_methods = len(METHOD_ORDER)


def plot_panel(ax, rates, counts, title, notes):
    for j, method in enumerate(METHOD_ORDER):
        if method not in rates:
            continue
        vals = [rates[method].get(tt, 0) for tt in TASK_ORDER]
        offset = (j - (n_methods - 1) / 2) * bar_width
        label = method
        if method in notes:
            label += f" ({notes[method]})"
        bars = ax.bar(
            x + offset, vals, bar_width,
            label=label,
            color=COLORS[method],
            edgecolor='white',
            linewidth=0.5,
            zorder=3,
        )
        # Value labels on bars
        for bar, v in zip(bars, vals):
            if v > 3:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.2,
                    f'{v:.0f}',
                    ha='center', va='bottom',
                    fontsize=6, fontweight='bold',
                    color=COLORS[method],
                )

    ax.set_ylabel('Success Rate (%)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(x)

    # Build x-tick labels with count annotation
    xlabels = []
    for tt, label in zip(TASK_ORDER, TASK_LABELS):
        n = counts.get(tt, 0)
        xlabels.append(f'{label}\n(n={n})')
    ax.set_xticklabels(xlabels, fontsize=9)

    ax.set_ylim(0, 109)
    ax.legend(fontsize=8.5, loc='upper right', framealpha=0.9, ncol=1)
    ax.grid(axis='y', alpha=0.25, linestyle='--', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


plot_panel(axes[0], rates_1_5b, counts_1_5b,
           'Qwen2.5-1.5B (step 100)', notes_1_5b)
plot_panel(axes[1], rates_3b, counts_3b,
           'Qwen2.5-3B (step 100)', notes_3b)

plt.tight_layout(w_pad=2.0)

# Save
for ext in ['pdf', 'png']:
    outpath = os.path.join(OUT_DIR, f'fig_task_type_breakdown.{ext}')
    plt.savefig(outpath, bbox_inches='tight', dpi=200)
    print(f"Saved: {outpath}")

print("\nDone.")
