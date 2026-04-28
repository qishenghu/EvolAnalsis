"""
WS Sweep Result Analyzer.

Scans experiments/webshop/ws_swA_* / ws_sweepB_* / ws_sweepC_*  validation_log/100.jsonl
files. Computes (reward, success_rate) per run, groups by config (strip seed
suffix), reports mean±std.

Usage:
    python scripts/analyze_ws_sweep.py [--phase A|B|C|all]
"""
import argparse
import glob
import json
import os
import re
import statistics
from collections import defaultdict


def compute_metrics(path: str):
    n = 0
    sr = 0
    rw = 0.0
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            n += 1
            s = d.get("score", d.get("reward", 0))
            rw += s
            sr += 1 if s >= 1.0 else 0
    return n, rw / n, sr / n * 100


def strip_seed(name: str) -> str:
    """ws_swA_v39b_seed42 → ws_swA_v39b"""
    return re.sub(r"_seed\d+$", "", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["A", "B", "C", "all"], default="all")
    ap.add_argument("--root", default="experiments/webshop")
    args = ap.parse_args()

    if args.phase == "all":
        pattern = f"{args.root}/ws_sw*/validation_log/100.jsonl"
    else:
        pattern = f"{args.root}/ws_sweep{args.phase}_*/validation_log/100.jsonl"

    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found matching {pattern}")
        return

    # Per-run results
    print("═" * 90)
    print(f"Per-run results ({len(files)} runs):")
    print("─" * 90)
    print(f"  {'Run':<48s} {'N':<5s} {'reward':<10s} {'success':<10s}")
    print("─" * 90)

    runs = {}
    for f in files:
        name = os.path.basename(os.path.dirname(os.path.dirname(f)))
        n, rw, sr = compute_metrics(f)
        runs[name] = (n, rw, sr)
        print(f"  {name:<48s} {n:<5d} {rw:<10.4f} {sr:<8.1f}%")

    # Aggregate by config (strip seed)
    print()
    print("═" * 90)
    print("Aggregated by config (stripped seed):")
    print("─" * 90)
    print(f"  {'Config':<42s} {'n_seeds':<10s} {'reward_mean±std':<25s} {'success_mean±std':<25s}")
    print("─" * 90)

    grouped = defaultdict(list)
    for name, (n, rw, sr) in runs.items():
        grouped[strip_seed(name)].append((rw, sr))

    rows = []
    for config, vals in sorted(grouped.items()):
        rewards = [v[0] for v in vals]
        succs = [v[1] for v in vals]
        n_seeds = len(vals)
        rw_mean = statistics.mean(rewards)
        sr_mean = statistics.mean(succs)
        if n_seeds >= 2:
            rw_std = statistics.stdev(rewards)
            sr_std = statistics.stdev(succs)
        else:
            rw_std = 0.0
            sr_std = 0.0
        rows.append((config, n_seeds, rw_mean, rw_std, sr_mean, sr_std))
        rw_str = f"{rw_mean:.4f}±{rw_std:.4f}"
        sr_str = f"{sr_mean:.1f}%±{sr_std:.1f}%"
        print(f"  {config:<42s} {n_seeds:<10d} {rw_str:<25s} {sr_str:<25s}")

    # Headline: best by success_rate mean
    print()
    print("═" * 90)
    print("Top 3 by success_rate mean:")
    print("─" * 90)
    rows_by_sr = sorted(rows, key=lambda r: -r[4])[:3]
    for i, (config, n_seeds, rw_mean, rw_std, sr_mean, sr_std) in enumerate(rows_by_sr):
        rw_str = f"{rw_mean:.4f}±{rw_std:.4f}"
        sr_str = f"{sr_mean:.1f}%±{sr_std:.1f}%"
        print(f"  {i+1}. {config:<42s} n={n_seeds}  reward={rw_str}  success={sr_str}")

    # Compare to baselines
    print()
    print("═" * 90)
    print("Baselines reference (single-run from H100):")
    print("  DUET v1     53.0%   (target to beat)")
    print("  LUFFY       49.5%")
    print("  CHORD       39.0%")
    print("  SFT+RL      24.0%")
    print("  OnPolicy     2.0%")
    print()
    print("Status:")
    if rows_by_sr:
        best = rows_by_sr[0]
        sr_mean = best[4]
        sr_std = best[5]
        if sr_mean - sr_std > 53.0:
            print(f"  🏆 Best config '{best[0]}' MEAN-STD = {sr_mean - sr_std:.1f}% > DUET v1 (53%)")
        elif sr_mean > 53.0:
            print(f"  ✓ Best config '{best[0]}' mean = {sr_mean:.1f}% > DUET v1 (53%) (within noise)")
        elif sr_mean > 49.5:
            print(f"  ~ Best config '{best[0]}' mean = {sr_mean:.1f}% > LUFFY (49.5%) but < DUET v1 (53%)")
        else:
            print(f"  ✗ Best config '{best[0]}' mean = {sr_mean:.1f}% < LUFFY (49.5%). Need more sweep.")


if __name__ == "__main__":
    main()
