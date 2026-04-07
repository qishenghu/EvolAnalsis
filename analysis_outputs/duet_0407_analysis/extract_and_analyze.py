#!/usr/bin/env python3
"""
Extract wandb metrics for DUET 0407 variants + LUFFY and build comprehensive comparison.
"""

import json
import os
import sys
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

WANDB_DIR = "wandb"

# Target runs
RUNS = {
    "LUFFY": "o405qtk1",
    "DUET_0407_SC": "u55gm1e5",
    "DUET_0407_Alpha": "qsrb20qf",
}

# Metrics to extract
METRICS = [
    "critic/success_onpolicy/mean",
    "critic/avg_reward_onpolicy/mean",
    "duet/teacher_gradient_share",
    "dr3/disc_acc",
    "dr3/w_off_mean",
    "dr3/w_off_std",
    "dr3/alpha",
    "dr3/dual_lambda",
    "dr3/ess_ratio",
    "dr3/w_clip_upper",
    "actor/kl_loss",
    "actor/entropy",
    "state_channel/coverage",
    "state_channel/progress_mean",
    "state_channel/bonus_mean",
    "state_channel/bonus_vs_reward_ratio",
    "diag/adv_teacher_sample_mean",
    "diag/adv_onpolicy_sample_mean",
    "diag/teacher_sample_ratio",
    "actor/pg_loss",
]


def find_run_dir(run_id):
    """Find wandb run directory by run ID."""
    for d in os.listdir(WANDB_DIR):
        if d.endswith(f"-{run_id}"):
            return os.path.join(WANDB_DIR, d)
    return None


def extract_metrics_from_run(run_dir, metrics_list):
    """Extract metrics from wandb run files."""
    results = {}
    run_files_dir = os.path.join(run_dir, "files")

    # Try reading from wandb-history.jsonl
    history_file = os.path.join(run_files_dir, "wandb-history.jsonl")
    if not os.path.exists(history_file):
        # Try the run directory itself
        for fname in os.listdir(run_dir):
            if fname.endswith(".wandb"):
                # Binary format - need wandb SDK
                pass

    # Try to read from the wandb event files
    events_dir = run_dir
    event_files = []
    for root, dirs, files in os.walk(run_dir):
        for f in files:
            if f == "wandb-history.jsonl":
                event_files.append(os.path.join(root, f))

    if event_files:
        for ef in event_files:
            with open(ef) as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        step = record.get("_step", 0)
                        for metric in metrics_list:
                            if metric in record:
                                if metric not in results:
                                    results[metric] = {"steps": [], "values": []}
                                results[metric]["steps"].append(step)
                                results[metric]["values"].append(record[metric])
                    except json.JSONDecodeError:
                        continue
        return results

    # Fallback: try reading from run summary
    summary_file = os.path.join(run_files_dir, "wandb-summary.json")
    if os.path.exists(summary_file):
        with open(summary_file) as f:
            summary = json.load(f)
        for metric in metrics_list:
            if metric in summary:
                results[metric] = {"steps": [-1], "values": [summary[metric]]}

    return results


def extract_from_tfevents(run_dir, metrics_list):
    """Alternative extraction using raw log files if wandb-history not available."""
    # Check for jsonl log files in the run directory
    results = {}
    log_dir = os.path.join(run_dir, "logs")
    if os.path.exists(log_dir):
        for fname in os.listdir(log_dir):
            if fname.endswith(".jsonl"):
                with open(os.path.join(log_dir, fname)) as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            step = record.get("step", record.get("_step", 0))
                            for metric in metrics_list:
                                if metric in record:
                                    if metric not in results:
                                        results[metric] = {"steps": [], "values": []}
                                    results[metric]["steps"].append(step)
                                    results[metric]["values"].append(record[metric])
                        except json.JSONDecodeError:
                            continue
    return results


def main():
    all_data = {}

    for name, run_id in RUNS.items():
        run_dir = find_run_dir(run_id)
        if not run_dir:
            print(f"WARNING: Could not find run directory for {name} (ID: {run_id})")
            continue

        print(f"Extracting {name} from {run_dir}...")

        # Try primary extraction
        metrics = extract_metrics_from_run(run_dir, METRICS)

        # If no history file, try alternative
        if not metrics:
            metrics = extract_from_tfevents(run_dir, METRICS)

        if not metrics:
            print(f"  WARNING: No metrics found for {name}")
            # List what files exist
            for root, dirs, files in os.walk(run_dir):
                for f in files:
                    fpath = os.path.join(root, f)
                    size = os.path.getsize(fpath)
                    print(f"    {fpath} ({size} bytes)")
        else:
            print(f"  Found {len(metrics)} metrics, examples:")
            for m in list(metrics.keys())[:5]:
                vals = metrics[m]["values"]
                print(f"    {m}: {len(vals)} points, last={vals[-1]:.4f}" if vals else f"    {m}: empty")

        all_data[name] = {
            "_meta": {
                "run_id": run_id,
                "run_dir": run_dir,
            },
            **metrics
        }

    # Save extracted data
    output_file = "analysis_outputs/duet_0407_analysis/wandb_0407_data.json"
    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nSaved extracted data to {output_file}")

    # Print comparison summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    for name in ["LUFFY", "DUET_0407_SC", "DUET_0407_Alpha"]:
        data = all_data.get(name, {})
        success = data.get("critic/success_onpolicy/mean", {})
        vals = success.get("values", [])
        if vals:
            print(f"\n{name}:")
            print(f"  Training success: peak={max(vals):.4f}, final={vals[-1]:.4f}, "
                  f"mean_last10={np.mean(vals[-10:]):.4f}, steps={len(vals)}")

        # DR3 metrics
        for metric in ["dr3/disc_acc", "dr3/w_off_mean", "duet/teacher_gradient_share"]:
            mdata = data.get(metric, {})
            mvals = mdata.get("values", [])
            if mvals:
                print(f"  {metric}: final={mvals[-1]:.4f}, mean={np.mean(mvals):.4f}")

        # SC metrics
        for metric in ["state_channel/coverage", "state_channel/bonus_vs_reward_ratio"]:
            mdata = data.get(metric, {})
            mvals = mdata.get("values", [])
            if mvals:
                print(f"  {metric}: final={mvals[-1]:.4f}, mean={np.mean(mvals):.4f}")

        # Actor metrics
        for metric in ["actor/kl_loss", "actor/entropy"]:
            mdata = data.get(metric, {})
            mvals = mdata.get("values", [])
            if mvals:
                print(f"  {metric}: final={mvals[-1]:.4f}")

        # Advantage metrics
        for metric in ["diag/adv_teacher_sample_mean", "diag/adv_onpolicy_sample_mean"]:
            mdata = data.get(metric, {})
            mvals = mdata.get("values", [])
            if mvals:
                print(f"  {metric}: final={mvals[-1]:.4f}, mean={np.mean(mvals):.4f}")


if __name__ == "__main__":
    main()
