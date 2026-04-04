import wandb
import json

api = wandb.Api()
entity = "qisheng001-nanyang-technological-university-singapore"
project = "agentevolver"

# Metrics we want
target_metrics = [
    "critic/success_onpolicy/mean",
    "critic/rewards/mean",
    "critic/rewards_onpolicy/mean",
    "state_channel/coverage_mean",
    "state_channel/bonus_vs_reward_ratio",
    "state_channel/progress_mean",
    "state_channel/progress_onpolicy_mean",
    "state_channel/bonus_mean",
    "state_channel/bonus_onpolicy_mean",
    "state_channel/bonus_total_mean",
    "state_channel/step_delta_mean",
    "state_channel/step_delta_onpolicy_mean",
    "duet/teacher_gradient_share",
    "duet/adv_onpolicy_effective_mean",
    "duet/adv_teacher_effective_mean",
    "diag/adv_teacher_sample_mean",
    "diag/adv_onpolicy_sample_mean",
    "diag/adv_std",
    "diag/teacher_sample_ratio",
    "actor/kl_loss",
    "actor/entropy_loss",
    "actor/pg_loss",
    "dr3/disc_acc",
    "dr3/ess_ratio",
    "dr3/gap_gate_enabled",
]

runs_info = {
    "DUET_0401": "4jwrx73g",
    "DUET_orig": "bgokw3m6",
    "LUFFY": "o405qtk1",
}

all_data = {}

for name, run_id in runs_info.items():
    print(f"\n{'='*60}")
    print(f"Fetching {name} ({run_id})...")
    run = api.run(f"{entity}/{project}/{run_id}")
    
    # Get ALL history without key filtering
    history = list(run.scan_history(page_size=500))
    print(f"  Total rows: {len(history)}")
    
    run_data = {}
    for m in target_metrics:
        run_data[m] = {"steps": [], "values": []}
    
    for i, row in enumerate(history):
        step = row.get("_step", i)
        for m in target_metrics:
            if m in row and row[m] is not None:
                run_data[m]["steps"].append(step)
                run_data[m]["values"].append(float(row[m]))
    
    all_data[name] = run_data
    
    for m in target_metrics:
        n = len(run_data[m]["values"])
        if n > 0:
            vals = run_data[m]['values']
            print(f"  {m}: {n} pts, first={vals[0]:.6f}, last={vals[-1]:.6f}")
        else:
            print(f"  {m}: NO DATA")

# Also dump all available keys from first run for reference
if history:
    all_keys = set()
    for row in history:
        all_keys.update(row.keys())
    all_data["_available_keys_0401"] = sorted(all_keys)

output_path = "/data/code/exp/EvolAnalsis/analysis_outputs/duet_0401_diagnosis/wandb_data.json"
with open(output_path, "w") as f:
    json.dump(all_data, f, indent=2)
print(f"\nSaved to {output_path}")
