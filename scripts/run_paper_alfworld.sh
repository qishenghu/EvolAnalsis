#!/bin/bash
# ============================================================================
# Paper Experiments: AlfWorld (all methods, 100 steps)
# ============================================================================
# Usage:
#   bash scripts/run_paper_alfworld.sh [method]
#
# Methods:
#   all        - Run all experiments sequentially
#   onpolicy   - Pure on-policy GRPO baseline
#   luffy      - LUFFY (teacher no logprob + policy shaping)
#   chord      - CHORD (SFT + RL convex combination)
#   dr3        - DR³ (our method)
#   uniform    - Uniform mixing baseline (no correction)
#   dr3_no_dual  - Ablation: DR³ without ESS dual
#   dr3_no_gate  - Ablation: DR³ without gap gate
#   dr3_pq       - Ablation: DR³ with p/q ratio
#
# Prerequisites:
#   - AlfWorld env service running at 127.0.0.1:8081
#   - Teacher data at data/teacher_trajectories/qwen72b/alfworld_qwen72b_filtered.pkl
# ============================================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3

METHOD=${1:-all}

run_experiment() {
    local name=$1
    local config=$2
    echo "============================================"
    echo "Running: $name"
    echo "Config:  $config"
    echo "============================================"
    python launcher.py --conf "$config"
    echo ""
    echo "$name finished with exit code $?"
    echo ""
}

case "$METHOD" in
    onpolicy)
        run_experiment "AlfWorld On-Policy" config/paper_alfworld_onpolicy.yaml
        ;;
    luffy)
        run_experiment "AlfWorld LUFFY" config/paper_alfworld_luffy.yaml
        ;;
    chord)
        run_experiment "AlfWorld CHORD" config/paper_alfworld_chord.yaml
        ;;
    dr3)
        run_experiment "AlfWorld DR3" config/paper_alfworld_dr3.yaml
        ;;
    uniform)
        run_experiment "AlfWorld Uniform Mix" config/paper_alfworld_uniform_mix.yaml
        ;;
    dr3_no_dual)
        run_experiment "AlfWorld DR3 No Dual" config/paper_alfworld_dr3_no_dual.yaml
        ;;
    dr3_no_gate)
        run_experiment "AlfWorld DR3 No Gate" config/paper_alfworld_dr3_no_gate.yaml
        ;;
    dr3_pq)
        run_experiment "AlfWorld DR3 P/Q Ratio" config/paper_alfworld_dr3_pq_ratio.yaml
        ;;
    baselines)
        # Run all baselines (no ablations)
        run_experiment "AlfWorld On-Policy"    config/paper_alfworld_onpolicy.yaml
        run_experiment "AlfWorld LUFFY"        config/paper_alfworld_luffy.yaml
        run_experiment "AlfWorld CHORD"        config/paper_alfworld_chord.yaml
        run_experiment "AlfWorld Uniform Mix"  config/paper_alfworld_uniform_mix.yaml
        run_experiment "AlfWorld DR3"          config/paper_alfworld_dr3.yaml
        ;;
    ablations)
        # Run ablation experiments only
        run_experiment "AlfWorld DR3 No Dual"    config/paper_alfworld_dr3_no_dual.yaml
        run_experiment "AlfWorld DR3 No Gate"    config/paper_alfworld_dr3_no_gate.yaml
        run_experiment "AlfWorld DR3 P/Q Ratio"  config/paper_alfworld_dr3_pq_ratio.yaml
        ;;
    all)
        # Run everything
        run_experiment "AlfWorld On-Policy"      config/paper_alfworld_onpolicy.yaml
        run_experiment "AlfWorld LUFFY"          config/paper_alfworld_luffy.yaml
        run_experiment "AlfWorld CHORD"          config/paper_alfworld_chord.yaml
        run_experiment "AlfWorld Uniform Mix"    config/paper_alfworld_uniform_mix.yaml
        run_experiment "AlfWorld DR3"            config/paper_alfworld_dr3.yaml
        run_experiment "AlfWorld DR3 No Dual"    config/paper_alfworld_dr3_no_dual.yaml
        run_experiment "AlfWorld DR3 No Gate"    config/paper_alfworld_dr3_no_gate.yaml
        run_experiment "AlfWorld DR3 P/Q Ratio"  config/paper_alfworld_dr3_pq_ratio.yaml
        ;;
    *)
        echo "Unknown method: $METHOD"
        echo "Usage: bash scripts/run_paper_alfworld.sh [onpolicy|luffy|chord|dr3|uniform|dr3_no_dual|dr3_no_gate|dr3_pq|baselines|ablations|all]"
        exit 1
        ;;
esac

echo "============================================"
echo "All requested AlfWorld experiments completed."
echo "============================================"
