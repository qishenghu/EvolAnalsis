#!/bin/bash
# ============================================================================
# Paper Experiments: ScienceWorld (all methods, 100 steps)
# ============================================================================
# Usage:
#   bash scripts/run_paper_sciworld.sh [method]
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
#   - SciWorld env service running at 127.0.0.1:8084
#   - Teacher data at data/teacher_trajectories/sciworld_gold_qwen72b_800_filtered.pkl
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
        run_experiment "SciWorld On-Policy" config/paper_sciworld_onpolicy.yaml
        ;;
    luffy)
        run_experiment "SciWorld LUFFY" config/paper_sciworld_luffy.yaml
        ;;
    chord)
        run_experiment "SciWorld CHORD" config/paper_sciworld_chord.yaml
        ;;
    dr3)
        run_experiment "SciWorld DR3" config/paper_sciworld_dr3.yaml
        ;;
    uniform)
        run_experiment "SciWorld Uniform Mix" config/paper_sciworld_uniform_mix.yaml
        ;;
    dr3_no_dual)
        run_experiment "SciWorld DR3 No Dual" config/paper_sciworld_dr3_no_dual.yaml
        ;;
    dr3_no_gate)
        run_experiment "SciWorld DR3 No Gate" config/paper_sciworld_dr3_no_gate.yaml
        ;;
    dr3_pq)
        run_experiment "SciWorld DR3 P/Q Ratio" config/paper_sciworld_dr3_pq_ratio.yaml
        ;;
    baselines)
        run_experiment "SciWorld On-Policy"    config/paper_sciworld_onpolicy.yaml
        run_experiment "SciWorld LUFFY"        config/paper_sciworld_luffy.yaml
        run_experiment "SciWorld CHORD"        config/paper_sciworld_chord.yaml
        run_experiment "SciWorld Uniform Mix"  config/paper_sciworld_uniform_mix.yaml
        run_experiment "SciWorld DR3"          config/paper_sciworld_dr3.yaml
        ;;
    ablations)
        run_experiment "SciWorld DR3 No Dual"    config/paper_sciworld_dr3_no_dual.yaml
        run_experiment "SciWorld DR3 No Gate"    config/paper_sciworld_dr3_no_gate.yaml
        run_experiment "SciWorld DR3 P/Q Ratio"  config/paper_sciworld_dr3_pq_ratio.yaml
        ;;
    all)
        run_experiment "SciWorld On-Policy"      config/paper_sciworld_onpolicy.yaml
        run_experiment "SciWorld LUFFY"          config/paper_sciworld_luffy.yaml
        run_experiment "SciWorld CHORD"          config/paper_sciworld_chord.yaml
        run_experiment "SciWorld Uniform Mix"    config/paper_sciworld_uniform_mix.yaml
        run_experiment "SciWorld DR3"            config/paper_sciworld_dr3.yaml
        run_experiment "SciWorld DR3 No Dual"    config/paper_sciworld_dr3_no_dual.yaml
        run_experiment "SciWorld DR3 No Gate"    config/paper_sciworld_dr3_no_gate.yaml
        run_experiment "SciWorld DR3 P/Q Ratio"  config/paper_sciworld_dr3_pq_ratio.yaml
        ;;
    *)
        echo "Unknown method: $METHOD"
        echo "Usage: bash scripts/run_paper_sciworld.sh [onpolicy|luffy|chord|dr3|uniform|dr3_no_dual|dr3_no_gate|dr3_pq|baselines|ablations|all]"
        exit 1
        ;;
esac

echo "============================================"
echo "All requested SciWorld experiments completed."
echo "============================================"
