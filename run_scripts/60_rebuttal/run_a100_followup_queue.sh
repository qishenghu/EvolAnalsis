#!/bin/bash
# ============================================================================
# Follow-up queue: runs AFTER the main A100 rebuttal queue finishes.
# Launch it now; it waits for the main queue's PID file to clear, then runs
# experiments added after the main queue was already in flight (the running
# script cannot be edited safely — bash reads it incrementally).
#
#   PRIORITY CHANGE 2026-07-27. A reviewer simulation found that Reviewer bDeY's blocker for moving
#   3 -> 4 is that SFT+GRPO trains on 400 distinct tasks against DUET's 800: the headline margin is
#   compute-matched but not task-matched. `*_sft_rl_a100_taskmatched` gives the baseline the same 800
#   tasks AND more optimisation than DUET (50 SFT + 100 GRPO vs DUET's 100), so the comparison is
#   conservative in the baseline's favour. ALFWorld goes first because that cell reproduces.
#   Its SFT stage must be regenerated first (`alfworld_qwen1.5b_sft_a100`) — the original
#   checkpoint is no longer on disk.
#
#   PRIORITY CHANGE 2026-07-26 13:20. The first like-for-like replicate (seed 2027, identical 800
#   tasks, verified-identical code and config) scored 2.5% against the paper run's 35.5%, while both
#   were indistinguishable at val@50 (1.0% each). The runs diverge only across the last 50 steps,
#   i.e. the 100-step budget cuts through the phase transition. The decisive question is therefore
#   no longer "how much do replicates vary at 100 steps" but "does a longer budget make them
#   converge", so the 150-step run leads the queue.
#
#   OLD F0a/F0b. webshop_qwen1.5b_duet_a100_fixedtask_seed{2025,2027} — the LIKE-FOR-LIKE seed
#       replicates. `data.seed` alone shuffles the training split and keeps the first 800, so two
#       seeds train on largely disjoint curricula (WebShop: 89 of 800 shared). These configs pin
#       `data.task_seed: 2026`, so they train on exactly the paper run's 800 tasks and only the
#       run-time randomness differs. Compare them against the paper's 35.5%.
#   F3. webshop_qwen1.5b_duet_a100_seed2025_long150 — diagnostic for the WebShop
#       seed sensitivity found on 2026-07-26: seed 2025 scores 3.5% strict vs the
#       paper seed's 35.5% at the 100-step budget, but its training curve has the
#       same shape ~10 steps behind and 32.5% of its episodes sit in the 0.75-0.90
#       band (vs 20.5% for seed 2026), i.e. it is mid-phase-transition at step 100.
#       This run extends the budget to 150 steps to separate "slower" from "worse".
#       Reported as a diagnostic alongside the protocol-matched 100-step number,
#       never as a replacement for it.
#   F1. alfworld_qwen1.5b_duet_a100_shuffled_sc — matched-magnitude shaping
#       control: the progress values are permuted among each task's own states,
#       so coverage (90.4%) and bonus magnitude (mean P 0.507 vs 0.523) are held
#       fixed while the state->progress correspondence is destroyed
#       (corr(position, Phi) 0.772 -> 0.045). Answers y9x6's "compare against
#       simpler reward-shaping baselines" — it isolates whether the gain comes
#       from teacher-derived *progress* or merely from a dense bonus of that size.
#   F2. alfworld_llama3b_duet_a100_rebuttal — DUET (paper recipe) on a
#       Llama-3.2-3B student with the unchanged Qwen2.5-72B teacher cache.
#       Answers UyKJ's "non-Qwen student" question. We already have the LUFFY
#       and GRPO Llama points on disk (19.5% vs 5.5% at step 50), so this
#       completes the cross-family comparison.
#
# Launch: nohup bash run_a100_followup_queue.sh > logs/a100_followup.log 2>&1 &
# Stop:   touch logs/A100_FOLLOWUP_STOP
# ============================================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs
PIDFILE=logs/.a100_followup.pid
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: follow-up queue already running (PID $(cat "$PIDFILE"))."; exit 1
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

source "$SCRIPT_DIR/env_config.sh"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_DUET}"

GPUS="0,1,2,4"
CFG_ROOT="config/duet_paper_experiments_configs/rebuttal_neurips"
RESULTS_LOG="NeurIPS_2026_Latex/data/a100_rebuttal_results.md"
STOP_FILE="logs/A100_FOLLOWUP_STOP"

echo "[$(date '+%m-%d %H:%M')] follow-up queue armed; waiting for the main queue to finish..."
for i in $(seq 1 2880); do   # up to 48h
    if [ -f "$STOP_FILE" ]; then echo "STOP file set before start — exiting."; exit 0; fi
    main_pid=$(cat logs/.a100_queue.pid 2>/dev/null || echo "")
    if [ -z "$main_pid" ] || ! kill -0 "$main_pid" 2>/dev/null; then
        echo "[$(date '+%m-%d %H:%M')] main queue finished (after ${i} min of waiting)"
        break
    fi
    sleep 60
done

kill_ray_stragglers() {
    local n
    n=$(ps -ef | grep -E "ray::TaskRunner|ray::AsyncvLLMServer|ray::WorkerDict|ray::WorkerGroupRegisterCenter" | grep -v grep | wc -l)
    if [ "$n" -gt 0 ]; then
        ps -ef | grep -E "ray::TaskRunner|ray::AsyncvLLMServer|ray::WorkerDict|ray::WorkerGroupRegisterCenter" | grep -v grep | awk '{print $2}' | xargs -r kill 2>/dev/null
        sleep 5
        ps -ef | grep -E "ray::TaskRunner|ray::AsyncvLLMServer|ray::WorkerDict|ray::WorkerGroupRegisterCenter" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
        sleep 3
    fi
}

parse_and_log() {
    local name=$1
    python - <<PYEOF
import json, os, datetime
name = "${name}"
env = "webshop" if name.startswith("webshop_") else "alfworld"
val_log = f"experiments/{env}/{name}/validation_log/150.jsonl"
if not os.path.exists(val_log): val_log = f"experiments/{env}/{name}/validation_log/100.jsonl"
alt     = f"experiments/{env}/{name}/validation_log/50.jsonl"
p = val_log if os.path.exists(val_log) else alt
ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
if not os.path.exists(p):
    line = f"| {ts} | 4xA100(0124) | {name} | - | - | - | - | val MISSING |"
else:
    n=s=l=0; rw=0.0
    for ln in open(p):
        try: x=json.loads(ln)
        except Exception: continue
        n+=1; sc=float(x.get("score", x.get("reward",0.0))); rw+=sc
        s+= sc>=1.0; l+= sc>=0.9
    step = os.path.basename(p).split('.')[0]
    line = (f"| {ts} | 4xA100(0124) | {name} (val@{step}) | {s/n*100:.1f}% | {l/n*100:.1f}% | "
            f"{rw/n:.4f} | {n} | OK |") if n else f"| {ts} | 4xA100(0124) | {name} | - | - | - | 0 | EMPTY |"
print(line)
open("${RESULTS_LOG}","a").write(line+"\n")
PYEOF
}

QUEUE=(
    webshop_qwen1.5b_duet_a100_fixedtask_seed2027_s150
    alfworld_qwen1.5b_sft_a100
    alfworld_qwen1.5b_sft_rl_a100_taskmatched
    webshop_qwen1.5b_duet_a100_fixedtask_seed2025
    alfworld_qwen1.5b_duet_a100_shuffled_sc
    alfworld_qwen1.5b_duet_a100_teacher14b
    webshop_qwen1.5b_sft_a100
    webshop_qwen1.5b_sft_rl_a100_taskmatched
    alfworld_llama3b_duet_a100_rebuttal
    webshop_qwen1.5b_duet_a100_fixedtask_seed2028
    alfworld_qwen1.5b_duet_a100_soft_clean
)

for name in "${QUEUE[@]}"; do
    [ -f "$STOP_FILE" ] && { echo "STOP file set — exiting."; break; }
    case "$name" in webshop_*) env=webshop ;; *) env=alfworld ;; esac
    cfg="${CFG_ROOT}/${env}/${name}.yaml"
    [ -f "$cfg" ] || { echo "MISSING config $cfg — skipping"; continue; }
    ray_tmp="${RAY_TMPDIR}/r$(echo "$name" | md5sum | head -c 8)"
    echo "[$(date '+%m-%d %H:%M')] PREP: $name"
    kill_ray_stragglers
    bash start_env_alfworld.sh stop 2>/dev/null || true
    bash start_env_webshop.sh stop 2>/dev/null || true
    sleep 8
    if [ "$env" = "webshop" ]; then bash start_env_webshop.sh; else bash start_env_alfworld.sh; fi
    sleep 12
    mkdir -p "$ray_tmp"; rm -rf "$ray_tmp"/session_* 2>/dev/null || true
    echo "[$(date '+%m-%d %H:%M')] RUN: $name"
    (CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR="$ray_tmp" python launcher.py --conf "$cfg" > "logs/${name}.log" 2>&1)
    echo "[$(date '+%m-%d %H:%M')] rc=$? : $name"
    parse_and_log "$name"
    kill_ray_stragglers
done

bash start_env_alfworld.sh stop 2>/dev/null || true
bash start_env_webshop.sh stop 2>/dev/null || true
echo "[$(date '+%m-%d %H:%M')] FOLLOW-UP QUEUE COMPLETE"
