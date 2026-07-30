#!/bin/bash
cd /data/home/qisheng/EvolAnalsis
# wait for the current minus_bc run to release GPUs 4-7
while ps -eo args | grep -q "[m]ain_ppo.*minus_bc_s2027"; do sleep 120; done
sleep 60
PY=/data/home/qisheng/miniconda3/envs/duet/bin/python
export RAY_TMPDIR=/tmp/ray_gpu47_minussc_retry; mkdir -p "$RAY_TMPDIR"
echo "[$(date '+%m-%d %H:%M')] RETRY alfworld_qwen1.5b_duet_minus_sc_s2027"
CUDA_VISIBLE_DEVICES=4,5,6,7 $PY launcher.py \
  --conf config/duet_paper_experiments_configs/rebuttal_neurips/alfworld/alfworld_qwen1.5b_duet_minus_sc_s2027.yaml \
  > logs/alfworld_qwen1.5b_duet_minus_sc_s2027_retry.log 2>&1
echo "[$(date '+%m-%d %H:%M')] DONE (rc=$?)"
