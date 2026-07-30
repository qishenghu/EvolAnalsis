#!/bin/bash
cd /data/home/qisheng/EvolAnalsis
# wait for the minus_sc retry to release GPUs 4-7
while ps -eo args | grep -q "[m]ain_ppo.*minus_sc_s2027"; do sleep 180; done
sleep 120
PY=/data/home/qisheng/miniconda3/envs/duet/bin/python
export RAY_TMPDIR=/tmp/ray_llama8; mkdir -p "$RAY_TMPDIR"
echo "[$(date '+%m-%d %H:%M')] LAUNCH alfworld_llama3b_duet_a100_both8 (8 GPUs, micro=1)"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PY launcher.py \
  --conf config/duet_paper_experiments_configs/rebuttal_neurips/alfworld/alfworld_llama3b_duet_a100_both8.yaml \
  > logs/alfworld_llama3b_duet_a100_both8.log 2>&1
echo "[$(date '+%m-%d %H:%M')] DONE (rc=$?)"
