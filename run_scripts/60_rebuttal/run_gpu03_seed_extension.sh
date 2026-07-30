#!/bin/bash
# Sequential DUET seed extension on GPUs 0-3 (vLLM 8124, main env 8081, own RAY_TMPDIR).
# Parallel-safe with the gpu4-7 ablation runner (8125/18091). NEVER calls launcher.py --kill.
cd /data/home/qisheng/EvolAnalsis
source env_config.sh >/dev/null 2>&1
PY=/data/home/qisheng/miniconda3/envs/duet/bin/python
CFG=config/duet_paper_experiments_configs/rebuttal_neurips/alfworld
for name in alfworld_qwen1.5b_duet_s2029 alfworld_qwen1.5b_duet_s2030; do
  echo "[$(date '+%m-%d %H:%M')] RUN $name"
  export RAY_TMPDIR="/tmp/ray_gpu03_${name:0:24}"
  mkdir -p "$RAY_TMPDIR"
  CUDA_VISIBLE_DEVICES=0,1,2,3 $PY launcher.py --conf $CFG/${name}.yaml > logs/${name}.log 2>&1
  echo "[$(date '+%m-%d %H:%M')] DONE $name (rc=$?)"
done
echo "[$(date '+%m-%d %H:%M')] gpu03 queue complete"
