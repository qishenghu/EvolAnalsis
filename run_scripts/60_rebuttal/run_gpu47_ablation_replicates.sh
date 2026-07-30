#!/bin/bash
# Sequential replicate runner on GPUs 4-7. Parallel-safe with the GPUs 0-3 run:
# vLLM on 8125 (main run uses 8124), env on aux 18091 (main uses 8081),
# own RAY_TMPDIR per run. NEVER calls launcher.py --kill (it sweeps globally).
cd /data/home/qisheng/EvolAnalsis
source env_config.sh >/dev/null 2>&1
PY=/data/home/qisheng/miniconda3/envs/duet/bin/python
CFG=config/duet_paper_experiments_configs/rebuttal_neurips/alfworld
for name in alfworld_qwen1.5b_duet_minus_sc_s2027 alfworld_qwen1.5b_bconly_s2027 alfworld_qwen1.5b_duet_minus_bc_s2027; do
  echo "[$(date '+%m-%d %H:%M')] RUN $name"
  export RAY_TMPDIR="/tmp/ray_gpu47_${name:0:24}"
  mkdir -p "$RAY_TMPDIR"
  CUDA_VISIBLE_DEVICES=4,5,6,7 $PY launcher.py --conf $CFG/${name}.yaml > logs/${name}.log 2>&1
  echo "[$(date '+%m-%d %H:%M')] DONE $name (rc=$?)"
done
echo "[$(date '+%m-%d %H:%M')] gpu47 queue complete"
