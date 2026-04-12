#!/bin/bash
# ==============================================================================
# DUET Paper: 7B DUET Experiments (WebShop + ALFWorld)
#
# Server: 8x A100-80G, conda env "duet"
# Config: bs=16, 1600 tasks, rollout.n=8, 100 steps/epoch
# Algorithm: DUET (DR3 + SC), aligned with 3B best config 0409_ema
#
# Each experiment kills stale processes, restarts the environment, then trains.
# ==============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Redirect all temp files to /data to avoid filling up root partition (99GB)
export TMPDIR=/data/home/qisheng/tmp
export TEMP=/data/home/qisheng/tmp
export TMP=/data/home/qisheng/tmp
mkdir -p "$TMPDIR"

cleanup_processes() {
    echo ">>> Cleaning up stale processes..."
    python -c "
import subprocess, os, signal
for kw in ['ray', 'python']:
    result = subprocess.run(['bash', '-c', f'ps -eo pid,cmd | grep {kw} | grep -v grep | grep -v vscode | grep -v \$\$'],
                            capture_output=True, text=True)
    for line in result.stdout.strip().split('\n'):
        if not line.strip(): continue
        try:
            pid = int(line.strip().split()[0])
            if pid == os.getpid(): continue
            os.kill(pid, signal.SIGTERM)
        except: pass
" 2>/dev/null || true
    sleep 3
}

echo "============================================"
echo " DUET Paper: 7B DUET Experiments"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# ---------- 1. WebShop DUET ----------
echo ""
echo "[1/2] WebShop 7B DUET"
echo "============================================"
cleanup_processes
bash start_env_webshop.sh
echo ""
python launcher.py --conf config/duet_paper_experiments_configs/webshop/webshop_7b_duet.yaml
bash start_env_webshop.sh stop

# ---------- 2. ALFWorld DUET ----------
echo ""
echo "[2/2] ALFWorld 7B DUET"
echo "============================================"
cleanup_processes
bash start_env_alfworld.sh
echo ""
python launcher.py --conf config/duet_paper_experiments_configs/alfworld/alfworld_7b_duet.yaml
bash start_env_alfworld.sh stop

echo ""
echo "============================================"
echo " All 7B DUET experiments done."
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
