#!/bin/bash
# Periodically health-check whichever rebuttal run is live, and emit a line ONLY when there is
# something to act on (a mechanism fault) or to note (drift). Silence means healthy.
cd /data/home/qisheng/EvolAnalsis
PY=/data/home/qisheng/miniconda3/envs/duet/bin/python
INTERVAL=${INTERVAL:-1200}
while true; do
    live=$(ps -ef | grep "[l]auncher.py --conf" | grep -oE "rebuttal_neurips/[a-z]+/[a-zA-Z0-9_.]+\.yaml" | head -1 | xargs -r basename | sed 's/\.yaml$//')
    if [ -n "$live" ]; then
        case "$live" in
            webshop_*) ref=webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06; env=webshop ;;
            *)         ref=alfworld_qwen1.5b_duet_v39c_postfix;            env=alfworld ;;
        esac
        out=$(PYTHONPATH=. timeout 600 $PY scripts/monitor_run_health.py --run "$live" --reference "$ref" --env "$env" 2>/dev/null)
        echo "$out" | grep -qE "FAULT" && { echo "$out"; }
        echo "$out" | grep -qE "WATCH" && { echo "$out" | grep -E "^\[|WATCH"; }
    fi
    sleep "$INTERVAL"
done
