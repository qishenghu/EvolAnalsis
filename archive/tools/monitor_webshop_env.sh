#!/bin/bash
# Lightweight WebShop env service monitor — polls every 30s and logs:
#  - env service PID + RSS memory
#  - /create response time (if env up)
#  - /delete error count
#  - launcher.py PID + state
#
# Intended for use during the v39b sanity rerun. Outputs to logs/env_monitor.log.

LOG="logs/env_monitor.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] WebShop env monitor started" >> "$LOG"

while true; do
    ts="$(date '+%H:%M:%S')"

    # find webshop env service
    env_info="$(ps -eo pid,etime,rss,cmd 2>/dev/null | grep 'agentenv-webshop' | grep -v grep | head -1)"
    if [ -n "$env_info" ]; then
        env_pid=$(echo "$env_info" | awk '{print $1}')
        env_etime=$(echo "$env_info" | awk '{print $2}')
        env_rss_kb=$(echo "$env_info" | awk '{print $3}')
        env_rss_mb=$((env_rss_kb / 1024))
    else
        env_pid="-"
        env_etime="-"
        env_rss_mb="-"
    fi

    # try a /create then /release with timing
    if curl -s --max-time 1 http://127.0.0.1:8083 > /dev/null 2>&1; then
        # Don't spam create — only check listening; reading agentgym log periodically instead
        env_status="up"
    else
        env_status="DOWN"
    fi

    # find launcher
    launcher="$(ps -eo pid,etime,cmd 2>/dev/null | grep 'launcher.py' | grep -v grep | grep -v 'monitor_webshop' | head -1)"
    if [ -n "$launcher" ]; then
        l_pid=$(echo "$launcher" | awk '{print $1}')
        l_etime=$(echo "$launcher" | awk '{print $2}')
        l_conf=$(echo "$launcher" | grep -oE 'webshop_qwen3b_duet_[A-Za-z0-9_]+' | head -1)
    else
        l_pid="-"
        l_etime="-"
        l_conf="-"
    fi

    # latest training step (from latest matching log)
    latest_log=$(ls -t logs/webshop_qwen3b_duet_*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        latest_step=$(grep -oE "Training Progress: *[0-9]+%" "$latest_log" 2>/dev/null | tail -1 | grep -oE "[0-9]+%")
        latest_succ=$(grep -oE "step:[0-9]+ - .*critic/success_onpolicy/mean:[0-9.]+" "$latest_log" 2>/dev/null | tail -1 | grep -oE "critic/success_onpolicy/mean:[0-9.]+" | grep -oE "[0-9.]+$")
    fi

    # error count in env service log (last 60s of new lines)
    err_count=$(tail -200 logs/webshop_envservice.log 2>/dev/null | grep -ciE "error|Exception|MemoryError|fail" || echo "0")

    printf "%s  env_pid=%s rss=%sM etime=%s status=%s | launcher_pid=%s etime=%s conf=%s | step=%s succ=%s | recent_errs=%s\n" \
        "$ts" "$env_pid" "$env_rss_mb" "$env_etime" "$env_status" "$l_pid" "$l_etime" "$l_conf" "${latest_step:-?}" "${latest_succ:-?}" "$err_count" >> "$LOG"

    sleep 30
done
