#!/bin/bash
# Remove tmpfs weight-sync scratch dirs for runs that are no longer training.
# The sync dir holds a full bf16 actor copy (~9GB for 4B) in RAM per run.
for d in /dev/shm/duet_rollout_sync/*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  if pgrep -af "main_ppo" | grep -q "$name"; then
    echo "keep (running): $name"
  else
    echo "removing: $name ($(du -sh "$d" 2>/dev/null | cut -f1))"
    rm -rf "$d"
  fi
done
