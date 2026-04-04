#!/usr/bin/env python3
"""Find the actual wandb run for DUET 0404."""
import wandb

api = wandb.Api()
entity = "qisheng001-nanyang-technological-university-singapore"
project = "agentevolver"

# Search recent runs
runs = api.runs(f"{entity}/{project}", order="-created_at")
print("Recent runs:")
for r in runs[:30]:
    print(f"  {r.id} | {r.name} | {r.state} | {r.created_at}")

print("\nSearching for '0404' in run names...")
for r in runs:
    if "0404" in r.name:
        print(f"  FOUND: {r.id} | {r.name} | {r.state} | {r.created_at}")

print("\nSearching for 'duet' in run names (recent)...")
for r in runs[:50]:
    if "duet" in r.name.lower():
        print(f"  {r.id} | {r.name} | {r.state} | {r.created_at}")
