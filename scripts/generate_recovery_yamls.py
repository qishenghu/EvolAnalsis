#!/usr/bin/env python3
"""Generate 3 recovery yamls with SHORT names (avoid ray socket 107-byte limit):
  1. ws_3b_luffy_v          — LUFFY 3B WS verification (user wants to confirm 49.5%)
  2. ws_3b_bok_v10          — best-of-k retry (FAILED last time due to long name)
  3. ws_1_5b_swC02_da       — 1.5B WS recovery using disc_acc level mode (SOTA recipe)
"""
from pathlib import Path
import re

ROOT = Path("/data/home/qisheng/EvolAnalsis")
OUT_DIR = ROOT / "config/duet_paper_experiments_configs/sota_hunt_2026_05_03"

# 1) LUFFY 3B WS verify: use existing webshop_qwen3b_luffy.yaml, lower gpu_mem
luffy_base = ROOT / "config/duet_paper_experiments_configs/webshop/webshop_qwen3b_luffy.yaml"
text = luffy_base.read_text()
text = re.sub(r"^(\s*experiment_name:\s*).+$", r"\g<1>ws_3b_luffy_v", text, count=1, flags=re.MULTILINE)
text = re.sub(r"(\s*workspace_id:\s*).+$", r"\g<1>ws_3b_luffy_v", text, count=1, flags=re.MULTILINE)
text = re.sub(r"(\s*gpu_memory_utilization:\s*)[\d.]+", r"\g<1>0.6", text)
out_luffy = OUT_DIR / "ws_3b_luffy_v.yaml"
out_luffy.write_text(text)
print(f"  OK: {out_luffy.name} (LUFFY 3B WS verify)")

# 2) best-of-k retry: copy existing bok yaml and rename
bok_base = OUT_DIR / "ws_3b_gap_bok_pk02_v10_NOtw_dr3fast.yaml"
text = bok_base.read_text()
text = re.sub(r"^(\s*experiment_name:\s*).+$", r"\g<1>ws_3b_bok_v10", text, count=1, flags=re.MULTILINE)
text = re.sub(r"(\s*workspace_id:\s*).+$", r"\g<1>ws_3b_bok_v10", text, count=1, flags=re.MULTILINE)
out_bok = OUT_DIR / "ws_3b_bok_v10.yaml"
out_bok.write_text(text)
print(f"  OK: {out_bok.name} (best-of-k retry, short name)")

# 3) 1.5B WS recovery: clone swC_02 SOTA template (disc_acc level mode, NOT gap mode)
swC02_base = ROOT / "config/duet_paper_experiments_configs/webshop/sweep_1.5b/webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06.yaml"
text = swC02_base.read_text()
text = re.sub(r"^(\s*experiment_name:\s*).+$", r"\g<1>ws_1_5b_swC02_da", text, count=1, flags=re.MULTILINE)
text = re.sub(r"(\s*workspace_id:\s*).+$", r"\g<1>ws_1_5b_swC02_da", text, count=1, flags=re.MULTILINE)
text = re.sub(r"(\s*gpu_memory_utilization:\s*)[\d.]+", r"\g<1>0.6", text)
# Also patch model path to local 1.5B
remote = "/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/models/Qwen/Qwen2.5-1.5B-Instruct"
text = text.replace(remote, "/data/shared_models/Qwen2.5-1.5B-Instruct")
out_swC02 = OUT_DIR / "ws_1_5b_swC02_da.yaml"
out_swC02.write_text(text)
print(f"  OK: {out_swC02.name} (1.5B WS recovery, swC_02 disc_acc level recipe)")
