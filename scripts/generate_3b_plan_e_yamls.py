#!/usr/bin/env python3
"""Generate 3B Plan E yamls — untested BC schedule corners."""
from pathlib import Path

ROOT = Path("/data/home/qisheng/EvolAnalsis")
BASE = ROOT / "config/duet_paper_experiments_configs/webshop/sweep/ws_swA_01_v39b_default.yaml"
OUT = ROOT / "config/duet_paper_experiments_configs/webshop/sweep_3b"

REMOTE = "/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/models/Qwen/Qwen2.5-3B-Instruct"
LOCAL = "/data/shared_models/Qwen2.5-3B-Instruct"

CELLS = [
    ("01", "pk03_v05_ema01",  0.3, 0.05, 0.5, 0.1),  # very slow EMA
    ("02", "pk02_v10",        0.2, 0.10, 0.5, 0.5),  # low peak + raised valley
]

def patch(text, cell_id, tag, peak, valley, d_floor, d_ema_alpha):
    name = f"webshop_qwen3b_duet_swE_{cell_id}_{tag}"
    text = text.replace("experiment_name: ws_swA_01_v39b_default", f"experiment_name: {name}")
    text = text.replace("workspace_id: webshop_qwen3b_duet_v39b", f"workspace_id: {name}")
    text = text.replace(REMOTE, LOCAL)
    text = text.replace("gpu_memory_utilization: 0.65", "gpu_memory_utilization: 0.6")
    text = text.replace("chord_mu_peak: 0.3", f"chord_mu_peak: {peak}")
    text = text.replace("chord_mu_valley: 0.05", f"chord_mu_valley: {valley}")
    text = text.replace("chord_mu_d_floor: 0.5", f"chord_mu_d_floor: {d_floor}")
    text = text.replace("chord_mu_d_ema_alpha: 0.5", f"chord_mu_d_ema_alpha: {d_ema_alpha}")
    return text

OUT.mkdir(parents=True, exist_ok=True)
base = BASE.read_text()
for cell_id, tag, peak, valley, d_floor, d_ema_alpha in CELLS:
    out = patch(base, cell_id, tag, peak, valley, d_floor, d_ema_alpha)
    path = OUT / f"webshop_qwen3b_duet_swE_{cell_id}_{tag}.yaml"
    path.write_text(out)
    print(f"[swE_{cell_id}] {tag}: peak={peak} valley={valley} d_floor={d_floor} ema_a={d_ema_alpha} -> {path.name}")
