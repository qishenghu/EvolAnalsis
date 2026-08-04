#!/usr/bin/env python3
"""Generate 3B WebShop sweep yamls.

Strategy: port the 1.5B SOTA recipe (swC_02 = peak=0.3, valley=0.10, d_floor=0.6,
ema=0.2 → 36.0% on 1.5B WS) to 3B, plus 3 nearby variations to handle the
possibility that 3B's sweet spot differs from 1.5B's.

Base: config/duet_paper_experiments_configs/webshop/sweep/ws_swA_01_v39b_default.yaml
This is the 3B server's existing v39b sweep template (3B v39b cfg infrastructure).

Patches applied per cell:
  1. experiment_name + workspace_id  -> 'webshop_qwen3b_duet_swD_*'
  2. model.path                      -> /data/shared_models/Qwen2.5-3B-Instruct
  3. (peak, valley, d_floor, d_ema_alpha) per cell

Output: config/duet_paper_experiments_configs/webshop/sweep_3b/

Target: beat LUFFY 49.5% on 3B WS val@100 success_rate.
"""
from pathlib import Path

ROOT = Path("/data/home/qisheng/EvolAnalsis")
BASE = ROOT / "config/duet_paper_experiments_configs/webshop/sweep/ws_swA_01_v39b_default.yaml"
OUT = ROOT / "config/duet_paper_experiments_configs/webshop/sweep_3b"

REMOTE_MODEL_PATH = "/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/models/Qwen/Qwen2.5-3B-Instruct"
LOCAL_MODEL_PATH = "/data/shared_models/Qwen2.5-3B-Instruct"

# (cell_id, tag, peak, valley, d_floor, d_ema_alpha)
# swD_01 is the direct port of 1.5B SOTA winner swC_02 (36.0%).
# swD_02-04 probe nearest neighbors.
CELLS = [
    ("01", "pk03_v10_floor06",  0.3, 0.10, 0.6, 0.2),  # = 1.5B SOTA
    ("02", "pk03_v10_floor07",  0.3, 0.10, 0.7, 0.2),  # push d_floor higher
    ("03", "pk03_v10_ema02",    0.3, 0.10, 0.5, 0.2),  # = 1.5B swB_01 (21.5%)
    ("04", "pk03_v15_floor06",  0.3, 0.15, 0.6, 0.2),  # more BC valley
]


def patch(base_text: str, cell_id: str, tag: str,
          peak: float, valley: float, d_floor: float, d_ema_alpha: float) -> str:
    name = f"webshop_qwen3b_duet_swD_{cell_id}_{tag}"
    text = base_text
    text = text.replace(
        f"experiment_name: ws_swA_01_v39b_default",
        f"experiment_name: {name}",
    )
    text = text.replace(
        f"workspace_id: webshop_qwen3b_duet_v39b",
        f"workspace_id: {name}",
    )
    text = text.replace(REMOTE_MODEL_PATH, LOCAL_MODEL_PATH)
    # gpu_memory_utilization=0.65 OOMs on this server's 3B; lower to 0.6
    text = text.replace("gpu_memory_utilization: 0.65", "gpu_memory_utilization: 0.6")
    text = text.replace("chord_mu_peak: 0.3",  f"chord_mu_peak: {peak}")
    text = text.replace("chord_mu_valley: 0.05",  f"chord_mu_valley: {valley}")
    text = text.replace("chord_mu_d_floor: 0.5",  f"chord_mu_d_floor: {d_floor}")
    text = text.replace("chord_mu_d_ema_alpha: 0.5",  f"chord_mu_d_ema_alpha: {d_ema_alpha}")
    return text


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base_text = BASE.read_text()
    for cell_id, tag, peak, valley, d_floor, d_ema_alpha in CELLS:
        out = patch(base_text, cell_id, tag, peak, valley, d_floor, d_ema_alpha)
        path = OUT / f"webshop_qwen3b_duet_swD_{cell_id}_{tag}.yaml"
        path.write_text(out)
        print(f"[swD_{cell_id}] {tag}: peak={peak} valley={valley} d_floor={d_floor} ema_a={d_ema_alpha}")
        print(f"       -> {path.relative_to(ROOT)}")
    print(f"\nWrote {len(CELLS)} 3B WS configs to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
