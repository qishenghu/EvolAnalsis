#!/usr/bin/env python3
"""Generate 3 extension velocity-mode WS yamls for the 4xA100 server.

Fills gaps L20X+mandatory queue did not cover:
  pk04_v00            : peak=0.4 valley=0  K=10 vt=0.01   (interp pk03/pk05)
  pk05_v00_K5_vt005   : peak=0.5 valley=0  K=5  vt=0.005  (aggr x high peak)
  pk04_v00_K5_vt005   : peak=0.4 valley=0  K=5  vt=0.005  (aggr x mid peak)

Base: ws_swC_v_pk03_v00.yaml (already patched with local model path + gpu_mem 0.6).
Output: same dir.
"""
from pathlib import Path

ROOT = Path("/data/home/qisheng/EvolAnalsis")
SWEEP = ROOT / "config/duet_paper_experiments_configs/webshop/sweep_phase_c"
BASE = SWEEP / "ws_swC_v_pk03_v00.yaml"

CELLS = [
    ("ws_swC_v_pk04_v00",          0.4, 0.0, 10, 0.01),
    ("ws_swC_v_pk05_v00_K5_vt005", 0.5, 0.0, 5,  0.005),
    ("ws_swC_v_pk04_v00_K5_vt005", 0.4, 0.0, 5,  0.005),
]


def patch(text: str, name: str, peak: float, valley: float, k: int, vt: float) -> str:
    text = text.replace(
        "experiment_name: ws_swC_v_pk03_v00",
        f"experiment_name: {name}",
    )
    text = text.replace("chord_mu_peak: 0.3", f"chord_mu_peak: {peak}")
    text = text.replace("chord_mu_valley: 0.0", f"chord_mu_valley: {valley}")
    text = text.replace("chord_mu_velocity_window: 10", f"chord_mu_velocity_window: {k}")
    text = text.replace("chord_mu_velocity_target: 0.01", f"chord_mu_velocity_target: {vt}")
    return text


def main():
    base = BASE.read_text()
    for name, peak, valley, k, vt in CELLS:
        out = patch(base, name, peak, valley, k, vt)
        path = SWEEP / f"{name}.yaml"
        path.write_text(out)
        print(f"  -> {path.relative_to(ROOT)}  peak={peak} valley={valley} K={k} vt={vt}")


if __name__ == "__main__":
    main()
