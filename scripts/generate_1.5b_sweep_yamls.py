#!/usr/bin/env python3
"""Generate 1.5B v39b sweep yamls for ALFWorld + WebShop.

Base configs:
  config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet_v39_postfix.yaml
  config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_duet_v39_postfix.yaml

These bases are 1.5B v24 winner infrastructure + adaptive-mu patch (= 1.5B's v39b cfg).
Sweep varies ONLY (peak, valley, d_floor, d_ema_alpha) per handoff Section 5.

3 cells (#01, #07, #09) already covered by existing v39b/v39/v39c postfix runs;
this script generates the remaining 9 cells for each env.
"""
from pathlib import Path

ROOT = Path("/data/home/qisheng/EvolAnalsis")
WS_BASE = ROOT / "config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet_v39_postfix.yaml"
AF_BASE = ROOT / "config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_duet_v39_postfix.yaml"
WS_OUT = ROOT / "config/duet_paper_experiments_configs/webshop/sweep_1.5b"
AF_OUT = ROOT / "config/duet_paper_experiments_configs/alfworld/sweep_1.5b"

# (id, tag, peak, valley, d_floor, d_ema_alpha)
CELLS = [
    ("02", "peak02",          0.2, 0.05, 0.5, 0.5),
    ("03", "peak04",          0.4, 0.05, 0.5, 0.5),
    ("04", "peak05",          0.5, 0.05, 0.5, 0.5),
    ("05", "peak06",          0.6, 0.05, 0.5, 0.5),
    ("06", "peak07",          0.7, 0.05, 0.5, 0.5),
    ("08", "ema08",           0.3, 0.05, 0.5, 0.8),
    ("10", "pk05_ema02",      0.5, 0.05, 0.5, 0.2),
    ("11", "pk05_v10",        0.5, 0.10, 0.5, 0.5),
    ("12", "pk05_ema02_v10",  0.5, 0.10, 0.5, 0.2),
]


def patch(base_text: str, env: str, cell_id: str, tag: str,
          peak: float, valley: float, d_floor: float, d_ema_alpha: float) -> str:
    name = f"{env}_qwen1.5b_duet_swA_{cell_id}_{tag}"
    text = base_text
    # 1) experiment_name
    text = text.replace(
        f"experiment_name: {env}_qwen1.5b_duet_v39_postfix",
        f"experiment_name: {name}",
    )
    # 2) workspace_id
    text = text.replace(
        f"workspace_id: {env}_qwen1.5b_duet_v39_postfix",
        f"workspace_id: {name}",
    )
    # 3) chord schedule params (use exact-line replacements for safety)
    text = text.replace(
        "chord_mu_peak: 0.3",
        f"chord_mu_peak: {peak}",
    )
    text = text.replace(
        "chord_mu_valley: 0.05",
        f"chord_mu_valley: {valley}",
    )
    text = text.replace(
        "chord_mu_d_floor: 0.5",
        f"chord_mu_d_floor: {d_floor}",
    )
    text = text.replace(
        "chord_mu_d_ema_alpha: 0.2",
        f"chord_mu_d_ema_alpha: {d_ema_alpha}",
    )
    return text


def main():
    ws_base = WS_BASE.read_text()
    af_base = AF_BASE.read_text()
    WS_OUT.mkdir(parents=True, exist_ok=True)
    AF_OUT.mkdir(parents=True, exist_ok=True)

    for cell_id, tag, peak, valley, d_floor, d_ema_alpha in CELLS:
        ws_text = patch(ws_base, "webshop", cell_id, tag, peak, valley, d_floor, d_ema_alpha)
        af_text = patch(af_base, "alfworld", cell_id, tag, peak, valley, d_floor, d_ema_alpha)
        ws_path = WS_OUT / f"webshop_qwen1.5b_duet_swA_{cell_id}_{tag}.yaml"
        af_path = AF_OUT / f"alfworld_qwen1.5b_duet_swA_{cell_id}_{tag}.yaml"
        ws_path.write_text(ws_text)
        af_path.write_text(af_text)
        print(f"[{cell_id}] {tag}: peak={peak} valley={valley} d_floor={d_floor} ema_a={d_ema_alpha}")
        print(f"        WS: {ws_path.relative_to(ROOT)}")
        print(f"        AF: {af_path.relative_to(ROOT)}")

    print(f"\nWrote {len(CELLS)} configs per env to:")
    print(f"  {WS_OUT.relative_to(ROOT)}")
    print(f"  {AF_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
