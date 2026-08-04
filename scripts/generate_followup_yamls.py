#!/usr/bin/env python3
"""Generate 3 follow-up SOTA-hunt yamls based on agent diagnostic findings.

Agent verdict on Run A (38% gap mode): token_weighting suppressed effective
BC dose by ~15× (φ_mean = 0.033, not 0.15-0.25 as predicted). Single-knob
fix: drop token_weighting. DR3 acceleration is benign and can stay.

A_revised: drop TW, raise valley to 0.10, keep everything else
B_revised: A_revised + best-of-k gap signal (orthogonal novelty test)
C_revised: defensive — keep TW but boost peak 10× to compensate (insurance)
"""
from pathlib import Path
import re

ROOT = Path("/data/home/qisheng/EvolAnalsis")
OUT_DIR = ROOT / "config/duet_paper_experiments_configs/sota_hunt_2026_05_03"
BASE = ROOT / "config/duet_paper_experiments_configs/sota_hunt_2026_05_03/ws_3b_gap_pk02_v05_tw_dr3fast.yaml"

CANDIDATES = [
    {
        "name": "ws_3b_gap_pk02_v10_NOtw_dr3fast",
        "peak": 0.20,
        "valley": 0.10,                # 0.05 → 0.10 (match swE_02's late-train dose)
        "token_weighting": False,      # ⭐ PRIMARY FIX
        "best_of_k": False,
        "rationale": "Agent's #1 recommendation. Drop TW = restore 15× effective BC dose. Predicted 44-47%.",
    },
    {
        "name": "ws_3b_gap_bok_pk02_v10_NOtw_dr3fast",
        "peak": 0.20,
        "valley": 0.10,
        "token_weighting": False,
        "best_of_k": True,             # ⭐ test best-of-k as orthogonal improvement
        "rationale": "A_revised + best-of-k gap. Tests whether capability gap (vs mean gap) further accelerates late-training BC fade in a useful way.",
    },
    {
        "name": "ws_3b_gap_pk20_v50_TW_dr3fast",
        "peak": 2.0,                   # 10× compensation for φ ≈ 0.033 suppression
        "valley": 0.5,
        "token_weighting": True,       # keep TW for ablation
        "best_of_k": False,
        "rationale": "Agent's insurance #2. Keep TW but boost peak 10× to compensate. Predicted 41-44% (lower than A_revised).",
    },
]


def patch_yaml(text: str, cfg: dict) -> str:
    text = re.sub(
        r"^(\s*experiment_name:\s*).+$",
        rf"\g<1>{cfg['name']}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"(\s*workspace_id:\s*).+$",
        rf"\g<1>{cfg['name']}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = re.sub(r"(\s*chord_mu_peak:\s*)[\d.]+", rf"\g<1>{cfg['peak']}", text)
    text = re.sub(r"(\s*chord_mu_valley:\s*)[\d.]+", rf"\g<1>{cfg['valley']}", text)
    text = re.sub(
        r"(\s*chord_use_token_weighting:\s*)(true|false|True|False)",
        rf"\g<1>{'true' if cfg['token_weighting'] else 'false'}",
        text,
    )
    # Add best_of_k flag if needed (after gap_decay_gamma line)
    if cfg["best_of_k"]:
        # Strip any existing best_of_k line
        text = re.sub(r"^\s*chord_mu_gap_use_best_of_k:.*\n", "", text, flags=re.MULTILINE)
        text = re.sub(
            r"(\s*chord_mu_gap_decay_gamma:\s*[\d.]+\n)",
            r"\1    chord_mu_gap_use_best_of_k: true\n",
            text,
            count=1,
        )
    return text


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_text = BASE.read_text()
    for cfg in CANDIDATES:
        out = patch_yaml(base_text, cfg)
        path = OUT_DIR / f"{cfg['name']}.yaml"
        path.write_text(out)
        print(f"  OK: {cfg['name']}")
        print(f"      peak={cfg['peak']} valley={cfg['valley']} TW={cfg['token_weighting']} best_of_k={cfg['best_of_k']}")
        print(f"      → {cfg['rationale']}")


if __name__ == "__main__":
    main()
