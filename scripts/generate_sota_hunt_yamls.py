#!/usr/bin/env python3
"""Generate 4 SOTA-hunt yamls (one per env × scale) using gap-mode + token weighting.

Per-setting hyperparams chosen from analysis:
- 3B WS: peak=0.2 (swE_02 evidence), valley=0.05, fast gap decay (γ=0.93), DR3 accelerated
- 1.5B WS: peak=0.3 (preserve swC_02 dose), valley=0.10, slow gap decay (γ=0.97), DR3 default
- 3B AF: peak=0.2, valley=0.02 (preserve v_gap_af_b), gap decay 0.95
- 1.5B AF: peak=0.3, valley=0.05 (preserve v39c_postfix dose), gap decay 0.95

Universal: chord_use_token_weighting=true (filter unlearnable tokens)
"""
from pathlib import Path
import re

ROOT = Path("/data/home/qisheng/EvolAnalsis")
OUT_DIR = ROOT / "config/duet_paper_experiments_configs/sota_hunt_2026_05_03"

LOCAL_MODEL_3B = "/data/shared_models/Qwen2.5-3B-Instruct"
LOCAL_MODEL_15B = "/data/shared_models/Qwen2.5-1.5B-Instruct"

REMOTE_PATHS = [
    "/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/models/Qwen/Qwen2.5-3B-Instruct",
    "/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/models/Qwen/Qwen2.5-1.5B-Instruct",
]

# (key, base_template_relative_path, model_path, gap_decay_gamma, peak, valley, dr3_accel, name)
CONFIGS = [
    # 3B WS: peak from swE_02 evidence; γ=0 lets gap-mode auto-fade match swE_02's low-constant μ
    # geometry (~0.13 avg). DR3 acceleration closes the 20-30 step "danger window" where BC + un-faded
    # DR3 double-pull damages 3B's already-decent intrinsic policy.
    {
        "key": "3b_ws",
        "base": "config/duet_paper_experiments_configs/webshop/sweep_phase_c/ws_swC_v_pk03_v00.yaml",
        "model": LOCAL_MODEL_3B,
        "peak": 0.20,
        "valley": 0.05,
        "decay_gamma": 0.0,   # gap-mode alone; fading via gap closure (0.79 → 0.16 over 100 steps)
        "dr3_accel": True,    # disc_lr 0.0003 → 0.001, steps_per_call 2 → 4 (close DR3 lag on 3B WS)
        "name": "ws_3b_gap_pk02_v05_tw_dr3fast",
        "env": "webshop",
    },
    # 1.5B WS: preserve swC_02 SOTA dose (peak=0.3 valley=0.10). γ=0 because 1.5B benefits from BC
    # throughout — early time-cap would cripple the curriculum scaffold that 1.5B critically depends on.
    {
        "key": "1.5b_ws",
        "base": "config/duet_paper_experiments_configs/webshop/sweep_1.5b/webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06.yaml",
        "model": LOCAL_MODEL_15B,
        "peak": 0.30,
        "valley": 0.10,
        "decay_gamma": 0.0,   # let gap mode alone (gap closes slowly on 1.5B WS, BC stays useful)
        "dr3_accel": False,   # 1.5B disc_acc saturates fast, no acceleration needed
        "name": "ws_1_5b_gap_pk03_v10_tw",
        "env": "webshop",
    },
    # 3B AF: matches existing v_gap_af_b template (proven schedule), only adds token weighting.
    # γ=0.95 because AF gap closes slowly (student never reaches teacher), need time-cap as safety.
    {
        "key": "3b_af",
        "base": "config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_duet_v_gap_af_b.yaml",
        "model": LOCAL_MODEL_3B,
        "peak": 0.20,
        "valley": 0.02,
        "decay_gamma": 0.95,
        "dr3_accel": False,
        "name": "af_3b_gap_pk02_v02_g095_tw",
        "env": "alfworld",
    },
    # 1.5B AF: preserve v39c_postfix dose envelope (effective μ ≈ 0.10 throughout). γ=0.97 (gentler
    # than 3B AF's 0.95) — 1.5B AF needs BC longer because student is weaker; 0.95 would fade too fast.
    {
        "key": "1.5b_af",
        "base": "config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_duet_v39_postfix.yaml",
        "model": LOCAL_MODEL_15B,
        "peak": 0.30,
        "valley": 0.05,
        "decay_gamma": 0.97,
        "dr3_accel": False,
        "name": "af_1_5b_gap_pk03_v05_g097_tw",
        "env": "alfworld",
    },
]


def patch_yaml(text: str, cfg: dict) -> str:
    """Apply universal + per-setting patches."""
    # 1. Model path
    for remote in REMOTE_PATHS:
        text = text.replace(remote, cfg["model"])

    # 2. Force experiment_name and workspace_id
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

    # 3. gpu_memory_utilization → 0.6 (safety)
    text = re.sub(
        r"(\s*gpu_memory_utilization:\s*)[\d.]+",
        r"\g<1>0.6",
        text,
    )

    # 4. chord adaptive mode → gap (with all gap params)
    text = re.sub(
        r"(\s*chord_mu_adaptive_mode:\s*)\"?\w+\"?",
        r'\g<1>"gap"',
        text,
    )

    # 5. peak / valley
    text = re.sub(r"(\s*chord_mu_peak:\s*)[\d.]+", rf"\g<1>{cfg['peak']}", text)
    text = re.sub(r"(\s*chord_mu_valley:\s*)[\d.]+", rf"\g<1>{cfg['valley']}", text)

    # 6. chord_use_token_weighting → true
    text = re.sub(
        r"(\s*chord_use_token_weighting:\s*)(true|false|True|False)",
        r"\g<1>true",
        text,
    )

    # 7. Add/update gap-mode params (after chord_mu_adaptive_mode line)
    gap_block = (
        '    chord_mu_gap_ema_alpha: 0.2\n'
        '    chord_mu_gap_anchor_n: 5\n'
        '    chord_mu_gap_anchor_min: 0.05\n'
        f'    chord_mu_gap_decay_gamma: {cfg["decay_gamma"]}\n'
    )
    # Strip any existing gap params first to avoid dupes
    for k in ("chord_mu_gap_ema_alpha", "chord_mu_gap_anchor_n",
             "chord_mu_gap_anchor_min", "chord_mu_gap_decay_gamma"):
        text = re.sub(rf"^\s*{k}:.*\n", "", text, flags=re.MULTILINE)
    # Insert after adaptive_mode line
    text = re.sub(
        r'(\s*chord_mu_adaptive_mode:\s*"gap"\n)',
        r"\1" + gap_block,
        text,
        count=1,
    )

    # 8. Strip any v1/v2 latch params (legacy from velocity mode) - they're inert in gap mode but cleaner without
    for k in ("chord_mu_velocity_window", "chord_mu_velocity_target",
             "chord_mu_velocity_latch_threshold", "chord_mu_velocity_min_warmup_steps",
             "chord_mu_velocity_plateau_level_min", "chord_mu_velocity_latch_persist_steps",
             "chord_mu_d_floor", "chord_mu_d_ema_alpha"):
        text = re.sub(rf"^\s*{k}:.*\n", "", text, flags=re.MULTILINE)

    # 9. DR3 acceleration if requested
    if cfg["dr3_accel"]:
        text = re.sub(r"(\s*disc_lr:\s*)[\d.eE+-]+", r"\g<1>0.001", text)
        text = re.sub(r"(\s*disc_steps_per_call:\s*)\d+", r"\g<1>4", text)

    return text


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for cfg in CONFIGS:
        base_path = ROOT / cfg["base"]
        if not base_path.exists():
            print(f"  SKIP (missing): {cfg['key']} → {cfg['base']}")
            continue
        text = base_path.read_text()
        out = patch_yaml(text, cfg)
        out_path = OUT_DIR / f"{cfg['name']}.yaml"
        out_path.write_text(out)
        print(f"  OK: {cfg['key']:8s} → {out_path.relative_to(ROOT)}")
        print(f"        peak={cfg['peak']}  valley={cfg['valley']}  γ={cfg['decay_gamma']}  "
              f"dr3_accel={cfg['dr3_accel']}  env={cfg['env']}")


if __name__ == "__main__":
    main()
