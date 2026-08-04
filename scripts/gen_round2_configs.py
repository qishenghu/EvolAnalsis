#!/usr/bin/env python3
"""
Generate DUET WebShop 1.5B Round 2 configs (v10-v21).
All based on v8 (best variant: 0.574) with targeted changes.
"""
import yaml
import copy
import os

CONFIG_DIR = "/data/home/qisheng/EvolAnalsis/config/duet_paper_experiments_configs/webshop"
TEMPLATE = os.path.join(CONFIG_DIR, "webshop_qwen1.5b_duet_v8.yaml")

with open(TEMPLATE) as f:
    base = yaml.safe_load(f)

def make_variant(base_cfg, version, description, changes):
    """Create a variant config with specified changes."""
    cfg = copy.deepcopy(base_cfg)
    name = f"webshop_qwen1.5b_duet_v{version}"
    cfg["trainer"]["experiment_name"] = name
    cfg["exp_manager"]["reme"]["workspace_id"] = name

    # Apply changes
    for path, value in changes:
        parts = path.split(".")
        obj = cfg
        for p in parts[:-1]:
            obj = obj[p]
        obj[parts[-1]] = value

    # Write
    out_path = os.path.join(CONFIG_DIR, f"{name}.yaml")
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"  Created v{version}: {description}")
    print(f"    -> {out_path}")
    return out_path


print("Generating Round 2 configs from v8 base...\n")

# ============================================================================
# v10: Softer DR3 discriminator (disc_temperature 1.0 -> 1.5)
# Rationale: ALFWorld uses 1.5 and works well. Higher temperature = softer
# discriminator probabilities = more gradual teacher weighting. At 1.5B scale
# with a large expert-policy gap, softer discrimination may prevent the
# discriminator from becoming too sharp too early, which would cut off teacher
# signal prematurely. The data shows v8 teacher_gradient_share may fade too
# fast (teacher_adv_sample_mean drops from 0.32 to 0.13 by step 100).
# ============================================================================
make_variant(base, 10,
    "disc_temperature 1.0->1.5 (softer DR3, slower teacher fade)",
    [("actor_rollout_ref.actor.dr3.disc_temperature", 1.5)])

# ============================================================================
# v11: Lower clip_max (5.0 -> 2.0)
# Rationale: clip_max bounds the importance weight w_hat. At 1.5B scale the
# policy is far from the expert, so some trajectories get very high w_hat.
# Reducing clip_max to 2.0 reduces variance of the importance-weighted
# gradient. This is the standard bias-variance tradeoff in IS: we accept
# slight bias for much lower variance. CHORD doesn't have this variance
# issue at all (SFT loss is deterministic), which may be part of why it's
# more stable.
# ============================================================================
make_variant(base, 11,
    "clip_max 5.0->2.0 (reduce IS variance)",
    [("actor_rollout_ref.actor.dr3.clip_max", 2.0)])

# ============================================================================
# v12: Softer DR3 + lower clip_max (combine v10 + v11)
# Rationale: These two changes are theoretically complementary. Softer
# discriminator prevents premature teacher fade-out; lower clip_max
# reduces gradient variance. Together they should make DR3 more stable
# and keep teacher signal useful longer.
# ============================================================================
make_variant(base, 12,
    "disc_temp=1.5 + clip_max=2.0 (combined DR3 stabilization)",
    [("actor_rollout_ref.actor.dr3.disc_temperature", 1.5),
     ("actor_rollout_ref.actor.dr3.clip_max", 2.0)])

# ============================================================================
# v13: Mild KL + softer DR3
# Rationale: v3 showed KL=0.01 is too much, but v8 at 0.001 may allow
# too much policy drift in late training (we see mild overfitting at steps
# 88+). A mild increase to 0.003 plus softer DR3 could stabilize late
# training without the early-phase suppression that killed v3.
# ============================================================================
make_variant(base, 13,
    "KL=0.003 + disc_temp=1.5 (late-training stabilization)",
    [("actor_rollout_ref.actor.kl_loss_coef", 0.003),
     ("actor_rollout_ref.actor.dr3.disc_temperature", 1.5)])

# ============================================================================
# v14: SC beta 0.2 -> 0.15 (without step deltas, lower beta might work)
# Rationale: v2 showed beta=0.1 hurt (0.521), but v2 had step_deltas ON.
# The total SC injection in v2 was beta*P + eta*delta ~ 0.1*0.4 + 0.05*0.5
# = ~0.065. In v8 (no deltas), SC is only beta*P ~ 0.2*0.4 = ~0.08.
# Trying 0.15 gives ~0.06, which is similar to v2's trajectory component.
# The analysis shows v8's SC bonus ratio is ~0.13 at step 100, borderline
# high. Reducing to 0.15 keeps the directional signal but reduces the
# reward inflation.
# ============================================================================
make_variant(base, 14,
    "SC beta=0.15 (slightly lower SC, no step deltas)",
    [("exp_manager.state_channel.beta", 0.15)])

# ============================================================================
# v15: SC beta decay + softer DR3
# Rationale: ALFWorld uses beta_decay=True and works well. The idea is to
# let SC guide exploration early (high beta), then let the task reward
# dominate late (low beta). Combined with softer DR3, this creates a
# well-shaped curriculum: strong teacher signal + strong SC early, then
# natural fade of both.
# ============================================================================
make_variant(base, 15,
    "beta_decay=true + disc_temp=1.5 (dual curriculum)",
    [("exp_manager.state_channel.beta_decay", True),
     ("actor_rollout_ref.actor.dr3.disc_temperature", 1.5)])

# ============================================================================
# v16: Gap gate + softer DR3
# Rationale: ALFWorld uses gap_gate_enable=True (WebShop v8 has it false).
# Gap gate modulates DR3's influence based on the reward gap between teacher
# and on-policy. When the gap is small (policy is close to teacher on a
# task), DR3 reduces its influence on that task. This is more targeted
# than the global discriminator temperature. Combined with disc_temp=1.5,
# this could provide per-task-adaptive teacher fading.
# ============================================================================
make_variant(base, 16,
    "gap_gate=true + disc_temp=1.5 (per-task adaptive DR3)",
    [("actor_rollout_ref.actor.dr3.gap_gate_enable", True),
     ("actor_rollout_ref.actor.dr3.disc_temperature", 1.5)])

# ============================================================================
# v17: Lower policy_shaping_beta (0.1 -> 0.05)
# Rationale: policy_shaping_beta controls how much the density ratio
# modifies the teacher's effective learning rate. At 1.5B scale, the
# policy is very far from the expert, so density ratios can be extreme.
# A lower shaping beta means more aggressive down-weighting of teacher
# samples whose trajectories are implausible under the current policy.
# This could reduce harmful gradient signal from "impossible" teacher
# actions at 1.5B scale where the model can't yet reproduce expert behavior.
# ============================================================================
make_variant(base, 17,
    "policy_shaping_beta=0.05 (more aggressive DR3 down-weighting)",
    [("actor_rollout_ref.actor.dr3.policy_shaping_beta", 0.05)])

# ============================================================================
# v18: Temperature 0.7 (more exploration)
# Rationale: v8 uses temp=0.6. Higher temperature means more diverse
# rollouts, which (1) increases on-policy coverage of expert states
# (helping SC), and (2) provides more diverse gradient signal. The risk
# is more noise, but v8's high adv_pos_ratio (0.84 at step 100) suggests
# the policy may benefit from more exploration. CHORD also uses temp=0.6,
# so this is a differentiation axis.
# ============================================================================
make_variant(base, 18,
    "temperature=0.7 (more exploration for SC coverage)",
    [("actor_rollout_ref.rollout.temperature", 0.7)])

# ============================================================================
# v19: The "kitchen sink" config combining best hypotheses
# disc_temp=1.5 + clip_max=3.0 + beta_decay=true + KL=0.002
# Rationale: Each change addresses a specific failure mode:
# - disc_temp=1.5: prevent premature teacher fade
# - clip_max=3.0: reduce IS variance (moderate, not as aggressive as 2.0)
# - beta_decay: natural SC curriculum
# - KL=0.002: mild late-training stabilization
# Together they create a "stable learning" profile that should maintain
# teacher signal longer while preventing late-training divergence.
# ============================================================================
make_variant(base, 19,
    "disc_temp=1.5 + clip=3.0 + decay + KL=0.002 (stable combo)",
    [("actor_rollout_ref.actor.dr3.disc_temperature", 1.5),
     ("actor_rollout_ref.actor.dr3.clip_max", 3.0),
     ("exp_manager.state_channel.beta_decay", True),
     ("actor_rollout_ref.actor.kl_loss_coef", 0.002)])

# ============================================================================
# v20: Warmup 10->20 + disc_temp=1.5
# Rationale: DR3 apply_warmup_steps=10 means the discriminator starts
# affecting training at step 11. At 1.5B scale, the discriminator may not
# be well-calibrated yet at step 11 (it has seen only 10*8=80 trajectory
# pairs). Extending warmup to 20 gives the discriminator 160 pairs to
# learn from before affecting training. This prevents bad early-stage
# importance weights from corrupting the policy. Combined with softer
# discrimination.
# ============================================================================
make_variant(base, 20,
    "warmup=20 + disc_temp=1.5 (more DR3 calibration time)",
    [("actor_rollout_ref.actor.dr3.apply_warmup_steps", 20),
     ("actor_rollout_ref.actor.dr3.disc_temperature", 1.5)])

# ============================================================================
# v21: grpo_decouple=false + disc_temp=1.5
# Rationale: grpo_decouple=true means SC bonus is NOT included in GRPO's
# advantage computation, only added separately. When false, SC bonus
# inflates the reward used for GRPO normalization. This creates stronger
# advantage signals for trajectories that match expert states. The risk
# is that it distorts the relative ranking of trajectories, but with
# beta=0.2 and no step deltas, the distortion is mild. Worth testing
# against the decoupled version, especially since CHORD's SFT loss
# provides a similarly "coupled" signal.
# ============================================================================
make_variant(base, 21,
    "grpo_decouple=false + disc_temp=1.5 (coupled SC in GRPO)",
    [("exp_manager.state_channel.grpo_decouple", False),
     ("actor_rollout_ref.actor.dr3.disc_temperature", 1.5)])

print("\nDone. 12 configs generated (v10-v21).")
