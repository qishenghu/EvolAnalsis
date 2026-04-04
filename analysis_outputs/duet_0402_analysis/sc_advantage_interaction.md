# SC-GRPO Interaction Bug: Sign-Inverted Teacher Advantages

**Date**: 2026-04-02
**Severity**: Critical — actively harms training when SC + teacher mixing are both enabled
**Status**: Diagnosed, fix specified

## 1. The Bug

### Code Path
```
ae_ray_trainer.py:
  3248: token_level_rewards = token_level_scores          # task reward
  3375: token_level_rewards[on-policy] += SC_bonus         # SC trajectory bonus (on-policy only)
  3552: token_level_rewards[on-policy] += step_deltas      # SC step deltas (on-policy only)
  3588: compute_advantage(batch) → scores = token_level_rewards.sum()  ← INCLUDES SC!
```

### Effect
GRPO compares `score_onpolicy = task_reward + SC_bonus` vs `score_teacher = task_reward` (no SC).

When on-policy task_reward ≈ teacher_reward (both near 1.0 on WebShop):
```
on-policy score = 1.0 + 0.08 = 1.08
teacher score   = 1.0
all_mean        = 1.07
non_teacher_std = ~1e-7 (on-policy scores cluster at 1.08)
teacher_advantage = (1.0 - 1.07) / 1e-7 = -700,000
```

**The policy is trained to AVOID teacher behavior.**

### Why This Is Worse Than Std Collapse Alone

Even without std collapse, when on-policy + SC > teacher:
```
teacher_advantage = (1.0 - 1.07) / 0.1 = -0.7  (negative even with healthy std)
```

The SIGN is wrong, not just the magnitude. Teacher samples become negative examples
that GRPO pushes away from. This is the exact opposite of the intended behavior.

## 2. Interaction with Other Components

### DR3 Cannot Compensate
DR3 provides density ratio w_hat that corrects old_log_prob. But the PPO loss is:
```
loss = -advantage × clip(ratio, 1-ε, 1+ε)
```
If advantage is negative, the loss encourages DECREASING π(a_teacher|s) — moving
away from teacher actions. DR3's w_hat correction doesn't change the sign.

### Gap Gate Makes It Worse
Gap_gate computes: gate = (teacher_reward - on_policy_reward) / tau
When teacher_reward < on_policy_reward (because on-policy has SC bonus):
- gap is negative → gate clips to 0
- Teacher advantages are zeroed out entirely

But if gate_level=group and rewards are pre-SC, the gate may still be positive while
advantages are negative — inconsistent signals.

### Teacher Baseline Separation Doesn't Help
Config: `teacher_baseline: all_mean`
The all_mean includes on-policy's SC-inflated scores, making it higher than teacher_reward.
Even `teacher_baseline: non_teacher_mean` wouldn't help — non_teacher_mean is even higher.

## 3. Fix: Decouple SC from GRPO Advantage

### Principle
GRPO advantage should compare trajectories on equal footing (task reward only).
SC should influence learning through a separate channel (post-GRPO advantage adjustment).

### Implementation (Minimal Change)

**Before SC injection** (insert before line 3251):
```python
# Save task-only rewards for fair GRPO advantage computation
batch.batch["token_level_rewards_for_grpo"] = batch.batch["token_level_rewards"].clone()
```

**In compute_advantage** (modify line 3588 area):
```python
# Use task-only rewards for GRPO scoring
grpo_rewards = batch.batch.get("token_level_rewards_for_grpo", batch.batch["token_level_rewards"])
# ... pass grpo_rewards to compute_grpo_outcome_advantage_with_teacher_separation
```

**After GRPO advantage computation** (add after line 3597):
```python
# Re-inject SC bonus into advantages for on-policy samples
# This preserves SC's learning signal without contaminating teacher comparison
if _use_state_channel and "_sc_bonus" in batch.batch:
    sc_bonus_vals = batch.batch["_sc_bonus"]  # (bs,)
    is_teacher = (batch.batch.get("teacher_mask", torch.zeros(1)).sum(dim=-1) > 0)
    advantages = batch.batch["advantages"]
    resp_len = advantages.shape[-1]
    rmask = batch.batch.get("response_mask", torch.ones_like(advantages))
    if rmask.shape[-1] > resp_len:
        rmask = rmask[:, -resp_len:]

    for idx in range(sc_bonus_vals.shape[0]):
        if not is_teacher[idx] and sc_bonus_vals[idx].abs() > 1e-8:
            n_valid = rmask[idx].sum().clamp(min=1)
            advantages[idx] += (sc_bonus_vals[idx] / n_valid) * rmask[idx]
```

### Why Post-GRPO SC Injection Works

1. GRPO advantage uses task_reward → fair teacher vs on-policy comparison
2. SC bonus added post-GRPO → on-policy trajectories following expert states get extra positive advantage
3. Teacher advantages remain positive (teacher_reward > on-policy task_reward typically)
4. The SC bonus (≤0.2) is small relative to GRPO advantages (O(1)) → doesn't dominate
5. Step-level deltas still flow through token_level_rewards → they shape within-trajectory credit
   BUT they should also be excluded from GRPO scoring (same save/restore approach)

### Potential-Based Shaping Preservation

Ng et al. (1999): F(s,s') = γΦ(s') - Φ(s) preserves optimal policy when added to reward.

The decoupled version:
- Task advantage: (R_task - baseline) / std → trajectory-level signal (which trajectory is good)
- SC addition: β·P(τ) distributed across tokens → extra advantage for expert-state-following
- Step deltas: η·[Φ(s_{t+1})-Φ(s_t)] at step boundaries → within-trajectory credit

The shaping theorem holds because SC additions don't affect the GRPO normalization
constant (std), which was the source of the distortion.

## 4. Residual Issues After SC Decoupling

### Std Collapse (Teacher Adv Clip Needed)

Even with task-only rewards, std can collapse when on-policy rewards cluster:
```
task rewards: [0.72, 0.73, 0.71, 0.72, ...] → std ≈ 0.008
teacher_adv = (1.0 - 0.74) / 0.008 = 32.5
```

Teacher adv clip at C=5 handles this residual.

### Step-Level Deltas in GRPO Scoring

Step deltas (line 3552) also modify token_level_rewards before GRPO.
The `token_level_rewards_for_grpo` save should happen BEFORE both SC trajectory bonus
AND step-level deltas — at line 3248 (right after assignment from token_level_scores).

## 5. Priority Stack for DUET 0403

| Priority | Fix | Failure Mode | Impact |
|----------|-----|-------------|--------|
| P0 | **SC decouple from GRPO** | Sign inversion (teacher = negative example) | Critical |
| P1 | **Teacher adv clip (C=5)** | Magnitude explosion from std collapse | High |
| P2 | **gap_gate OFF** | Double suppression via DR3 × gap_gate | Medium |
| P3 | **disc_temperature=2.5** | Keep (validated in 0402) | Keep |

P0 without P1: Training goes in right direction but is noisy.
P1 without P0: Advantage clipped to -5 → still training AWAY from teacher.
**Both P0 and P1 are needed.**
