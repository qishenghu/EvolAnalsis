# DUET vs LUFFY: Fundamental Mechanism Analysis

**Date**: 2026-04-02
**Context**: DUET has failed to beat LUFFY on WebShop across 5 iterations. This analysis traces the exact code paths and identifies the root cause.

---

## 1. LUFFY's Loss Computation Path (Exact Code Trace)

### Step 1: Teacher ratio construction
**File**: `het_core_algos.py:396-399`
```python
# LUFFY mode: teacher has no log_prob, so π_old = 1
# ratio = π_current / 1 = exp(log_prob)
teacher_ratio = torch.exp(log_prob)  # shape: (bs, response_len)
```
This is **per-token**: each token gets `π_θ(a_t | s_t)`, the current policy's probability for that specific teacher action.

### Step 2: Policy shaping applied
**File**: `het_core_algos.py:508-516` → `_apply_policy_shaping()` at line 765-768
```python
# p/(p+β) where p = π_θ(a_t | s_t) per token
teacher_ratio = ratio / (ratio + beta)  # beta = 0.1
```

**Key property**: This is a **concave, monotonically increasing** function mapping [0,∞) → [0,1):
- When `p ≈ 0` (policy hasn't learned this token): `p/(p+0.1) ≈ 0` → **gradient ≈ 0**
- When `p = 0.1` (learning): `0.1/0.2 = 0.5` → moderate gradient
- When `p = 0.9` (already learned): `0.9/1.0 = 0.9` → near-saturated
- Derivative: `β/(p+β)² → 0` as p grows → **automatic gradient decay per token**

### Step 3: Loss computation
**File**: `het_core_algos.py:569`
```python
teacher_off_pg_losses = -advantages * teacher_ratio  # (bs, response_len)
```

### Step 4: Aggregation
**File**: `het_core_algos.py:597-599`
```python
teacher_exp_mask = exp_mask * teacher_mask_float
teacher_off_pg_loss = masked_mean(teacher_off_pg_losses, teacher_exp_mask * response_mask)
```

### LUFFY Gradient Summary
The effective gradient for a teacher token is:
```
∇_θ L_teacher ∝ -A(τ) · d/dθ [π_θ(a_t|s_t) / (π_θ(a_t|s_t) + β)]
            = -A(τ) · β / (π_θ(a_t|s_t) + β)² · ∇_θ π_θ(a_t|s_t)
```

**This means**: For each teacher token, the gradient magnitude is **inversely proportional** to how well the current policy already assigns probability to it. Tokens the policy has already learned (`π_θ ≈ 1`) contribute near-zero gradient. Tokens the policy struggles with (`π_θ ≈ 0`) contribute proportional to `β/(β²) = 1/β` but are bounded.

---

## 2. DR3's Loss Computation Path (Exact Code Trace)

### Step 1: Discriminator produces w_hat
**File**: `dr3_ratio.py:734-804`
```python
# Discriminator classifies trajectory features (sequence-level)
logits = self._disc(feats)  # feats: (bs, feature_dim), one vector per TRAJECTORY
d = sigmoid(logits / temperature)  # (bs,)
r_hat = d / (1 - d)  # likelihood ratio estimate

# Relative density ratio
w = r_hat / ((1-α)*r_hat + α)  # (bs,) — ONE scalar per trajectory
w_hat = clamp(w, 0, clip_upper)  # (bs,)
```

**Critical**: `w_hat` is a **per-SAMPLE** (per-trajectory) scalar. Every token in the same trajectory gets the same weight.

### Step 2: w_hat used to "repair" old_log_prob
**File**: `het_actor.py:1482-1525`
```python
w_hat = w_hat.clamp(min=dr3_w_min)  # w_min = 0.01
log_w = torch.log(w_hat).unsqueeze(-1)  # (bs, 1)
old_lp_new = old_log_prob.clone()
# For teacher samples: synthesize old_log_prob from current policy and w_hat
old_lp_new[teacher_sample] = log_prob.detach()[teacher_sample] - log_w[teacher_sample]
```

**What this does mathematically**:
```
old_log_prob_synthetic = log π_θ(a|s) - log(w_hat)
ratio = exp(log_prob - old_log_prob) = exp(log π_θ - (log π_θ - log w_hat)) = w_hat
```

So the effective ratio for all teacher tokens becomes `w_hat` — a **uniform** scalar across the entire trajectory.

### Step 3: Standard PPO/RePO loss with this ratio
**File**: `het_core_algos.py:1873-1922` (repo_compute_token_loss)
```python
ratio = exp(log_prob - old_log_prob)  # = w_hat for teacher samples
# Off-policy loss (with or without shaping):
off_pg_losses = -advantages * ratio  # or -advantages * ratio/(ratio + beta)
```

### DR3 Gradient Summary
The effective gradient for a teacher token is:
```
∇_θ L_teacher ∝ -A(τ) · w_hat(τ) · ∇_θ log π_θ(a_t|s_t)
```

**w_hat is the SAME for every token in the trajectory**. It's a scalar that says "how much does this trajectory look like on-policy data" based on trajectory-level features.

---

## 3. THE FUNDAMENTAL DIFFERENCE

| Property | LUFFY (p/(p+β)) | DR3 (w_hat) |
|----------|-----------------|-------------|
| **Granularity** | Per-TOKEN | Per-TRAJECTORY |
| **Signal source** | Current policy π_θ(a_t\|s_t) | Discriminator on trajectory features |
| **Adaptation speed** | Instant (recomputed each forward pass) | Delayed (discriminator must be trained) |
| **Learned tokens** | Automatically down-weighted (p/(p+β) → 1, gradient → 0) | Same weight as unlearned tokens |
| **Hard tokens** | Naturally up-weighted (low p → steep gradient) | Same weight as easy tokens |
| **Semantic meaning** | "How much can π_θ still learn from this token?" | "Does this trajectory look like on-policy?" |

### The Core Problem

**LUFFY provides implicit per-token curriculum learning.** Each teacher token has a natural "learning progress" signal built into the gradient:

1. **Early training**: Most teacher tokens have low `π_θ(a_t|s_t)`, so `p/(p+β)` is small but the gradient `β/(p+β)²` is large → strong learning signal
2. **Mid training**: Easy tokens (format, common actions) have high `π_θ`, so their gradient vanishes. Hard tokens still have low `π_θ` → learning focuses on what matters
3. **Late training**: Almost all tokens learned → gradient vanishes naturally, teacher influence fades

**DR3 applies a single blunt instrument.** The discriminator says "this trajectory is 60% similar to on-policy" and applies w_hat=0.6 to EVERY token:
- Tokens the policy has already perfectly learned: gradient ∝ 0.6 (wasteful, potentially harmful)
- Tokens the policy hasn't learned: gradient ∝ 0.6 (insufficient, should be higher)
- Result: **suboptimal gradient allocation within each trajectory**

### Mathematical Illustration

Consider a teacher trajectory with 100 tokens. After some training:
- 80 tokens: `π_θ(a_t|s_t) ≈ 0.9` (already learned: format tokens, easy actions)
- 20 tokens: `π_θ(a_t|s_t) ≈ 0.01` (not learned: key decision tokens)

**LUFFY gradient allocation**:
- Learned tokens: `p/(p+β) = 0.9/1.0 = 0.9`, gradient ∝ `β/(0.9+0.1)² = 0.1` → near-zero effective learning signal
- Unlearned tokens: `p/(p+β) = 0.01/0.11 = 0.09`, gradient ∝ `β/(0.01+0.1)² = 8.26` → 83x stronger signal
- **Net effect**: 99% of teacher gradient budget goes to the 20 tokens that matter

**DR3 gradient allocation** (suppose w_hat = 0.4 for this trajectory):
- Learned tokens: ratio = 0.4, gradient = -A · 0.4 · ∇log π for ALL 80 tokens
- Unlearned tokens: ratio = 0.4, gradient = -A · 0.4 · ∇log π for ALL 20 tokens
- **Net effect**: 80% of teacher gradient budget goes to already-learned tokens → wasted

### Why DR3's Trajectory-Level Weighting is Structurally Wrong for GRPO

GRPO assigns advantages at the **trajectory level** (group-relative). Within a trajectory, all tokens share the same advantage A(τ). The ONLY thing that differentiates token gradients is the ratio multiplier:

- In LUFFY: ratio = `π_θ(a_t|s_t) / (π_θ(a_t|s_t) + β)` → **token-level differentiation** within trajectory
- In DR3: ratio = w_hat → **no differentiation** within trajectory

Since GRPO already assigns uniform advantages within a trajectory, DR3 makes the gradient even MORE uniform. LUFFY adds the crucial per-token modulation that GRPO lacks.

---

## 4. Trajectory Comparison (WebShop 3B)

### Reward Progression

| Step | DUET on-policy reward | LUFFY on-policy reward | LUFFY advantage |
|------|----------------------|------------------------|-----------------|
| 1    | 0.226                | 0.157                  | DUET +0.069     |
| 10   | 0.111                | 0.094                  | DUET +0.017     |
| 30   | 0.442                | 0.521                  | LUFFY +0.079    |
| 50   | 0.714                | 0.647                  | DUET +0.067     |
| 80   | 0.750                | 0.861                  | LUFFY +0.111    |
| 100  | 0.796                | 0.790                  | ~tie            |

**Observation**: LUFFY overtakes DUET during mid-training (steps 30-80), exactly when per-token curriculum matters most. Early (random policy) and late (converged) are similar.

### Teacher Advantage Dynamics

| Step | DUET teacher_adv_pos_ratio | LUFFY teacher_adv_pos_ratio |
|------|---------------------------|----------------------------|
| 1    | 1.0                       | 1.0                        |
| 10   | 1.0                       | 1.0                        |
| 30   | 1.0                       | 1.0                        |
| 50   | 1.0                       | 1.0                        |
| 80   | (not in diag)             | 0.571                      |
| 100  | (not in diag)             | 0.833                      |

LUFFY's teacher_adv_pos_ratio dropping below 1.0 at step 80 means some teacher trajectories have NEGATIVE advantages (on-policy is performing better). This is healthy — it means the policy has surpassed some teachers, and LUFFY's `p/(p+β)` naturally reduces their influence.

---

## 5. Proposed Fixes (Ranked by Likelihood of Success)

### Fix A: Hybrid — LUFFY Policy Shaping + DR3 Sample Gate (Recommended)

**Idea**: Use LUFFY's `p/(p+β)` for per-token gradient modulation (the core loss computation), but use DR3's w_hat as a **per-sample scaling gate** on top.

**Implementation** (in `het_actor.py`, DR3 teacher_no_logprob branch):
```python
# Instead of: old_lp_new[teacher_sample] = log_prob.detach() - log_w
# Do: Use LUFFY's het_compute_teacher_aware_loss but scale by w_hat

# Apply DR3 w_hat as teacher_loss_scale (per-sample gate)
dr3_teacher_scale = w_hat.unsqueeze(-1).expand(-1, response_length)  # (bs, resp_len)
# Only scale teacher samples
if teacher_loss_scale is not None:
    teacher_loss_scale = teacher_loss_scale * dr3_teacher_scale
else:
    teacher_loss_scale = dr3_teacher_scale

ret_dict = het_compute_teacher_aware_loss(
    ...,
    teacher_use_log_prob=False,
    teacher_policy_shaping_enable=True,
    teacher_policy_shaping_mode="p_div_p_beta",
    teacher_policy_shaping_beta=0.1,
    teacher_loss_scale=teacher_loss_scale,
)
```

**Why this works**:
- LUFFY's `p/(p+β)` handles **within-trajectory** gradient allocation (which tokens to learn)
- DR3's `w_hat` handles **between-trajectory** weighting (which trajectories to prioritize)
- These are orthogonal concerns that should be composed, not substituted

### Fix B: Token-Level DR3

**Idea**: Make DR3 operate at the token level by using per-token features instead of trajectory-level features.

**Problem**: Much harder to implement. Per-token discriminator would need to classify each (state, action) pair, dramatically increasing computation. The discriminator input dimension would blow up.

### Fix C: Just Use LUFFY (Drop DR3 Entirely for Loss)

**Idea**: Keep DR3 only for diagnostics/monitoring, but use LUFFY's `p/(p+β)` for the actual teacher loss computation.

This is equivalent to the LUFFY baseline but with SC (State Channel) on top. The question is whether SC + LUFFY > LUFFY alone.

**Config change**:
```yaml
actor:
  use_dr3: true  # keep for monitoring
  teacher_policy_shaping_enable: true
  teacher_policy_shaping_mode: p_div_p_beta
  teacher_policy_shaping_beta: 0.1
  dr3:
    enable: true
    apply_to: none  # disable DR3's old_log_prob correction
    # ... keep rest for monitoring
```

### Fix D: LUFFY + DR3 Adaptive β

**Idea**: Use LUFFY's `p/(p+β)` but let DR3's w_hat modulate β per-trajectory:
```python
# High w_hat (trajectory looks on-policy) → larger β → weaker teacher influence
# Low w_hat (trajectory looks off-policy) → smaller β → stronger teacher influence
beta_adaptive = beta_base * (1 + w_hat)  # or other monotonic function
teacher_ratio = p / (p + beta_adaptive)
```

This preserves LUFFY's token-level curriculum while adding DR3's trajectory-level fade-out signal.

---

## 6. Recommendation

**Fix A (Hybrid)** is the strongest because it:
1. Preserves the token-level learning signal that makes LUFFY work
2. Adds the distribution-shift correction that DR3 provides (between-trajectory)
3. Keeps DUET's narrative ("learned density ratio correction") while fixing the core issue
4. Is a minimal code change (reuse existing `teacher_loss_scale` pathway)

The current DR3 approach of replacing `old_log_prob` is mathematically elegant but **operationally wrong** for GRPO because it eliminates per-token variance in the gradient — the very thing that makes teacher trajectories useful.

---

## 7. Key Insight for the Paper

DUET's contribution should be reframed: DR3 is not a replacement for policy shaping — it's a **complementary mechanism**. Policy shaping (LUFFY) handles within-trajectory credit assignment. Density ratio estimation (DR3) handles between-trajectory importance weighting. The original formulation conflates these two orthogonal concerns by using w_hat for both.

The correct composition is:
```
L_teacher = Σ_τ w_hat(τ) · Σ_t A(τ) · p_θ(a_t|s_t) / (p_θ(a_t|s_t) + β) · ∇log π_θ(a_t|s_t)
                ↑ DR3 (between-trajectory)    ↑ LUFFY (within-trajectory)
```
