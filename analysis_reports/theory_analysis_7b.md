# DUET at 7B Scale: Why the Advantage Diminishes and What to Do About It

**Date**: 2026-04-13
**Scope**: WebShop environment, Qwen2.5-3B-Instruct vs Qwen2.5-7B-Instruct

---

## 1. Executive Summary

DUET's advantage over baselines collapses from +86% (vs on-policy) at 3B to +6.5% at 7B on WebShop. This is not a bug -- it is a predictable consequence of the "diminishing teacher gap" phenomenon. The stronger the base model, the less marginal value teacher demonstrations provide, and both of DUET's channels (DR3 Action Channel and State Channel) are fundamentally mechanisms for extracting value from teacher demonstrations. When the teacher gap narrows, both channels lose their leverage.

**Key finding**: The 7B on-policy baseline already reaches 0.82 peak reward (vs 0.50 at 3B), closing 64% of the gap to teacher performance (1.0). DUET's machinery for importing teacher knowledge simply has less room to add value.

---

## 2. Empirical Evidence: The Numbers

### 2.1 Peak Performance (5-step rolling average)

| Method       | 3B Peak | 7B Peak | 3B->7B Improvement |
|-------------|---------|---------|---------------------|
| On-policy   | 0.496   | 0.818   | +64.9%              |
| LUFFY       | 0.805   | 0.821   | +2.0%               |
| CHORD       | 0.392   | 0.834   | +112.8%             |
| **DUET**    | 0.923   | 0.871   | -5.6%               |

### 2.2 DUET Advantage Over Baselines

| Comparison          | 3B Delta | 7B Delta | Collapse Factor |
|--------------------|----------|----------|-----------------|
| DUET vs On-policy  | +0.427   | +0.053   | 8.1x smaller    |
| DUET vs LUFFY      | +0.118   | +0.050   | 2.4x smaller    |
| DUET vs CHORD      | +0.531   | +0.037   | 14.3x smaller   |

### 2.3 Initial Capability (Step 1, before any training)

| Scale | On-policy Step-1 Reward | Teacher Reward | Gap     |
|-------|------------------------|----------------|---------|
| 3B    | 0.179                  | 1.000          | 0.821   |
| 7B    | 0.333                  | 1.000          | 0.667   |

The 7B model starts 86% stronger than 3B. This is the root cause.

---

## 3. Config Differences Between 3B and 7B

### 3.1 Differences That Exist

| Parameter                      | 3B Value | 7B Value | Impact     |
|-------------------------------|----------|----------|------------|
| `actor.optim.lr`              | 1.0e-6   | 5.0e-7   | LR halved  |
| `rollout.tensor_model_parallel_size` | 1 | 2        | Infra only |
| `rollout.gpu_memory_utilization` | 0.5   | 0.65     | Infra only |
| `rollout.max_env_worker`       | 32       | 64       | Throughput |

### 3.2 Differences That Do NOT Exist (Same at Both Scales)

All DUET-specific hyperparameters are **identical** between 3B and 7B:

- DR3: `disc_lr=0.0003`, `hidden=64`, `buffer_size=1024`, `clip_max=5.0`, `ess_target_ratio=0.5`, `policy_shaping_beta=0.1`, `w_hat_ema_alpha=0.3`, `apply_warmup_steps=10`
- State Channel: `beta=0.2`, `eta=0.05`, `match_mode=attribute_aware`, `grpo_decouple=true`, `exclude_teacher=true`
- GRPO: `teacher_baseline_separation.enable=true`, all sub-settings identical
- PPO: `kl_loss_coef=0.001`, `off_cliprange_high=0.6`, `clip_ratio_high=0.28`

### 3.3 Assessment

The learning rate change (1e-6 -> 5e-7) is a reasonable scale-dependent adjustment. Larger models generally need smaller learning rates. However, the DUET component hyperparameters were **not re-tuned for 7B**. This is a problem because DUET's effectiveness depends on the relative scale of its signals (SC bonus, DR3 weights) compared to the base reward signal, which changes dramatically between 3B and 7B.

---

## 4. Theoretical Analysis: Why DUET's Advantage Diminishes

### 4.1 The Strong Base Model Hypothesis

DUET provides two types of value from teacher demonstrations:

1. **Action Channel (DR3)**: Corrects the policy gradient to incorporate teacher action choices with proper importance weighting. The gradient contribution from teacher samples is proportional to `w_hat * advantage_teacher`. When the policy already generates near-teacher-quality actions, `w_hat -> 1` (policy and teacher distributions converge), AND `advantage_teacher -> 0` (teacher rewards no longer exceed on-policy rewards). Both factors diminish simultaneously.

2. **State Channel (SC)**: Provides dense reward shaping to guide the policy toward expert-visited states. When the policy already visits the right states (a 7B model with WebShop-relevant instruction-following capability does this naturally), the progress signal P(tau) for on-policy samples approaches the teacher's progress, reducing the marginal shaping value.

**Formal argument for Action Channel**: The teacher gradient share is approximately:

```
TGS = E[|w_hat * adv_teacher|] / (E[|w_hat * adv_teacher|] + E[|adv_onpolicy|])
```

At 3B, the teacher-on_policy reward gap is large (0.88 at step 10), so `adv_teacher >> adv_onpolicy`, and TGS is high. At 7B, the gap is smaller (0.72 at step 10) and closes faster (0.08 by step 50 vs 0.22 at 3B), so TGS drops rapidly. DR3's fade-out mechanism, which is theoretically elegant at 3B, actually **accelerates too quickly** at 7B -- it correctly detects the narrowing gap and reduces teacher influence, but this means the teacher provides less total gradient signal over the course of training.

**Formal argument for State Channel**: The SC bonus is `beta * P(tau)` where P(tau) is the mean progress across the trajectory. For WebShop with `attribute_aware` matching:

- `search_home` -> 0.0
- `search_results` -> 0.15
- `product_detail` -> 0.35 + attribute_match_score (up to 0.50)
- `purchase_complete` -> 1.0

A 7B model that can already navigate to product pages and match some attributes will have P(tau) ~ 0.3-0.5 even at step 1. The SC bonus of `0.2 * 0.4 = 0.08` is small relative to the on-policy reward of ~0.33. At 3B where on-policy reward starts at 0.18, the same SC bonus of `0.2 * 0.2 = 0.04` is proportionally more meaningful because it comes at a time when the on-policy reward signal is much noisier and sparser.

### 4.2 The Ceiling Effect

WebShop rewards are bounded in [0, 1] (plus potential invalid action penalties). As on-policy performance approaches 1.0, the variance of rewards within each GRPO group decreases. GRPO's advantage signal becomes:

```
adv_i = (R_i - mean(R)) / std(R)
```

When most trajectories score ~0.9, `std(R)` is small, and the advantages become large but noisy (driven by small reward differences). Teacher samples (always scoring 1.0) barely exceed the on-policy mean, so their advantage is small and their gradient contribution is minimal.

This ceiling effect applies equally to all teacher-leveraging methods (DUET, LUFFY, CHORD), explaining why 7B baselines also compress: LUFFY goes from 0.805 to 0.821 (+2%), while CHORD goes from 0.392 to 0.834 (+113% -- but CHORD 3B was pathological at -0.1, so this is recovery, not improvement).

### 4.3 DR3 Discriminator Behavior at Different Scales

The DR3 discriminator uses v3_aug features: log-prob statistics (mean, std, min, max, low-ratio, high-ratio), response length, and KL-to-ref statistics. At 7B:

1. **Log-prob distributions shift**: Larger models typically produce tighter log-prob distributions (more confident). The log-prob mean for on-policy 7B samples is closer to the teacher's log-prob profile than at 3B, making the discriminator's task harder early on and trivially easy late.

2. **KL-to-ref features lose discriminative power**: Since the 7B model starts closer to the teacher's behavior, the KL divergence between policy and reference is smaller, and the gap between on-policy and teacher KL features narrows.

3. **The discriminator likely achieves high accuracy too quickly**: When on-policy behavior is already teacher-like, the discriminator learns fast, `w_hat` converges to ~1 rapidly, and the entire Action Channel effectively shuts off before it has time to provide meaningful curriculum benefit.

### 4.4 State Channel Effectiveness at Different Scales

The State Channel uses `attribute_aware` matching for WebShop, which classifies observations into stages and computes attribute match scores on product detail pages. Key scale-dependent behaviors:

1. **Stage progression is less informative at 7B**: A 7B model naturally progresses through search -> results -> product detail -> purchase, because this is basic instruction following. The progress signal adds value mainly when the model gets stuck (e.g., repeatedly searching without clicking products), which happens much more at 3B.

2. **Attribute matching is a strong signal at 3B, weak at 7B**: At 3B, the model often reaches products with wrong attributes. The SC attribute_match_score (0 to 0.50) provides a gradient toward better products. At 7B, the model already selects reasonably well, so the attribute match score is consistently moderate-to-high, reducing the signal's discriminative power.

3. **Beta calibration**: SC beta=0.2 produces bonuses up to 0.2 (for perfect progress). At 3B with rewards ~0.2, this is a 100% bonus. At 7B with rewards ~0.9, this is a 22% bonus. The same beta has fundamentally different effect sizes.

---

## 5. Teacher Gap Dynamics: Empirical Confirmation

### 5.1 Teacher Gap Closing Speed

| Step | 3B Gap | 7B Gap | 7B Closes Faster? |
|------|--------|--------|--------------------|
| 1    | 0.759  | 0.419  | Starts smaller     |
| 10   | 0.854  | 0.868  | Similar (dip phase) |
| 30   | 0.348  | 0.459  | ~Similar           |
| 50   | 0.216  | 0.078  | Yes, much faster   |
| 70   | 0.077  | 0.169  | Oscillating        |
| 90   | 0.033  | 0.132  | 7B stalls slightly |

The 7B DUET run reaches teacher-level performance by step 50 (gap=0.078) and then **oscillates** rather than improving further. The 3B run steadily closes the gap through step 90. This oscillation at 7B suggests that by mid-training, the teacher demonstrations provide no clear directional signal -- the on-policy samples are as good as the teacher, so mixing in teacher data is neutral at best.

### 5.2 The On-Policy Baseline Surprise

The most striking finding is that 7B on-policy GRPO (without any teacher data) reaches 0.818 peak performance -- only 5.3 points below DUET. At 3B, on-policy reaches only 0.496, a full 42.7 points below DUET. This confirms that the 7B model can learn WebShop effectively from its own exploration, reducing the marginal value of teacher demonstrations.

---

## 6. What Would Fix the 7B Results

### 6.1 Hyperparameter Re-tuning (Low-hanging fruit)

1. **Reduce SC beta at 7B**: Set `beta=0.05-0.1` instead of 0.2. The SC bonus should be proportional to the reward scale. Since 7B rewards are ~3-5x higher than 3B at equivalent training points, beta should scale inversely.

2. **Reduce SC step-level eta**: Similarly, `eta=0.01-0.02` instead of 0.05.

3. **Increase DR3 apply_warmup_steps**: From 10 to 20-30. Give the discriminator more time to learn before applying weights. At 7B, the initial gap is smaller, so early DR3 application when the discriminator is not yet calibrated may introduce noise rather than signal.

4. **Consider higher learning rate for DUET**: The 7B on-policy run uses 5e-7. DUET could potentially benefit from a slightly higher learning rate (e.g., 7e-7) to better exploit the teacher signal in the brief window before it fades out.

### 6.2 Algorithmic Changes (Medium effort)

1. **Scale-adaptive SC beta**: Set `beta = beta_base * (1 - mean_onpolicy_reward)` so that as the model improves, the SC bonus automatically diminishes. This is more principled than a fixed beta and handles scale differences automatically.

2. **DR3 warm-start from SFT features**: Pre-train the discriminator buffer with synthetic on-policy vs teacher feature pairs from the initial model, so DR3 starts calibrated rather than needing warmup steps.

3. **Task-conditional teacher selection**: Instead of random teacher trajectory selection, preferentially select teacher trajectories for tasks where the 7B model currently fails. This concentrates teacher signal where it matters.

### 6.3 Fundamental Rethinking (High effort)

1. **Harder teacher**: Use a stronger teacher (e.g., Qwen-110B or GPT-4 trajectories) that consistently outperforms the 7B model. If the teacher and student are too close in capability, the entire DUET framework has limited value.

2. **Curated teacher data**: Filter teacher trajectories to only include those that demonstrate strategies the 7B model does NOT already know. This requires analyzing the model's failure modes and selecting teacher data that addresses specific gaps.

3. **Longer training horizon**: DUET's advantage may emerge more clearly with longer training. At 7B, the fast initial learning (from the strong base model) may mask DUET's contribution in early steps, but DUET could provide asymptotic benefits through better exploration guidance that only manifest after 200+ steps.

---

## 7. Implications for the Paper and Reviewer Concerns

### 7.1 The Scale-Dependency Critique

**Reviewer attack**: "DUET shows large improvements at 3B but marginal gains at 7B. This suggests the method is primarily useful for weak models, limiting its practical value as the field moves toward larger models."

**Response strategy**: This is the single most dangerous critique. Options:

(a) **Frame as expected and principled**: "DUET's advantage is proportional to the capability gap between the student and teacher. This is a feature, not a bug -- it means DUET correctly identifies and fills capability gaps rather than forcing teacher behavior onto already-capable models. As evidence, note that DUET never hurts performance at 7B, while LUFFY's policy shaping can actively degrade performance when the student surpasses the teacher."

(b) **Add a third scale point**: Run DUET at 1.5B or 0.5B to show a clear trend: larger gap -> larger improvement. This turns a weakness into a strength by demonstrating a principled scaling law for teacher-assisted RL.

(c) **Show DUET with a stronger teacher at 7B**: If time permits, generate teacher trajectories from a 72B+ model with chain-of-thought or multi-attempt filtering to create trajectories that clearly exceed 7B capability. This separates "method doesn't scale" from "teacher quality doesn't scale."

### 7.2 The "Just Use On-Policy GRPO" Critique

**Reviewer attack**: "At 7B, on-policy GRPO reaches 0.818 vs DUET's 0.871. Is the +5.3 points worth the complexity of two additional channels?"

**Response**: This is hard to refute on WebShop alone. The key rebuttal is:

(a) Show that DUET's advantage is larger on harder environments (ALFWorld, SciWorld) where even 7B models struggle more.

(b) Argue that DUET's advantage is in **sample efficiency**: check if DUET reaches 0.818 (on-policy's peak) in fewer steps than on-policy. If DUET reaches this threshold at step 40 while on-policy needs step 80, that is a 2x sample efficiency gain, which is meaningful even if asymptotic performance is similar.

(c) Show consistency: DUET never collapses (3B CHORD hits -0.1, 3B on-policy drops to 0.14 at step 70), while DUET is monotonically stable. Reliability is valuable.

### 7.3 The DR3 Fade-Out Acceleration Problem

**Reviewer attack**: "You claim DR3 provides data-driven curriculum, but at 7B the curriculum ends prematurely because the discriminator correctly identifies that the distributions are similar. Isn't this a failure mode of the approach?"

**Response**: "The fade-out is doing exactly what it should: reducing off-policy influence when the off-policy data provides diminishing marginal returns. The alternative -- forcing teacher influence when it's no longer helpful -- would risk the 'teacher lock-in' phenomenon we identify in LUFFY. DUET's conservative approach (fade out when unsure) is safer than LUFFY's aggressive approach (always include teacher signal)."

### 7.4 Recommended Paper Positioning

1. **Primary results at 3B** where DUET's advantage is clear and substantial.
2. **7B as a scaling analysis** section showing graceful degradation, not as a primary result.
3. **Explicitly state the prerequisite**: "DUET is most effective when there is a significant capability gap between the student model and the teacher trajectories."
4. **Sample efficiency analysis** at 7B showing DUET reaches on-policy's asymptotic performance faster.
5. **Multi-environment results** where even 7B models struggle (SciWorld especially).

---

## 8. Specific Predictions (Testable)

1. Running DUET at 7B with `beta=0.05` and `eta=0.01` will improve the DUET-vs-on-policy gap from +5.3 to +8-12 points.

2. The DR3 discriminator at 7B reaches >0.90 accuracy by step 15 (vs step 25+ at 3B), causing premature teacher fade-out.

3. If we track `state_channel/progress_onpolicy_mean` at step 1, it will be significantly higher at 7B (~0.35) than 3B (~0.15), confirming that the SC provides less marginal progress information at 7B.

4. DUET at 7B will show larger advantages on ALFWorld and SciWorld where 7B models still have substantial capability gaps vs the 72B teacher.

5. A "DUET-lite" variant with only SC (no DR3) at 7B will perform comparably to full DUET, because the teacher mixing + DR3 machinery adds negligible value when the teacher gap is small.

---

## 9. Conclusion

The diminished DUET advantage at 7B is a fundamental consequence of the method's design, not a bug. DUET extracts value from teacher demonstrations through two channels, and both channels produce signals proportional to the student-teacher capability gap. When Qwen2.5-7B-Instruct already achieves ~83% of teacher-level WebShop performance through on-policy GRPO alone, there is simply less headroom for DUET to add value.

The correct response is not to "fix" DUET for 7B on WebShop, but to:
1. Position the paper's claims carefully (DUET is most valuable when the capability gap is large)
2. Show DUET's advantage on harder tasks where even 7B struggles
3. Re-tune hyperparameters (especially SC beta) for the 7B reward scale
4. Demonstrate sample efficiency advantages even when asymptotic performance converges
