# DUET Unified DR3 — Implementation Plans (5 Specs + Debug)

Target: drop the separate BC/SFT term. Make DR3 do both channels of the "Action" work (importance correction AND per-token teacher-fit signal). If unification fails, make mu adaptive without calibration.

Audience: implementer. All file paths absolute, all line numbers reference current HEAD.

---

## 0. Ground truth on current code

### DR3 discriminator (trajectory level, confirmed)

File: `/data/home/qisheng/EvolAnalsis/agentevolver/module/exp_manager/dr3_ratio.py`

- `compute_sequence_features()` at L73 pools per-sequence stats (log_prob mean/std/min/max, adv_abs_mean, len, KL-to-ref) into a single vector per trajectory (L86-226).
- `DR3Discriminator` at L229 is an MLP over that per-sequence feature vector.
- `DR3RatioEstimator.step()` at L577 returns `w_hat` of shape `(bs,)` — one scalar per trajectory.
- `w_hat` then shifts `old_log_prob` uniformly across response tokens: `het_actor.py:1496-1500` —
  `old_lp_new[apply_mask] = log_prob.detach()[apply_mask] - log_w[apply_mask]` where `log_w` is broadcast from shape `(bs,1)`.
- Hidden-feature mode (`_dr3_pooled_hidden`, `het_actor.py:288-308`) already extracts pooled last-layer hidden state per sequence — token-level hidden states exist but are currently pooled away.

### BC / CHORD SFT term (what we want to eliminate)

File: `/data/home/qisheng/EvolAnalsis/agentevolver/module/exp_manager/het_core_algos.py`

- `compute_chord_sft_loss()` at L1723: `-phi(p) * log_prob` masked by `exp_mask`, `phi(p)=p*(1-p)`.
- `chord_mu_scheduler()` at L1627: cosine warmup/decay over global_step.
- Called in `het_actor.py:1731` and merged at L1785 — DR3+CHORD mode: `pg_loss = grpo_loss + mu * sft_loss`.
- Adaptive mu at `het_actor.py:1748-1772` uses `V_A = std(per_sample_mean_abs_advantage)` — v37's bug (target 0.035 is ~20-50x smaller than observed 0.7-3.0).

Key insight: the CHORD SFT term is exactly `-mu * log pi_theta(a_teacher | s)`. This is the expert-NLL gradient. The current DR3 teacher PG has gradient proportional to `w_hat * A * d log pi`. They differ only in (a) sign flip because A is group-relative not ≥0; (b) per-token weighting. If we can make DR3 deliver a **positive, token-level "imitate this token harder when policy under-fits it"** signal, we subsume BC.

---

## Spec 1: Token-level discriminator DR3

**Goal**: replace `w_hat(τ)` with `w_hat(s,a)` computed per token. Higher `w_hat` on tokens where the teacher distribution dominates the mixture (i.e. tokens policy has not yet learned). Natural fade-out: once policy matches teacher at that token, `D → 0.5` and `w → 1`.

**Feature construction (per token)**:
- Existing: `log_prob[b, t]` (current policy), `ref_log_prob[b, t]` if KL enabled, token position, response length.
- Proposed addition: pooled last-layer hidden `h[b, t, :H]` (already partially computed; must stop the pooling-over-t reduction).
- Per-token feature vector: `[log_prob, log_prob - ref_log_prob, 1/resp_len, one-hot(pos_bucket=4), pooled_h]`. If `feature_mode=v3`, ~7 stat dims + projected hidden.

**Discriminator input/output**:
- Input: `(B, T, F)` flattened to `(B*T, F)` with response mask.
- Output: `logits(B, T)` → `D(B, T)` → `w(B, T) = D / (1 - D) / (1 - alpha)` with dual clip.

**Training** (token-level paired classification):
- Label: teacher token = 0, on-policy token = 1. Use class-balanced sampling (already exists at L544-552 in `dr3_ratio.py`, extend to token axis).
- Buffer now holds per-token features, size `buffer_size * avg_resp_len`. Need to shrink `buffer_size` from 2048 → 256 or cap samples per sequence (random subsample 32 tokens per trajectory) to avoid memory blow-up.

**Exact code patch (sketch)**:

1. `dr3_ratio.py` — add a new function `compute_token_features(log_prob, ref_log_prob, response_mask, hidden_tokens=None) -> (bs, T, F)` next to existing `compute_sequence_features()`. ~40 LOC.
2. `dr3_ratio.py:229` `DR3Discriminator.forward()`: already token-agnostic (just an MLP); only callers need to reshape.
3. `dr3_ratio.py:577` `DR3RatioEstimator.step()`: add parameter `token_mode: bool = False`. When true:
   - Flatten features `(bs, T, F) -> (bs*T, F)` and labels `(bs, T) -> (bs*T,)` masked by `response_mask`.
   - Push flattened into buffer (same buffer, unaware of shape).
   - Return `w` reshaped to `(bs, T)`.
4. `het_actor.py:1282-1299` — when `dr3_cfg.get("token_level", False)` is true, call with new mode; pipe `_dr3_pooled_hidden_tokens` (currently pooled-reduced; keep unreduced path). ~60 LOC.
5. `het_actor.py:1496-1500` — remove the `.unsqueeze(-1)` broadcast; use `log_w` as `(bs, T)` directly. Same for `clamp(min=dr3_w_min)` already shape-agnostic.

**LOC**: ~150 net (including parallel sequence path for backward compat), **risk**: medium — buffer size blow-up, FSDP gather over `(bs*T)` tensors needs reshape care at L658-687 in `dr3_ratio.py`. Hidden-state pooling change must not break v5_hidden mode.

**Backward compat**: yes — `dr3.token_level: false` (default) keeps current sequence-level path. v24 runs unchanged.

**Config template** (new file `/data/home/qisheng/EvolAnalsis/config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet_v38.yaml`):
```yaml
# Inherit v24 fields. Override:
use_chord: false     # <-- drop BC term entirely
dr3:
  enable: true
  token_level: true
  feature_mode: v3
  disc_hidden: 64
  disc_steps_per_call: 1
  buffer_size: 4096    # holds tokens, not trajectories
  train_batch_size: 256
  token_subsample_per_seq: 32  # cap memory
  w_min: 0.01
  clip_max: 10.0
```

**Validation**: run `webshop_qwen1.5b_duet_v38.yaml` for 30 steps. Success metrics:
- `critic/success_onpolicy/mean` within 5% of v24 at step 30.
- `dr3/w_off_mean` has per-token std > 0.2 (proves w varies across tokens, unlike v24 where w is per-traj).
- `actor/kl_loss` < 0.5 (policy stable).

Failure modes to watch: discriminator overfits to token position (always predicts from `t/resp_len`); mitigate with label smoothing 0.1 (already supported via `disc_label_smoothing`).

---

## Spec 2: Surprise-weighted DR3 (cheapest)

**Goal**: keep trajectory-level discriminator; multiply per-token teacher-ratio by token surprise `(1 - pi_theta(a|s))` so unfinished teacher tokens get more weight and converged tokens get ~0 weight. This is mathematically equivalent to applying `phi(p) = 1 - p` on the PG ratio — half of CHORD's Bernoulli variance, without the explicit SFT term.

**Exact code patch**:

`het_core_algos.py` — inside `het_compute_teacher_aware_loss`, at L509-521 (after policy shaping, before loss computation):

```python
# NEW: surprise-weighted token coefficient (subsumes BC)
teacher_surprise_enable = kwargs.get("teacher_surprise_enable", False)
if teacher_surprise_enable:
    # token_surprise = (1 - p_teacher_at_current_policy).detach()
    # high surprise -> policy hasn't learned this token yet -> amplify grad
    p_curr = torch.exp(log_prob.clamp(max=0))      # (bs, resp_len), in (0,1]
    surprise = (1.0 - p_curr).detach()              # (bs, resp_len)
    # clamp to avoid zero-gradient at converged tokens (optional)
    surprise_min = float(kwargs.get("teacher_surprise_min", 0.0))
    surprise = surprise.clamp(min=surprise_min)
    # apply only on teacher tokens (via teacher_mask_float); on_ratio untouched
    teacher_ratio = teacher_ratio * (teacher_mask_float * surprise + (1.0 - teacher_mask_float))
```

Then at `het_actor.py:1700-1805`, the CHORD branch can be skipped entirely if `teacher_surprise_enable=true` and `use_chord=false`.

**LOC**: ~12. **Risk**: low.

**PPO clip interpretation**: yes preserved. `teacher_ratio` already goes into `-advantages * teacher_ratio` then optional clip at L577. Surprise is detached so ratio clipping still makes sense: the clip acts on `(pi/pi_old) * surprise` which lies in `[0, ratio_max]`. Since surprise ≤ 1, clip thresholds remain conservative (bound only shifts down, never up).

**Surprise outside [0, 1]**: `p = exp(log_prob.clamp(max=0))` ∈ (0, 1], so `1 - p ∈ [0, 1)`. Safe.

**Backward compat**: `teacher_surprise_enable: false` is default → identical behavior to v24.

**Config template** (v39):
```yaml
# Based on v24 but drop CHORD
use_chord: false
teacher_surprise_enable: true
teacher_surprise_min: 0.01  # so even learned tokens get tiny grad; prevents sharp cutoff
```

**Validation**: `critic/success_onpolicy/mean` ≥ v24 at step 50. Log new metric `duet/teacher_surprise_mean` (add 2 lines in existing `_masked_stats` block at L714). If mean decays from ~0.8 → ~0.2 over training, the BC functionality is being absorbed naturally.

---

## Spec 3: Token-level value baseline (learned)

**Goal**: add a tiny value head producing `V_phi(s, t)`, advantage = `reward - V_phi`. Teacher PG uses per-token `A(s, a)`.

**Scope concern**: introduces a second learned model, needs a second optimizer, value loss, hyperparameters, target updates. Full PPO has critic; veRL ppo trainer supports this via `critic` role — but DUET uses critic-free GRPO, so enabling a critic means re-architecting `ae_ray_trainer.fit()` to create and train a critic worker.

**LOC**: ~800-1200 including worker wiring, value loss, scheduler. **Risk**: high.

**17-day calendar**: NOT recommended. Return to this only after v38/v39 fail. A lighter hack: use the `ref_log_prob - log_prob` per-token as a *zero-shot* advantage proxy (no new model) — but this collapses to Spec 2 with minor differences.

**Config template**: skip. Out of budget.

---

## Spec 4: Fix v37 adaptive mu bug

v37 set `chord_mu_VA_target=0.035`. Actual `V_A = std(per_sample |adv|)` in our runs was 0.7-3.0 on webshop (group-relative GRPO advantages are roughly in `[-1, 1]` after std normalization; per-sample absolute mean has std ~0.3-1.0). Result: `excess = (VA_ema - 0.035) / 0.035 ≈ 20-80`, sigmoid saturates at 1 → mu pinned at peak (0.5) forever, never decays.

### 4a (recommended): recalibrate VA_target against v24 decay

From v24 logs (see `analysis_reports/duet_v24_theory_and_framing.md` and `duet_curriculum_empirical_validation.md`):
- Step 1-5: group-relative adv |mean| ~0.15-0.25, std across samples ~0.6-1.0 → `V_A ≈ 0.8`.
- Step 25+: convergence reduces dispersion, `V_A ≈ 0.2-0.3`.

For sigmoid to transition cleanly from 1 (peak) at early training to 0 (valley) at convergence, center target near mean:
- `VA_target = 0.5` (midpoint).
- `sigmoid_k = 3.0` unchanged; gives ~90% of range covered as V_A walks 0.2→0.8.

Patch `het_actor.py:1761`:
```python
VA_star = float(self.config.get("chord_mu_VA_target", 0.5))   # was 0.035
```

### 4b: use already-logged `duet/adv_onpolicy_effective_abs_mean`

From logs (v24): Q1 ≈ 0.12, Q4 ≈ 0.18. Smaller dynamic range; cleaner signal (post-gate, post-SC).

Patch `het_actor.py:1752-1756`:
```python
# Use effective on-policy adv (post SC, post gate), not raw adv
_rm = response_mask.float()
_onpolicy_mask = (1.0 - exp_mask.float()) * _rm
_denom = _onpolicy_mask.sum(-1).clamp_min(1.0)
_adv_per_sample = (advantages.detach().abs() * _onpolicy_mask).sum(-1) / _denom
# keep only on-policy rows
_has_on = (_onpolicy_mask.sum(-1) > 0)
if _has_on.any():
    _va_current = float(_adv_per_sample[_has_on].std().item()) if _has_on.sum() > 1 else 0.0
else:
    _va_current = 0.0
VA_star = float(self.config.get("chord_mu_VA_target", 0.05))   # tighter range
```

Likelihood of v24-like mu trajectory: **high**. The sigmoid now operates over [0.02, 0.20] with target 0.05 — mu high at start, falls as V_A compresses.

### 4c: grad-norm ratio

Requires extra backward passes to measure `BC_grad_norm / DR3_grad_norm`. Expensive under FSDP. Not recommended.

### 4d: pure grad-norm EMA

`mu_t = valley + (peak - valley) * sigmoid(k * (grad_norm_ema - g*) / g*)`. Requires reading total grad norm post-backward — veRL exposes `actor/grad_norm`. Feasible but delayed-feedback and noisy under per-micro-batch updates.

**Top pick for Spec 4: option (b)** — uses signal already computed, narrow range, matches v24 closely.

### Minimal v37b patch (fix in place)

```python
# het_actor.py:1748-1772 — replace the block
use_adaptive_mu = bool(self.config.get("chord_mu_adaptive", False))
if use_adaptive_mu:
    import math
    with torch.no_grad():
        _rm = response_mask.float()
        _on_mask = (1.0 - exp_mask.float()) * _rm
        _denom = _on_mask.sum(-1).clamp_min(1.0)
        _adv_ps = (advantages.detach().abs() * _on_mask).sum(-1) / _denom
        _has_on = (_on_mask.sum(-1) > 0)
        if _has_on.sum() > 1:
            _va_current = float(_adv_ps[_has_on].std().item())
        else:
            _va_current = 0.0
    if not hasattr(self, "_VA_ema"):
        self._VA_ema = _va_current
    else:
        self._VA_ema = 0.9 * self._VA_ema + 0.1 * _va_current
    VA_star = float(self.config.get("chord_mu_VA_target", 0.05))
    sigmoid_k = float(self.config.get("chord_mu_sigmoid_k", 3.0))
    excess = (self._VA_ema - VA_star) / max(1e-6, VA_star)
    gated = 1.0 / (1.0 + math.exp(-sigmoid_k * excess))
    mu = chord_mu_valley + (chord_mu_peak - chord_mu_valley) * gated
    adaptive_metrics = {
        "chord/mu_mode": 2.0,
        "chord/VA_ema": float(self._VA_ema),
        "chord/VA_current": float(_va_current),
        "chord/VA_star": VA_star,
        "chord/mu_adaptive_gated": float(gated),
    }
```

**v37b config** (`webshop_qwen1.5b_duet_v37b.yaml`): same as v37 with `chord_mu_VA_target: 0.05`. Validation: `chord/mu` should rise from valley then decay to valley by step 40, matching v24's scheduler shape.

---

## Spec 5: Closed-form mu (no calibration)

Requirement: derive mu from quantities already in the loss, no tunable target.

### 5a: teacher-vs-onpolicy NLL gap

```python
# At het_actor.py, inside update_policy loop after sft_ret computed:
with torch.no_grad():
    # per-token NLL on teacher tokens
    teacher_tok = (exp_mask * response_mask).float()
    on_tok = ((1.0 - exp_mask) * response_mask).float()
    nll_teacher = ((-log_prob) * teacher_tok).sum() / teacher_tok.sum().clamp_min(1.0)
    nll_on      = ((-log_prob) * on_tok).sum()      / on_tok.sum().clamp_min(1.0)
    # mu rises when teacher NLL > on-policy NLL, zero when equal
    gap = (nll_teacher - nll_on).clamp_min(0.0)
    # normalize by on-policy NLL to make dimensionless
    mu_closed = (gap / nll_on.clamp_min(0.1)).clamp(max=1.0).item()
mu = float(mu_closed) * chord_mu_peak
```

LOC: ~10. Mu is **zero** once policy fits teacher as well as itself — exactly the "natural fade-out" story.

Interpretation: `nll_teacher - nll_on` ≈ KL(empirical teacher || policy) relative to KL(self || policy). When they match, policy has nothing to learn from teacher.

Catch: needs at least one teacher sample and one on-policy sample in the micro batch. With `ppo_micro_batch_size_per_gpu=1`, NLL-teacher is undefined on on-policy-only micro-batches → mu=0 (skip BC for that batch). That is fine.

### 5b: KL-to-teacher softmax temperature

```python
# teacher loss already produces KL proxy: mean(log_prob_teacher - log_prob_policy_on_teacher_tokens)
# Use that as mu signal directly:
# mu = sigmoid(k * (kl_teacher_to_policy - kl_policy_to_policy))
```

Harder to compute cleanly because we have only 1-side log_probs (no teacher logits). Skip.

### 5c: mu = 1 - D(average teacher sample)

Reuse DR3 discriminator! When D predicts teacher samples are barely distinguishable from on-policy, teacher no longer useful, mu → 0.

```python
with torch.no_grad():
    # Average D(teacher) ≈ 0 when distinguishable, → 0.5 when policy matches teacher
    d_teacher_mean = dr3_metrics.get("dr3/w_off_mean", 0.0) if dr3_metrics else 0.0
    # map to mu: w_off is already "policy/teacher" ratio. high w_off = policy ≈ teacher → low mu.
    # mu_closed = exp(-k * w_off_mean)
    mu_closed = math.exp(-3.0 * d_teacher_mean)
mu = chord_mu_valley + (chord_mu_peak - chord_mu_valley) * mu_closed
```

LOC: ~5. Reuses DR3 work, zero new computation. **Narrative**: DR3 and BC share a gating mechanism → confirms single-operator story.

**Most implementable**: **5a** (NLL gap). Truly closed form, no tunables, works with existing tensors.

**Config template** (v40):
```yaml
chord_mu_closed_form: "nll_gap"   # new key; "nll_gap" | "dr3_w" | "scheduler"
chord_mu_peak: 0.5                # still used as max cap
# chord_mu_VA_target, warmup/decay steps all ignored
```

Wire-up: in `het_actor.py:1748`, branch on `chord_mu_closed_form` instead of `chord_mu_adaptive`.

---

## Ranked recommendations

| Rank | Spec | Effort | Risk | Eliminates BC? | Match v24? | Narrative win |
|------|------|--------|------|----------------|------------|----------------|
| 1 | Spec 2 (surprise) | 12 LOC | low | yes | high | DR3 absorbs per-token teacher fit weight — clean |
| 2 | Spec 4b (fix mu) | 10 LOC | low | no | very high | fixes v37 but keeps BC term — safer fallback |
| 3 | Spec 5a (NLL gap) | 10 LOC | low | no | medium | closed-form mu story (DUET selling point) |
| 4 | Spec 1 (token DR3) | 150 LOC | medium | yes | medium | strongest paper narrative, some engineering |
| 5 | Spec 3 (value head) | 1k LOC | high | yes | unknown | out of budget |

### Top pick: Spec 2 first, Spec 4b as safety net

Why Spec 2 first:
- **12 lines**, one-afternoon patch.
- Single-operator narrative: "DR3's per-token surprise weighting (1-p) times group-advantage times importance ratio = ONE loss, no SFT term."
- Direct correspondence to CHORD's phi: `phi(p) = p(1-p)` splits as `p * (1-p)`; we keep the `(1-p)` part that pushes unlearned tokens and drop the `p` part that dampened already-learned ones (the PPO ratio already dampens them via near-1 ratio at converged tokens).
- If v39 matches v24, we can drop CHORD entirely. If v39 underperforms, we fall back to Spec 4b.

Why Spec 4b (fix mu) as safety: guarantees we have something better than v37 regardless of Spec 2's outcome.

---

## Debug plan for v37 adaptive mu

### Root cause (confirmed)

Code at `/data/home/qisheng/EvolAnalsis/agentevolver/module/exp_manager/het_actor.py:1752-1764`:

```python
_adv_per_sample = (advantages.detach().abs() * _rm).sum(-1) / _denom
_va_current = float(_adv_per_sample.std().item()) if _adv_per_sample.numel() > 1 else 0.0
# ...
VA_star = float(self.config.get("chord_mu_VA_target", 0.035))
excess = (self._VA_ema - VA_star) / max(1e-6, VA_star)
gated = 1.0 / (1.0 + math.exp(-sigmoid_k * excess))
```

For `VA_ema = 0.7` and `VA_star = 0.035`, `excess = 18.6` → `gated = 1.0` → `mu = mu_peak`. mu never decays.

### Minimal patch (v37b)

Two-line change — copy above Spec 4b block. Or simpler: override via config only:

```yaml
# v37b.yaml — copy v37 then add
chord_mu_VA_target: 0.5           # match raw |adv| std scale (4a option)
# optional: also restrict signal to on-policy tokens (4b)
```

No code change needed if using 4a variant.

### Re-run command

```bash
cd /data/home/qisheng/EvolAnalsis
cp config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet_v37.yaml \
   config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet_v37b.yaml
# edit chord_mu_VA_target: 0.5 (or 0.05 with 4b code change)
python launcher.py --conf config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet_v37b.yaml
```

### Success criteria

Within 50 steps:
- `chord/VA_ema` descends from ~0.7 → ~0.3 (monotone enough).
- `chord/mu` decays from ~0.5 (peak) to ~0.05 (near valley).
- `chord/mu_adaptive_gated` traverses at least 30% of [0,1] range.
- `critic/success_onpolicy/mean` within 3% of v24 at step 50.

If `chord/mu` still stuck near peak after step 20, measure actual `chord/VA_current` distribution and recenter `VA_target` between the observed p25 and p75.

---

## File locations (absolute paths)

- `/data/home/qisheng/EvolAnalsis/agentevolver/module/exp_manager/dr3_ratio.py` — L73 (features), L229 (disc), L577 (step). Spec 1 main edits.
- `/data/home/qisheng/EvolAnalsis/agentevolver/module/exp_manager/het_actor.py` — L288-308 (hidden pooling), L1282-1299 (DR3 call site), L1494-1500 (w_hat apply), L1700-1805 (CHORD branch), L1748-1772 (adaptive mu block).
- `/data/home/qisheng/EvolAnalsis/agentevolver/module/exp_manager/het_core_algos.py` — L393-580 (teacher surrogate, teacher_ratio built), L509-521 (surprise hook point, Spec 2), L1627 (chord_mu_scheduler), L1723 (compute_chord_sft_loss).
- `/data/home/qisheng/EvolAnalsis/config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet_v24.yaml` — baseline.
- `/data/home/qisheng/EvolAnalsis/config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet_v37.yaml` — bugged adaptive mu config.
- New configs to create: `v37b.yaml` (fix), `v38.yaml` (token DR3), `v39.yaml` (surprise), `v40.yaml` (closed-form mu).

Done.
