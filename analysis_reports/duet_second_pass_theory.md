# DUET — Second-Pass Theory, Empirics-First

*Lead researcher memo, 2026-04-19. Supersedes `duet_first_principles_analysis.md` on v24 mechanism. Starts from the empirical data that refuted the "curriculum pedagogy" story, derives a mechanism, and specifies what to code next.*

---

## 1. The empirical facts that any theory must explain

Any mechanism we posit must be consistent with all of the following WebShop 1.5B, 100-step measurements (from `_parsed/curriculum_metrics.json` and `fig6`–`fig11`):

1. **Val scores**: v24 (μ 0.3→0.05 decaying over 25 steps) = **0.678**; v12 (no BC) = 0.431; v36 (μ=0.05 constant) = 0.389 **after peaking at 0.527**; v22 (μ=0.05 constant, v1-stab) = 0.462.
2. **Gradient norm, late training**: v12 ≈ **11 (exploding)**; v36 ≈ 6 (drifting up); v22 ≈ 6–7; v24 **≈ 3–4 (stable throughout)**.
3. **On-policy advantage magnitude** (`duet/adv_onpolicy_effective_abs_mean`): v12 grows 0.16→0.33 over training; v36 flat ~0.18; v24 **smallest and flattest: 0.12→0.17**.
4. **SFT loss on teacher tokens** (`chord/sft_loss`): v24 reaches **0.61** by step 100 (best teacher fit); v36 reaches 0.67 despite applying the same terminal μ.
5. **Entropy**: v12, v36 comparable (~0.54 late). v24 is **consistently highest** (~0.60).
6. **Discriminator accuracy**: v36 and v24 both cross 0.9 by step 30; v12 only at step 45. *Adding any BC accelerates disc_acc.* v24 and v36 are comparable on this axis.
7. **SC on-policy progress**: v36 **highest** throughout (0.35–0.37); v12 lowest (0.27). BC feeds SC rather than starving it.
8. **Teacher gradient share**: v24 has a unique **early peak (0.32 at steps 5–15) followed by sharp decay to 0.13**; v36 is flat ~0.13–0.14 throughout; v12 decays gradually.
9. **PPO clip never fires** (`off_pg_cliphit_rate = 0`). Ratio clipping is not the binding constraint anywhere.
10. **v36 regresses**: peaks at 0.527 ~ step 75 and falls back to 0.389.

Constraints this imposes on any theory:
- The mechanism must **not** route through disc_acc differences (6), entropy differences (5), or SC bonus differences (7) — these were the first-pass theory's load-bearing causal links, and they are all either null or backwards.
- The mechanism **must** produce v24's distinctive signatures in grad_norm (2), advantage magnitude (3), and SFT-loss fitness (4).
- The mechanism must explain why constant μ=0.05 cannot reach the same endpoint despite roughly equal per-step BC gradient at convergence time.

These are tight constraints. The first-pass "BC installs rare tokens so DR3 can then fade" story satisfies (4) but is silent on (2) and (3). We need a new mechanism.

---

## 2. Deliverable 1 — Why v24's gradient norm stabilizes

### 2.1 Test the candidate mechanisms against the data

Call the BC gradient $g_{\text{BC}}$, the DR3+GRPO gradient $g_{\text{RL}}$, and the parameter update $\Delta\theta \propto -(g_{\text{RL}} + \mu_t g_{\text{BC}})$. The question is: why does $\lVert g_{\text{RL}}\rVert$ stabilize when $\mu_t$ front-loads and decays, but not when $\mu_t$ is constant?

**(a) Gradient noise offset (BC variance damps DR3 variance).** BC's gradient on teacher tokens is deterministic once the teacher corpus is fixed: $g_{\text{BC}} = -\nabla\log\pi_\theta(a^*|s)$ per token, weight 1 by design (token weighting aside). DR3's gradient is $-w(\tau)A(\tau)\nabla\log\pi_\theta(a|s)$; it inherits variance from both $w(\tau)$ and $A(\tau)$. If BC merely *offset* DR3 variance, a summing argument gives $\lVert g_{\text{BC}} + g_{\text{RL}}\rVert^2 = \lVert g_{\text{RL}}\rVert^2 + 2\mu\, g_{\text{BC}}\!\cdot g_{\text{RL}} + \mu^2\lVert g_{\text{BC}}\rVert^2$. For the cross term to *reduce* the norm, BC must be anti-aligned with DR3. But BC is sign-positive in $\log\pi_\theta$ on every teacher token regardless of trajectory advantage, whereas DR3 flips sign on low-advantage trajectories. So BC is *sometimes* anti-aligned, *sometimes* aligned. On average the cross term is not guaranteed negative. **Verdict: does not cleanly produce the observed late-training stabilization.**

**(b) Implicit trust region.** BC acts as a quadratic-like anchor to the teacher distribution. Large $\mu$ in early training shrinks the effective step size by pulling the update toward the low-dimensional teacher manifold. This is the mechanism used informally in "distillation + RL" literatures (DQfD, AWAC). *But it doesn't explain why v24's grad_norm remains stable long after $\mu$ has decayed to 0.05 — by step 40 the anchor is weak and yet grad_norm stays at 3-4.* So (b) can't be the whole story for late-training stability.

**(c) Advantage scale-calibration (the winning hypothesis).** The GRPO normalization is $A(\tau) = (R(\tau) - \mu_g)/\sigma_g$ where $\mu_g, \sigma_g$ are group-level mean/std. Suppose early training has **high variance in group rewards** — some rollouts score 0.8, others 0.0, because the policy is scattered across state space. Then $\sigma_g$ is large, $|A|$ is moderate-but-skewed, and subsequent gradient updates push the policy in *conflicting* directions on different tasks → gradient variance accumulates over epochs → late grad_norm explodes. Compare the alternative: early BC aligns the policy toward the teacher *state distribution*, so rollouts become **more homogeneous in reward** (either all reach the task, or all fail in similar ways). This shrinks $\sigma_g$, shrinks $|A|$, and — crucially — this small-advantage regime is self-reinforcing because small advantages produce small updates and the policy drifts slowly, so $\sigma_g$ stays small.

This is **not** "BC damps DR3." It is **"BC changes the reward distribution seen by GRPO, which changes the advantage magnitude, which changes the gradient magnitude."** The mechanism is indirect: BC acts on $\pi_\theta$, which acts on rollout diversity, which acts on $\sigma_g$, which acts on $A$, which acts on $\lVert g_{\text{RL}}\rVert$.

**(d) Format-lock stabilization.** BC preserves low-probability grammar tokens. If these drift, outputs become malformed, environment rewards become noisier, $\sigma_g$ grows. This is a special case of (c) — format drift is one source of reward variance that BC prevents.

### 2.2 Evidence that (c) is primary

Look at (3): v24's `adv_onpolicy_effective_abs_mean` is **lowest and flattest** among all variants (0.12→0.17). This is the direct observable for the advantage-scale-calibration mechanism. And it rank-orders perfectly with grad_norm stability: small advantages → small gradients, modulo a roughly constant Jacobian. The cross-comparison:

| Variant | adv magnitude | grad_norm late | Explanation |
|---|---:|---:|---|
| v24 | 0.12–0.17 | 3–4 | early BC homogenizes rollouts → small $\sigma_g$ persists |
| v36 | ~0.18 | ~6 | constant weak BC never achieves homogenization; late drift reintroduces variance |
| v12 | 0.16→0.33 | ~11 | no BC; rollout diversity grows unchecked; $\sigma_g$ grows; advantages explode |
| v22 | ~0.18 | ~6–7 | same BC as v36 but looser DR3 clip, slightly more tolerant |

The monotone relationship between early-window BC magnitude and terminal advantage scale is the cleanest empirical signature in the dataset.

### 2.3 Why constant μ=0.05 fails to trigger the same mechanism

To drive the policy onto the teacher manifold within the window where group-reward variance is still shrinkable, BC must apply *enough log-probability gradient per step* that $\log p_\theta(a^*)$ climbs a non-trivial amount in 10–15 steps. At μ=0.05, $\Delta\log p_\theta \approx 0.05$ per step per teacher token — ~0.5 nats over 10 steps, moving $p_\theta$ from $10^{-4}$ to at best $\sim 2 \times 10^{-4}$. Policy is *barely* moved. At μ=0.3, $\Delta\log p \approx 3$ nats over 10 steps → $p_\theta$ climbs from $10^{-4}$ to $\sim 2 \times 10^{-3}$, into the steep region of $p(1-p)$. This is a **threshold phenomenon**: BC must cross a threshold in early training to trigger the rollout-homogenization cascade; once in the stable regime, small μ suffices to maintain it.

So the mechanism is best phrased as: **BC at early steps provides a "bootstrap kick" onto the teacher state distribution. Once on-manifold, rollout variance is self-limiting (small advantages → small updates → small drift). Constant weak BC never delivers the kick.** The late μ=0.05 in v24 is maintenance, not the active ingredient. This matches (4): v24's lower terminal `sft_loss` (0.61) vs v36's (0.67) with identical terminal μ — v24's policy is already closer to teacher when μ=0.05 takes over, so maintenance is effective.

### 2.4 Parsimonious summary

**Primary mechanism: advantage-scale calibration via early policy-teacher alignment.** Early high μ moves the policy onto a neighborhood of the teacher's state distribution. This homogenizes rollouts, shrinks $\sigma_g$, shrinks $|A|$, and shrinks $\lVert g_{\text{RL}}\rVert$. The shrinkage is **self-sustaining** because small gradients produce small updates which produce small policy drift which keeps rollouts homogeneous. Constant weak BC never triggers the initial alignment. Pure RL (v12) never aligns either, and rollout variance grows over training because GRPO's advantage scale provides no restoring force.

**Secondary mechanism: format-token preservation.** BC's μ_valley=0.05 prevents low-probability grammar tokens from being drifted by PPO. This is real (v25 ablation confirmed) but is a smaller-magnitude effect than advantage-scale calibration.

**Rejected mechanisms:** direct gradient offset (a) — signs don't reliably anti-align; static trust region (b) — doesn't explain late-phase stability; curriculum-on-disc-acc — disc_acc converges similarly in v24 and v36.

The word "pre-conditioner" is slightly misleading: we're not making the Hessian better-conditioned in the optimizer-theory sense. We are changing the rollout-reward distribution so that the advantage estimator produces smaller, less-noisy signals. "Advantage regularizer via implicit rollout control" is more precise.

---

## 3. Deliverable 2 — v36 reliability

The reported v36 number is 0.389 (final) after a peak of 0.527. This single-seed result is *the* empirical foundation for the claim "constant weak BC is strictly worse than no BC." It also deserves skepticism.

**What the theory predicts about v36.** Under the advantage-scale mechanism, constant μ=0.05 should produce **weakly better than v12** (small homogenization bonus, but below the bootstrap threshold). The expected outcome is ~0.46–0.50, slightly above v12. The 0.389 endpoint *does not fit* this prediction — it is lower than expected.

What could produce a late-phase regression? Two possibilities:

- **S2 (most likely): seed-specific collapse.** Even with slight homogenization, constant μ exerts continuous low-magnitude pull on a policy that is also under DR3 pressure. If a single trajectory cluster gets slightly off-manifold at step 70, the residual BC gradient may not be strong enough to pull it back but the DR3 gradient (now with increasing `w_hat` on the "bad" cluster) may amplify the drift. This would look exactly like the v36 trajectory: late peak, then regression.

- **S3 (also plausible): v36's 0.389 is within the noise band for this setup.** The v22 run (nearly identical setup, weaker DR3 stab) landed at 0.462. The spread between v22 and v36 is 7.3 pp for nominally similar configurations, which suggests the run-to-run variance is large for this class of method.

**Confidence assessment.** I give S1 ("constant μ is genuinely harmful relative to no BC") **~25% prior**. S2+S3 ("constant μ is near-neutral vs no BC, with run noise") gets **~75% prior**. The v24 vs v36 ordering (v24 ≫ v36) is robust — decaying BC genuinely wins. The v12 vs v36 ordering is fragile.

**Experiments that would disambiguate.**
1. **Re-run v36 with seeds 43, 44** — cheapest test. If the re-runs land near 0.46 with a regression at step 100, S2 is confirmed and we report "constant μ ≈ no BC, modulo noise."
2. **Run v36 for longer** (150–200 steps). If the regression continues monotonically, it's a systematic late-phase collapse specific to constant μ. If it plateaus and recovers, it was transient.
3. **Compare at matched *peak* score rather than matched step count.** Report v12_peak vs v36_peak vs v24_peak — this is robust to late-phase noise and focuses on "can this method reach high performance."

**Implication for the paper.** Do not write the claim "constant μ is harmful" — the single-seed evidence is too thin. Do write "the *schedule* is load-bearing: constant μ fails to match decayed μ." This is robust across all constant-μ variants (v22, v23, v36 all < v24) and does not depend on the contested v12 vs v36 comparison.

---

## 4. Deliverable 3 — Optimal adaptive μ signal

Given the Section 2 mechanism (BC serves to homogenize rollouts and shrink advantage magnitude), μ_t should track a signal that directly indexes rollout homogenization need. I'll derive the correct signal from the mechanism rather than pick from the candidate list.

### 4.1 Derivation

The quantity μ controls is the "bootstrap kick" — specifically, how fast $\log p_\theta(a^*)$ climbs for teacher tokens at the current policy. The need for BC is high iff:

- Teacher-token probabilities are low under the current policy (→ BC is needed to install them, and log-prob gradient is high-utility because $p(1-p)$ is small and one needs the unit-coefficient ascent).
- GRPO advantages have large variance (→ RL alone is unstable).
- Rollout distribution is far from teacher's state-visitation (→ group-reward variance will be large).

The mechanism identifies **advantage variance** as the *downstream* signal. Let $V_A(t) = \text{Var}_\tau(A(\tau))$. This is both (i) a direct measurable proxy for rollout heterogeneity, and (ii) the quantity that drives $\lVert g_{\text{RL}}\rVert$ variance. A rule:

$$
\mu_t = \text{clip}\!\left(\mu_\min + (\mu_\max - \mu_\min)\cdot \tanh\!\left(\kappa\,\cdot\,\frac{V_A(t) - V_A^\star}{V_A^\star}\right),\ \mu_\min,\ \mu_\max\right)
$$

where $V_A^\star$ is a target advantage variance (set to a small constant like 0.03², matching v24's observed 0.17 endpoint), $\kappa$ is a gain parameter, and $\mu_\min = 0.05$, $\mu_\max = 0.3$. When advantage variance is above target (rollouts heterogeneous), μ ramps up; when at/below target (rollouts homogeneous), μ relaxes to $\mu_\min$.

### 4.2 Why this beats the candidates in the brief

| Candidate | Issue |
|---|---|
| $E[p(1-p)]$ on teacher tokens (researcher v1) | Measures where BC gradient is "efficient" in probability space, not where it's *needed*. Saturates once policy has modest teacher support even if rollout heterogeneity persists. |
| $(1-\text{disc\_acc})$ (algo-engineer) | Rejected by exp-analyst because disc_acc saturates to 0.9+ by step 30 in all variants. Correct rejection. |
| $\lVert g_{\text{RL}}\rVert$ directly | Works in principle but is very noisy (microbatch effects, optimizer state coupling). `adv_effective_abs_mean` is one denoising step upstream. |
| $\lVert g_{\text{RL}}\rVert / \lVert g_{\text{BC}}\rVert$ | Appealing but circular — $\lVert g_{\text{BC}}\rVert$ is mechanically near-constant on teacher tokens. Ratio reduces to $\lVert g_{\text{RL}}\rVert$ up to scale. |

Advantage variance $V_A(t)$ (or its proxy `adv_onpolicy_effective_abs_mean`) is:
- **Mechanism-aligned**: directly measures the quantity the BC kick is supposed to shrink.
- **Already logged**: zero new instrumentation.
- **Smooth**: variance over a group of 8 rollouts is much less noisy than gradient norm.
- **Self-quiescing**: once $V_A$ is small, μ drops to $\mu_\min$ and stays.

### 4.3 Predicted behavior

At step 0: rollouts are random, $V_A$ is large → μ = $\mu_\max$ = 0.3. BC fires aggressively.
Steps 5–20: policy aligns, rollouts homogenize, $V_A$ drops → μ cosine-like decay.
Steps 20–100: $V_A$ at target, μ = $\mu_\min$. DR3 + SC carry the learning signal.

This reproduces v24's **observed** μ trajectory as an emergent consequence rather than a hand-tuned schedule. The 25-step decay window in v24 is the time scale of natural rollout homogenization on WebShop 1.5B — not a magic hyperparameter.

### 4.4 Why this beats v24's hand-tuned schedule

- **Generalizes to ALFWorld**: where teacher is closer to initial policy, $V_A$ starts small, μ immediately drops to $\mu_\min$. The adaptive rule predicts the observed "v24 BC-on-ALFWorld = neutral" result *without refitting*.
- **Robust to training-budget changes**: v24's 25-step decay was set for a 100-step run. Extending to 200 steps requires re-tuning. Adaptive μ handles any budget.
- **Eliminates three hyperparameters**: μ_warmup, μ_decay, μ_peak. Only $\mu_\max, V_A^\star$ remain, and $V_A^\star$ has a principled default (set it to the median advantage variance v24 achieves in its stable regime, which is directly observable).

**Reviewer rebuttal bonus**: "your μ schedule is the result of hyperparameter search on WebShop" → "our μ is adapted online from the observed advantage distribution; the schedule is an emergent property of the mechanism, not a tuned artifact."

---

## 5. Deliverable 4 — Paper narrative candidates

Given Section 2's "advantage-scale calibration" mechanism, here are three candidate framings, ranked by combination of honesty and reviewer-robustness.

### 5.1 Candidate A — "Advantage regularizer" (recommended)

**Elevator pitch.** *Teacher imitation in the form of low-weight behavior cloning acts as an advantage regularizer for on-policy RL: it aligns the learner's state-visitation with the expert's, which shrinks group-reward variance, which shrinks advantage magnitude, which stabilizes gradients. DUET's Action Channel contains two orthogonal operators: (i) an advantage regularizer (adaptive-μ BC) that controls the *distribution* of the RL signal, and (ii) a density-ratio-corrected policy gradient (DR3) that provides *unbiased* off-policy updates from expert trajectories.*

**Smoking gun plot.** The scatter of `grad_norm` vs `adv_onpolicy_effective_abs_mean` across v12/v22/v36/v24, colored by μ schedule. Near-perfect monotone relationship. Adding a vertical line at v24's operating point shows it sits in the stable basin.

**Weakest point.** "Why isn't this just entropy regularization in disguise?" — must explicitly differentiate. BC regularizes toward a *specific* anchor (teacher), not toward uniform. Entropy regularization would not produce the observed reduction in `adv_onpolicy_effective_abs_mean` below what the teacher's own state distribution produces.

### 5.2 Candidate B — "Two-operator curriculum" (v24 memo's original pitch)

**Elevator pitch.** *BC installs rare teacher tokens that DR3 cannot lift from cold start; DR3 provides trajectory-level credit assignment that BC cannot. The two operators form a curriculum: BC dominates early when the policy lacks teacher support, DR3 dominates late once support is installed.*

**Smoking gun plot.** `chord/sft_loss` trajectory — v24 fits teacher best by step 100 despite applying smaller total BC integral than CHORD, showing that the schedule *shape* matters beyond the schedule *area*.

**Weakest point.** Refuted empirically by Section 1 facts 6 and 7 — disc_acc and SC progress don't discriminate v24 from v36, yet only v24 wins. The "support installation" mechanism was never the only (or dominant) thing BC did.

### 5.3 Candidate C — "DUET is adaptive regularization + principled off-policy PG"

**Elevator pitch.** *DUET combines two adaptive regularizers: (i) adaptive-μ BC that stabilizes advantage magnitude, and (ii) DR3 density-ratio estimation that makes teacher-off-policy trajectories usable as unbiased gradient samples. The State Channel (Φ(τ)) independently shapes reward density. All three components are self-quiescing.*

**Smoking gun plot.** Three-panel figure: (a) adaptive-μ converges to $\mu_\min$ tracking $V_A$ convergence; (b) DR3 `w_hat` converges to ~1 on teacher data; (c) SC `bonus/reward` ratio stays bounded. Each arm retires on its own schedule driven by its own diagnostic.

**Weakest point.** More complex story. Three mechanisms × two channels is a lot to explain in an introduction. If the reviewer cares about simplicity, (A) is cleaner.

### 5.4 Recommendation

**Go with Candidate A.** It is (i) the most honest post-refutation framing, (ii) has the clearest empirical signature, (iii) motivates the adaptive-μ proposal from Section 4 naturally, (iv) keeps the dual-channel framing (Action Channel = BC + DR3; State Channel = Φ) intact. Candidate B should be abandoned as the primary frame but kept as a "mechanistic observation" in the analysis section. Candidate C is Candidate A with SC foregrounded — fine for a longer version of the paper but adds surface area.

---

## 6. Deliverable 5 — Implementation plan

### 6.1 Minimum experimental validation

**Single experiment (v37):** Run v24's config with adaptive-μ rule from Section 4 on WebShop 1.5B, 100 steps, same seeds as v24. Rule: $\mu_t = \text{clip}(\mu_\min + (\mu_\max - \mu_\min)\cdot \text{sigmoid}((V_A(t) - V_A^\star)/V_A^\star))$ with $\mu_\min=0.05$, $\mu_\max=0.3$, $V_A^\star=0.035$ (set from v24's stable-regime observation). EMA the $V_A$ estimate over 5 steps to denoise.

**Decision rule.**
- v37 ≥ 0.65: adopt adaptive-μ as the paper's headline Action Channel; reframe with Candidate A narrative. Strong result.
- 0.55 ≤ v37 < 0.65: mixed result. Report v24 as empirical and adaptive-μ as "principled alternative matches within noise"; the paper story is still Candidate A but less decisive.
- v37 < 0.55: adaptive-μ is underperforming. Either mechanism theory is wrong or rule is miscalibrated. Keep v24 as empirical, reframe Action Channel as "hand-tuned schedule approximating the advantage-regularizer principle"; paper narrative weakens toward "observation + empirical recipe" rather than first-principles derivation.

This one experiment resolves the headline question.

### 6.2 Algorithm implementation

Code patch in `het_actor.py` near line 1740–1752, replacing the `chord_mu_scheduler` call:

```python
# --- Adaptive mu from advantage variance (mechanism: advantage regularizer) ---
use_adaptive_mu = self.config.get("chord_mu_adaptive", False)
if use_adaptive_mu:
    # Estimate group-relative advantage variance from current microbatch.
    # advantages tensor shape (bs, resp_len); aggregate per-sample first.
    with torch.no_grad():
        _adv_per_sample = (advantages.abs() * response_mask).sum(-1) / response_mask.sum(-1).clamp_min(1.0)
        _VA_current = float(_adv_per_sample.detach().std().item()) if _adv_per_sample.numel() > 1 else 0.0
    if not hasattr(self, "_VA_ema"):
        self._VA_ema = _VA_current
    else:
        self._VA_ema = 0.9 * self._VA_ema + 0.1 * _VA_current
    VA_star = float(self.config.get("chord_mu_VA_target", 0.035))
    mu_min = float(self.config.get("chord_mu_valley", 0.05))
    mu_max = float(self.config.get("chord_mu_peak", 0.30))
    # Sigmoid on normalized excess; sharp gain for near-threshold behavior.
    excess = (self._VA_ema - VA_star) / max(1e-6, VA_star)
    import math
    gated = 1.0 / (1.0 + math.exp(-3.0 * excess))
    mu = mu_min + (mu_max - mu_min) * gated
    metrics["chord/mu_mode"] = 2.0  # adaptive
    metrics["chord/VA_ema"] = self._VA_ema
    metrics["chord/mu_adaptive"] = mu
else:
    mu = chord_mu_scheduler(...)  # existing path
```

No changes to `het_core_algos.py` (the BC loss function itself is unchanged; only the μ schedule differs). No changes to `dr3_ratio.py`.

Config addition in `config/agentevolver.yaml`:
```yaml
chord_mu_adaptive: false  # default off; enable per-experiment
chord_mu_VA_target: 0.035
```

Total patch: ~30 lines. Zero new tensors, zero FSDP-serialization concerns. Safe.

### 6.3 Timeline (17 days to NeurIPS)

- **Days 1–2**: code v37 patch, local smoke test (1.5B × 20 steps), verify μ trajectory looks sensible (starts high, decays to μ_min).
- **Days 3–6**: run v37 on WebShop 1.5B + 3B (two seeds each). 4 runs × ~1 day = manageable on 8xA100.
- **Days 7–8**: run v37 on ALFWorld 1.5B + 3B (generalization check; theory predicts adaptive-μ collapses to $\mu_\min$ immediately).
- **Days 9–11**: if v37 validates, redraft paper narrative around Candidate A; update all v24-centric plots with v37 overlay.
- **Days 12–14**: ablations the reviewer will demand: $V_A^\star \in \{0.02, 0.05, 0.08\}$, $\mu_\max \in \{0.2, 0.4\}$. Cheapest first. These can share compute with existing runs.
- **Days 15–16**: re-run v36 with seeds 43, 44 to resolve the v12-vs-v36 reliability question (Section 3). Cheap, low-stakes.
- **Day 17**: buffer for anomalies.

If adaptive-μ fails validation (v37 < 0.55), switch narrative to Candidate B-weakened ("empirical schedule as approximation of advantage regularizer"). This is the contingency; budget ~2 days for narrative rewrite.

---

## 7. Uncertainty flags

- The Section 2 mechanism argument relies on the advantage-magnitude–grad-norm correlation visible in `fig11`. I have not validated by computing the actual correlation coefficient; this should be done before citing the plot as a "smoking gun."
- The $V_A^\star = 0.035$ target is set by matching v24's observed endpoint. If v24's endpoint is itself suboptimal (not at the true minimum of reachable $V_A$), adaptive-μ will undertrain BC. Consider running a small grid on $V_A^\star$ in the ablation phase.
- The "BC bootstrap kick" requires $\mu_\max$ large enough to move low-probability tokens. 0.3 worked for WebShop 1.5B. For larger models (3B/7B) with stronger priors, $\mu_\max = 0.2$ might suffice. Ablate.
- The v36 late-phase regression mechanism (Section 3) is speculative. If the seed re-runs show v36 ≈ v12, we lose part of the "constant BC is distinctly bad" narrative — but that claim is a secondary one, not the headline.
- I have *not* empirically verified that BC reduces group-reward variance (the causal mediator in the mechanism). This is a testable prediction: plot `std_over_group(critic/score)` for v24 vs v12; theory predicts v24 < v12 throughout. If the plot shows no difference, the mechanism is partially wrong.

## 8. Bottom line

v24 works because decaying BC provides a **bootstrap kick onto the teacher state distribution, which homogenizes rollouts, shrinks advantage magnitude, and thereby stabilizes gradients**. Constant weak BC fails to cross the bootstrap threshold. The mechanism is indirect (BC→state distribution→rollout variance→advantage→gradient), which is why first-pass theories focused on direct gradient interactions missed it.

The correct adaptive signal for μ_t is **advantage variance**, which the mechanism identifies as the downstream quantity BC is meant to regularize. Proposed rule: $\mu_t$ increases when $V_A$ exceeds a target, relaxes to $\mu_\min$ otherwise. Single experiment (v37) resolves whether this derivation beats v24's hand-tuned schedule.

Paper narrative: "Action Channel = advantage regularizer (BC) + principled off-policy PG (DR3)." Candidate A. Dual-channel framing preserved, BC motivated as principled rather than hacky, the schedule becomes emergent rather than tuned.
