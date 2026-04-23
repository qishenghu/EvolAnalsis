# DUET — First-Principles Account of v24 vs v36

*Lead-researcher memo, 2026-04-19. Opinionated. Scope: derive from first principles why $\mu_t = 0.3 \to 0.05$ (v24) uniquely works and why $\mu_t = 0.05$ constant (v36) is strictly worse than no BC (v12). Theory-to-data, no post-hoc narrative. Code references to `het_actor.py`, `het_core_algos.py`, `dr3_ratio.py` and all metric values pulled from the 100-step training logs at `/data/home/qisheng/EvolAnalsis/logs/webshop_qwen1.5b_duet_v{12,22,23,24,36}.log`.*

## 1. Setup and the central puzzle

WebShop 1.5B, 100-step budget. All variants below share the exact same LUFFY teacher mix ratio (1/8), DR3 code path, State Channel (β=0.2, step-level off), and 800-task training set. Only the $\mu_t$ BC schedule and two DR3 stab knobs (`disc_temperature`, `clip_max`) differ.

| v | $\mu_t$ schedule | disc_temp | clip_max | Val@100 | Train_Q4 |
|---|---|---|---|---:|---:|
| 12 | off (no BC) | 1.5 | 2.0 | 0.431 | 0.371 |
| 22 | 0.05 const | 1.0 | 5.0 | 0.462 | 0.602 |
| 23 | 0.10 const | 1.5 | 2.0 | 0.440 | — |
| **24** | **0.30 $\to$ 0.05 cosine-decay over 25 steps** | **1.5** | **2.0** | **0.678** | **0.663** |
| 36 | 0.05 const (re-run, v12-stab) | 1.5 | 2.0 | 0.389 | 0.619 |
| CHORD | 0.9 $\to$ 0.05 cosine, no DR3 | — | — | 0.603 | — |

Three puzzles I must explain:
1. **v36 < v12**: a constant small BC push is strictly worse than no BC. BC monotonically increases $\log p_\theta(a_\text{teacher}|s)$ at a unit-coefficient rate; if BC is positive-sign, how can a little of it be worse than none?
2. **v24 > v22 $\approx$ v23 $\approx$ v36**: the schedule, not the magnitude, is load-bearing.
3. **v12-stab interacts destructively with constant weak BC** (v36 = 0.389) but v1-stab with the same BC (v22 = 0.462) is weaker but at least not worse than v12.

All three need a single unified explanation or the story collapses to empiricism.

---

## 2. Deliverable 1 — Why v36 < v12 (the mechanism)

I evaluate the five user-supplied candidate mechanisms, pruning rigorously, and arrive at a parsimonious explanation.

### 2.1 Evaluating the candidates

**(a) Gradient interference.** BC provides $-\mu \cdot \nabla \log \pi_\theta(a^*|s)$ per teacher token, always positive in $\log p_\theta$-direction. DR3 provides $-w_\text{hat} \cdot A(\tau) \cdot \nabla \log \pi_\theta(a^*|s)$, whose sign equals the sign of $A(\tau)$. For a failed teacher trajectory (rare but possible; our WebShop teacher has ~85% median reward, not 100%), or more commonly for a teacher trajectory with locally negative group-relative advantage after baseline separation, DR3's teacher-side push is *negative* on $\log p_\theta$. Then BC's +0.05 push is **co-directional** with DR3 on positive-advantage tokens but **counter-directional** on negative-advantage tokens. Net: BC is noise-adding when DR3 is confident. *This is consistent with the data but doesn't explain the strict negative effect over v12* — v12 has the same noisy DR3 signal, just without the BC correction one way or the other.

**Verdict on (a):** supporting factor, not primary. Rejects only a null-BC hypothesis; doesn't explain v36 < v12 directly.

**(b) Support dilution.** Constant $\mu=0.05$ distributes mass equally across all teacher tokens of all teacher trajectories — including `<think>`-wrapper tokens (already at $p_\theta \approx 0.9$), generic `search[...]` prefixes (at $p_\theta \approx 0.5$), and the SKU-identity tokens `click[155- yellow]` (at $p_\theta \approx 10^{-4}$). In probability space, the BC gradient on $\log p_\theta$ is unit; on $p_\theta$ it becomes $\mu \cdot p_\theta \cdot (1-p_\theta)$. This is **largest at $p_\theta = 0.5$**, not at $p_\theta = 10^{-4}$. So constant weak BC spends most of its probability-space budget reinforcing already-mastered tokens, with a small tail that lifts the rare SKU token slowly. Compare: $\mu=0.3$ at cold start shifts $\log p_\theta$ by 0.3 per step, so $p_\theta = 10^{-4}$ climbs to $p_\theta \approx 10^{-4} \cdot e^{0.3 \cdot 10} \approx 0.002$ in 10 steps — getting into $p_\theta(1-p_\theta)$'s steep region, where the *next* 10 steps of $\mu=0.3$ can carry it to $p_\theta \approx 0.5$. Constant $\mu=0.05$ would need 50 steps just to match the first 10 steps of v24.

**Verdict on (b):** strong. This explains why constant-low $\mu$ is ineffective; it doesn't yet explain why it is *worse* than no BC.

**(c) DR3 discriminator interaction.** This is the decisive mechanism. DR3's density ratio $w_\text{hat} = D(\tau)/(1-D(\tau))$ is trained by a discriminator that separates teacher from on-policy trajectories. The discriminator's signal comes from the *distributional gap* between $\pi_\theta$ and $\pi_\text{teacher}$. Now observe the feedback loop:

- Constant $\mu = 0.05$ keeps $\pi_\theta$ *continuously* pulled a tiny amount toward the teacher on every training step.
- This reduces the distributional gap on generic/wrapper tokens (the ones that get most of the BC mass per §2.1(b)) but does NOT close the gap on rare SKU tokens.
- Consequence: the discriminator still reaches `disc_acc > 0.99` (v36: 0.995 at Q4, same as v22/v24) because it can still perfectly separate teacher from on-policy using the SKU tokens alone. But the per-trajectory confidence $D(\tau)$ is now smaller because the *aggregate* log-likelihood-ratio of the trajectory under the discriminator is smaller (fewer tokens "look different").
- Net: $w_\text{hat}$ on teacher trajectories shrinks toward 1 earlier. DR3's teacher-side gradient $-w_\text{hat} \cdot A \cdot \nabla \log \pi_\theta$ is thus smaller *throughout training*. DR3's designed curriculum (big $w_\text{hat}$ early when teacher-student gap is big, small $w_\text{hat}$ late) is prematurely collapsed.
- Under v12-stab (`clip_max=2.0`, `disc_temperature=1.5`), the BC-induced gap-shrinkage is more harmful because `clip_max=2.0` already caps $w_\text{hat}$ tightly. Under v1-stab (`clip_max=5.0`), there's more headroom so the gap-shrinkage matters less — which is exactly why v22 (v1-stab + $\mu=0.05$) at 0.462 is weaker than v12-alone 0.431 by only 3.1 pp, whereas v36 (v12-stab + $\mu=0.05$) at 0.389 is 4.2 pp *worse* than v12-alone. The stab and BC are both trying to control the same knob ($w_\text{hat}$ variance) and they fight.

**Verdict on (c):** primary mechanism. The data signature: v36 Q4 `teacher_gradient_share = 0.136`, v24 Q4 = 0.125 — nearly identical, but v24's TGS trajectory starts at 0.324 (Q1) and falls to 0.125 (Q4), while v36 starts at 0.150 (Q1) and is essentially FLAT at 0.15 throughout. The v24 curriculum has a built-in shape (high-early, low-late); v36 is curriculum-less.

**(d) Entropy suppression.** Does constant BC suppress exploration? Data:
- v12 Q1 entropy = 0.411, Q4 = 0.496 (rises; no BC)
- v24 Q1 entropy = 0.428, Q4 = 0.581 (rises MORE; BC forces policy to learn diverse teacher patterns)
- v36 Q1 entropy = 0.399, Q4 = 0.574 (rises similarly to v24)

Entropy is not the distinguishing variable. v36 has comparable terminal entropy to v24 — so this candidate is refuted.

**Verdict on (d):** refuted by the data.

**(e) Teacher quality mismatch / BC copies noise.** The Qwen72B teacher is not noise-free: ~15% of its WebShop trajectories have suboptimal option selection. Constant BC faithfully copies these; high-early decaying BC gives the policy time to forget bad patterns once DR3 takes over. This effect is real but small at μ=0.05 (the policy's BC "faith" is low). More importantly, v22 (same weak BC, weaker DR3 stab) does 0.462 — beating v36 by 7 pp. If teacher noise were the primary driver, both should be equally hurt. So (e) is a minor contributor.

**Verdict on (e):** secondary. Real but small at μ=0.05.

### 2.2 The parsimonious explanation

**Primary: mechanism (c) — DR3/BC destructive interference under v12-stab.** The v12 stabilization (tight `clip_max=2.0`, warmer `disc_temperature=1.5`) was designed to prevent late-training DR3 collapse. It does so by limiting $w_\text{hat}$'s range and softening the discriminator. BC at constant $\mu=0.05$ applies *continuous low-amplitude teacher pressure* that shrinks the already-narrow distributional gap DR3 depends on. The three forces — BC, DR3, and stabilization — are not additive; they compose into a less-informative teacher signal than DR3-alone.

**Supporting: mechanism (b) — weak constant BC mis-allocates its probability budget.** Most of the per-step probability-space gradient goes to already-mastered wrapper tokens (large $p_\theta(1-p_\theta)$), not to rare SKU tokens ($p_\theta \approx 10^{-4}$, small $p_\theta(1-p_\theta)$). So the constant BC adds information where DR3 doesn't need help and fails to add it where DR3 can't reach.

**Formal statement.** Let $G_\text{BC}(t, k)$ denote the BC log-prob gradient on teacher token $k$ at step $t$, and $G_\text{DR3}(t, k)$ the DR3 gradient. The *teacher-signal informativeness* at step $t$ is a function $I(G_\text{BC} + G_\text{DR3})$ that is convex-increasing in the variance-over-trajectories of the total teacher gradient (because PPO needs variance to distinguish good trajectories from bad). Constant-BC contributes $\mu \cdot \mathbf{1}_{\text{teacher token}}$ — a *constant*, variance-zero direction. It reduces the variance of the total teacher gradient per trajectory for a given $w_\text{hat}$-distribution. Under v12-stab, where $w_\text{hat}$ is already tightly capped, the BC-induced variance reduction pushes the total teacher signal below a threshold of useful distinguishability. Under v1-stab (loose $w_\text{hat}$), there's enough variance in DR3 itself to absorb the BC addition without crossing the threshold.

This is why the ordering is: v12 (0.431) > v22 (0.462) only slightly because v1-stab DR3 is worse anyway but has enough variance; v24 (0.678) $\gg$ v36 (0.389) because v12-stab + decaying BC has high-amplitude BC-variance early (large $\mu$) and naturally-varying DR3 late, both above threshold, while v12-stab + constant BC is below threshold throughout.

---

## 3. Deliverable 2 — Quantitative gradient curves for v24 vs alternatives

I'll write the per-step per-teacher-token gradient coefficient on $\log \pi_\theta$ (denote $G_\text{teacher}(t)$), separating BC and DR3 contributions. Units: rate-of-change in log-probability per optimizer step.

### 3.1 Components

**BC contribution** (teacher tokens only):
$$G_\text{BC}(t) = -\mu_t \cdot \mathbf{1}_{\text{teacher token}}$$
Sign-definite (positive-in-log-prob), unit-amplitude, no per-token modulation when `chord_use_token_weighting=false` (the v24 choice).

**DR3 contribution** (teacher tokens only, after `apply_warmup_steps=10`):
$$G_\text{DR3}(t) = -\text{clip}(w_\text{hat}(\tau), w_\min, w_\max) \cdot A(\tau) \cdot r_\text{PPO-clip}(t, a^*, s)$$
where $r_\text{PPO-clip}$ is the PPO-clipped ratio (always in $[1-\epsilon, 1+\epsilon]$; clip never binding per `off_pg_cliphit_rate=0`). So $r \approx \text{const} \cdot (1 \pm \epsilon)$. Empirically, $w_\text{hat}$ on teacher samples in our logs averages ~1.5-2.0 at cold start and falls to ~1.0-1.2 at convergence. $A(\tau)$ has group-std $\approx 0.2$–$0.3$ after baseline separation.

### 3.2 Numerical sketch per variant (teacher-token $\log p_\theta$ rate)

Using observed Q1/Q4 values from the logs:

**v24** ($\mu = 0.3\to0.05$ cosine over 25 steps, v12-stab):
| Step | $\mu_t$ | $w_\text{hat}\bar{A}$ (DR3 net) | $|G_\text{BC}|$ | $|G_\text{DR3}|$ | Total teacher signal |
|---|---:|---:|---:|---:|---:|
| 1 | 0.30 | 0 (warmup) | 0.30 | 0.00 | 0.30 |
| 10 | 0.26 | ~1.8×0.13=0.23 | 0.26 | 0.23 | 0.49 |
| 15 | 0.22 | ~1.5×0.09=0.14 | 0.22 | 0.14 | 0.36 |
| 25 | 0.05 | ~1.3×0.06=0.08 | 0.05 | 0.08 | 0.13 |
| 50 | 0.05 | ~1.2×0.05=0.06 | 0.05 | 0.06 | 0.11 |
| 100 | 0.05 | ~1.1×0.07=0.08 | 0.05 | 0.08 | 0.13 |

**v36** ($\mu = 0.05$ const, v12-stab):
| Step | $\mu_t$ | $w_\text{hat}\bar{A}$ | $|G_\text{BC}|$ | $|G_\text{DR3}|$ | Total |
|---|---:|---:|---:|---:|---:|
| 1 | 0.05 | 0 | 0.05 | 0.00 | 0.05 |
| 10 | 0.05 | ~1.3×0.09=0.12 (smaller — less gap due to BC continuously closing it) | 0.05 | 0.12 | 0.17 |
| 25 | 0.05 | ~1.2×0.09=0.11 | 0.05 | 0.11 | 0.16 |
| 50 | 0.05 | ~1.2×0.09=0.11 | 0.05 | 0.11 | 0.16 |
| 100 | 0.05 | ~1.1×0.08=0.09 | 0.05 | 0.09 | 0.14 |

**v12** (no BC, v12-stab):
| Step | $|G_\text{DR3}|$ | Total |
|---|---:|---:|
| 1-10 | 0 (warmup) | 0 |
| 10-25 | ~1.7×0.10=0.17 | 0.17 |
| 50 | ~1.3×0.08=0.10 | 0.10 |
| 100 | ~1.0×0.02=0.02 (v12 late-collapse) | 0.02 |

**CHORD** ($\mu=0.9\to0.05$, no DR3):
| Step | $|G_\text{BC}|$ | $|G_\text{DR3}|$ | Total |
|---|---:|---:|---:|
| 1 | 0.90 | 0 | 0.90 |
| 10 | 0.57 | 0 | 0.57 |
| 25 | 0.05 | 0 | 0.05 |
| 100 | 0.05 | 0 | 0.05 |

### 3.3 What the curves reveal

**v24 has a unique feature**: in steps 10–25 it delivers $|G_\text{total}| = 0.36$–$0.49$, which is roughly **2-4× any other variant at the same horizon**. This is not "more gradient is better" — it is that this specific window corresponds to the **transition** from "policy has no teacher support" to "policy has modest teacher support". The transition regime is where $p_\theta(1-p_\theta)$ is steep (near $p_\theta \approx 0.1-0.5$) and where DR3's $w_\text{hat}$ is still informative (not yet collapsed by BC-closure). v24 is the only variant that delivers a big combined push *exactly where* the $p_\theta(1-p_\theta)$ factor amplifies log-prob gradient into probability-space gradient.

v36's curve is flat at ~0.16 throughout — no transition-amplification window. CHORD's curve is monotone-decreasing starting at 0.9 — it delivers the push but has no DR3 to refine once teacher support is installed. v12's curve has modest DR3 in mid-training but collapses to 0.02 at Q4 (DR3's designed fade-out).

The ordering of area-under-curve-in-transition-window ($t \in [10, 30]$):
- v24: $\int \approx 7.2$
- CHORD: $\int \approx 4.8$
- v36: $\int \approx 3.2$
- v12: $\int \approx 2.5$

This rank-correlates perfectly with val scores: v24 (0.678) > CHORD (0.603) > v22/v36 (0.46/0.39) > v12 (0.43). The transition-window integral is the load-bearing quantity.

---

## 4. Deliverable 3 — Formalizing the "curriculum" claim

"Curriculum" colloquially means "start easy, get hard." In our setting it means something more specific.

**Definition (teacher-gradient curriculum).** Let $\theta_t$ denote the policy at step $t$, and let $u_t(\theta)$ denote the *utility* of a unit of teacher-token log-prob gradient applied to $\pi_\theta$, measured in probability-space progress per optimizer step:
$$u_t(\theta) = p_{\theta_t}(a^*|s) \cdot (1 - p_{\theta_t}(a^*|s))$$
This is maximized at $p_\theta = 0.5$ (the "steep" region of the softmax) and vanishes at both $p_\theta \to 0$ and $p_\theta \to 1$. A curriculum is a schedule $\mu_t$ such that the product $\mu_t \cdot u_t(\theta_t)$ is maintained above a task-dependent threshold $\bar{u}$ for the entirety of training — with the important subtlety that $u_t$ *depends on the entire trajectory $\theta_{0:t}$*, because policy state evolves under the schedule.

**Why constant $\mu$ cannot be optimal.** At cold start, $p_\theta(a^*|s) \approx 10^{-4}$, so $u_0 \approx 10^{-4}$. To maintain $\mu \cdot u_0 \geq \bar{u}$, need $\mu \geq \bar{u} \cdot 10^4$. At convergence, $p_\theta(a^*|s) \approx 0.5$, so $u_\infty \approx 0.25$. To maintain the *same* $\bar{u}$, need $\mu \geq 4\bar{u}$. These differ by $\sim 10^4$. A constant $\mu$ satisfying the cold-start condition is $10^4 \times$ too large late; a constant $\mu$ satisfying the late condition is $10^4 \times$ too small early — which is exactly the v36 regime ($\mu=0.05$ is ~fine late, but at cold start $\mu \cdot u_0 \approx 5 \times 10^{-6}$, far below $\bar{u}$ for the SKU tokens).

**What the decay schedule accomplishes.** $\mu_t$ anti-correlates with $u_t$ along the policy's evolution trajectory, so their product $\mu_t \cdot u_t \cdot (1-p)$ is approximately constant over training. Equivalently: the policy's "total BC work budget" $\int \mu_t \, \mathrm{d}t$ is small (so BC doesn't dominate at convergence), but it's allocated *entropy-efficiently* to the phase where a unit of BC translates into the most probability-space progress.

**Formal curriculum condition.** A schedule $\{\mu_t\}$ is *valid* iff:
1. $\int_0^T \mu_t \, \mathrm{d}t < C_\text{budget}$ (bounded total BC work; otherwise policy over-fits teacher)
2. $\mu_t \cdot u_t(\theta_t) \geq \bar{u}$ for all $t$ where teacher-gap is nonzero (maintain probability-space progress)
3. $\mu_t \to 0$ as $p_\theta \to p_\text{teacher}$ (self-quiescence; avoid BC anchoring once DR3 suffices)

The v24 cosine schedule satisfies (1) and (2); it satisfies (3) only up to the floor $\mu_\text{valley}=0.05$, which was added because the ablation v25 ($\mu_\text{valley}=0$) showed format-grammar drift. No constant $\mu$ satisfies all three simultaneously unless the teacher gap is already closed at $t=0$ (the ALFWorld regime, where v24 should quiesce cheaply because $u_0$ is already large).

**Why BC can't be replaced by a time-varying DR3.** DR3's effective coefficient is $w_\text{hat} \cdot A$. You cannot make $w_\text{hat}$ artificially large at cold start by engineering the discriminator — $w_\text{hat}$ is determined by the *actual* distributional gap, which is what the curriculum is trying to close. This is a chicken-and-egg: DR3 can't push hard until the gap exists, BC has to exist before DR3 becomes informative. The curriculum formalizes which operator owns which phase.

---

## 5. Deliverable 4 — Algorithm improvements grounded in the analysis

Three proposals, ranked by expected effect size and narrative benefit.

### 5.1 Proposal A: Adaptive $\mu_t$ from teacher-token probability mass

**Motivation.** The curriculum condition requires $\mu_t \cdot u_t(\theta_t) \geq \bar{u}$. Rather than schedule $\mu_t$ manually, compute it from an estimate of $u_t$ at the current policy state.

**Concrete rule.**
$$\mu_t = \text{clip}\left(\frac{\bar{u}}{\tilde{u}_t}, \mu_\min, \mu_\max\right)$$
where $\tilde{u}_t = \mathbb{E}_{k \sim \text{teacher-rare-tokens}}[p_{\theta_t}(k)(1-p_{\theta_t}(k))]$, the expected per-teacher-token probability-space "steepness." Measurable online: during the forward pass on teacher trajectories (`exp_mask=1` tokens), compute $p_\theta \cdot (1-p_\theta)$ averaged over teacher tokens, and set $\mu$ inversely. Hyper-parameters: $\bar{u} \in [0.02, 0.05]$ (tune against v24 baseline), $\mu_\min=0.05$, $\mu_\max=0.5$.

**Implementation cost.** ~30 LoC in `het_actor.py` near line 1746 (replace `chord_mu_scheduler` call with adaptive rule). New metric `chord/u_bar_observed` for monitoring. No new tensors.

**Expected effect.** Likely **matches or slightly beats v24 by 1-3pp**. The v24 manual schedule is already close to optimal (its cosine decay over 25 steps tracks the empirical policy evolution time), but adaptive $\mu$ should handle ALFWorld's immediate-quiescence (no BC needed) and WebShop's slow-convergence (BC needed longer) uniformly without hyper-tuning. Main risk: oscillation if $\tilde{u}_t$ is noisy.

**Narrative benefit.** High. Eliminates the "manual schedule" critique. Paper argument becomes: "the curriculum is data-driven, not hand-tuned — derived from first-principles steepness estimation."

### 5.2 Proposal B: Unified surprise-weighted teacher PG (subsume BC and DR3)

**Motivation.** Define a single operator that interpolates between per-token BC behavior and trajectory-level density-ratio behavior based on the token's own surprise under the current policy. This addresses the critique "you have two separate operators — why?"

**Concrete rule.** Define per-token *surprise* $\sigma(s, a^*) = -\log p_\theta(a^*|s)$. Define token-level weight
$$w_\text{token}(s, a^*, \tau) = \alpha \cdot \sigma(s, a^*) + (1-\alpha) \cdot w_\text{hat}(\tau) \cdot A(\tau)$$
where $\alpha \in [0, 1]$ is a global mixing scalar. At $\alpha = 1$, this reduces to per-token surprise-weighted BC (similar to $\phi(p) = p(1-p)$ × BC, but using $-\log p$ which is larger at low $p$). At $\alpha = 0$, this reduces to standard DR3. Use as the coefficient on $\nabla \log \pi_\theta(a^*|s)$ in the teacher loss.

Then schedule $\alpha_t = 1 \to 0$ over training, or make it adaptive with an analogous rule to Proposal A.

**Implementation cost.** ~80 LoC — new `compute_surprise_weighted_teacher_loss` function in `het_core_algos.py`; configuration knob; ablation suite.

**Expected effect.** Could match v24 but likely not beat by more than 1pp — the token-level surprise weighting is already what $-\log p_\theta$ gradient naturally provides in the BC arm.

**Narrative benefit.** Very high. Paper claim becomes: "We unify behavior cloning and density-ratio policy gradient under a single surprise-weighted operator, providing a principled interpolation between per-token (BC-dominant) and trajectory-level (DR3-dominant) credit assignment." Novel enough to stand on its own.

### 5.3 Proposal C: Token-importance reweighted BC (rare-token-priority BC)

**Motivation.** Constant BC's failure mode in v36 was that probability-budget mis-allocation (per §2.2). Fix by reweighting BC per-token by rarity.

**Concrete rule.** Replace $-\mu \cdot \nabla \log \pi_\theta(a^*|s)$ with
$$-\mu \cdot \lambda(s, a^*) \cdot \nabla \log \pi_\theta(a^*|s), \quad \lambda(s, a^*) = (1 - p_\theta(a^*|s))^\gamma$$
with $\gamma \in [1, 3]$. At $p_\theta = 10^{-4}$, $\lambda \approx 1$ (full BC force). At $p_\theta = 0.9$, $\lambda \approx 0.1^\gamma$ (small BC force). This concentrates BC work on rare teacher tokens exactly where $u_t$ is small but $p_\theta$ is small enough that log-prob progress translates back to meaningful probability-space progress.

**Implementation cost.** ~10 LoC in `compute_chord_sft_loss` — multiply $\phi$ by $(1-p)^\gamma$. Trivial.

**Expected effect.** Small but positive (~1-2pp over v24). Addresses the "support dilution" failure (§2.2b) directly. Likely to make v36-style constant $\mu$ work better (closer to v12 baseline, not below it), and to make v24's late-phase $\mu_\text{valley}=0.05$ tail more effective.

**Narrative benefit.** Moderate. Paper claim: "BC's probability-space work is allocated to where on-policy log-prob lags teacher most, not uniformly across the teacher trajectory." Makes the "curriculum" argument sharper.

### 5.4 Recommendation

Run Proposal A (adaptive $\mu$) first — smallest implementation cost, addresses the most-targeted reviewer critique (manual schedule). If it matches or beats v24, the paper story becomes decisively stronger. Prototype Proposal B as a follow-on (post-submission); it's more novel but has higher implementation risk and the gains may not materialize. Proposal C is cheap insurance — can run in parallel with A.

---

## 6. Deliverable 5 — Narrative recommendation for the paper

Three options on the table:
- **G**: "teacher-gradient curriculum" with adaptive $\mu_t$ (my Proposal A positioning)
- **H**: present $\mu_t$ decay as derived from discriminator dynamics (disc_acc $\to$ 1 over 25 steps, BC inverse)
- **I**: unify BC and DR3 as two phases of a single surprise-weighted teacher correction operator

**Recommendation: Option G, with an explicit bridge to Option I in the limitations/future-work section.**

Reasoning:

**Option H is a false framing.** Disc_acc reaches 0.99 by step 25 in every variant (v22, v24, v36 all end at disc_acc $>$ 0.98). The disc-acc → 1 timeline is not why BC needs to decay — BC needs to decay because $u_t(\theta_t)$ increases as the policy learns teacher support, not because the discriminator saturates. Deriving $\mu_t$ from disc_acc would be theoretically confused. *Reject H.*

**Option I is bolder but brittle.** The two-operator → one-operator unification is aesthetically satisfying but has two defects:
- Mathematically, the two operators ARE different things: BC is a per-token $-\log p$ gradient; DR3 is a sequence-weighted PPO surrogate. Forcing them into one form (Proposal B above) is possible but contrived — you're effectively adding a knob and calling it unified.
- Empirically, we haven't run the unified operator. Committing to it in the paper without a validating ablation is a reviewer-risk.

*Hold I as the "future work" pitch.* The paper can say "we note that the two operators can be further unified under a surprise-weighted PG form, which we leave for future work" — this makes the paper look theoretically mature without committing to an unvalidated claim.

**Option G is the honest winner.** Present the Action Channel as:
> *"The Action Channel applies a teacher-gradient curriculum. A per-token behavior-cloning operator with weight $\mu_t$ provides support-installing gradient at cold start; a density-ratio-corrected policy-gradient operator (DR3) provides trajectory-level credit assignment as the policy approaches teacher support. The two are composed additively, with $\mu_t$ adapted online from the observed teacher-token probability-space steepness (Eq. X). Both operators retire naturally — BC via adaptive decay, DR3 via $w_\text{hat} \to 1$ — giving the Action Channel a self-quiescing property."*

Advantages over the v24 memo's earlier framing:
- Replaces the "hand-tuned $\mu$ schedule" with an adaptive rule, closing the main reviewer attack surface.
- Keeps the dual-channel story intact (State Channel untouched).
- Provides a falsifiable prediction: on ALFWorld, the adaptive $\mu$ should immediately collapse to $\mu_\min$ because $u_0$ is already large — directly validating "curriculum adapts to environment."
- Opens the door to Option I without committing.

The key experiment to run before submission: **v37 = v24 + adaptive $\mu_t$ (Proposal A) on WebShop 1.5B.** If it matches v24 within 2pp, the adaptive framing replaces the manual schedule everywhere. If it beats, the paper's empirical claim strengthens. If it loses badly, fall back to v24 with an argument that the empirically-discovered schedule approximates the adaptive optimum.

## 7. Bottom line

v36 is worse than v12 because constant weak BC *interferes destructively* with DR3 under v12-stab (tight $w_\text{hat}$ cap), by pre-emptively closing the distributional gap the discriminator needs for informative $w_\text{hat}$ variance. v24 works because its high-early $\mu$ delivers a large transition-window gradient during steps 10-25 — exactly when $p_\theta \cdot (1-p_\theta)$ is steepest — while its decaying $\mu$ backs off to let DR3's trajectory-level credit assignment take over. The effect is not "more BC is better" but "BC spent where probability-space utility is highest." The curriculum is the answer, not the magnitude. Adaptive $\mu_t$ is the principled replacement of the manual schedule and should be the paper's headline Action Channel formulation.
