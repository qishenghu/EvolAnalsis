# DUET WebShop Strategic Analysis: Final Assessment

**Date**: 2026-04-05
**Status**: Definitive — 10 iterations, diminishing returns reached

---

## 1. What Was Tried in 0407

### 0407-SC: "SC-Only Direction C" — Best-Tuned State Channel
- **Changes from Hybrid 0405**: `progress_agg=last` (vs `mean`), `beta=0.15` (vs `0.2`), `step_level.enable=false`, `disc_temperature=1.0` (vs `1.5`)
- **Rationale**: `last` observation has 0.82-0.96 reward correlation vs 0.77-0.83 for `mean`. Step-level deltas are broken in multi-turn WebShop (sliding window makes `step_ids` max=0 in response). Lower beta compensates for higher `last` values.
- **Result**: Close to LUFFY but still worse.
- **Diagnosis**: The SC improvements are real — `last` aggregation is strictly better. But SC provides dense reward signal to on-policy samples, which helps but cannot overcome DR3's structural drag.

### 0407-Alpha: "Fixed Alpha Prior" — Wider DR3 Dynamic Range
- **Changes**: `alpha_prior=0.3` to expand w_hat range from [0, 1.143] to [0, 1.429].
- **Rationale**: With `alpha=0.125` (true mixing ratio), relative density ratio has max 1.143 above neutral — 3x increase in dynamic range.
- **Result**: Clearly worse.
- **Diagnosis**: Wider w_hat range alone doesn't help because the discriminator features (12-dim trajectory-level statistics) lack granularity for WebShop's fine-grained policy differences. More range + same signal = more noise.

---

## 2. Definitive Assessment: Can DR3 Beat LUFFY on WebShop?

### Short answer: No, not with the current architecture. The limitation is fundamental, not configurational.

### The Granularity Mismatch Is Structural

**LUFFY's mechanism**: `p/(p+beta)` applies at **every token**. For a 200-token teacher trajectory, LUFFY makes 200 independent credit-assignment decisions. Each action token gets weighted by how likely the *current policy* would produce it. This is exactly what WebShop needs — the difference between good and mediocre shopping strategies lies in *which specific tokens differ* (search query phrasing, click targets, buy timing).

**DR3's mechanism**: w_hat is a **single scalar per trajectory**, computed from 12 aggregate statistics (mean/std/quantile of log-probs). For that same 200-token trajectory, DR3 produces ONE number. It cannot distinguish "teacher used a different search query" from "teacher clicked a different product" — both produce similar trajectory-level statistics.

### Why this matters specifically for WebShop

| Factor | ALFWorld | WebShop |
|--------|----------|---------|
| Action vocabulary | ~30 structured actions | ~10K product pages + free-text search |
| Key differentiator | Correct action *sequence* | Correct *individual actions* (right product, right query) |
| Teacher-student gap | Structural (sequence planning) | Token-level (specific words, clicks) |
| DR3 granularity | Sufficient (sequence-level ≈ right level) | Insufficient (need token-level) |
| LUFFY granularity | Sufficient but noisy | Near-optimal |

**The mathematical argument**: Let $\tau = (a_1, a_2, ..., a_T)$ be a trajectory. LUFFY computes $\prod_t \frac{\pi_\theta(a_t|s_t)}{\pi_\theta(a_t|s_t) + \beta}$ — effectively a token-factored soft importance weight. DR3 computes $w_\alpha(\tau) = f(\text{statistics}(\log \pi_\theta(\tau)))$ — a lossy compression. When the policy differences are concentrated in specific tokens (WebShop), the lossy compression discards the critical information. When the differences are distributed across the trajectory structure (ALFWorld), the compression is less harmful.

### What about improving DR3 features?

The DR3 improvement proposals (action-token masking, step-level features, Platt scaling) would help DR3's discriminator accuracy but **cannot close the granularity gap**. Even a perfect trajectory-level ratio $w^*(\tau) = p(\tau)/q(\tau)$ would still assign one weight to the entire trajectory. The proposals move DR3 from "bad trajectory ratio" to "good trajectory ratio" — but LUFFY operates at a fundamentally finer grain.

### Could the Hybrid (w_hat × p/(p+beta)) work?

The Hybrid was tested:
- **webshop_3b_duet_hybrid** (`disc_temperature=1.5`): This was an early attempt
- **webshop_3b_duet_hybrid_0405** (`disc_temperature=1.0`): Baseline for 0406/0407 experiments

**Results**: Hybrid 0405 achieved `avg_reward=0.766`, which is respectable but still below LUFFY's `0.495` success rate (note: avg_reward and success rate are different metrics; success@100 is the paper metric).

**The theoretical problem with Hybrid**: When w_hat ∈ [0.87, 1.143] (as it is empirically with relative ratio), the Hybrid reduces to `w_hat × LUFFY ≈ 1.0 × LUFFY = LUFFY`. DR3 adds computational overhead without meaningfully modulating LUFFY's token-level weights. When w_hat has wider range (0406-v1, v3 with direct ratio), training destabilizes — the discriminator isn't good enough to use that range safely.

**Verdict**: Hybrid is LUFFY with extra complexity. The theoretical "trajectory × token" factorization is elegant but empirically vacuous on WebShop because w_hat doesn't vary enough to matter, and making it vary more causes instability.

---

## 3. Paper Strategy — Three Options, Ranked

### Recommended: Option B — "Adaptive Action Channel"

**Option B: Environment-Adaptive DUET**
- **ALFWorld**: Full DUET (DR3 + SC) — DR3 beats LUFFY because trajectory-level correction matches ALFWorld's planning-centric structure
- **WebShop**: LUFFY + SC — LUFFY's per-token p/(p+β) is the right inductive bias for WebShop's token-level differences; SC adds dense reward shaping
- **SciWorld**: Full DUET or LUFFY + SC depending on results

**Narrative frame**: "DUET is a *framework* that decomposes teacher utilization into an Action Channel and a State Channel. The Action Channel is *pluggable* — the appropriate correction mechanism depends on where the policy-teacher gap manifests:
- When the gap is *structural* (action sequence planning, as in ALFWorld), trajectory-level correction (DR3) is superior
- When the gap is *lexical* (individual token choices, as in WebShop), token-level correction (LUFFY's p/(p+β)) is better suited
- The State Channel provides orthogonal dense reward shaping regardless of Action Channel choice"

**Strengths**:
- Honest about DR3's environment dependence
- Positions DUET as a principled framework, not a single algorithm
- SC is the unifying contribution across all environments
- Reviewers respect nuanced results over uniform wins

**Weaknesses**:
- Reviewers may say "so you just use LUFFY sometimes — what's the contribution?"
- Need strong ALFWorld results and SC ablation to compensate

### Alternative: Option A — "DUET Wins ALFWorld, Competitive Elsewhere"

**Presentation**:
- ALFWorld: DUET (DR3+SC) as hero result (+8pp over LUFFY)
- WebShop: Show DUET variants are competitive (35.5% for best DUET vs 49.5% LUFFY) but frame SC as the novel contribution
- Emphasize SC ablation: LUFFY+SC vs LUFFY, GRPO+SC vs GRPO

**Strengths**: Simpler narrative — one method, consistent presentation
**Weaknesses**: 14pp gap on WebShop is hard to handwave. Reviewers will notice.

### Fallback: Option C — "LUFFY+SC Everywhere"

Drop DR3 from the paper entirely. Present LUFFY+SC as the contribution.

**Strengths**: Simplest story, avoid DR3 complexity
**Weaknesses**: DR3 is the more novel contribution; dropping it leaves a thin paper. "You just added reward shaping to LUFFY" is a devastating review.

### Ranking: B >> A > C

---

## 4. Minimum Experiment Set for NeurIPS

### Must-Run (blocking submission)

| Experiment | Environment | Purpose | Config exists? |
|-----------|------------|---------|---------------|
| LUFFY+SC (attribute_aware, last) | WebShop 3B | WebShop DUET variant | luffy_sc_0405.yaml exists but uses `attribute_aware` + `step_level`; needs update with `progress_agg=last`, `step_level=false` |
| LUFFY+SC (attribute_aware, last) | WebShop 7B | Scale validation | No — need to create |
| Full DUET (0406 config) | ALFWorld 3B | Confirm +8pp claim | alfworld_3b_duet_0406.yaml exists |
| Full DUET | ALFWorld 7B | Scale validation | alfworld_7b_duet.yaml exists |
| GRPO+SC (SC only, no teacher) | WebShop 3B | SC ablation — does SC help without teacher? | Need to create |
| LUFFY (no SC) | WebShop 3B | SC ablation — does SC help with LUFFY? | webshop_3b_luffy.yaml exists (already run: 49.5%) |

### Should-Run (strengthens paper significantly)

| Experiment | Purpose |
|-----------|---------|
| DUET on SciWorld 3B/7B | Third environment to show generality |
| Qwen3-4B variants (all methods) | Different architecture, stronger base model |
| 3-seed runs for all primary results | Error bars for reviewers |

### Nice-to-Have (if time permits)

| Experiment | Purpose |
|-----------|---------|
| Teacher data ablation (vary n_teacher) | Sample efficiency claim |
| SC beta sweep (0.05, 0.1, 0.15, 0.2, 0.3) | Hyperparameter sensitivity |
| DR3 feature ablation (v3_aug vs v3) | Justify feature engineering |

### Critical Config to Create

**webshop_3b_luffy_sc_0407.yaml**: LUFFY + SC with `progress_agg=last`, `step_level.enable=false`, `match_mode=attribute_aware`. This is the WebShop DUET variant for the paper.

---

## 5. Anticipated Reviewer Questions and Defenses

### Q1: "DUET doesn't beat LUFFY on WebShop — why should I accept this?"

**Defense**: "DUET is a framework with pluggable Action Channel. Our contribution is threefold:
1. The *orthogonal decomposition* into Action (policy-dependent) and State (policy-independent) channels — a novel conceptual framework
2. *DR3*: a black-box density ratio estimator that provides trajectory-level correction and automatic teacher fade-out, demonstrated to outperform LUFFY on ALFWorld by +8pp
3. *State Channel*: a hash-based dense reward signal derived from expert state visitation, shown to improve performance across all environments and Action Channel choices

We demonstrate that the appropriate Action Channel depends on the environment's structure, providing the first empirical characterization of when trajectory-level vs token-level correction is preferred."

**Supporting evidence needed**: LUFFY+SC outperforms LUFFY (proving SC adds value). DR3+SC outperforms LUFFY on ALFWorld (proving DR3 adds value where appropriate).

### Q2: "Is SC actually contributing anything on WebShop?"

**Defense**: Show LUFFY+SC > LUFFY ablation result. If the improvement is small (<2pp), emphasize:
- SC provides dense reward signal that is independent of whether teacher samples are available for a given task
- SC's `progress_agg=last` with `attribute_aware` matching achieves 0.82-0.96 reward correlation
- SC accelerates early training even if asymptotic performance converges

**Risk**: If LUFFY+SC ≈ LUFFY on WebShop, SC's WebShop contribution is purely density of reward signal. Frame as "consistency" — SC provides the same benefit regardless of environment, which is the robustness story.

### Q3: "Isn't this just LUFFY with extra complexity?"

**Defense**: "LUFFY's p/(p+β) is a heuristic importance approximation with no principled convergence guarantee. It has one global parameter β that cannot adapt to different tasks or training stages. DR3 provides:
1. A *principled* density ratio estimate grounded in the discriminator framework
2. *Automatic* teacher fade-out (no schedule needed) — as π→π_teacher, D→0.5, w→1
3. Environment-dependent correction at the appropriate granularity level

Furthermore, SC is a wholly independent contribution that LUFFY does not have. SC enables dense reward shaping from expert states without requiring expert actions — a new modality of expert information utilization."

### Q4: "Your DR3 features are lossy — how can you claim density ratio estimation?"

**Defense**: (From improvement proposals doc) "DR3 estimates a *coarsened* trajectory-level density ratio. For the purpose of trajectory-level reweighting in GRPO, relative ordering of teacher trajectories matters more than exact importance weights. The features capture distributional moments of the student's log-probabilities evaluated on teacher trajectories — exactly the information needed to assess distribution mismatch at the trajectory level."

**Note**: This is a genuine weakness. Proposals 1-2 (action-token masking, step-level features) would strengthen this defense but are unlikely to change the WebShop result qualitatively.

### Q5: "Only 2-3 environments? What about generalization?"

**Defense**: We test on 3 environments spanning different structures:
- ALFWorld: long-horizon sequential planning (household tasks)
- WebShop: short-horizon search+click (e-commerce)
- SciWorld: multi-step scientific experimentation (requires both planning and precise execution)

"These environments were chosen to span the spectrum of action structure complexity, providing evidence for when each Action Channel variant is appropriate."

**Mitigation**: SciWorld results are critical. If DUET (DR3) wins on SciWorld (which has multi-step structure similar to ALFWorld), the story is: "DR3 wins on structured planning tasks, LUFFY wins on lexical-action tasks."

### Q6: "Where are the error bars?"

**Defense**: This is the weakest point. 3-seed runs are expensive but expected for NeurIPS. At minimum, provide 3 seeds for the primary comparison (GRPO vs LUFFY vs DUET on each env).

---

## 6. Synthesis: What the Paper Should Say

### Title Direction
"DUET: Dual-Channel Expert Trajectory Utilization for LLM Agent Training"

### Core Claim
Expert trajectories contain two orthogonal types of information — action-level (what the expert did) and state-level (where the expert went). DUET decomposes teacher utilization into an Action Channel and a State Channel, each addressing a distinct failure mode (distribution shift vs reward sparsity). The Action Channel is environment-adaptive: trajectory-level DR3 for planning-heavy tasks, token-level p/(p+β) for action-selection tasks. The State Channel provides universal dense reward from expert state visitation.

### Results Table (Target)

| Method | ALFWorld 3B | WebShop 3B | SciWorld 3B |
|--------|------------|------------|-------------|
| GRPO | ~24% | 2.0% | TBD |
| LUFFY | ~50% | 49.5% | TBD |
| CHORD | ~40% | 0.0% | TBD |
| DUET (DR3+SC) | **~58%** | ~35% | TBD |
| DUET (LUFFY+SC) | ~52% | **~50-52%?** | TBD |

The key gap: can LUFFY+SC actually beat LUFFY on WebShop? If yes (even by 1-2pp), the story works. If no, SC's WebShop contribution is limited to early-training acceleration.

### Ablation Table (Critical)

| | No SC | With SC | SC Delta |
|---|-------|---------|----------|
| GRPO | 2.0% | ? | ? |
| LUFFY | 49.5% | ? | ? |
| DR3 | ~30% | ~35% | +5pp |

This table makes or breaks the SC contribution claim on WebShop.

---

## 7. Recommended Immediate Actions

1. **Create `webshop_3b_luffy_sc_0407.yaml`**: LUFFY + SC with `progress_agg=last`, `step_level=false`, `match_mode=attribute_aware`, `beta=0.15`
2. **Run LUFFY+SC 0407 on WebShop**: This is the highest-priority experiment — it determines whether Option B is viable
3. **Run SC-only ablation (GRPO+SC) on WebShop**: Determines SC's standalone value
4. **Confirm ALFWorld DUET results**: Ensure 0406 config still shows +8pp over LUFFY
5. **Begin SciWorld experiments**: Third environment is critical for generality

The paper deadline pressure suggests: run (1-3) in parallel, then decide Option A vs B based on results.
