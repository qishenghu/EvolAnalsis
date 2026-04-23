# Adaptive-μ Master Plan — 4 Frameworks + Execution

**Last updated**: 2026-04-22
**Purpose**: Consolidate all the thinking behind adaptive-μ design for DUET, and lay out the NeurIPS-ready experiment plan.

---

## 1. Background: Why Adaptive μ?

### DUET's current loss (simplified)

```
L_total = L_DR3(on teacher trajectories)    ← density-ratio corrected PG
        + L_SC(on on-policy trajectories)   ← state channel reward shaping
        + μ · L_BC(on teacher tokens)       ← behavior cloning
```

### Three concepts

- **DR3** = density-ratio correction via discriminator. Computes `w_hat ≈ π_θ/π_teacher`, then uses `w_hat × advantage × log π_θ` as the teacher-sample gradient. **Indirect** teacher use — modulated by trajectory advantage.

- **BC** = direct imitation via `-log π_θ(a_teacher|s)`. Pushes teacher-token probability up regardless of advantage. **Direct** teacher use.

- **μ** = the weight on BC. 
  - Low μ → DR3 dominates (advantage-weighted, may fade teacher signal too fast)
  - High μ → BC dominates (always-on teacher imitation, may override RL signal)
  - v24 hand-tuned `μ=0.3→0.05 over 25 steps` — empirically works on WebShop but doesn't generalize to ALFWorld

### The adaptive problem

We want μ to **automatically adjust** based on training state:
- Closed-form (uses existing observables, no new compute)
- Cross-environment (works on WebShop AND ALFWorld without re-tuning)
- Cross-scale (works on 1.5B, 3B, 7B without re-tuning)
- Theoretically elegant (passes NeurIPS review)

---

## 2. Four Frameworks (4 different signals for μ)

All 4 frameworks share the same goal (auto-set μ) but use different **signals** derived from the training state.

### Framework 1: Discriminator-based (v39)

**Signal**: `dr3/disc_acc` (EMA) — the DR3 discriminator's classification accuracy

**Rule**:
```python
μ = clamp(μ_max · (1 − d) / (1 − d_floor), μ_min, μ_max)
where d = EMA(dr3/disc_acc)
```

**Intuition**:
- `disc_acc ≈ 0.5` (can't distinguish policy from teacher) → policy ≈ teacher → keep μ high
- `disc_acc → 1.0` (fully distinguishable) → policy has moved → retire μ

**Relationship to DR3**: **Directly reuses DR3's discriminator**. One discriminator, two purposes (density ratio + μ control).

**Empirical result**: v39 WebShop = 0.605 reward / 11.5% success (ties CHORD, misses v24). Phase-lag identified: μ knee at step 24 vs v24's step 17.

---

### Framework 2: Teacher NLL (v40)

**Signal**: `chord/sft_loss = -E_teacher[log π_θ(a|s)]` — how well policy predicts teacher tokens

**Rule**:
```python
# sigmoid variant
μ = μ_min + (μ_max − μ_min) · σ(k · (NLL_ema − τ))

# linear variant (from empirical fit)
μ = clamp(0.02 + 0.156 · NLL_ema, μ_min, μ_max)
```

**Intuition**:
- NLL high (policy can't predict teacher) → need BC → μ high
- NLL low (policy fits teacher) → BC redundant → μ low

**Relationship to DR3**: **Independent** of DR3. Uses BC's own loss signal.

**Empirical result**: Best cross-env signal (ALF/WS = 0.30, i.e. naturally much smaller on ALFWorld).

---

### Framework 3: KL-Lagrangian (v43) ⭐ Paper narrative

**Signal**: `KL(π_θ || π_teacher)` — approximated via NLL in black-box teacher setting

**Rule** (dual-ascent):
```python
μ_{t+1} = clamp(μ_t · exp(η · (KL − ε_t)), μ_min, μ_max)
ε_t = ρ · ε_{t-1} + (1−ρ) · KL    # auto-tuned budget (moving mean)
```

**Intuition**: μ is the Lagrange multiplier enforcing `E[KL(π_θ || π_teacher)] ≤ ε`. Dual ascent:
- KL > budget → raise μ (more BC to pull policy back)
- KL < budget → lower μ (relax BC, let RL explore)

**Relationship to DR3**: **Parallel and complementary**.
- DR3 = unbiased advantage-weighted imitation (varies per teacher trajectory)
- BC Lagrangian = hard KL constraint (uniform per teacher token)
- Both regularize policy toward teacher but through different mechanisms

**Why this is the paper's narrative**:
- TRPO / CPO family — reviewers identify immediately
- Formal constraint optimization (strong math)
- Auto-budget ε_t is self-calibrating → "closed-form" claim

**Crucial reduction**: In black-box teacher setting (we don't have π_teacher), `KL ≈ NLL + constant`. So **Framework 3's theory can be implemented using Framework 2's signal**. This is NOT a hack — it's a formal reduction.

---

### Framework 4: DR3 density-ratio quality (v41 ESS)

**Signal**: `dr3/ess_off_window` — effective sample size of teacher importance weights

**Rule**:
```python
μ = μ_max · (1 − ESS / N_window)
```

**Intuition**:
- High ESS → density ratio is uniform across teacher samples → DR3 is reliable → BC not needed
- Low ESS → density ratio is skewed → DR3 noisy → need BC variance reduction

**Relationship to DR3**: **Quality monitor on DR3**. BC acts as safety net when DR3 is unstable.

**Empirical result**: ESS saturates to `N` in both envs → plateau identical. But **time-to-saturation differs** (WebShop 23 steps, ALFWorld 18 steps). Demoted to ablation.

---

## 3. Summary Table

| Framework | Signal | DR3 relationship | Intuition |
|---|---|---|---|
| 1. Disc (v39) | `dr3/disc_acc` | DR3 drives μ | "when discriminator separates them, retire BC" |
| 2. NLL (v40) | `chord/sft_loss` | independent | "when policy fits teacher, retire BC" |
| **3. KL-Lagrangian (v43)** | `KL ≈ NLL` | **parallel complement** | **"μ is Lagrange multiplier on KL constraint"** |
| 4. ESS (v41) | `dr3/ess_off_window` | DR3 quality monitor | "when density ratio is reliable, retire BC" |

---

## 4. Paper Narrative — Framework 3 Story + Framework 2 Implementation

**Title pitch**:
> *"Constrained Dual-Channel Imitation RL: BC as a Dual-Ascent Lagrangian on Teacher KL, Realized via a Mechanism-Matched Surprise Signal"*

**Contribution**:
- **Action Channel = DR3 + Lagrangian BC**, where BC's weight μ is the Lagrange multiplier enforcing an auto-calibrated KL-to-teacher budget
- **State Channel = expert progress shaping** (unchanged)
- **Closed-form**: μ, DR3, SC all use existing observables; no new compute
- **Self-adaptive**: μ_t, budget ε_t, and DR3 w_hat all self-tune via their own error signals

**Why KL-Lagrangian (Framework 3) preserves the dual-channel story**:
- DR3 and BC both live in the **Action Channel** (both use teacher trajectory data)
- They are mathematically distinct operators:
  - DR3: density-ratio correction (varies per trajectory, unbiased)
  - BC: token-level KL enforcement (uniform, biased but trust-region-like)
- Lagrangian duality unifies them under one principle: "constrained policy optimization against teacher distribution"

**Why Framework 2 is the honest implementation**:
- True `KL(π_θ || π_teacher)` requires access to π_teacher (we don't have it — black-box)
- **Reduction**: `KL(π_θ || π_teacher) = E_teacher[log π_teacher - log π_θ] = −NLL + H(π_teacher)`
- `H(π_teacher)` is a constant (teacher is fixed)
- So monitoring NLL is equivalent to monitoring KL up to a constant
- Paper Section 3.X: "Since π_teacher is a black box, we track KL via its reducible component, the teacher-token NLL."

---

## 5. Experiment Plan

### Phase 1: One variant per framework (~12 GPU-hours on WebShop)

Goal: Identify which framework's basic implementation best matches v24's performance.

| Run | Framework | Rule | Rationale |
|-----|-----------|------|-----------|
| **v39b** | 1 (Disc) | `d_ema_alpha: 0.2 → 0.5` | Fix v39's phase-lag (faster EMA) |
| **v40b** | 2 (NLL) | linear: `μ=0.02+0.156·NLL` | Empirical top signal |
| **v41b** | 4 (ESS) | saturating: `(1-(ess/ess_0)^0.5)` | Rescue ESS via ratio |
| **v43a** | 3 (Lagrangian) ⭐ | `η=0.3, ρ=0.9` | **Paper's primary narrative** |

**Success criterion**: Val@100 success ≥ v24's 22.0%.

### Phase 2: Parameter sweeps within best framework (~12 GPU-hours)

Depending on Phase 1 results, run 3-5 variants of the winning framework. Examples:

**If v43 (Lagrangian) wins**: test `η ∈ {0.1, 0.3, 0.5}`, `ρ ∈ {0.8, 0.9, 0.95}`, plus fixed-ε ablation
**If v40 (NLL) wins**: test sigmoid vs linear, τ ∈ {0.5, 0.65, 0.8}, k ∈ {3, 6, 10}

### Phase 3: Cross-env + scaling (~16 GPU-hours + 3B on user's other servers)

- Winner → ALFWorld 1.5B (confirm cross-env generalization, ~5h)
- Winner → WebShop 3B (confirm cross-scale, ~8h on other server)
- Winner → ALFWorld 3B (final cross-env + scale, ~8h on other server)

### Phase 4: Ablations for paper (~10 GPU-hours)

- Fixed-ε Lagrangian (ablation of auto-budget)
- Framework 1 (disc) and Framework 4 (ESS) as comparisons
- v24 hand-tuned schedule as comparison

---

## 6. Config Files Ready to Launch

Generated by algo-engineer, located at `config/duet_paper_experiments_configs/webshop/`:

- `webshop_qwen1.5b_duet_v39b.yaml` — fast-EMA disc
- `webshop_qwen1.5b_duet_v39d.yaml` — earlier-knee disc
- `webshop_qwen1.5b_duet_v39f.yaml` — sigmoid disc
- `webshop_qwen1.5b_duet_v40b.yaml` — linear NLL
- `webshop_qwen1.5b_duet_v41b.yaml` — saturating ESS ratio
- `webshop_qwen1.5b_duet_v41c.yaml` — sigmoid ESS ratio
- **`webshop_qwen1.5b_duet_v43a.yaml` — Lagrangian η=0.3, ρ=0.9** ⭐
- `webshop_qwen1.5b_duet_v43b.yaml` — Lagrangian η=0.1, ρ=0.95

---

## 7. Code Changes (already implemented)

**File**: `agentevolver/module/exp_manager/het_actor.py:1757-1976`

- Extended `disc_acc` mode with sigmoid option + **bug fix for warmup** (disc_acc=0 → 0.5 substitution)
- Extended `nll` mode with linear + ratio-to-initial mappings
- Added `ess_ratio` mode (saturating, sigmoid, velocity variants)
- Added `kl_lagrangian` mode (dual ascent with moving-mean or fixed budget)

All gated behind `chord_mu_adaptive: true` + `chord_mu_adaptive_mode`. Back-compat: v24 baseline still runs unchanged.

---

## 8. Expected Results & Decision Tree

### Ideal outcome: v43a ≥ v24's 22.0% success on WebShop
→ **Paper's Primary narrative is validated**
→ Proceed to Phase 2 (parameter sweeps of Lagrangian)
→ Phase 3 on ALFWorld + 3B

### If v43a fails but v40b matches v24
→ **Paper still claims Lagrangian framing**, acknowledging NLL as the practical implementation
→ Phase 2 explores NLL hyperparameters

### If no Phase 1 variant matches v24
→ **Re-examine implementation**: is there a bug? Is KL proxy wrong?
→ Consider combining signals (v40 + v43 hybrid)

### If NONE of the 4 frameworks works
→ Fall back to v24 as empirical recipe, Lagrangian framing as "principled motivation" in method section

---

## 9. Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| v43 dual-ascent oscillates | Medium | Start with conservative η=0.1-0.3 |
| ESS anchor fails to set | Low | Falls back to μ_max during warmup |
| FSDP state serialization | Low | μ_lagrange_state is per-rank plain float |
| KL proxy (NLL) insufficient | Medium | Can add `NLL_onpolicy` subtraction |
| v39 bug fix changes baseline | Low | Config flag to revert if needed |

---

## 10. Reference Documents

- `analysis_reports/neurips_adaptive_framework_audit.md` — theory-researcher's final audit
- `analysis_reports/adaptive_implementation_plan.md` — algo-engineer's implementation plan
- `analysis_reports/adaptive_signal_discovery.md` — initial signal correlation analysis
- `analysis_reports/adaptive_signal_expansion_empirical.md` — cross-env empirical validation
- `analysis_reports/duet_third_pass_adaptive.md` — theory-researcher's NLL proposal
- `analysis_reports/v39_run_plan.md` — v39 experimental plan
- `analysis_reports/v24_alfworld_dynamics_analysis.md` — v24-ALFWorld diagnosis

---

## TL;DR

**4 frameworks** answer "how to auto-set μ":
1. Disc-driven — uses DR3's discriminator
2. NLL-driven — uses teacher-prediction loss
3. **KL-Lagrangian — μ as Lagrange multiplier (paper's narrative)**
4. ESS-driven — uses DR3's density-ratio quality

**Choice**: Framework 3's theory + Framework 2's implementation (via KL→NLL reduction).

**Phase 1**: Run v39b, v40b, v41b, v43a on WebShop 1.5B (~12h total).
Priority watch: **v43a** (Lagrangian, paper's primary narrative).
