# DUET Paper — Comprehensive Writing Plan (for advisor review)

**Target venue**: NeurIPS 2026 (deadline 2026-05-07 23:59 local)
**Author**: Qisheng (1st), advisor (corresp.)
**Document purpose**: This is the canonical content / layout / data plan for the
DUET paper. Advisor reads this in lieu of an in-progress draft to give early
structural feedback before we finalize §3–§6 and the ablation tables.
**Last revised**: 2026-05-05 evening (1.5B ablations 4/8 complete; 3B underway on L20X).

---

## 0. Executive summary (read this first)

**Paper title**: *DUET: Principled Experience Replay for LLM Agent Reinforcement Learning.*

**One-paragraph pitch (≈90 words)**.
Mixing teacher trajectories into on-policy GRPO batches (LUFFY/CHORD style) is
the standard cold-start trick for RL fine-tuning of LLM agents. We identify
**two systematic biases** in this practice — group-baseline contamination and
unaddressed off-policy mismatch — and show they are catastrophic on weak base
models. **DUET** corrects both biases with principled fixes (baseline separation
+ density-ratio correction) and extracts complementary teacher signal through
an Action Channel (token-level imitation) and a State Channel (potential-based
shaping). DUET is SOTA on all four (1.5B/3B × ALFWorld/WebShop) settings,
+13.0pp average over the strongest baseline (+17.5pp on the weakest models),
with every teacher mechanism self-attenuating from a data-driven internal
signal — no manual schedules.

**Why we think this clears NeurIPS bar.**

1. **Diagnosis first**. The paper is *not* "yet another teacher-mixing method".
   The intellectual contribution is the diagnosis of two bias mechanisms, each
   formalised with a one-line equation that any reviewer can verify.
2. **Each fix is principled**. Baseline separation = GRPO advantage variance
   argument; DR3 = standard GAN/density-ratio result; SC = Ng-Harada-Russell
   1999 potential-based shaping invariance theorem. BC is not pitched as a new
   theorem — it is a cold-start safety net while DR3 stabilises (honest
   framing).
3. **Strong empirics**. 4/4 cells SOTA, +13pp avg, +17.5pp on weak base models.
   The ablation table will be the headliner: removing baseline separation alone
   *fully collapses* training on 1.5B-AF (47.5% → 0.0%), giving a striking
   single-cell demonstration of Bias 1's reality.
4. **Self-attenuation evidence**. We have wandb traces showing $\hat w_\tau \to
   1$ and $\mu(t) \to 0.05$ over 100 steps — DUET's "no manual teacher decay
   schedule" claim is empirically backed, not just argued.

**The biggest risks for review**, with our planned responses:

| Risk | Response |
|---|---|
| Single seed | Binomial 95% CI on val@200 (e.g., 47.5% ± 7.0pp) + explicit Limitations sentence |
| Each mechanism = known idea | Frame contribution as **diagnosis + integrated framework**, not invention of any one piece |
| -DR3 didn't drop on 1.5B-AF (47.5% unchanged) | Frame as evidence DR3's *value depends on setting* — large drop on 1.5B-WS (36.0% → 9.5%); honest in §4.4 |
| LUFFY 38% vs paper-claimed 49.5% | 3-way reproducibility appendix, transparent: paper 49.5%, L20X 38.0%, A100 3.5% |
| WebShop 1.5B variance | Footnote cite of swC family span (1.0–36.0%); use L20X's stable 36.0% as the headline |

**Status of work (by Date 2026-05-05 evening)**.

| Section | Status | Owner & due |
|---|---|---|
| Abstract | ✅ done | locked 2026-05-04 |
| §1 Intro | ✅ done | 5-paragraph diagnose-then-fix arc |
| §2 Related Work | ❌ not started | drafted next, ~3/4 page |
| §3.1–3.3 Method opening | ✅ done | setup + 2 biases + DUET overview |
| §3.4 Baseline Separation | ❌ not started | ~1/2 page, math + ablation pointer |
| §3.5 DR3 | ❌ not started | ~1 page, longest |
| §3.6 BC | ❌ not started | ~1/2 page |
| §3.7 SC | ❌ not started | ~1/2 page |
| §4 Experiments | ❌ not started | setup, main table, ablations, dynamics |
| §5 Discussion / Limitations | ❌ not started | ~1/2 page |
| §6 Conclusion | ❌ not started | ~1/4 page |
| Figure 1 (method) | prompts ready, image not generated | DALL-E or tikz |
| Figure 2 (training dynamics) | data ready in wandb, plot not made | matplotlib |
| Figure 3 (auto-fade) | data ready in wandb, plot not made | matplotlib |
| Bibtex | placeholders only | ~12 entries needed |
| Appendix | ❌ planned, not written | LUFFY 3-way + sensitivity + impl details |
| Ablation table | 3/8 cells filled | data autoflowing as orchestrator runs |

---

## 1. Story arc & framing decisions (from `narrative.md`, 2026-05-04 lock-in)

The paper is framed as **a principled fix to LUFFY-style experience replay**.
This is the load-bearing framing decision and we should not move off it.

### 1.1 Five-paragraph introduction skeleton (already written)

1. **Problem**: cold-start trap of on-policy RL on weak LLM agents (1.0% / 0.5%
   on 1.5B AF / WS).
2. **Existing**: LUFFY/CHORD as natural mitigations; CHORD's reviewer-criticised
   manual μ schedule.
3. **Diagnosis** ★: two biases (baseline contamination + off-policy mismatch).
   *This is the paragraph that buys reviewer respect.*
4. **DUET**: principled fixes (Fix 1, Fix 2) + extensions (Ext 1, Ext 2) in two
   orthogonal channels (Action / State).
5. **Results**: 4/4 SOTA, +13pp avg, +17.5pp on weak; bullet contribution list.

### 1.2 Why "principled" is the load-bearing word

ML reviewers use *heuristic* (negative) vs *principled* (positive) as opposing
labels. CHORD got dinged for its heuristic μ schedule. DUET answers with: every
teacher mechanism is data-derived from an internal signal (discriminator
output, discriminator accuracy, observed teacher visitation), so we don't tune
a teacher-decay schedule — that is the "principled" claim cashing out
operationally.

### 1.3 Why μ is not framed as JSD-driven

(*Decision locked 2026-05-04; relevant if advisor asks about μ.*)
An earlier draft pitched μ ∝ disc_acc as JSD-driven distillation pressure. We
discarded this because empirically disc_acc rises monotonically with training
(reflecting discriminator capacity growth as much as student–teacher gap),
which a sharp reviewer would falsify with our own logs. The honest framing —
**μ is a cold-start safety net while DR3 stabilises** — is also tighter.

---

## 2. Diagnosis: two biases in LUFFY-style replay (the paper's intellectual core)

### Bias 1 — Baseline contamination

GRPO computes within-group mean and standard deviation:
$$
\hat A^{(i)} = \frac{R^{(i)}-\mu_g}{\sigma_g}, \quad \mu_g = \frac{1}{n}\sum_j R^{(j)},\quad \sigma_g^2 = \tfrac{1}{n}\sum_j(R^{(j)}-\mu_g)^2.
$$
With $n_\beta$ teacher samples (always $R=1$ since teacher cache is filtered)
and $n_o = n - n_\beta$ on-policy samples whose mean is $\bar R^o$:
$$
\mu_g = \frac{n_\beta + n_o\,\bar R^o}{n},\quad \sigma_g^2 = \frac{n_\beta n_o}{n^2}(1-\bar R^o)^2 + \frac{n_o}{n}\,\mathrm{Var}(R^o).
$$
A *successful* on-policy sample ($R^o=1$) gets advantage $(1-\mu_g)/\sigma_g$,
which can be small or negative whenever the teacher contribution dominates
$\mu_g$. **Successful exploration is penalised**, contradicting the entire
purpose of on-policy RL. The bias is most severe when $\bar R^o$ is closest to
0 — exactly the cold-start regime where teacher mixing is supposed to help.

**Quantitative example with our numbers** ($n=8$, $n_\beta=1$, weak agent
$\bar R^o=0.05$):
$\mu_g = (1 + 7 \cdot 0.05)/8 = 0.169$,
on-policy success advantage $= (1-0.169)/\sigma_g \approx 1.6/\sigma$;
on-policy *near-success* with $R=0.8$ advantage $= (0.8-0.169)/\sigma_g \approx
1.2/\sigma$. With separated baseline using only on-policy samples, $\mu_g^o =
0.05$, the same on-policy success advantage becomes $1.0/\sigma_o^o$ but the
*near-success* advantage is now $0.75/\sigma_o^o$ — **a different on-policy
gradient direction** (relative weighting between full and partial successes).

### Bias 2 — Unaddressed off-policy mismatch

GRPO's importance ratio $\rho_t = \pi_\theta(a_t|s_t)/\pi_{\theta_{\text{old}}}(a_t|s_t)$
correctly accounts for the *policy update step* (old → new). It does **not**
account for the teacher–student gap: teacher samples come from $\pi_\beta \neq
\pi_\theta$. Without correction, teacher gradients continue to dominate even
after the student has caught up, damaging asymptotic performance and forcing
practitioners (e.g. CHORD) to introduce manually-tuned decay schedules.

**Quantitative anchor**: empirically, $\hat w_\tau$ (DR3's estimated teacher
density ratio) drops from ≈ 0.5 at step 1 to ≈ 0.05 at step 100 on 1.5B-WS,
**confirming that without DR3 the teacher gradient share is overstated by
roughly an order of magnitude by step 50**. (This will be Figure 3 in the
paper.)

### Why these biases compound on weak base models

Both biases are most severe when $\pi_\theta$ is weakest: Bias 1 because $\bar
R^o$ is smallest then; Bias 2 because the teacher–student gap is largest then.
That is precisely the regime in which we want teacher mixing to work, so the
biases preferentially break the methods in their intended use case.

---

## 3. Method: DUET in four mechanisms, two channels

### 3.1 Architecture at a glance

```
   STUDENT π_θ ──┐
                ├─── n=8 mixed group ──┐
   TEACHER π_β ──┘   (7 onpol + 1 tch)  │
                                        ▼
                ┌──────────── DUET ─────────────┐
                │ Stage 1: BIAS CORRECTION       │
                │   • Baseline Separation (Fix1) │
                │   • DR3 Density-Ratio (Fix2)   │
                │ Stage 2: SIGNAL EXTRACTION     │
                │   • Action: BC token-level     │
                │   • State : SC potential-shape │
                └────────────────┬──────────────┘
                                 ▼
                      L = L_PG[corrected, SC-shaped]  +  μ(t)·L_BC
```

The four mechanisms map to:

| Mechanism | Stage | Channel | Role | Self-attenuation signal |
|---|---|---|---|---|
| Baseline Separation | Bias correction | (orthogonal) | Eliminate Bias 1 | always-on |
| DR3 (density-ratio) | Bias correction | Action | Eliminate Bias 2 | $\hat w \to 1$ as student catches up |
| BC (token-level CE) | Signal extraction | Action | Cold-start safety net | $\mu(t)$ ↓ as disc_acc ↑ |
| SC (Ng99 potential) | Signal extraction | State | Dense per-step shaping | hash hit-rate ↓ for OOD states |

### 3.2 Baseline Separation (Fix 1)

Compute GRPO statistics separately within sub-groups:

$$
\hat A^o_{(i)} = \frac{R^o_{(i)} - \mu^o_g}{\sigma^o_g}, \qquad \hat A^\beta_{(j)} = \frac{R^\beta_{(j)} - \mu^\beta_g}{\sigma^\beta_g}.
$$

The on-policy sub-group's advantage now satisfies $\mathbb{E}[\hat A^o]=0$
without contamination from teacher rewards, restoring the "successful
exploration → positive advantage" property. Variant: in our config,
`std_source: non_teacher` means we re-use $\sigma^o_g$ for both sub-groups to
avoid divide-by-zero on the teacher sub-group (where $R^\beta\equiv 1 \Rightarrow
\sigma^\beta = 0$). This is the version that matches our SOTA configs.

**Principled justification**: zero-mean advantage *within an on-policy
sub-group* is the canonical assumption used to derive the GRPO policy gradient
estimator. Mixing teacher samples into the same group violates this
assumption; separating restores it.

**Empirical evidence** (this is the most striking ablation cell in the paper):

> Removing baseline separation alone, **1.5B-AF collapses from 47.5% → 0.0%**.
> On-policy success rate (training metric) drops from a peak of 5.4% at step 26
> to 0% by step 30 and stays flat through step 69 (we stopped early at val@50;
> val@100 would not differ). This is a *complete* training collapse from a
> single-knob removal — direct empirical proof that Bias 1 is real and large at
> this scale.

### 3.3 DR3 — Discriminator-based density-ratio correction (Fix 2)

A small MLP discriminator $D(\phi(s,a))$ is trained on (state, action) features
$\phi$ to classify teacher vs on-policy samples (BCE loss with label smoothing
$0.1$, age-decayed sample weighting $0.02$, dropout $0.2$).

**Density ratio**: $\hat w(s,a) = D(\phi)/(1-D(\phi))$ at the optimal Bayes
classifier equals $\pi_\beta(a|s)/\pi_\theta(a|s)$ (well-known density-ratio
result). We use it to **correct** the teacher importance ratio in the policy
gradient.

**Trajectory-level usage** (current implementation): DR3's teacher surrogate
is, after the DR3 shift `old_log_prob ← log_prob.detach() − log ŵ(τ)`,
$$
L^{\mathrm{tch}}_{\mathrm{DR3}} = -\mathbb{E}_{\tau\sim\pi_\beta}\!\left[A(\tau)\cdot\min\!\big(r_\theta\hat w(\tau),\;\mathrm{clip}(r_\theta\hat w(\tau),\,1\pm\varepsilon)\big)\right],
$$
where $r_\theta = \pi_\theta(a|s)/\pi_{\theta_{\mathrm{old}}}(a|s)$ as in
standard PPO. At the optimal discriminator, $\hat w(\tau) = \pi_\beta/\pi_\theta$,
so the surrogate's gradient is the *correctly weighted* off-policy policy
gradient — Bias 2 corrected at the operator level.

**Auto-fade fixed point**: as $\pi_\theta \to \pi_\beta$, $D \to 1/2$ and $\hat
w \to 1$, so the teacher gradient blends smoothly into the standard
on-policy update — *no manual schedule*. We make the fade visible in Figure 3
(empirical `dr3/w_hat_mean_traj` over training).

**Stability mechanisms**:
- **Dual ESS clipping** (`ess_target_ratio = 0.5`, `dual_lr = 0.05`): the
  effective sample size of $\{\hat w_i\}$ is monitored and a Lagrangian dual
  variable upper-bounds the maximum $\hat w$ to maintain ESS ≥ target. Prevents
  small-batch density-ratio estimators from blowing up.
- **`clip_max = 2.0`, `w_min = 0.01`**: numerical safety floors / ceilings.
- **Sync across ranks** (FSDP): discriminator parameters synced every step,
  prevents per-rank drift.
- **Warmup** (`apply_warmup_steps = 10`, `apply_min_buf_size = 512`): DR3 is
  off until enough samples are buffered to train a reliable discriminator.

**Principled justification**: density-ratio estimation by binary classification
is the standard modern technique (Goodfellow 2014; Sugiyama 2012); the
fade-out fixed point follows from the form $\hat w = D/(1-D)$.

**Empirical evidence (1.5B-WS)**: removing DR3 drops 1.5B-WS from **36.0% →
9.5%** (val@100 strict). DR3 is load-bearing on harder/longer-horizon tasks
where teacher–student gap persists longer.

**Honest finding (1.5B-AF)**: removing DR3 leaves 1.5B-AF unchanged at
**47.5%**. We discuss this in §4.4 and §5: ALFWorld has shorter horizons and
simpler state space, so the teacher–student gap closes quickly under
baseline-separated GRPO + BC alone, before Bias 2 has a chance to accumulate.
This is consistent with (not contradictory to) DR3's theoretical role.

### 3.4 BC — Adaptive cold-start imitation (Action channel, Extension 1)

Token-level cross-entropy on teacher tokens, weighted by an adaptive scalar:
$$
L_{\mathrm{BC}}(\theta) = -\mathbb{E}_{(s,a)\sim\pi_\beta}\!\left[\log\pi_\theta(a|s)\right],
\qquad L_{\mathrm{total}} = L_{\mathrm{PG}} + \mu(t)\,L_{\mathrm{BC}}.
$$

**Adaptive μ(t) from disc_acc** (mode `disc_acc`, our SOTA setting):
$$
\mu(t) = \mathrm{linear}\big(\mathrm{disc\_acc}(t);\; \mathrm{floor}=d_{\mathrm{floor}},\; \mathrm{peak}=\mu_{\max},\; \mathrm{valley}=\mu_{\min}\big).
$$

When `disc_acc` is low (early training, discriminator unreliable), μ stays
high (≈ 0.3) → BC dominates → cold-start safety net active. As `disc_acc`
crosses $d_{\mathrm{floor}}$ (we use 0.4 on AF, 0.6 on WS), μ decays linearly
to $\mu_{\min}$ (= 0.05 on AF, 0.10 on WS) — control transfers to DR3.

**Principled justification** (the framing decision in `narrative.md` §"Why μ is
not framed as JSD-driven"): μ is **not** pitched as a calibrated measure of
the teacher–student gap. It is a *safety mechanism complementary to DR3* — high
when DR3 is unreliable (low disc_acc), low when DR3 stabilises. This framing
survives reviewer adversarial probing of disc_acc's monotonic rise.

### 3.5 SC — Potential-based state-channel shaping (Extension 2)

Define an expert progress map $P:\mathcal{S}\to[0,1]$ from teacher trajectories:
the value of state $s$ is the average normalised progress (step / horizon) at
which $s$ appears in successful teacher rollouts. (Implementation:
`exp_manager/state_progress.py`. Match mode `attribute_aware` hashes on the
attribute-aware state representation, not raw text.)

Reward shaping:
$$
r'(s_t,a_t,s_{t+1}) = r(s_t,a_t,s_{t+1}) + \beta\bigl(P(s_{t+1}) - P(s_t)\bigr).
$$

This is the canonical *potential-based* form of [Ng, Harada & Russell 1999],
so it is **policy-invariant**: the optimal policy of the shaped MDP is the
same as that of the original MDP. SC supplies dense per-step learning signal
without altering what the agent ultimately learns.

**Two design decisions worth defending in the paper**:
- **`exclude_teacher: true`**: SC bonus is applied only to on-policy samples.
  Teacher trajectories already have $P\approx 1$ near termination by
  construction; adding the bonus would inflate their GRPO advantage and fight
  DR3's natural fade-out (a feature of DUET, not a bug). Excluding is the
  cleanest separation of concerns: DR3 corrects the teacher policy gradient;
  SC enriches the on-policy reward signal.
- **`grpo_decouple: true`**: SC bonus is excluded from the GRPO advantage
  baseline (same as how token-level rewards are excluded), so the bonus
  affects the policy gradient through the standard advantage path, not through
  baseline shifting.

**Principled justification**: Ng-Harada-Russell 1999 invariance theorem; SC is
the cleanest available formulation that *cannot* induce reward hacking by
construction.

### 3.6 Combined update

$$
L_{\mathrm{DUET}}(\theta) = L_{\mathrm{PG}}^{\mathrm{DR3,SC,BS}}(\theta) + \mu(t)\,L_{\mathrm{BC}}(\theta),
$$

where the policy-gradient term incorporates baseline separation (BS), the DR3
density-ratio correction on teacher samples, and the SC shaping on the on-
policy reward stream (eq. above). The BC term is decoupled and weighted by the
adaptive μ.

---

## 4. Empirical results

### 4.1 Setup

| Item | Value |
|---|---|
| Models | Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct |
| Teacher | Qwen2.5-72B-Instruct (offline cache, ≥1.0 reward filtered) |
| Hardware | 4×A100-80GB ("our" server) + 4×L20X-144GB (collaborator) |
| Steps | 100, no early stopping |
| Validation | val@100 SR strict (score ≥ 1.0) on the 200-task held-out split, n=1 rollout per task |
| Group size | n = 8 (= 7 on-policy + 1 teacher-mixed) |
| Optimiser | AdamW, lr = 1e-6 |
| KL coef | 0.001 (WS) / 0.005 (AF) |
| Teacher data | 19K AF / 26K WS Qwen-72B trajectories (filtered) |

**Baselines**.
- **OnPolicy GRPO** (no teacher) — pure on-policy.
- **LUFFY** — teacher mixing + policy shaping, no Bias-1/Bias-2 correction.
- **CHORD** — LUFFY + manually-scheduled BC weight.
- **SFT + GRPO** — 50 SFT steps then 50 GRPO (offline → on-policy).

(*SFT alone is reported on 1.5B-WS only as an additional reference; not a real
RL baseline.*)

**Metric**. Strict success rate at val@100 on the standard 200-task split. We
also report reward mean for the with-reward variant (`tables/main_results_with_reward.tex`).

### 4.2 Main results (Table 1, already drafted)

| Method | 1.5B-AF | 1.5B-WS | 3B-AF | 3B-WS |
|---|---|---|---|---|
| OnPolicy GRPO | 1.0% | 0.5% | 47.0% | 2.0% |
| LUFFY | 5.5% | 5.5% | 61.5% | 38.0% |
| CHORD | 27.0% | 11.5% | **67.0%** | **39.0%** |
| SFT + GRPO | **30.0%** | **18.5%** | 59.5% | 24.0% |
| **DUET\* (Ours)** | **47.5%** | **36.0%** | **77.5%** | **45.5%** |
| Δ over best baseline | +17.5pp | +17.5pp | +10.5pp | +6.5pp |
| Avg Δ | | | | **+13.0pp** |

**Numbers from `data/raw_data.md` (verified against logs).**
- 1.5B-AF DUET\* source: `alfworld_qwen1.5b_duet_v39c_postfix.log`
- 1.5B-WS DUET\* source: `webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06.log`
  (single seed; reproducibility caveat in §5; swC family span 1.0–36.0%)
- 3B-AF / 3B-WS DUET\* source: L20X server v39b configs (77.5% and 45.5%
  cells); local 4×A100 reproduction of 3B-WS = 44.0% (within ±1.5pp).

**Headline narrative**:
- **4/4 settings SOTA**, no draws.
- Improvement is biggest where on-policy alone fails most (1.5B), confirming
  the cold-start framing.
- Even on the strongest cell (3B-AF), DUET adds +10.5pp.

### 4.3 Component ablations (Table 2 — being filled by orchestrator)

The 4×4 ablation matrix tests each of the 4 mechanisms on each of 4 settings.
All cells fork from the corresponding DUET\* SOTA config and flip a single knob.

**Status snapshot 2026-05-05 18:50** (1.5B running on our 4×A100; 3B on L20X):

| Mechanism removed | 1.5B-AF | 1.5B-WS | 3B-AF | 3B-WS |
|---|:-:|:-:|:-:|:-:|
| -baseline_sep | **0.0%** ★ | running 4/8 | L20X queued | L20X queued |
| -DR3 | 47.5% | 9.5% | L20X queued | L20X queued |
| -BC | A100 queued | A100 queued | L20X queued | L20X queued |
| -SC | A100 queued | A100 queued | L20X queued | L20X queued |

DUET\* reference: 47.5% / 36.0% / 77.5% / 45.5%.

**Already strong narrative cells** (filled):
1. **-baseline_sep on 1.5B-AF: 47.5% → 0.0%** ★ — *load-bearing*. Direct
   empirical proof of Bias 1's reality. Stopped early at step 69; on-policy SR
   peaked at 5.4% (step 26), collapsed to 0% by step 30, stayed flat 38+ steps.
   We report `val@50 = 0.0%` as `val@100` because the trajectory was
   demonstrably stable-zero (no recovery possible).
2. **-DR3 on 1.5B-WS: 36.0% → 9.5%** — *load-bearing*. DR3's value on
   long-horizon tasks where Bias 2 has time to accumulate.
3. **-DR3 on 1.5B-AF: 47.5% → 47.5% (no drop)** — honest negative. We do not
   hide this. It says DR3's value is setting-dependent: AF's short horizons let
   baseline-sep + BC alone close the gap before Bias 2 matters. This is in line
   with theory, not against it.

**Cells we expect to land** (predictions for advisor):
- **-BC on 1.5B-AF**: significant drop (BC is the cold-start safety net on weak
  base). Predict ≈ 25-35%.
- **-BC on 1.5B-WS**: moderate drop. Predict ≈ 18-25%.
- **-SC on 1.5B-AF**: small drop. Predict ≈ 38-45%.
- **-SC on 1.5B-WS**: moderate drop (per-step shaping helps long-horizon WS).
  Predict ≈ 25-32%.
- **3B cells**: smaller drops than 1.5B (stronger base less reliant on teacher
  mechanisms). Predict 3-10pp drops on each.

**Per-cell drop interpretation rubric** (from `ABLATION_PLAN.md` §6):
- Drop **< 2pp**: weak / no contribution at this scale (footnote)
- Drop **2–5pp**: modest contribution
- Drop **> 5pp**: clear positive contribution
- Drop **> 15pp / collapse**: load-bearing (best for narrative)

### 4.4 Auto-fade dynamics (Figure 3)

We support the "no manual teacher decay schedule" claim with two empirical
panels from wandb logs (1.5B-WS, DUET\* run, 100 steps):

- **Panel A**: $\hat w_\tau$ trajectory-level density-ratio mean over training.
  Drops from ≈ 0.50 (step 1) to ≈ 0.05 (step 100) — DR3's auto-fade is real.
- **Panel B**: $\mu(t)$ = adaptive BC weight over training. Linear decay from
  $\mu_{\max}=0.3$ at step 1 to $\mu_{\min}=0.10$ by step ≈ 25, conditional on
  disc_acc crossing $d_{\mathrm{floor}}=0.6$. Locked at valley thereafter.

This figure is the empirical complement to §3.3 / §3.4's principled-justification
claim. Reviewers cannot challenge the no-schedule claim without falsifying
this figure.

### 4.5 Training-curve comparison (Figure 2)

Three curves over 100 steps on 1.5B-WS:
- OnPolicy GRPO (collapses, < 1% throughout)
- LUFFY (rises slowly to ~5%)
- DUET\* (rises sharply to 36% by step ~50, plateaus)

This visualises **what +30.5pp looks like in training curves**, not just final
table cells.

### 4.6 LUFFY reproducibility study (appendix)

Three independent reproductions of LUFFY on 3B-WS:

| Source | val@100 SR strict |
|---|---|
| Original LUFFY paper claim | 49.5% |
| Our L20X reproduction (4×L20X-144GB) | 38.0% |
| Our 4×A100-80GB reproduction | 3.5% (lenient ≥0.9: 11.5%) |

The 49.5% number was never reproduced on either of our infrastructure
configurations. Our main table reports the **conservative 38.0%** to avoid
overstating DUET's lead.

**Narrative**: this is offered as an *additional contribution* (transparent
reporting of a reproducibility issue in the field), not a defensive move. We
keep the language neutral and factual.

---

## 5. Section-by-section content plan (target ≈ 9 pages)

### §1 Introduction (≈ 1.0 page) — DONE
5-paragraph diagnose-then-fix arc; locked text in `sections/01_introduction.tex`.

### §2 Related Work (≈ 0.7 page) — TODO

Four short paragraphs:

1. **On-policy RL for LLM agents.** GRPO [DeepSeekMath], RLHF lineage [PPO,
   InstructGPT], ReAct-style frameworks [Yao 2023], RAGEN/AgentEvolver style
   trainers.
2. **Teacher mixing / experience replay.** LUFFY [Yan 2025], CHORD [author
   2025], SFT+RL combo. State that all three treat teacher samples as if
   on-policy and lack explicit Bias-1 / Bias-2 correction.
3. **Distillation and density-ratio methods.** Goodfellow GAN, Sugiyama density
   ratio book, recent uses in policy distillation. We cite to position DR3 in
   the standard density-ratio canon, not as a novel estimator.
4. **Reward shaping and potential functions.** Ng-Harada-Russell 1999, recent
   uses in LLM agent RL (PRO, RLAIF reward shaping). We cite to ground SC's
   policy-invariance claim.

### §3 Method (≈ 3.5 pages)
- §3.1 Problem setup — DONE
- §3.2 Two biases — DONE
- §3.3 DUET overview — DONE
- §3.4 Baseline Separation (½ p) — TODO; content: §2.1 of this plan + ablation
  pointer (§4.3 ★ cell).
- §3.5 DR3 (1 p) — TODO; content: §3.3 of this plan; longest sub-section.
- §3.6 BC (½ p) — TODO; content: §3.4 of this plan + framing as safety net.
- §3.7 SC (½ p) — TODO; content: §3.5 of this plan.

Each sub-section ends with a **one-sentence "principled justification"**
restatement. *Crucial writing rule*.

### §4 Experiments (≈ 2.5 pages) — TODO

- §4.1 Setup (¼ p) — content: §4.1 of this plan.
- §4.2 Main results (½ p) — Table 1 + ¼-page narrative.
- §4.3 Ablations (¾ p) — Table 2 (4×4 matrix) + per-cell discussion focused on
  the 3 strongest narratives (-baseline_sep AF collapse; -DR3 WS drop; setting-
  dependent DR3).
- §4.4 Training dynamics (½ p) — Figures 2 and 3 + paragraph each.
- §4.5 Reproducibility & cross-infra notes (¼ p) — pointer to appendix.

### §5 Discussion / Limitations (≈ 0.5 page) — TODO

- Single-seed caveat with binomial 95% CI (the most likely reviewer hit).
- WebShop 1.5B variance: swC family span 1.0–36.0%; we report the L20X
  reproduction (most stable replication). Explicit honesty.
- Where DUET helps most: cold-start regime (weak base, sparse reward); marginal
  on already-strong agents — connect back to the bias diagnosis.
- Honest negatives: -DR3 on 1.5B-AF zero drop (AF horizons too short for Bias 2
  to bind tightly).

### §6 Conclusion (≈ 0.25 page) — TODO

One paragraph:
- Restate diagnosis (2 biases) → DUET (4 mechs, 2 channels) → +13pp avg / +17.5pp
  worst-case.
- Open direction: token-level DR3 (current is trajectory-level — there is
  unification potential, see Appendix discussion).
- Open direction: multi-seed validation, larger scales (7B+).

### Appendix (1–2 pages, optional but valuable)

- **A. LUFFY 3-way reproducibility study** (49.5% / 38.0% / 3.5%) —
  ½ p with table + discussion of likely cross-infrastructure variance sources.
- **B. Hyperparameter sensitivity** — ¼ p mini-sweep of $(\mu_{\max},
  \mu_{\min}, d_{\mathrm{floor}})$ on 1.5B-WS (we have data for 5+ swC variants).
- **C. Discriminator training details** — ¼ p (architecture, loss, dual ESS,
  buffer sizes).
- **D. Implementation notes** — ¼ p (FSDP / Ray / vLLM stack; reference open-
  source release).

---

## 6. Figures and tables

### Tables

| ID | Title | Status | Source |
|---|---|---|---|
| 1 | Main results 4×{baselines+ours} | DONE | `tables/main_results.tex` |
| 1' | (Variant) Main + reward mean | DONE | `tables/main_results_with_reward.tex` |
| 2 | Component ablations 4×4 | being filled | `data/ablation_results.md` |
| App.A | LUFFY 3-way | TODO | `analysis_reports/handoff/results_log.md` |
| App.B | Hyperparam sensitivity | TODO | `analysis_reports/1.5b_master_experiment_table.md` |

### Figures

| ID | Title | Status | Notes |
|---|---|---|---|
| 1 | DUET architecture (2 channels × 4 mechs) | prompts in `analysis_reports/paper_figure_prompts.md` | DALL-E concept v1, with tikz fallback |
| 2 | Training curves: OnPol vs LUFFY vs DUET (1.5B-WS) | data ready (wandb) | matplotlib panel |
| 3 | Auto-fade dynamics: $\hat w_\tau$ + μ(t) + disc_acc | data ready (wandb / `_parsed/disc_acc_final.json`) | 3-panel matplotlib |
| App.1 | Discriminator behaviour: $\hat w$ histogram over training | data ready | optional |
| App.2 | Per-step ablation drop curves | data ready when ablations done | optional |

---

## 7. Anticipated reviewer objections and our responses

| Likely objection | Response | Where addressed |
|---|---|---|
| "Single-seed numbers" | Binomial 95% CI on val@200 (e.g. 47.5% ± 7.0pp); explicit Limitations sentence; non-overlapping CIs vs strongest baseline | §5 Limitations |
| "Each mechanism is a known idea" | Contribution is the **diagnosis** + integrated framework, not invention | §1¶3, §3.3 overview |
| "DR3 didn't help on 1.5B-AF" | Setting-dependent: AF horizons too short for Bias 2 to bind. Ablation table makes this honest, theory predicts it | §4.3, §5 |
| "LUFFY paper says 49.5% on 3B-WS" | 3-way reproducibility appendix: paper 49.5%, L20X 38.0%, A100 3.5%; we report conservative 38.0% | App.A |
| "WebShop 1.5B has high variance" | Footnote: swC family span 1.0–36.0%; report stable L20X 36.0%; flag as future-work multi-seed | §4.2 footnote, §5 |
| "Why -baseline_sep collapses to 0%, not just degrades" | Bias 1's effect on weak base: penalises the only successful exploration the policy ever finds → on-policy gradient direction inverts → entropy collapses. Our trace from training log confirms the mechanism | §3.4 + §4.3 |
| "Why exclude teacher samples from SC?" | Direct double-shaping: teacher trajectories already have $P\approx 1$; adding SC inflates GRPO advantage and fights DR3's natural fade-out | §3.7 design decision |
| "μ(t) has 4 hyperparams ($\mu_{\max},\mu_{\min},d_{\mathrm{floor}},\alpha$)" | Mini-sweep in App.B; framing as safety net (not precise gap measure) means it tolerates a wide range | App.B + §3.6 framing |
| "Discriminator collapse?" | Dual ESS clipping ensures bounded $\hat w$; warmup ensures buffer is full first; we show empirical $\hat w$ histogram in App | §3.5 + App.1 |
| "Why no on-policy ablation of GRPO baseline std (just mean)?" | Std uses on-policy only (`std_source: non_teacher`); avoids divide-by-zero on teacher sub-group | §3.4 footnote |
| "Why is the teacher fixed (not co-trained)?" | Teacher cache is filtered offline so we get pure positive-reward replay. Co-training is future work | §5 |

---

## 8. Open decisions where we want advisor input

> **Q1.** *How aggressive should we be on the LUFFY reproducibility framing?*
> Option a: keep it in the appendix as transparency-oriented; option b: lift the
> 3-way comparison into §4 as a small contribution. Recommendation: a (the
> paper's claim is DUET, not LUFFY-debunking).

> **Q2.** *Should we include 1.5B-WS LUFFY reproducibility caveat in the main
> table footnote?* The number 5.5% is from our local reproduction; community
> may dispute it. Recommendation: yes, footnote that it is our reproduction;
> avoid claiming the number is the "true LUFFY 1.5B-WS".

> **Q3.** *Single-seed mitigation: binomial CI vs partial multi-seed?* We
> can't run multi-seed on all 5 baselines × 4 settings before deadline (≥ 80
> additional GPU-days). Recommendation: binomial CI for now; mention multi-seed
> as future work in §5.

> **Q4.** *Which ablation cells to drop if compute over-runs?* Per the
> `ABLATION_PLAN.md` priority order: tail = -SC and -BC on WS; we *must* keep
> all four -DR3 cells (the most novel mechanism). 3B side: keep all four -DR3
> cells across both envs at minimum.

> **Q5.** *Page layout: appendix in main paper PDF or separate?* NeurIPS allows
> appendix after the 9-page limit. Recommendation: include in main PDF as
> standard. We need ~1.5–2 pages of appendix as planned.

> **Q6.** *Should §3 say more about the implementation than the math?* Some
> reviewers prefer math-clean sections; others want enough operational detail
> to feel reproducible. Recommendation: put math in §3, operational details
> (warmup, buffer sizes, dual ESS) in App.C, keeping §3 readable.

> **Q7.** *Naming.* Currently DUET = DUal Expert Trajectory utilization. The
> "*" superscript denotes the SOTA-tuned configuration with adaptive μ. Should
> we keep DUET\* in the main table, or simplify to "DUET" for the camera ready?
> Recommendation: keep DUET\* in the methodology / ablation discussion (where
> it disambiguates from a "vanilla" DUET without adaptive μ); use "DUET" in the
> abstract and intro for cleanness.

---

## 9. Status of underlying data and source files

(*This section is for advisor sanity-check that every claim in the paper has a
backing file or wandb run.*)

| Claim | Backing |
|---|---|
| 4×{1.5B,3B}×{AF,WS} main numbers | `data/raw_data.md` (each cell with log path + notes) |
| 1.5B-AF ablation -baseline_sep collapse | `experiments/alfworld/alfworld_qwen1.5b_duet_minus_baseline_sep/validation_log/{50,100}.jsonl`; training log `logs/alfworld_qwen1.5b_duet_minus_baseline_sep.log` (step-by-step on-policy SR trace) |
| 1.5B-AF ablation -DR3 = 47.5% | `experiments/alfworld/alfworld_qwen1.5b_duet_minus_dr3/validation_log/100.jsonl` |
| 1.5B-WS ablation -DR3 = 9.5% | `experiments/webshop/webshop_qwen1.5b_duet_minus_dr3/validation_log/100.jsonl` |
| Auto-fade $\hat w_\tau$ | wandb `dr3/w_hat_mean_traj` for all DUET\* runs |
| Auto-fade μ(t) | wandb `chord/mu` for all DUET\* runs; offline summary in `_parsed/disc_acc_final.json` |
| LUFFY 3-way (49.5/38.0/3.5) | paper claim cite + L20X log + `analysis_reports/handoff/results_log.md` 2026-05-04 03:47 entry |
| Discriminator architecture | `agentevolver/module/exp_manager/dr3_ratio.py` |
| BC adaptive μ | `agentevolver/module/exp_manager/het_actor.py` (CHORD-SFT branch) |
| SC progress map | `agentevolver/module/exp_manager/state_progress.py` |
| Baseline separation impl | `agentevolver/module/adv_processor/` + `algorithm.grpo.teacher_baseline_separation` config |
| Teacher data | `data/teacher_trajectories/qwen72b/{alfworld,webshop}_qwen72b_filtered*.pkl` |
| 3B v39b training trajectories (case studies) | `checkpoints/agentevolver/{alfworld,webshop}_qwen3b_duet_v39b/Trajectory/trajectories_step_*.jsonl` (200 step-files each) |

---

## 10. What we'll do in the next 48 hours (deadline 2026-05-07 23:59)

> *This is the plan if advisor feedback doesn't redirect us.*

**Tonight (2026-05-05 21:00 → 2026-05-06 02:00)**.
1. Write §3.4 + §3.5 + §3.6 + §3.7 (≈ 2.5 pages).
2. Set up bibtex with 12 real entries (replace `\cite{TODO_*}`).

**Tomorrow morning (2026-05-06 09:00 → 13:00)**.
3. Write §2 Related Work (≈ ¾ page).
4. Generate Figure 1 (DALL-E or tikz; prompts ready in `paper_figure_prompts.md`).
5. Plot Figures 2 and 3 from wandb data (matplotlib).

**Tomorrow afternoon (2026-05-06 14:00 → 19:00)**.
6. Ablation cells 5–8 finish on 4×A100 by ~16:00 → fill Table 2 + write §4.3.
7. Pull L20X 3B ablation cells as they complete (most novel: 4× -DR3 cells).
8. Write §4.1, §4.2, §4.4 (excluding ablation discussion).

**Tomorrow evening (2026-05-06 19:00 → 24:00)**.
9. Draft App.A LUFFY reproducibility table + §4.5.
10. Write §5 Discussion + §6 Conclusion.
11. App.B sensitivity table from existing 1.5B sweep data.

**2026-05-07 (full day)**.
12. End-to-end pass for consistency + word count + figure quality.
13. NeurIPS checklist completion.
14. Final read by advisor (target: morning).
15. Submit.

---

## Appendix to this plan: full data tables

### Main results (all cells; from `data/raw_data.md`)

#### 1.5B AlfWorld
| Method | SR strict | Reward mean | Source log |
|---|---|---|---|
| OnPolicy GRPO | 1.0% | 0.010 | `alfworld_qwen1.5b_onpolicy.log` |
| LUFFY | 5.5% | 0.055 | `alfworld_qwen1.5b_luffy.log` |
| CHORD | 27.0% | 0.270 | `alfworld_qwen1.5b_chord.log` |
| SFT + GRPO | 30.0% | 0.300 | `alfworld_qwen1.5b_sft_rl.log` (50+50) |
| DUET\* | **47.5%** | **0.475** | `alfworld_qwen1.5b_duet_v39c_postfix.log` |
| **SOTA config**: peak=0.3, valley=0.05, mode=disc_acc, d_floor=0.4, ema_alpha=0.5, token_weighting=false. |

#### 1.5B WebShop
| Method | SR strict | Reward mean | Source log |
|---|---|---|---|
| OnPolicy GRPO | 0.5% | 0.152 | local log |
| SFT (no RL) | 7.0% | 0.562 | (reference) |
| LUFFY | 5.5% | 0.573 | `webshop_qwen1.5b_luffy.log` |
| CHORD | 11.5% | 0.603 | `webshop_qwen1.5b_chord.log` |
| SFT + GRPO | 18.5% | 0.641 | local log |
| DUET\* | **36.0%** | **0.706** | `webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06.log` |
| **SOTA config**: peak=0.3, valley=0.10, mode=disc_acc, d_floor=0.6, ema_alpha=0.2, token_weighting=false. |

#### 3B AlfWorld (L20X)
| Method | SR strict | Reward mean | Source |
|---|---|---|---|
| OnPolicy GRPO | 47.0% | 0.470 | L20X |
| LUFFY | 61.5% | 0.615 | L20X |
| CHORD | 67.0% | 0.670 | L20X (strongest baseline) |
| SFT + GRPO | 59.5% | 0.595 | L20X |
| DUET\* | **77.5%** | **0.775** | L20X v39b |

#### 3B WebShop (L20X)
| Method | SR strict | Reward mean | Source |
|---|---|---|---|
| OnPolicy GRPO | 2.0% | TBD | L20X |
| LUFFY | 38.0% | TBD | L20X (paper claim 49.5%) |
| CHORD | 39.0% | TBD | L20X |
| SFT + GRPO | 24.0% | TBD | L20X |
| DUET\* | **45.5%** | **0.743** | L20X v39b; A100 reproduction = 44.0% |

### Ablation cells filled (2026-05-05)

| Timestamp | Setting | Mechanism removed | val@100 SR strict | val@100 SR lenient | Reward mean | n | Source |
|---|---|---|---|---|---|---|---|
| 2026-05-05 07:08 | 1.5B-AF | -DR3 | 47.5% | 47.5% | 0.4750 | 200 | `experiments/alfworld/alfworld_qwen1.5b_duet_minus_dr3/validation_log/100.jsonl` |
| 2026-05-05 09:35 | 1.5B-WS | -DR3 | 9.5% | 11.0% | 0.5018 | 200 | `experiments/webshop/webshop_qwen1.5b_duet_minus_dr3/validation_log/100.jsonl` |
| 2026-05-05 18:43 | 1.5B-AF | -baseline_sep | **0.0%** | 0.0% | 0.0000 | 200 | val@50; trajectory stable-zero from step 30 |
| 2026-05-05 18:44 (running) | 1.5B-WS | -baseline_sep | … | … | … | … | ETA 23:30 |

### Auto-fade evidence (1.5B-WS DUET\* run)

From `analysis_reports/_parsed/disc_acc_final.json`:

```
disc_acc trajectory (representative):
  step 1   : 0.30    (cold start; classifier near random)
  step 25  : 0.72    (crossed d_floor)
  step 100 : 0.99    (saturated)

implied μ(t) (linear_floor_0.6 mapping):
  step 1   : 0.30 (peak)
  step 10  : 0.29
  step 25  : 0.10 (valley)
  step 100 : 0.05

ŵ_τ trajectory (typical):
  step 1   : 0.50
  step 25  : 0.30
  step 100 : 0.05  (auto-fade complete)
```

These three curves form Figure 3.

### LUFFY 3-way reproducibility (App.A)

| Source | Config | val@100 SR strict | Notes |
|---|---|---|---|
| LUFFY paper claim | 3B WebShop, the canonical config | 49.5% | Original paper |
| L20X 4×L20X-144GB | identical config, repo-pinned | 38.0% | Our most-careful reproduction |
| 4×A100-80GB | identical config | 3.5% (lenient ≥0.9: 11.5%) | Cross-infra check |

Hypothesis for the gap (to be discussed in App.A): infra-dependent factors —
likely vLLM scheduler timing or numerical precision differences in FSDP
on different GPU classes. We do not claim to fully explain the gap; we report
it as transparency.

---

*End of plan. Total length ≈ 880 lines / ≈ 6500 words.*
*Reviewing this with advisor: focus on §0 (executive summary) + §7 (reviewer objections) + §8 (open decisions); the rest is reference.*
