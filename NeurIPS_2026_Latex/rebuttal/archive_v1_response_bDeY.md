# Response to Reviewer bDeY

Thank you for the close reading — the Eq. 9 objection is correct and we are fixing it. Below we correct the Eq. 8/9 notation, narrow the "principled" claim and report how much of DUET survives without the State Channel (including where it does not), isolate SC's dependence on exact state matching, answer the three factual questions from configs and logs, and fix the Table 1 underlines.

## On the Eq. 9 weight composition

**You are right, and the error is ours.** Eq. 8 defines $\hat w \approx \pi_\theta/\pi_\beta$ and Eq. 9 multiplies it by $\rho_t$; your reading of that product is the only correct reading of what we printed. The implementation never forms it. For teacher samples DUET *replaces* the behaviour log-probability rather than multiplying in a second ratio (`het_actor.py:1501-1507`, `:1544`):

$$\log\hat\pi_\beta(a_t\mid s_t) := \mathrm{sg}\big[\log\pi_\theta(a_t\mid s_t)\big] - \log\hat w(\tau),$$

so the single clipped ratio at `het_core_algos.py:1969` is $\rho^\beta_t = \pi_\theta/\hat\pi_\beta = \hat w(\tau)$ exactly, with gradient $\hat w\,\nabla_\theta\log\pi_\theta$. $\rho_t$ (Eq. 7) applies to on-policy samples only. Teacher log-probabilities are unavailable (`use_log_prob: false` in every DUET config), so DR3 imputes one from $\hat w$ and then runs ordinary PPO: one correction, not two. We will rewrite Eq. 9 as this substitution.

Two further notation errors, both of which weaken rather than strengthen our claims about $\hat w$:

- Eq. 8's $D/(1-D)$ is an intermediate. The applied weight is the $\alpha$-relative ratio $\hat w_\alpha = \hat r/((1-\alpha)\hat r + \alpha)\in(0,1/(1-\alpha)]$, $\alpha$ being the teacher fraction estimated online. Measured $\alpha\approx0.10$–$0.12$, so $\hat w \le 1.13$ (logged as `dr3/w_clip_upper` $=1.105$–$1.135$): DR3 can only *down-weight* teacher samples, never amplify them — hence "bias-mitigating replay weight" rather than exact likelihood ratio.
- WebShop configs use a policy-shaping variant (`dr3.use_policy_shaping: true`) whose teacher term is $\hat w\cdot g(\cdot)\cdot\pi_\theta/(\pi_\theta+\beta)$ — still a single $\hat w$ factor, but not the form Eq. 9 prints. We will give the default form in the main text and this variant in the appendix.

## Claim about being "principled"

We will narrow the claim rather than defend it: "principled" should describe only the two corrections derived from the two biases we identify — baseline separation and the density-ratio repair. SC is heuristic; §3.6 already says we "do not claim exact policy invariance", and we will move that into the contribution statement and abstract.

On the substantive question, the ablation cuts both ways and we will report it that way.

- The corrections are not decorative. Removing baseline separation collapses both settings to 0.0% within ~30 steps; removing DR3 costs 26.5pp on WebShop (36.0% → 9.5%), though nothing on ALFWorld at 1.5B (47.5% → 47.5%) — an environment-dependent result we should have stated as such.
- With SC removed entirely, the core reaches **31.0%** on 1.5B-ALFWorld vs **CHORD 27.0%** and **SFT+GRPO 30.0%** (identical protocol: val@100, 200 tasks, strict success), retaining 64.5% of DUET's gain over on-policy GRPO (1.0% → 47.5%). We claim *parity* with CHORD, not a win: single seed, and our only 3-seed ablation cell at this scale has sd 4.9pp.
- **On WebShop SC is load-bearing and we will say so.** −SC gives 1.0% strict success, *below* CHORD's 11.5%. On mean reward the core still moves 0.152 → 0.450 (CHORD 0.603, full DUET 0.706) — ~54% of the reward gain — but rarely clears WebShop's all-attributes-exact threshold. We will not present that as a win.

So the defensible claim is narrower than the submission's: SC is an optional heuristic module on a CHORD-compatible core; without it DUET is baseline-competitive on ALFWorld and below CHORD on WebShop.

## Breadth of the evaluation and potential limitation of the applicability

We accept this. Three points.

**The core requires exactly CHORD's inputs.** Baseline separation, DR3 and adaptive BC read only teacher token ids, a teacher/on-policy token mask, and the scalar reward. The DR3 discriminator's features are twelve sequence-level statistics of $\log\pi_\theta$ and $\log\pi_{\text{ref}}$; teacher logits and tokenizer alignment are explicitly not required. Wherever CHORD runs, the core runs unchanged.

**SC's dependence is a property of the lookup operator, not of the shaping.** The trainer's only interface is $\Phi(\text{task\_id}, o)\to[0,1]$. ALFWorld uses the teacher-derived map (3 lines of per-environment code); our WebShop runs use no state matching at all — $\Phi$ there is a hand-written page/attribute potential of ~210 lines. That is genuine per-environment engineering and your criticism of it stands; we will report the cost of each instantiation rather than imply one free recipe. New measurement (ALFWorld, 300 tasks, held-out demo per task, word dropout applied to the matcher's view only): at 30% noise exact hashing retains 6.6% of the clean progress signal, while a dependency-free TF-IDF cosine matcher over the *same* map retains 101.4%. Caveat: at $\theta=0.6$ that matcher fires on ~34% of cross-task probe observations, so it is less discriminative (in training a task is only queried against its own map). The end-to-end runs confirm it, including the failure case. Training DUET with 30% observation noise and **exact** matching drives state coverage from 0.590 to 0.178 and strict success to **11.0%** — below the 31.0% obtained by removing the State Channel outright, since a matcher that misses scores on-path states as zero progress and misleads rather than merely abstains. Under the same noise with a **soft** matcher over the identical progress map, coverage recovers to 0.654 and strict success to **54.5%**, against the paper's clean exact-matching 47.5%. Same 800 training tasks, same 200 validation tasks, one config field changed. Your criticism of the hand-written WebShop potential stands and we will report its cost; what we can now show is that the *matching* dependence is a replaceable component rather than a property of the method.

**On math/reasoning we have no result and will not claim one.** Where no useful $\Phi$ exists the honest recommendation is DUET's core, and the 31.0% figure above is our best estimate of what that costs. We will state this as a limitation.

## Q1 — group composition; is $m$ fixed?

$n=7$ on-policy $+\;m=1$ teacher $=8$ per prompt, 8 prompts per step, 64 trajectories per update. **No resample-until-$m$-successes loop exists in training**: the cache is frozen and loaded once, and selection is sampling without replacement from a fixed per-task list, with no environment or LLM call. All 8 on-policy rollouts are generated first and up to $m$ are *replaced*, so a cache miss back-fills with on-policy data and the group is always 8 (verified: $782+5618=6400=100\times8\times8$). Realised rate: **0.978** teacher rollouts/prompt on ALFWorld (18/800 misses), **0.858** on WebShop (114/800). "Generate until success" lives in the *offline* collection script only.

## Q2 — does SFT+GRPO use the same teacher data? Plus the curve

Same cache file, same $n_{\text{teacher}}=1$, same selection mode and group size — so yes per prompt (0.975 vs DUET's 0.978 on ALFWorld; 0.838 vs 0.858 on WebShop). One difference we should have stated: SFT+GRPO is 50 SFT-stage steps then 50 pure-GRPO steps (100 total, budget-matched), so it consumes about half the total teacher volume (390 vs 782 trajectories on ALFWorld; 335 vs 686 on WebShop), and its two stages reuse the same 400 tasks, a subset of DUET's 800.

Per-step on-policy success (teacher excluded), 10-step block means, "|" marking the stage boundary:

- SFT+GRPO ALFWorld: 0.016, 0.007, 0.087, 0.264, 0.285 | 0.392, 0.398, 0.380, 0.473, 0.308
- DUET ALFWorld (no stage boundary): 0.011, 0.006, 0.019, 0.128, 0.272, 0.370, 0.397, 0.450, 0.408, 0.392
- SFT+GRPO WebShop: 0.014, 0.029, 0.012, 0.012, 0.049 | 0.030, 0.064, 0.062, 0.125, 0.136
- DUET WebShop (no stage boundary): 0.004, 0.019, 0.004, 0.026, 0.007, 0.028, 0.007, 0.046, 0.084, 0.125

The SFT stage is doing its job (`chord/sft_loss` 0.844 → 0.153 on ALFWorld, 1.380 → 0.561 on WebShop). On ALFWorld it lifts on-policy success from 0.016 to 0.285 before any RL, and the subsequent GRPO stage reaches roughly DUET's level on *training* tasks (0.473 vs DUET's 0.450 at the same point) while its *held-out* score is far lower (30.0% vs 47.5%) — a generalization gap from the repeated 400-task set, not an under-trained baseline. On WebShop both methods are still early in their trajectory at the 100-step budget, which is the same phenomenon behind the seed sensitivity we report to Reviewer y9x6. Held-out points are sparse (`test_freq: 50` gives two per run); an SFT rerun with denser held-out evaluation is running and we will add that curve.

## Q3 — teacher cache details

| | ALFWorld | WebShop |
|---|---|---|
| trajectories / distinct tasks | 19,497 / 2,348 | 26,178 / 5,691 |
| trajectories per covered task (mean / median) | 8.30 / 10 | 4.60 / 5 |
| turns per trajectory (mean / median / max) | 12.3 / 10 / 30 | 7.3 / 7 / 12 |
| max stored per task; reward | 10; 1.0 for 100% | 5; 1.0 for 100% |

Teacher is Qwen2.5-72B-Instruct for both. Only successful rollouts are kept (80.6% of 24,200 raw ALFWorld rollouts passed the reward-$=1.0$ filter), so tasks holding fewer than the per-task maximum simply had fewer successes. A 100-step run consumes ~780 trajectories: ~4% of the ALFWorld cache, ~3% of the WebShop cache. One disclosure we will add: the WebShop entries are verified gold action sequences replayed in the environment with 72B-authored rationales, not sampled 72B rollouts — hence their 100% keep-rate. All of this goes into the appendix.

## Table 1 underlines

Correct — the strongest non-DUET baseline in both 3B columns is CHORD (67.0% ALFWorld, 39.0% WebShop) and neither is underlined. The revision underlines all four columns; no number changes, since the $\Delta$ row was already computed against those cells ($77.5-67.0=10.5$, $45.5-39.0=6.5$). Separately, re-auditing every cited cell from the stored validation logs under one protocol (strict success $=$ score $\ge1.0$, $n=200$) found three WebShop cells off by $\le1.0$pp (DUET 36.0 → 35.5, SFT+GRPO 18.5 → 18.0, LUFFY 5.5 → 4.5), with the headline margin unchanged at 17.5pp and no ordering changes. The camera-ready table will be regenerated from the logs with that protocol stated in the caption.
