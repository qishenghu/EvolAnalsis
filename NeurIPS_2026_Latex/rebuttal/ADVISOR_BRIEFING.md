# DUET rebuttal — progress briefing

NeurIPS 2026 #32282. Scores: **UyKJ 4** (borderline accept, conf 3), **bDeY 3** (borderline reject,
conf 4), **y9x6 3** (borderline reject, conf 3). Prepared 2026-07-27.

Every number below is recomputed from raw validation logs; provenance is in
`rebuttal/verification_log.md`. Two open issues are listed at the end — they need a decision, and
the briefing is not usable without them.

---

## 1. The headline: what we can now answer that we could not at submission

**The generalisation objection — the top-listed weakness of both borderline-reject reviewers — now
has a direct experimental answer, and it is a positive one.**

Both y9x6 and bDeY argued that the State Channel depends on matching student states to teacher
states, so DUET may not transfer to noisier or open-ended environments. We tested it end to end on
ALFWorld 1.5B, same 800 training tasks, same 200 validation tasks, changing only the State
Channel's *lookup operator* and what the matcher sees:

| condition | SC state coverage | **val@100 strict** |
|---|---|---|
| clean observations, exact matching (the paper's setup) | 0.590 | 47.5% |
| **30% observation noise, soft (TF-IDF) matching** | **0.654** | **54.5%** |
| 30% observation noise, exact matching | 0.178 | **11.0%** |
| State Channel removed entirely | — | 31.0% |

Two things to draw out in discussion:

- Under noise, exact matching scores **below removing the State Channel altogether** (11.0% vs
  31.0%). A matcher that fails is not merely uninformative — states genuinely on the expert path get
  Φ = 0, so the shaping actively misleads. **We concede the reviewers' premise in a stronger form
  than they stated it.**
- Replacing the lookup operator — one config field, the same progress map, no new dependency, no
  learned model — restores coverage and performance (54.5%, against 47.5% clean). So the dependence
  is a property of a replaceable component, not of the method.

We read 54.5 vs 47.5 as "at least as good", not "better": our three-seed spread on a comparable
1.5B-ALFWorld cell is sd 4.9pp.

---

## 2. Reviewer-by-reviewer: concern → what we now have

### UyKJ (4 → best chance of 5)

| their concern | what we now have |
|---|---|
| Discriminator may separate *success* from *failure*, not teacher from student | Directly tested in both environments. Restricting the student side to **successes only** does not weaken separation ($D_{succ}$ 90.0% ALFWorld, 99.0% WebShop). Late in training successful vs failed student rollouts score near-identically as "student" (ALFWorld 0.861 vs 0.869; WebShop 0.947 vs 0.904) while teacher trajectories sit at 0.280 / 0.155. On WebShop the student already succeeds on 84% of rollouts, so this is the sharpest possible test. |
| Pick-Two exception unexplained | Dissolved statistically: n = 45 of 200, exact paired McNemar **p = 0.238**; three seeds of one identical configuration span 8.9pp on that same subset. It also does not replicate off the cell — 1.5B DUET 24.4% vs GRPO 0.0%; 7B 77.8% vs 75.6%. |
| "How does performance change when the teacher is weaker?" | The 7B student is that regime with the cache held fixed: measured teacher-student reward gap 0.93/0.55 (early/late) at 1.5B versus 0.42/0.06 at 7B; teacher share of the policy gradient 0.241 → 0.079. Outcome: **DUET +1.5pp over GRPO (86.5 vs 85.0) where LUFFY falls *below* the no-teacher baseline (82.5%)**. A genuinely weaker teacher (Qwen2.5-14B, 4,094 demos covering 90.9% of training tasks, 68.1% success rate vs 72B's 80.6%) is collected and queued. |
| DR3 called a "bias-mitigating weight" rather than a real ratio | Strengthened by the Eq. 8 correction: the applied weight is the bounded α-relative ratio (≤ 1.13 measured), so DR3 can only *down-weight* teacher samples, never amplify them. That is exactly what the term names. |

### bDeY (3, confidence 4 — the most technical reviewer)

| their concern | what we now have |
|---|---|
| **Eq. 9 double-counts the density ratio** | They are right about the notation and we concede it. The code never forms $\hat w\cdot\rho_t$: for teacher samples it *replaces* the behaviour log-probability (`het_actor.py:1507`), so the single clipped ratio evaluates to $\hat w$ — **exactly one correction**. Three independent code audits confirmed this and refuted our first (wrong) planned defence. Eq. 8's $D/(1-D)$ is likewise an intermediate. Both equations corrected in the revision. |
| SC is hand-designed and may be doing the real work | See §1. Also: with SC removed, the core still reaches 31.0% on ALFWorld against CHORD's 27.0% — but on WebShop −SC gives 1.0%, below CHORD. We report both. |
| Three factual questions (group composition, SFT data, cache details) | All answered from configs and logs: n = 8 = 7 on-policy + 1 teacher, drawn from a **frozen** cache with no resample-until-success loop; realised 0.977 (ALFWorld) / 0.857 (WebShop) teacher rollouts per prompt; caches 19,497 demos over 2,348 tasks and 26,178 over 5,691, with diversity statistics. |
| Missing underlines in Table 1 | Acknowledged and fixed in the revision. |
| *(found by us, not by them)* SFT+GRPO trains on 400 tasks against DUET's 800 | Disclosed, and a task-matched rerun is queued that also gives the baseline **more** optimisation than DUET (50 SFT + 100 GRPO vs DUET's 100) — deliberately conservative in the baseline's favour. |

### y9x6 (3)

| their concern | what we now have |
|---|---|
| Transfer to noisy / open-ended / partially observable environments | §1 — this is their headline weakness and our strongest result. |
| Is the gain just extra imitation? | Two-sided evidence. Removing behaviour cloning entirely still gives 34.0% / 16.5% (ALFWorld / WebShop) against GRPO's 1.0% / 0.5%. And going the other way — holding BC at **full weight** for the whole run — is *worse* at matched budget on ALFWorld: **31.0% vs DUET's 47.5%** (see §3). Headline margins are paired-significant: McNemar p = 2.2e-5 (ALFWorld), 7.9e-8 (WebShop). |
| DR3 principled or heuristic? | Reframed honestly as a bounded, variance-controlled replay weight, plus the confound diagnostic above. |
| Multi-seed | Partly answered, partly a problem — see §4. |
| Cache size / quality / diversity | Full statistics delivered; 10% and 1% cache ablations and the 14B teacher are queued. |
| Simpler shaping baselines | A matched-magnitude control is queued: the progress values are *permuted* among each task's own states, so coverage (90.4%) and bonus magnitude (mean 0.507 vs 0.523) are held fixed while the correlation between position and Φ collapses from +0.772 to +0.045. |

---

## 3. New result worth presenting on its own: evidence for the adaptive-μ design

We discovered that a configuration in our own experiment directory — GRPO + teacher mixing with the
behaviour-cloning weight held at **μ ≡ 1.0**, never fading — reaches 47.5% on ALFWorld in half the
budget, matching full DUET. That was a threat to the ALFWorld claim, so we ran it at DUET's exact
budget (800 tasks, 100 steps):

| ALFWorld 1.5B, 800 tasks / 100 steps | val@50 | val@100 |
|---|---|---|
| **DUET** (adaptive μ, 0.3 → 0.05, driven by discriminator accuracy) | 42.5% | **47.5%** |
| BC-only, μ ≡ 1.0 constant | **48.0%** | **31.0%** |
| CHORD (μ decays 0.9 → 0.05 on a fixed schedule) | 30.0% | 27.0% |
| on-policy GRPO | 16.5% | 1.0% |

**Constant full-weight imitation peaks at step 50 and then degrades by 17 points; DUET's adaptive
schedule keeps improving over the same interval.** This is direct evidence for the design choice the
paper argues for, and we did not have it before. It also answers y9x6's "is it just more imitation"
in the strongest form: turning the imitation signal up and leaving it up makes things worse.

Two supporting observations from the same table: on-policy GRPO *collapses* from 16.5% at step 50 to
1.0% at step 100, so teacher data is preventing a real failure mode; and continuing RL from the
BC checkpoint (the SFT+GRPO baseline) drops ALFWorld from 47.5% to 30.0%.

---

## 4. Open issues — needed for an honest discussion

**(a) The 1.5B-WebShop cell does not reproduce, and BC-only is strong there.**
Same 800 tasks, verified-identical code and config, only the run seed differing: the submitted
35.5% comes back as **2.5%** and **16.5%** in two reruns. Separately, BC-only at matched budget
reaches **30.0%** on WebShop (reward 0.690 against DUET's 0.706) — a strong baseline we never
reported, and above DUET's own replicates.

We do have a mechanism, which makes this a property of the cell rather than of the method: strict
success on WebShop requires *every* requested attribute to be clicked, 88.4% of training tasks
request at least two, and every run first converges on a three-action policy that earns partial
credit but is structurally incapable of an exact match (0 successes in 10,735 under-clicking
episodes). Escape is a late event: onset step 74 in the paper run, 81 in one replicate, never in the
other. We also identified a **format-collapse failure mode** — the policy regresses into emitting a
malformed action that costs exactly the 0.05 separating a 0.95 score from 1.0 — and across 68
historical WebShop runs, those above 30% malformed average 0.6% strict success versus 5.9% below
10%. It hits any method that enters it.

*Decision needed:* report this cell as a distribution over replicates, add mean reward alongside
strict success, and state that the submitted number was the top of that distribution. Options and
consequences: `rebuttal/DECISION_webshop_1p5b_cell.md`, `rebuttal/DECISION_bc_only_baseline.md`.

**(b) Table 1's 3B column cannot be verified on this machine.**
DUET 77.5%, CHORD 67.0% and SFT+GRPO 59.5% trace to runs on the H100 / remote 3B machine with no
raw logs here, and the Appendix-F task-type figure is computed from *different* runs (its DUET is
`alfworld_3b_duet_0329`, overall 69.5%, versus Table 1's 77.5%). Our Pick-Two answer quotes that
figure, so a reviewer cross-referencing would find the mismatch. **This needs the validation logs
recovered from the 3B machine** — it is the one item nobody here can resolve.
Detail: `rebuttal/paper_corrections.md` §C0.

---

## 5. Predicted score movement

Three agents were given each reviewer's own review and persona and asked to read our draft response
and re-score. Their verdicts, and what each said would move them further:

| reviewer | after reading the response | what they said would move them up |
|---|---|---|
| UyKJ | 4 held, confidence 3 → 4 | **DUET on a non-Qwen student, and a genuinely different teacher, with numbers.** Both are queued (Llama-3.2-3B; Qwen2.5-14B cache collected). |
| bDeY | 3 held, better informed | **A task-matched SFT+GRPO rerun, and WebShop reproducing.** The first is queued; the second is open issue (a). |
| y9x6 | high 3, confidence 3 → 4 | **The pinned-curriculum WebShop replicates and the shuffled-progress-map control.** Both queued. |

Every stated condition except "WebShop reproduces" is an experiment already running or queued.

---

## 6. Status of the machines

A100 (GPUs 0,1,2,4 — the paper's own hardware) is running a file-driven queue; remaining:
WebShop seed replicate → shuffled-map control → 14B weak teacher → WebShop SFT + task-matched
baseline → **DUET on Llama-3.2-3B** → further replicates. Roughly 20–25 hours.

The 2×H200 machine is **prepared but not deployed** (`/data/home/qisheng/DUET_H200`, 11 GB,
self-contained: environment, data, model, configs, and an agent handoff document). Deploying it
would run the ALFWorld seed replicates and the 10%/1% cache ablations in parallel, which are
y9x6's remaining asks. That is the single largest available speed-up.
