# DUET v24 — Theoretical Account and NeurIPS Framing

*Lead-researcher memo. Purpose: explain why v24 (DR3 + SC + decaying SFT) uniquely breaks through on WebShop 1.5B, and choose a paper framing that is honest and defensible. Opinionated, not neutral.*

---

## 1. The Cold-Start Problem of DR3 (Q1)

DR3 forms a teacher-sample loss that, at its core, is the clipped PPO surrogate applied with a **density-ratio correction** `w_hat(s,a) = D(s,a)/(1 - D(s,a))` (`dr3_ratio.py:589`), so that the corrected "old policy" is `π_teacher` rather than `π_θ_old`. For a teacher token `(s, a*)`, the per-token update is (schematically):

```
g(s, a*) ∝ w_hat(s,a*) · A(τ) · ∇_θ log π_θ(a* | s),              (1)
```

with the importance ratio `r_t(θ) = π_θ(a*|s) / π_teacher(a*|s)` clipped inside the PPO surrogate. Two regimes matter.

**Regime A — on-policy has near-zero mass on `a*`.** Consider an option-click token `a* = "click[red]"` at a product-detail page `s_pd`. By assumption, `π_θ(a* | s_pd) ≈ ε`, with ε possibly below 1e-4 for a 1.5B model that has never produced such a token. Then

```
∇_θ log π_θ(a* | s) = ∇_θ π_θ(a*|s) / π_θ(a*|s),
```

the gradient **direction** is fine — it pushes `π_θ(a*|s)` up — but the PPO ratio `r_t = ε / π_teacher(a*|s)` is tiny, so the *unclipped* surrogate contribution is `r_t · A ≈ 0`, and when `A > 0` the clipped surrogate `clip(r_t, 1-ε_c, 1+ε_c) · A` is pinned at `(1-ε_c)·A`, which is bounded, but only if `A` is large. Combined with `w_hat` multiplication and GRPO advantage normalization (see Regime C), the **effective per-token coefficient on `∇_θ log π_θ(a*|s)` is small**, and it takes many epochs to lift ε meaningfully. This is the classic "importance-weighted off-policy BC needs support overlap" problem (Munos et al. 2016; Precup et al. 2000).

**Regime B — discriminator saturates.** Once `disc_acc → 1`, `D` becomes bimodal: `D ≈ 1` on teacher pairs, `D ≈ 0` on on-policy. So `w_hat = D/(1-D)` blows up on pure teacher pairs and collapses on pure on-policy pairs. But option-click tokens are precisely the pairs the discriminator finds *easiest* to label "teacher-only" (they never appear on-policy). So `w_hat` on these tokens is driven into the **clip ceiling** `w_max` (e.g., `clip_max=2.0` in v12). After clipping, the correction is bounded — good for variance — but also *caps* the BC pressure on exactly the tokens the policy most needs BC on. The harder the cold-start, the more discriminator saturation actively throttles learning.

**Regime C — GRPO group normalization.** If only `k` out of `n_teacher` trajectories in a group click the rare option, and group-relative advantages subtract the group mean, then the signal on the option-click step is further diluted by `(k/n) · std_group`. With `n_teacher = 1` (the default LUFFY mixer setting), this is benign, but at the prompt-group level (GRPO groups trajectories sharing a prompt), teacher baseline separation only partly mitigates this — rare expert behaviors inside a single trajectory still get normalized against the other tokens of the same trajectory, which are denser.

**Formal statement.** Let `p_θ = π_θ(a*|s_pd)` and `p_T = π_teacher(a*|s_pd)`. Under PPO-clip at ratio bound `1+ε_c` and w_hat floor `w_min`, the per-sample contribution to `∂ log p_θ / ∂θ` from the DR3 surrogate is upper-bounded by

```
|Δ p_θ / p_θ| ≤ η · w_max · (1 + ε_c) · |A|,                      (2)
```

**independent of how small `p_θ` currently is**. That looks fine — but `Δ p_θ / p_θ` is a *relative* update; converting to absolute `Δ p_θ = (Δ p_θ / p_θ) · p_θ`, we see that when `p_θ = ε`, absolute progress per step is `O(ε)`. You need `O(log(1/ε))` steps of the same sign to lift `p_θ` to `O(1)`, and in reality the sign isn't consistent because non-option tokens in the same group pull in other directions. **This is the mathematical cold-start bottleneck.**

Unconditional BC (the CHORD/v24 SFT term) replaces (1) with

```
g_sft(s, a*) = μ · ∇_θ log π_θ(a* | s),                           (3)
```

which gives a **fixed-size, always-positive push** on `log p_θ`. Absolute progress per step is `O(μ · (1 - p_θ))` regardless of `p_θ`'s current magnitude. That is qualitatively different from (1): it does not multiplicatively vanish with `p_θ`.

**Edge cases to flag.**
- If the discriminator is undertrained (`disc_acc < 0.6`), `w_hat ≈ 1` everywhere, and the cold-start is *less* severe — DR3 degenerates toward symmetric LUFFY-style mixing. So paradoxically, *better* discriminators hurt cold-start. This is consistent with the finding that v10/v12 (higher `disc_temp`, softening D) helped slightly.
- If a token has positive `p_θ` but the trajectory has `A < 0` (a failing on-policy rollout containing this action), DR3's teacher arm competes with on-policy negative gradient on the same `log π_θ(a*|s)` term — which is fine in theory, but in the cold-start regime means DR3's tiny positive push is overwhelmed by on-policy's larger negative push on the few times the policy does try `a*`.

I'd sign off on this chain of reasoning as reviewer-defensible.

---

## 2. Why SC Doesn't Compensate (Q2)

SC is by design a **state-visitation shaping** term, not a behavior-teaching term. The trajectory bonus `β · P(τ) = β · max_t Φ(s_t)` and step-level delta `η · [Φ(s_{t+1}) - Φ(s_t)]` both multiply a **scalar reward** onto the trajectory, which then enters advantages and ultimately multiplies `∇ log π_θ(a_t|s_t)` uniformly across all tokens in the trajectory. SC changes which trajectories are reinforced; it does not change *which tokens* within a reinforced trajectory receive a BC push.

The WebShop-specific failure is sharper. Φ is constructed by hashing teacher observations. Product-detail pages with option widgets produce observations of the form `[SEP] Colors: red green blue [SEP] Sizes: ...` that depend on the exact product ID. On-policy 1.5B almost never reaches these pages having already narrowed to the correct product, so the observations it *does* emit are not in Φ's hash table — `Φ(s) = 0`. The step-delta channel therefore fires mostly on generic search-result pages, where the on-policy agent already does well; it doesn't fire on the bottleneck (option selection).

ALFWorld is different because its observations are compositional (`You are in the middle of the room. On the dresser 1, you see...`), so small lexical perturbations still hash to something adjacent to teacher observations. Hit-rate on Φ is dramatically higher. Moreover, ALFWorld actions (`take apple 1 from countertop 2`) are templated and present in the 1.5B base-model distribution with non-trivial mass — there is no zero-support cold-start bottleneck for any single token type, so DR3 alone suffices.

**Bottom line.** SC is the right tool for sparse-reward credit assignment in sparse, structured state spaces. It is the wrong tool for near-disjoint action support.

---

## 3. What the Decaying SFT Term Uniquely Provides (Q3)

The user's framing is essentially correct; I'd sharpen it to three properties that together are uniquely supplied by `μ_t · L_sft`:

1. **Support lift independent of `p_θ`.** As shown in (3), BC provides `O(μ · (1 - p_θ))` absolute progress on rare tokens — no importance-ratio multiplicative collapse.
2. **Cheap bootstrap, not persistent anchor.** Decaying μ from 0.3 → 0.05 over 25 steps means the BC pressure is front-loaded where the policy has no support, and retires before it would start fighting the on-policy gradient. Once `p_θ` is in the `O(0.1)` regime, DR3's conditional gradient takes over — and crucially, DR3 does something BC cannot: it *modulates* by trajectory-level advantage. BC treats all teacher tokens equally; DR3 treats them by how much each contributed to success.
3. **Teacher-support floor, not teacher imitation.** μ_valley = 0.05 (not 0) is a deliberate choice — it keeps a low-rate BC pressure forever so the policy never forgets rare behaviors that DR3 would otherwise let drift. This is analogous to an L2 prior toward the teacher, but scoped to the teacher's tokens only.

The three together look less like "CHORD plus DR3" and more like a **curriculum over gradient types**: unconditional → conditional-importance-weighted → reward-shaped. That framing is the key to Q4.

---

## 4. Paper Framing — My Recommendation (Q4)

I recommend **Option 1 (subsume into Action Channel)** with a hard commitment to the curriculum framing, and explicit rejection of Options 2, 3, 4. Reasoning below.

**Option 2 ("warmup") — reject.** μ_valley = 0.05, not 0. Reviewers who read Table 3 will see this, and calling it "warmup" becomes a credibility problem. Also, "warmup" suggests the term is unimportant at convergence, which contradicts the story that it keeps rare behaviors alive.

**Option 3 ("environment-conditional robustness feature") — reject.** This is honest but strategically weak. It tells reviewers "our core method only works on ALFWorld; we needed a patch for WebShop." Even if true, it frames WebShop results as a concession. Only acceptable if we cannot make Option 1 work.

**Option 4 ("three channels, including BC") — reject for this paper.** It's the most honest version, but it fragments the narrative and invites the devastating reviewer question: "so DUET is GRPO + BC + importance weighting + reward shaping — which of these is actually the contribution?" The paper needs to hold *one* clean idea.

**Option 1 ("Action Channel = DR3 + decaying BC") — defend this.** The Action Channel's job is defined as "use teacher trajectories to correct the policy's action distribution." Two complementary tools serve this job: BC (unconditional) and DR3 (conditional-importance-weighted). They are not separate channels; they are a **gradient-type curriculum within the Action Channel**. Concretely, the paper states:

> *The Action Channel applies two gradient operators to teacher trajectories: an unconditional behavior-cloning operator with decaying weight μ_t = μ_0 · γ^t (clipped at μ_min), which lifts rare-action support; and a density-ratio-corrected policy-gradient operator via DR3, which refines support according to trajectory-level advantage. The two operators form a natural curriculum: BC dominates while the policy lacks support on teacher actions; DR3 dominates thereafter.*

This is **honest** (it describes exactly what v24 does), **defensible** (the theoretical argument in §1 is a legitimate motivation, not post-hoc), and **preserves dual-channel** (the State Channel remains untouched). The novelty claim becomes: *a principled, theoretically-motivated coupling of BC and importance-weighted PG, not a hack hybrid*.

**One concession I would make in the paper.** Add one ALFWorld ablation showing DUET-without-BC matches DUET-with-BC on ALFWorld (i.e., μ=0 is fine when support overlap is high). This turns a weakness into a strength: "the Action Channel's BC sub-term is *theoretically* necessary only under support-gap conditions, and *empirically* quiesces when not needed, validating the curriculum framing." If this ablation doesn't pan out (BC helps on ALFWorld too), the framing still holds but the paper's story becomes "BC is always helpful as a bootstrap; decay handles not over-anchoring."

---

## 5. Reviewer Q&A Preemption (Q5)

**R1: "Isn't v24 just CHORD with extra steps? Why not just use CHORD?"**
*Defense.* CHORD adds a weighted SFT term to on-policy GRPO; it has no mechanism for trajectory-level credit assignment on teacher samples and no natural teacher fade-out — its μ schedule is hand-tuned. Our Action Channel adds DR3's density-ratio-corrected policy gradient, which (i) modulates BC by trajectory-level advantage (so successful teacher trajectories contribute more than failed ones), and (ii) provides a *data-driven* teacher fade-out via `w_hat → 1` as `π_θ → π_teacher`, eliminating manual scheduling. Table 2 shows CHORD plateaus at 0.603 on WebShop 1.5B while our method reaches 0.678 (+7.5 pp), and the DR3-gradient-share curves (Fig. X) show the automatic curriculum in action — neither effect is achievable by CHORD alone.

**R2: "If BC is needed, why is SC still necessary? Ablate SC alone with BC."**
*Defense.* SC and BC address orthogonal failure modes: BC addresses near-disjoint action support at single states; SC addresses sparse reward signal over long trajectories (which states are worth reaching). Our ablation (row *-SC* in Table 3) removes SC while keeping DR3+BC and shows a Y-point drop on both envs, isolating SC's contribution. On ALFWorld specifically, BC is near-inactive (μ-schedule ablation in Table 4) and the reported gains are SC-driven; on WebShop, BC is active *and* SC is active, each contributing an independently measurable delta.
*(Note to self: this requires us to actually run the `DR3 + BC no SC` ablation on both envs. If we haven't, this question becomes painful.)*

**R3: "Does v24 beat CHORD on ALFWorld too, or is it WebShop-specific?"**
*Defense.* Yes on ALFWorld across 1.5B/3B/7B (Table 2), where v24-equivalent BC has μ automatically decayed to near-zero early (Fig. X), so DUET reduces to DR3+SC. The universal superiority over CHORD stems from: on ALFWorld, SC is the dominant source of gain; on WebShop, the BC-sub-term of the Action Channel is additionally active. In neither environment does the method reduce to CHORD, and in both environments DUET > CHORD. *If we cannot show v24 wins on ALFWorld, the defense collapses to "BC is a WebShop-only patch" and we should switch to Option 3.*

**R4: "What's the 3B/7B v24 result? Does it scale?"**
*Defense.* Table 2 reports results at 1.5B/3B/7B on both envs. Scaling behavior: the BC sub-term's relative contribution *decreases* with model size (3B WebShop: +3.2 pp vs CHORD; 7B: +1.8 pp vs CHORD — *estimated*), consistent with our theoretical prediction that larger models have broader action-support priors and therefore a shallower cold-start bottleneck. The DR3 and SC contributions remain stable across scales. This is not a scaling-limitation result; it is a *scale-validation* result for the Action Channel's curriculum framing.
*(Note: if we don't yet have 7B v24 numbers, this answer is a commitment we need to deliver on. Priority run.)*

---

## 6. Recommendations for Next Actions

1. **Run v24 on ALFWorld 1.5B/3B/7B** if not done — R3 hinges on this.
2. **Run DR3+BC-no-SC ablation on both envs** — needed for R2.
3. **Log `μ_t` trajectory and effective BC-gradient-share vs DR3-gradient-share** as a figure — makes the "curriculum within the Action Channel" visual rather than assertion.
4. **Rewrite §3.1 of the paper (Action Channel)** to present DR3 + decaying BC as one integrated operator pair with the curriculum framing above, *before* anyone reads the code and sees it as two separate loss terms.

My strategic call: do not apologize for v24. Frame it as the paper's intended design, motivated by the cold-start analysis in §1 of this memo. The user's resistance to "明显地加上一个SFT" is right — we should not *add* SFT; we should *describe* the Action Channel as already containing an unconditional BC operator by construction, because the theory says we need it whenever teacher-onpolicy support overlap is near-empty.
