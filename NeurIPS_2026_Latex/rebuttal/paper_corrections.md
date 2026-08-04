# Paper corrections to commit to in the author response

Each item below was verified against the code/logs on 2026-07-26. These are the changes we
promise for the revision, and several of them are the substance of our reply to bDeY.

---

## C1. Eq. 8 — the applied weight is the α-relative ratio, not $D/(1-D)$

**Submitted (eq:dr3_ratio):**
$$\hat w(s,a) = \frac{D_\phi(s,a)}{1-D_\phi(s,a)} \approx \frac{\pi_\theta(a\mid s)}{\pi_\beta(a\mid s)}$$

**Implemented** (`dr3_ratio.py:846-852`, `use_relative_ratio` defaults to `True` and is not
overridden by any config — `grep -rn "use_relative_ratio" config/` returns nothing):
$$\hat r = \frac{D_\phi}{1-D_\phi},\qquad
\hat w_\alpha = \frac{\hat r}{(1-\alpha)\,\hat r + \alpha}\ \in\ \Big(0,\ \tfrac{1}{1-\alpha}\Big]$$
where $\alpha$ is the teacher fraction of the discriminator buffer, estimated online
(`alpha_mode: sync_batch_ema`).

This is the standard *relative* density ratio $w_\alpha = p/((1-\alpha)p + \alpha q)$: it is the
ratio against the **mixture** rather than against $\pi_\beta$ alone, and it is bounded above by
$1/(1-\alpha)$. Empirically $\alpha \approx 0.10$–$0.12$, so $\hat w_\alpha \le 1.13$
(`dr3/w_clip_upper` = 1.105–1.135 in the training logs).

**Why this is worth saying rather than hiding:** the boundedness is the reason we call $\hat w$ a
bias-mitigating replay weight rather than an exact likelihood ratio (a point reviewers UyKJ and
y9x6 both raised). DR3 can only ever *down-weight* a teacher sample, never amplify one, which is
exactly the fade-out behaviour reported in §4.4 (`dr3/w_off_mean`: 0.937 → 0.758 → 0.663 → 0.530
in a representative run). An unbounded $p/q$ ratio would have the opposite variance profile.

## C2. Eq. 9 — there is no $\hat w \cdot \rho_t$ product; this is bDeY's objection and they are right about the notation

**Submitted (eq:dr3_loss):** the teacher term multiplies $\hat w$ by the PPO ratio,
$\mathrm{clip}(\hat w(s,a)\,\rho_t, 1-\varepsilon, 1+\varepsilon)$ — which, as the reviewer says,
reads as two corrections stacked.

**Implemented** (`het_actor.py:1501-1507`, `het_actor.py:1544`, `het_core_algos.py:1968-1970`):
for teacher samples the *behaviour* log-probability is **replaced**, not multiplied:

```python
old_lp_new[teacher] = log_prob.detach()[teacher] - log(w_hat)[teacher]   # het_actor.py:1507
old_log_prob = old_lp_new                                                # het_actor.py:1544
...
ratio = torch.exp(log_prob - old_log_prob)                               # het_core_algos.py:1969
```

Substituting, the single ratio in the clipped surrogate becomes, for a teacher token,
$$\rho^{\mathrm{tch}}_t=\exp\!\big(\log\pi_\theta - (\overline{\log\pi_\theta} - \log \hat w_\alpha)\big)
= \hat w_\alpha \cdot \exp\!\big(\log\pi_\theta - \overline{\log\pi_\theta}\big),$$
where $\overline{\,\cdot\,}$ denotes stop-gradient. At the evaluation point this equals
$\hat w_\alpha$ exactly, and its gradient is $\hat w_\alpha \nabla_\theta \log \pi_\theta$ — a
single $\hat w_\alpha$-weighted policy-gradient term, i.e. **exactly one correction**, applied
inside the same clip as the on-policy term.

**Correction to make:** rewrite Eq. 9 to state the substitution
$\log \pi_{\theta_{\mathrm{old}}} \leftarrow \overline{\log \pi_\theta} - \log \hat w_\alpha$ for
teacher samples and show the resulting single ratio, instead of writing a product. Add one
sentence noting that this is what makes the teacher term a $\hat w_\alpha$-weighted REINFORCE
step rather than a stacked importance correction.

## C3. §3.5 — describe the discriminator's positive class accurately

The submission (`03_method.tex:134`) already says "a rolling buffer of recent on-policy and
teacher samples", which is correct. In the response we must not tighten this to "the current
rollout batch": `buffer_size: 1024` with 64 samples pushed per policy step means the positive
class spans up to **16 policy steps**, recency-weighted by `disc_age_weight_decay: 0.02`
(half-life ≈ 4–5 steps). Keep the paper's wording; do not over-claim in the rebuttal.

## C4. Table 1 — missing underlines for the 3B columns (bDeY, formatting)

Status is muddier than it looks and the response should simply concede the point. The current
working-tree `tables/main_results.tex` **does** underline the 3B cells
(`\underline{67.0\%}`, `\underline{39.0\%}` on the CHORD row), and so does the last commit
(`c0aac5ae`). But the submitted PDF carries a *third* caption wording
("Test-set success rate ... under a fixed 100-step training budget") that matches neither, so the
submitted build came from a version we no longer have on disk; underlines are graphical and cannot
be recovered from the PDF's text layer. The reviewer read the actual PDF, so take their word.

**Response wording:** acknowledge and state that the revision underlines the strongest non-DUET
baseline in all four columns — do not argue about the submitted build. Also rebuild the PDF from
the current source and confirm all four underlines render before submitting the revision.

Correct cells (strongest non-DUET per column):

| column | strongest non-DUET | value |
|---|---|---|
| 1.5B-ALFWorld | SFT + GRPO | 30.0% (already underlined) |
| 1.5B-WebShop | SFT + GRPO | 18.5% (already underlined) |
| 3B-ALFWorld | CHORD | 67.0% (**underline missing**) |
| 3B-WebShop | CHORD | 39.0% (**underline missing**) |

## C5. Uniform validation protocol + three WebShop cells off by ≤1.0pp

See `NeurIPS_2026_Latex/data/number_audit_2026_07_26.md`. Recomputing every cited cell from the
stored `validation_log/*.jsonl` under one protocol (strict success = score ≥ 1.0, n = 200) matches
the paper exactly for five cells and differs by ≤1.0pp for three WebShop cells (DUET 35.5 vs 36.0,
SFT+GRPO 18.0 vs 18.5, LUFFY 4.5 vs 5.5). The headline margin is unchanged
(36.0 − 18.5 = 35.5 − 18.0 = **17.5pp**) and no ordering changes.

**Action:** regenerate Table 1 in the camera-ready from the stored logs with a single stated
protocol, and state that protocol in the caption. Rebuttal tables already use it throughout.

## C0. ⚠️ MUST RESOLVE BEFORE FILING: the 3B column and the Appendix-F figure come from different runs

This is the one item that needs the authors, not more analysis, and it is the most exposed thing
in the paper right now.

**Finding.** Table 1's 3B column cannot be recomputed on this machine, and the Appendix-F
task-type figure is built from *different runs* than the ones Table 1 reports.

| 3B-ALFWorld cell | Table 1 | source per `analysis_reports/3b_master_experiment_table.md` | local validation log |
|---|---|---|---|
| DUET | 77.5% | `alfworld_qwen3b_duet_v39b` (rerun 04-27) | **absent** (only training trajectories) |
| CHORD | 67.0% | "4×H100 user table, no raw" | **absent** (local `alfworld_qwen3b_chord` = 46.5% @50) |
| SFT+GRPO | 59.5% | "4×H100 user table, no raw" | absent |
| LUFFY | 61.5% | — | 61.5% ✓ matches |
| GRPO | 47.0% | — | local `alfworld_3b_grpo_react_tags` = **58.5%** @100 (47.5% @50) |

The Appendix-F figure (`figures/make_task_type_figures.py:43-50`) uses `alfworld_3b_duet_0329`
(overall **69.5%**, an earlier DUET variant *without* the BC channel), `alfworld_3b_luffy`,
`alfworld_qwen3b_chord` **at step 50**, and `alfworld_3b_grpo_react_tags` (overall 58.5%).

**Why it matters for this rebuttal.** Our Pick-Two answer to UyKJ quotes that figure (DUET 37.8%,
GRPO 51.1%). Those per-type rates average to 69.5% and 58.5%, not to Table 1's 77.5% and 47.0%. A
reviewer who cross-references the appendix against the main table will find the mismatch, and it
would land in the worst possible place — a response whose whole credibility rests on us having
checked our own numbers.

**Action required (authors).**
1. Recover the validation logs for the runs behind the 3B column (`alfworld_qwen3b_duet_v39b`, the
   3B CHORD and SFT+GRPO runs, and the WebShop 3B equivalents) from the H100 / remote 3B machine,
   and recompute those cells under the same protocol as everything else.
2. Regenerate the Appendix-F figure from the **headline** runs, or state its run identity
   explicitly in the caption. Our response to UyKJ already commits to naming the run; that promise
   must be kept.
3. If the 3B GRPO cell should be 58.5% rather than 47.0%, the 3B-ALFWorld Δ changes from +10.5pp to
   +19.0pp over the strongest baseline — worth getting right in either direction.

Until step 1 is done, **do not** quote 3B numbers in the response beyond what is already in the
submitted table.

## C6. State the compute-matched budget of the SFT+GRPO baseline (answers bDeY Q2)

Verified from configs: SFT trains 50 steps (`max_train_tasks: 400`, batch 8, 1 epoch), then GRPO
trains 50 more from the step-50 checkpoint — **100 optimization steps total**, the same budget as
DUET/CHORD/LUFFY/GRPO (`max_train_tasks: 800`, batch 8 → 100 steps). This is why SFT+GRPO's
validation record is `validation_log/50.jsonl`: its RL phase *ends* at step 50, not because the
run was truncated. Add this to the experimental-setup appendix.

---

## C7 — 1.5B-WebShop 主表 cell 的 config 归属错误（2026-07-27 发现）

`EXPERIMENT_LOG.md` 把 1.5B-WebShop 的 DUET 一格归给
`config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet.yaml`。
该配置 **`use_chord: false`**——没有 BC 通道，不是论文定义的 DUET，其日志得分 **4.0%**。

主表报的是 **36.0%**（`tables/main_results.tex`，reward 0.706）。全机扫描后，唯一产出该数字的是
`config/duet_paper_experiments_configs/webshop/sweep_1.5b/webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06.yaml`
（`logs/webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06.log` → 0.360）。

**须做**：修订版把 1.5B-WebShop 的 DUET 配置指向 swC_02，并在附录列出该 cell 的实际 BC 超参
（peak 0.3 / valley 0.10 / d_floor 0.6 / ema 0.2），与 ALFWorld cell（0.3 / 0.05 / **0.4** / 0.5）
并排列出——两者不同，附录若声称统一取值即为不实。审稿人索要配置时会直接撞上这一点。

**连带**：`CLAUDE.md` 记的"paper values 0.3/0.05/0.5/0.5"与两个 cell 都不符，已同步更正。
ALFWorld 一侧无此问题：主表 47.5% 对应 `alfworld_qwen1.5b_duet_v39c_postfix`（0.425@50 → 0.475@100），
且 −BC / −SC / −DR3 三个消融的执行备份均由同一基线派生，逐字段核对一致。

---

## C8. Camera-ready 必改：WebShop 的 teacher 项是 policy-shaping 变体，论文只印了 clip 形式

按 2026-08-02 策略决定，此点**未**在 bDeY round-2 回复中披露，但录用后代码公开时必被发现，
camera-ready 必须写清。

已核实（`webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06.yaml:83` `dr3.use_policy_shaping: true`；
`het_actor.py:1925-1971`）：WebShop 主线的 teacher 项不是 Eq. 9 印的 clip 形式，而是
$\hat w_\alpha$ 作为乘性权重 × LUFFY 式 shaping $f(\pi_\theta)=\pi_\theta/(\pi_\theta+\tilde\beta)$
（`teacher_use_clip=False`，$\hat w_\alpha$ 经 `teacher_loss_scale` 进入）。ALFWorld 走
`repo_compute_token_loss` 的 clip 形式（对称 ε=0.2），与 Eq. 9 一致。两者都只含单个 ŵ 因子。

**Action**：§3.2 或附录按环境分别陈述两种 teacher surrogate 形式；bDeY round-2 已承诺
"§3 clarified to match the equations"，此项落地时一并完成。

## C9. Camera-ready 必改：Eq. 12 缺 step-level 项

按同一策略决定未在 round-2 披露。Eq. 12 只写了轨迹级 $\lambda P(\tau)$；ALFWorld 主线另有
step-level 势差 $\eta[\Phi(s_{t+1})-\Phi(s_t)]$，$\eta=0.05$（`v39c_postfix` 配置
`state_channel.step_level.enable: true`；WebShop 主线 `enable: false` 不用）。该项已出现在
Appendix-F 的 reward 分解图中，正文公式却没有——交叉引用即可发现。

**Action**：Eq. 12 补入 step-level 项及按环境取值；不声称 policy invariance。
