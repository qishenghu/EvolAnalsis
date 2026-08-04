# Author -> AC Confidential Comment (submission 32282)

> 用途：讨论期收尾时发给 Area Chair 的总结。走 "Author AC Confidential
> Comment" 按钮。已遵守红线：无破折号、无下划线数学、Eq.9 用
> denominator-convention 澄清口径（无 telescoping）、与 bDeY round-2
> 公开让步口径一致（SC 重新定位 + 改标题）。

---

**Title: Summary of rebuttal evidence and reviewer engagement**

Dear Area Chair,

As the discussion period closes, we would like to summarize what the rebuttal added, how each thread was resolved, and one engagement issue we hope you will take into account.

**New evidence added during rebuttal** (val@100 on the paper's settings unless noted):

1. **Multi-seed robustness** (requested by y9x6): six seeds on ALFWorld 1.5B give **44.3 ± 7.1**; the worst seed (34.5) still exceeds the strongest baseline (SFT+GRPO, 30.0). Five of six seeds improve during the second half of training, while all three baselines decline over the same interval.
2. **Robustness beyond exact state matching** (y9x6, bDeY): TF-IDF soft matching preserves clean performance (51.5 vs 47.5) and reaches **54.5 under 30% observation noise**, where exact matching degrades to 11.0. Exact matching is an implementation choice, not a requirement of the method.
3. **Teacher-quality ablation** (y9x6, UyKJ): with a substantially weaker 14B teacher (sample success 80.6% to 68.1%), DUET reaches 34.5, still above every baseline that uses the strong 72B teacher, and the run is still improving at the end of the fixed budget.
4. **Cross-family transfer** (UyKJ): a Llama-3.2-3B student trained with the unchanged Qwen-72B teacher cache reaches 15.0, versus 5.5 for no-teacher GRPO.
5. **Correction vs imitation attribution** (y9x6, bDeY): with the imitation term removed entirely, DUET still reaches 34.0 versus 1.0 (GRPO) on ALFWorld and 16.5 versus 0.5 on WebShop; removing baseline separation collapses training to 0.0 in both environments, and every teacher-mixing baseline in Table 1 relies on this correction as well.
6. **Discriminator probe** (y9x6, UyKJ): the learned features separate teacher from student more accurately when both sides contain only successful trajectories (84.7% to 90.0%), and successful versus failed student rollouts receive nearly identical weights. The discriminator tracks policy identity, not task outcome.
7. **Shaping control** (y9x6): randomly permuting progress values within tasks (same bonus magnitude and state coverage) drops performance from 47.5 to 41.0, isolating the value of teacher-derived ordering beyond generic dense reward.
8. **Full documentation** (bDeY): group composition (7 on-policy + 1 teacher rollout), a matched SFT budget with its training curve, and cache statistics (19,497 trajectories over 2,348 ALFWorld tasks; 4.4 distinct action sequences per task on average).

**Per-reviewer outcome:**

- **UyKJ (rating 4, confidence 4)**: reviewed our response, wrote that it "addressed my main concerns", maintained the score, and increased confidence in the assessment.
- **bDeY (rating 3, confidence 4)**: engaged in a substantive second round; we answered both follow-up questions the same day. The Eq. 9 concern was resolved by stating explicitly a denominator convention the submission left implicit (the reference policy for all samples is the previous student policy); under that convention the teacher term and the corrected ratio are the same quantity and no factor is double counted. On framing, we adopted the reviewer's point: we agreed that presenting the State Channel under the "principled" banner overclaims, and committed to rescope the paper, title included, as bias-corrected experience replay, with the State Channel as a clearly labeled extraction channel whose contribution is stated plainly. The reviewer has not replied since these answers.
- **y9x6 (rating 3, confidence 3)**: has not responded to the rebuttal or to a follow-up reminder. Their review drove most of the new experiments above (items 1, 2, 3, 5, 6, 7), including the noise-robustness result that exceeds our own paper number, and their listed weaknesses are each answered with a controlled experiment rather than argument. We would be grateful if the unanswered rebuttal could be weighed on its content when this review is interpreted.

In summary, the discussion record now shows that the reported gains are seed-robust, that the bias corrections carry standalone and load-bearing value beyond imitation, that the method degrades gracefully with teacher quality and transfers across model families, and that where criticism concerned framing rather than substance, we adopted it. Thank you for your time and consideration.
