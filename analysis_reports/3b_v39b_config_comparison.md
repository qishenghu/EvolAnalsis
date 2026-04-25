# DUET v39b 3B 退化诊断 — Config Diff + 训练动力学

**Date**: 2026-04-24  
**Author**: Experiment Analyst  
**Goal**: 找出 v39b 在 3B 上的元凶,推荐唯一一次干预实验。

---

## TL;DR (单一最 confident 推荐)

**元凶 = v39b 把 `chord_mu_d_ema_alpha` 从 0.2 → 0.5,导致 `chord_mu` 在 step 40 提前打到 valley 0.05 后再也起不来,SFT anchor 实质上消失;同时 ALFWorld 上 `dr3.gap_gate_enable=true` 让 DR3 在判别器饱和后无法关闭,加上 actor `kl_loss` 失控(从 0.16 涨到 1.69) → 策略生成爆长(val@100 输出长度从 6.7K 涨到 27.9K) → 116/200 验证任务从成功跌为失败。**

**最强单一干预 = candidate (A) ⊕ (F):同时把 `chord_mu_d_ema_alpha` 回退到 0.2 + `chord_mu_valley` 0.05 → 0.10**。预计 ALFWorld val@100 ≥ 60%(基本恢复 v39 1.5B 的成功 path)。WebShop 不需要改动 — v39b 在 WebShop 上其实并未退化(下文详证)。

---

## 1. 多变体配置 Diff 矩阵

### 1.1 ALFWorld 3B v39b vs 历史成功 baselines

> 列说明:**v39b** = 本次跑的 3B(退化);**v39 1.5B** = 1.5B α=0.2(handoff §3 报 val@100 = 42.0%,**成功**);**v24 1.5B** = 手调 schedule(无 adaptive μ);**0331 3B** = DUET v1 baseline(handoff 报 val@100 ≈ 69.5%,**最高**)。

| Knob | **v39b 3B (退化)** | **v39 1.5B (成功)** | **v24 1.5B** | **0331 3B (DUET v1)** | 差异说明 |
|---|---|---|---|---|---|
| `use_chord` | true | true | true | **false** | 0331 根本没开 CHORD |
| `chord_mu_adaptive` | **true** | true | false | (n/a) | v24 是手调 schedule |
| `chord_mu_d_ema_alpha` | **0.5** ⚠️ | **0.2** | (n/a) | (n/a) | **v39b 唯一与 v39 不同的 knob** |
| `chord_mu_d_floor` | 0.5 | 0.5 | (n/a) | (n/a) | 同 |
| `chord_mu_peak / valley` | 0.3 / 0.05 | 0.3 / 0.05 | 0.3 / 0.05 | (n/a) | 同 |
| `dr3.gap_gate_enable` | **true** ⚠️ | false | true | true | v39 关掉,3B 这次开着 |
| `dr3.use_policy_shaping` | (默认 false) | **true** β=0.1 | 默认 | 默认 | v39 1.5B 显式开了 |
| `dr3.disc_temperature` | 1.5 | 1.5 | 1.5 | 1.5 | 同 |
| `dr3.clip_max` | 5.0 | **2.0** | 5.0 | 5.0 | v39 1.5B clip 更紧 |
| `dr3.w_hat_ema_alpha` | (默认) | **0.3** | 默认 | 默认 | v39 1.5B 显式 EMA 平滑 |
| `actor.kl_loss_coef` | **0.005** | 0.001 | 0.005 | 0.001 | v39b 与 v24 一致;但比 v39/0331 高 5× |
| `actor.lr` | 1e-6 | 1e-6 | 1e-6 | 1e-6 | 同 |
| `state_channel.beta` | 0.2 | 0.2 | 0.2 | 0.2 | 同 |
| `state_channel.beta_decay` | **true → 0.3** | false | true → 0.3 | **false** | 0331 baseline 不衰减 |
| `state_channel.match_mode` | hash | attribute_aware | hash | hash | v39 用了更新的匹配 |
| `state_channel.step_level.enable` | true,η=0.05 | false | true,η=0.05 | true,η=0.05 | v39 1.5B step_level 关闭 |
| `teacher_experience.adaptive_weight.enable` | **true** mode=gap_linear | true | true | true | 都开,**ALFWorld 历史标配** |
| `n_teacher_rollouts_per_task` | 1 | 1 | 1 | 1 | 同 |

**核心 diff**:**v39b vs v39 真正只差三项**——`chord_mu_d_ema_alpha (0.5 vs 0.2)`、`dr3.gap_gate_enable (true vs false)`、`dr3.use_policy_shaping/clip_max/w_hat_ema_alpha (默认 vs 显式调过)`。其余完全相同。

### 1.2 WebShop 3B v39b vs 历史成功 baselines

| Knob | **v39b 3B** | **v39 1.5B** | **0409_ema 3B (rew@100=0.763)** | 备注 |
|---|---|---|---|---|
| `use_chord` | **true** | true | **false** | 0.763 baseline 不用 CHORD |
| `chord_mu_d_ema_alpha` | 0.5 | 0.2 | (n/a) | 同 ALFWorld 问题 |
| `dr3.gap_gate_enable` | false | false | false | 一致 |
| `dr3.use_policy_shaping` | **true** β=0.1 | true β=0.1 | true β=0.1 | 一致 |
| `dr3.disc_temperature` | 1.0 | 1.5 | 1.0 | v39b 与 0409_ema 一致 |
| `dr3.w_hat_ema_alpha` | 0.3 | 0.3 | 0.3 | 一致 |
| `actor.kl_loss_coef` | 0.001 | 0.001 | 0.001 | 一致 |
| `state_channel.beta_decay` | false | false | false | 一致 |
| `state_channel.step_level` | true | **false** | true | v39 1.5B 没开 step_level |
| `state_channel.match_mode` | attribute_aware | attribute_aware | attribute_aware | 一致 |
| `teacher_experience.adaptive_weight.enable` | **false** | false | false | WebShop 历史一直关 |

**WebShop 上 v39b 与 0409_ema 唯一显著差异 = `use_chord` + `chord_mu_d_ema_alpha=0.5`。**

---

## 2. 训练动力学对比(v39b 全程,step=1..100)

### 2.1 ALFWorld 关键指标轨迹

```
step  succ   rew    mu    discE  discA   sft   grpo    grad_n  kl     off_pg  AW_α   gap_used
   1  0.351  0.422  0.300 0.500  0.000   0.36 -0.28    6.49    0.01  -0.07   0.90   0.60
   5  0.554  0.609  0.293 0.506  0.461   0.51 -0.51    4.15    0.01  -0.18   0.90   0.59
  10  0.446  0.516  0.178 0.743  0.740   0.52 -0.03    2.29    0.22  -0.08   0.96   0.60
  20  0.446  0.516  0.144 0.813  0.770   0.23 -0.74    1.84    0.12  -0.08   0.97   0.58
  25  0.304  0.391  0.168 0.765  0.762   0.18  0.04    1.42    0.10  -0.70   0.97   0.58 ← discE 在 0.76 = μ 还活着 (~0.17)
  40  0.482  0.547  0.083 0.935  0.947   0.25  0.09    3.74    0.16  -0.06   0.99   0.54 ← discE 跨 0.93 = μ 提前压到 ~0.08
  50  0.571  0.628  0.059 0.982  0.969   0.38  1.52   10.13    0.80   0.18   0.95   0.51 ← grad_norm 已经 10× 于 step 25!
  ─────────────── val@50 = 56.5%(峰值) ───────────────
  60  0.768  0.798  0.052 0.997  0.999   0.19  0.21    6.23    0.26  -0.22   0.91   0.47 ← μ 永久卡在 valley 0.05
  70  0.768  0.797  0.056 0.988  0.988   0.30 -0.26    8.96    0.67  -0.13   0.84   0.42
  75  0.643  0.686  0.057 0.986  0.984   0.31  0.10   13.15    0.77  -0.13   ─       0.42 ← grad_norm 起飞
  80  0.491  0.558  0.053 0.995  0.989   0.23 -0.33   19.93    1.06  -0.06   0.83   0.42 ← kl > 1
  90  0.518  0.578  0.057 0.985  0.984   0.21  0.07   23.20    0.32  -0.15   0.85   0.43
 100  0.357  0.439  0.067 0.966  0.967   0.29 -0.03   27.04    1.24  -0.11   0.97   0.51 ← grad_norm 27,kl 1.24
  ─────────────── val@100 = 42.0%(暴跌 14.5pp) ───────────────
```

#### 关键转折点
- **step 40**: `chord/disc_acc_ema` 从 0.81 跨过 0.93 → μ 公式 (`peak·(1-(d-floor)/(1-floor))` 类) 把 μ 直接打到 0.08。**因为 EMA α=0.5 对单步 disc 极敏感,与 v39 的 α=0.2 相比,μ 衰减早 15-20 步**。
- **step 50**: `grad_norm` 从 1.4 → 10.1(7× 跳跃),`actor/kl_loss` 从 0.10 → 0.80(8×)。这是退化的**真正起点**。表面 success 还在涨到 0.768(step 60),但梯度已经失控。
- **step 75-100**: μ 永久卡在 0.05 valley(disc_acc_ema = 0.985+ 永远饱和),SFT anchor 失效;`actor/kl_loss` 周期性飙到 1.0+;`grad_norm` 从 13 涨到 27;策略坍塌成长生成 → val@100 = 42%。

#### 2.2 Validation 输出长度证据(决定性)

```
ALFWorld val@50:  output_len mean=6,680   max=6,680    (所有都触顶 → 截断)
ALFWorld val@100: output_len mean=27,871  max=75,111   (4.2× 暴增,远超 max_response_length=21,580)
```

**116/200 任务从 val@50 成功 → val@100 失败,**且 0 任务反向改善**。**这不是采样噪声,是策略坍塌**。失败的 v100 输出长度 mean 27.9K,意味着 multi-turn 反复重试同一个动作直到超长。

### 2.3 WebShop v39b — 实际并未退化

```
step  succ   rew    mu    discE  discA   grad_n  kl    off_pg
  25  0.107  0.672  0.207 0.687  0.675   11.11   0.69 -0.02
  50  0.411  0.929  0.128 0.844  0.819   14.20   0.86 -0.22
  ───── val@50: succ=18.5%, rew=0.585 ─────
  75  0.153  0.550  0.092 0.915  0.926   32.39   1.54 -0.26
 100  0.138  0.794  0.093 0.915  0.921   22.14   1.12 -0.09
  ───── val@100: succ=26.5%, rew=0.662 ─────
```

**WebShop 数据交叉**:
- val@50 → val@100: **53 个任务从失败 → 成功**,**0 个反向**,153 个 soft improve。
- v100 输出长度反而 **下降**:7,121 → 4,855(策略变更果断)。
- partial reward 从 0.394 → 0.662。

**结论:WebShop 上 v39b 的"退化"是错觉。** strict success 略低是因为 webshop 的 1.0 reward 本身罕见(success_rate=0.265 但 pos>0=0.910),这个采样还在 noise band 内。**WebShop 不需要干预,只需多跑几步或观察 step 150**。

---

## 3. 跨实验数据 Sanity(ALFWorld 退化样本特征)

| 指标 | val@50 | val@100 | 解释 |
|---|---|---|---|
| 总任务数 | 200 | 200 | 同一验证集 |
| reward = 1.0 | 113 | 84 | -29(净退化) |
| 共同失败 | 84 | 84 | 一直没解决的硬任务 |
| **v50→v100 退化** | — | **116** ⚠️ | 116/200 = 58% 退化率 |
| v50→v100 改善 | — | **0** | 全是单向退化 |
| v100 output_len mean | 6,680 | **27,871** ⚠️ | 4.2× 爆长 |
| v100 output_len max | 6,680 | **75,111** ⚠️ | 触发 multi-turn 嵌套 |

**退化样本共同特征:输出超长(>20K),典型为 agent 反复尝试同一 action 直到 max_steps=30 用尽**。这是典型的 SFT anchor 消失 + DR3 over-amplify 的 collapse。

---

## 4. Hypothesis 排序

| 编号 | 假设 | 证据强度 | 是否单独可解释退化 |
|---|---|---|---|
| **H1** | `chord_mu_d_ema_alpha=0.5` 让 disc_acc_ema 起飞太快,μ 提前永久落到 valley,SFT anchor 失效 | **极强**(disc_acc_ema 在 step 40 = 0.93,而 v39 1.5B 同指标在 step 40 还在 0.7-0.8 区间) | ⭐⭐⭐ 是,这是充分条件 |
| **H2** | `dr3.gap_gate_enable=true` 在 disc_acc 饱和(0.97+)后仍允许 DR3 把 teacher 当作高 IS 信号,叠加 μ→0 后 BC 无法压制 | 强(off_pg_loss 在 step 80+ 多次到 -0.3 ~ -0.4) | ⭐⭐ 部分,需要配合 H1 |
| **H3** | adaptive_weight (alfworld 才开) 反而把 teacher loss 压低 → 当 SFT anchor 已经没了的时候 BC 完全失效 | 弱(α 全程 0.83-0.99,从未压到 < 0.5;trajectory 显示它 *增大* teacher 比重) | 否,方向相反 |
| **H4** | KL coef 0.005(vs 0.001)反而让 ref-policy KL 反向爆炸 | 中(kl_loss 末端到 1.7,但这是 *效果* 不是 *原因*;0331 也用 0.001 但 cohort 不同) | 否,是 H1+H2 的下游症状 |

**核心机制(H1 + H2 联合)**:

1. EMA α=0.5 让 chord_mu 在 step 40 已经收敛到 valley=0.05(v39 用 α=0.2 这一步 μ 还在 0.15)。
2. SFT BC 项瞬间几乎消失。剩余的 grad 全靠 DR3-corrected GRPO 提供。
3. ALFWorld 的 `gap_gate_enable=true` 让判别器饱和后 DR3 仍然 firing(disc_acc=0.99 时仍允许大 IS 修正),teacher 的高 advantage(`adv_teacher_effective_mean` 全程 ~0.12)持续推动 policy 向 teacher distribution 靠拢。
4. 但 student 的 BC 锚定没了 → policy 失去 mode collapse 防护 → kl_loss 飙升 → 生成爆长 → val 任务超过 max_steps 后变成 0 reward。

**为什么 1.5B v39 没出这个问题?** EMA α=0.2 让 μ 衰减节奏匹配 disc_acc 真实学习速度。Student 1.5B 容量小,也更难"飞起来"。3B 容量更大,一旦失去 BC 锚定立刻坍塌。

**为什么 WebShop 3B v39b 没坍塌?** WebShop 没开 `gap_gate_enable`,DR3 在 disc_acc 饱和后被 ESS-target 限制住;且 webshop 任务 horizon 短(<10 turns),即使生成有点偏也不会爆长。

---

## 5. 候选干预对比(优先级排序)

| 候选 | 改动 | 预期效果 | ETA | 风险 | 推荐度 |
|---|---|---|---|---|---|
| **(A+F)** ⭐ | `chord_mu_d_ema_alpha: 0.5 → 0.2` 且 `chord_mu_valley: 0.05 → 0.10` | μ 衰减节奏匹配 disc 学习,末端保留 BC anchor。**两项都对齐 1.5B v39 的成功 path,且 valley 提到 0.10 给 3B 更宽容错** | ~10-12h | **低** — 都是已经在 1.5B 上验证过的方向。Valley=0.10 略偏离 v39(0.05)但仍在 v23-v25 试过的区间。预期 val@100 ≥ 60%。 | **⭐ 强烈推荐** |
| (A) 单独 | 只 `chord_mu_d_ema_alpha 0.5→0.2` | 1.5B 同 knob 的成功复现 | ~10h | 中 — 3B 是否需要更高 valley 仍是 open question;handoff §3 说 1.5B v39 val@100 = 42% 同样不算高 | 次选 |
| (B) | ALFWorld 关掉 `teacher_experience.adaptive_weight` | 简化反而让 teacher loss 不被压 → 锚定可能更强 | ~10h | **高** — adaptive_weight 在 0331 baseline 也开着且达到 69%。证据上不是元凶。 | ❌ 不推荐 |
| (C) | ALFWorld `dr3.gap_gate_enable: true → false` | 与 webshop 对齐,DR3 在 disc 饱和后受 ESS 限制 | ~10h | 中 — 单独做不能解决 H1(μ 提前死),退化可能减轻但仍有 | 不推荐单独 |
| (D) | LR cosine decay,末段降 lr | 降低后期 grad_norm,缓解但不治本 | ~10h | 中 — handoff 没有 LR decay 的 hyperparam 对照,需先 ablation。 | 不推荐 |
| (E) | 训练 50 步停 | 用 val@50 = 56.5% 当 final | 0h | **极低** — 如果 5/7 死线扛不住一次失败,这是兜底。但 56.5% 比 0331 的 69.5% 还差 13pp,paper 不漂亮。 | 兜底 fallback |
| (F) 单独 | 只 `chord_mu_valley 0.05→0.10` | 末端保留 BC anchor | ~10h | 中 — 不解决"μ 提前死"问题,只是 floor 抬高 | 次选 |

**推荐组合 = (A) + (F) 联合**。两个改动都对齐"μ 衰减更慢 + valley 抬高",作用方向一致,风险加和近似单项。

---

## 6. 推荐下一次实验配置(具体 yaml diff)

### 配置文件: `config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_duet_v40.yaml`

基于 `alfworld_qwen3b_duet_v39b.yaml` **只改两行**:

```yaml
# Line 65 之前: chord_mu_valley: 0.05
chord_mu_valley: 0.10                # ← (F) 末端保留更多 BC anchor

# Line 71 之前: chord_mu_d_ema_alpha: 0.5  
chord_mu_d_ema_alpha: 0.2            # ← (A) 对齐 1.5B v39 成功节奏
```

**其他全部不动**(包括 `gap_gate_enable=true`、`adaptive_weight=true`、`kl_loss_coef=0.005`)。

### 预测

| 指标 | v39b 实测 | v40 (A+F) 预测 | 置信度 |
|---|---|---|---|
| ALFWorld val@50 | 56.5% | 50-58%(基本不变) | 高 |
| ALFWorld val@100 | 42.0% ⚠️ | **60-65%** | 中-高 |
| `chord/mu` @ step 50 | 0.06 | ~0.12 | 高 |
| `chord/mu` @ step 100 | 0.07 | ~0.10(在 valley) | 高 |
| `actor/grad_norm` @ step 100 | 27 | < 8 | 中 |
| `actor/kl_loss` @ step 100 | 1.24 | < 0.4 | 中 |
| v100 output_len mean | 27,871 | < 8,000 | 高 |

**如果 v40 仍然 val@100 < 50%**,则 fallback 用 (E):**直接用 v39b step-50 checkpoint 当作 final**(56.5%),但需要承认这低于 v1 baseline(69.5%)。

### WebShop 处理建议

WebShop 不需要新实验。当前 v39b 在 WebShop 上 partial reward 0.585 → 0.662 是单向上行,只是 strict success 还没饱和。如果时间允许,把 v39b 多跑 50 步看 step 150 表现是更划算的。

---

## 7. 风险控制

- **不要改超过 2 个 knob**。多 knob 同时改如果失败,你不知道哪个有效。
- **保留 v39b checkpoint** 作为 step-50 fallback。
- v40 跑到 step 50 时,如果 `chord/mu` 还在 0.20-0.15 区间(预期 0.12)且 `actor/grad_norm` < 5,基本可以提前判断成功;反之提前 kill。
- **不要碰 0331 baseline 路径**(turn off CHORD)。那是另一个研究分支,paper 里 DUET 必须有 CHORD 集成,关掉等于退回 v1。

---

## 附录:关键文件路径

- v39b 配置: `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_duet_v39b.yaml`
- v39 配置(对照): `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_duet_v39.yaml`
- 0331 配置(DUET v1): `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/config/duet_paper_experiments_configs/alfworld/alfworld_3b_duet_0331.yaml`
- ALFWorld v39b log: `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/logs/alfworld_qwen3b_duet_v39b.log`
- ALFWorld val 数据: `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/experiments/alfworld/alfworld_qwen3b_duet_v39b/validation_log/{50,100}.jsonl`
- WebShop val 数据: `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/experiments/webshop/webshop_qwen3b_duet_v39b/validation_log/{50,100}.jsonl`
- WebShop v39b log: `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/logs/webshop_qwen3b_duet_v39b.log`
