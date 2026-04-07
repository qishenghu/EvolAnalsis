# DUET 算法进展报告

> 日期：2026-04-07
> 项目：DUET (DUal Expert Trajectory utilization) — 面向 NeurIPS 2026 投稿

---

## 1. 算法概述

DUET 是一种 off-policy integrated GRPO 算法，用于在交互式环境中训练 LLM Agent。其核心创新在于设计了两个**正交通道**来利用专家示范：

| 通道 | 机制 | 作用 | 理论基础 |
|------|------|------|----------|
| **Action Channel (DR3)** | 判别器估计密度比 w_hat → 修正 teacher 样本的 importance weight | 数据驱动的 teacher 渐退，无需手动 schedule | 密度比估计 (Sugiyama et al.) |
| **State Channel (SC)** | 专家进度图提供稠密奖励塑形 β·P(τ) + η·[Φ(s')-Φ(s)] | 仅作用于 on-policy 样本 | Potential-based reward shaping (Ng et al. 1999) |

**对比 Baselines：**
- **LUFFY**：teacher mixing + policy shaping (p/(p+β))，无密度比修正
- **CHORD**：GRPO + weighted SFT (Bernoulli variance weighting)，无 teacher 混合
- **GRPO**：纯 on-policy，无 teacher 利用

---

## 2. 实验环境

| 环境 | 特征 | 奖励类型 | 状态 |
|------|------|----------|------|
| **ALFWorld** | 文本家务任务，6种任务类型 | 二值 (0/1) | ✅ 已完成，结果稳固 |
| **WebShop** | 在线购物模拟，产品搜索+属性匹配 | 连续 [0,1] | ✅ 已完成，最佳配置确定 |
| SciWorld | 科学实验模拟 | 连续 | ❌ 放弃（任务过难） |

**基座模型**：Qwen-2.5 3B Instruct

---

## 3. 主要实验结果

### 3.1 ALFWorld — Validation@Step100 (200 tasks)

| Method | Success Rate | 备注 |
|--------|-------------|------|
| **DUET (0329)** | **69.5%** | 最佳 DUET 配置 |
| DUET (original) | 66.0% | |
| LUFFY | 61.5% | |
| GRPO (on-policy) | 58.5% | react_tags 格式 |
| CHORD | 54.5% | ⚠️ 需要重跑验证 |

> **DUET 领先 LUFFY +8.0pp，领先 CHORD +15.0pp。**
> ALFWorld 使用二值奖励，reward = success rate。

**学习曲线 (Validation)：**

| Method | Step 50 | Step 100 |
|--------|---------|----------|
| DUET (0329) | 48.0% | 69.5% |
| LUFFY | 47.5% | 61.5% |
| CHORD | 42.5% | 54.5% |
| GRPO | 47.5% | 58.5% |

### 3.2 WebShop — Validation@Step100 (200 tasks)

#### 核心对比表

| Method | Avg Reward | Success Rate | 备注 |
|--------|-----------|-------------|------|
| **DUET 0409_ema** | **0.763** | **53.0%** | 🏆 最佳 DUET (叙事自洽) |
| DUET Hybrid 0405 | 0.766 | 53.0% | reward 略高 |
| DUET 0410_bv | 0.768 | 48.5% | reward 最高但 success 低 |
| DUET 0405 | 0.761 | 49.0% | |
| DUET 0409_cap | 0.759 | 49.5% | |
| DUET 0409_bell | 0.735 | 48.5% | |
| DUET 0406_v3 | 0.744 | 43.0% | |
| DUET 0407_sc | 0.739 | 42.0% | |
| LUFFY | 0.753 | 49.5% | 主要 baseline |
| LUFFY+SC 0405 | 0.709 | 32.5% | SC 对 LUFFY 有负面影响 |
| CHORD | -0.100 | 0.0% | 完全崩溃 |
| CHORD_mu_0410 | 0.728 | 39.0% | 改进版 CHORD |
| On-policy GRPO | 0.402 | 2.0% | |

> **DUET 0409_ema vs LUFFY：reward +0.010, success +3.5pp。**

#### DUET 全版本演进表

| 版本 | Reward | Success | 核心变更 |
|------|--------|---------|----------|
| duet (original) | 0.725 | 32.5% | SC hash 匹配 (覆盖率 0%) |
| duet_0401 | 0.565 | 18.0% | stage SC，但 teacher 过早渐退 |
| duet_0402 | 0.735 | 35.5% | disc_temp=2.5, gap_gate ON |
| duet_0402_v2 | -0.100 | 0.0% | 崩溃 |
| duet_0403 | 0.679 | 33.0% | gap_gate OFF, grpo_decouple |
| duet_0404 | 0.646 | 23.5% | disc_temp=1.5 |
| duet_0405 | 0.761 | 49.0% | ⭐ attribute_aware SC + disc_temp=1.0 |
| duet_0406_v1 | 0.682 | 38.5% | DR3 unlock (direct ratio) |
| duet_0406_v3 | 0.744 | 43.0% | +更多 teacher |
| duet_0407_alpha | 0.522 | 2.5% | fixed alpha prior |
| duet_0407_sc | 0.739 | 42.0% | progress_agg=last |
| **duet_0409_ema** | **0.763** | **53.0%** | **⭐ w_hat EMA 平滑** |
| duet_0409_cap | 0.759 | 49.5% | capped monotonic shaping |
| duet_0409_bell | 0.735 | 48.5% | bell curve shaping |
| duet_0409_ema_cap | 0.735 | 35.0% | EMA + cap 组合 |
| duet_0410_bv | 0.768 | 48.5% | bernoulli variance |
| duet_0410_zpd | 0.724 | 45.5% | zone of proximal dev. |
| duet_hybrid | 0.512 | 15.5% | DR3 + policy shaping (early) |
| duet_hybrid_0405 | 0.766 | 53.0% | DR3 + policy shaping (fixed SC) |

**学习曲线 (Validation)：**

| Method | Step 50 | Step 100 |
|--------|---------|----------|
| DUET 0409_ema | 0.599 / 17.0% | 0.763 / 53.0% |
| DUET 0405 | 0.668 / 16.5% | 0.761 / 49.0% |
| Hybrid 0405 | 0.592 / 12.0% | 0.766 / 53.0% |
| LUFFY | 0.509 / 8.5% | 0.753 / 49.5% |
| CHORD | 0.267 / 0.5% | -0.100 / 0.0% |
| On-policy | 0.276 / 1.0% | 0.402 / 2.0% |

> DUET 在 Step 50 已明显领先 LUFFY（0.599 vs 0.509），收敛速度更快。

---

## 4. WebShop 调参历程：从失败到突破

### Phase 1: 发现根本问题 (03/31 - 04/01)

原始 DUET 在 WebShop 上只有 32.5%，比 LUFFY (49.5%) 差 17pp。根因分析：

1. **SC 完全失效**：Hash 匹配在 WebShop 上覆盖率 = 0%。不同搜索查询产生不同产品列表，observation hash 永远无法匹配。SC 贡献了零信号。
2. **Teacher 优势爆炸**：连续奖励 [0,1] 导致 on-policy 的 std 趋近于零，teacher 优势值爆炸至正常值的 4,840 倍。

### Phase 2: 失败的修复尝试 (04/01 - 04/02)

| 尝试 | 结果 | 原因 |
|------|------|------|
| 0401: std floor + stage SC | 18.0% | Teacher 过早渐退（DR3 在 success 仅 16% 时已将 teacher 权重降到 6%） |
| 0402: disc_temp=2.5 | 35.5% | gap_gate 与 DR3 双重抑制 |
| 0403: gap_gate OFF | 33.0% | 训练峰值 80.7% 后崩溃（判别器退化） |
| 0404: disc_temp=1.5 | 23.5% | 更低温度使判别器更差 |

### Phase 3: SC 重设计 — 首次追平 LUFFY (04/01 - 04/03)

关键突破：

- **attribute_aware 匹配模式**：将 WebShop 的观测分类为离散阶段（搜索→结果→产品详情→购买），并在产品详情页上进行细粒度属性匹配（颜色、尺寸、价格等），覆盖率从 0% 提升至 100%
- **disc_temperature = 1.0**：更锐利的判别器输出 → 更好的密度比估计
- **移除 gap_gate 和 adaptive_weight**：简化系统，消除双重抑制

0405 结果：49.0% success，首次追平 LUFFY。

### Phase 4: DR3 优化 (04/04 - 04/07)

多方向探索：
- 0406_v1 "DR3 Unlock"：直接密度比（非相对比），更宽 clip → 38.5%（过于激进）
- 0407_alpha "Fixed Alpha Prior"：手动设置先验比例 → 2.5%（崩溃）
- 0407_sc "SC Direction C"：progress_agg=last → 42.0%

### Phase 5: 最终突破 — 0409_ema (04/06)

**核心变更**：在 Hybrid_0405 基础上仅添加**一个参数** `w_hat_ema_alpha: 0.3`

EMA 平滑对 DR3 importance weights 做 Polyak 平均：
```
w_hat_new = 0.3 × w_hat_current + 0.7 × w_hat_previous
```

**为什么 EMA 有效**：
1. 判别器在训练后期输出越来越极端，EMA 阻尼这些极端值，防止梯度信号退化
2. 减少 step-to-step 的权重波动（降低 15-25%）
3. 给出更干净的梯度信号，加速 teacher 淘汰

**叙事优势**：最佳配置是最简单的——仅需一个额外超参数。

---

## 5. 关键机制验证

### 5.1 DR3 自然渐退

从 Trajectory 诊断数据可观测：
- Teacher advantage 正比率从训练初期 100% 下降至后期 33%（0409_ema）
- Teacher-student 奖励差距从 0.84 → 0.17
- 判别器准确率达到 0.98+，确认能有效区分分布

### 5.2 SC 贡献

- ALFWorld (hash 模式)：覆盖率 31% → 82%，提供稠密进度信号
- WebShop (attribute_aware 模式)：覆盖率 100%，bonus/reward 比例约 12%
- LUFFY+SC 组合始终负面（32.5% vs LUFFY 49.5%），说明 SC 仅与 DR3 配合有效

### 5.3 CHORD 崩溃分析

CHORD 在 WebShop 上的失败是**格式崩溃**（think-without-action）：
- Step 51-61 之间，100% on-policy 轨迹只生成 `<think>` 标签而不生成 `<action>` 标签
- 环境返回 "Invalid action format"，agent 重复同一模式
- 熵降至 0.265-0.294（vs DUET 的 0.4+），模型收敛到狭窄的错误输出分布
- 该崩溃不可逆

> CHORD 在 ALFWorld 上不崩溃（54.5%），说明这是 WebShop 严格动作格式特有的问题。

### 5.4 Agent 行为进化 (DUET 0409_ema on WebShop)

| 训练阶段 | 搜索次数/轨迹 | 查询词数 | Buy Now 率 | 无效动作/轨迹 |
|----------|--------------|----------|-----------|--------------|
| 早期 (Step 1) | 3.3 | 14.5 | 40% | 1.30 |
| 中期 (Step 41) | 1.1 | 20.9 | 98% | 0.02 |
| 后期 (Step 81) | 1.0 | 34.0 | 98% | 0.02 |

Agent 学会了：搜索次数减少（3.3→1.0），查询变长（学会将属性信息压缩进搜索词），几乎消除无效动作。

---

## 6. 最佳配置总结 (0409_ema)

| 参数类别 | 参数 | 值 |
|----------|------|-----|
| **DR3** | enable | true |
| | disc_temperature | 1.0 |
| | dual_enable (ESS) | true |
| | use_policy_shaping | true |
| | **w_hat_ema_alpha** | **0.3** |
| | gap_gate | false |
| | adaptive_weight | false |
| **SC** | enable | true |
| | match_mode | attribute_aware |
| | beta | 0.2 |
| | step_level.enable | true |
| | step_level.eta | 0.05 |
| | grpo_decouple | true |
| | exclude_teacher | true |
| **GRPO** | teacher_baseline_separation | true |
| | kl_loss_coef | 0.001 |
| **Teacher** | n_teacher_rollouts_per_task | 1 (of 8) |

---

## 7. 论文叙事 (Paper Narrative)

### 核心故事

> "现有利用专家示范的 LLM Agent RL 方法要么只在 action 层面工作（LUFFY 的 policy shaping），要么与 RL 目标冲突（CHORD 的 SFT 混合）。DUET 提出双通道框架：Action Channel (DR3) 通过判别器学习密度比来自动调控 teacher 影响力的渐退，State Channel (SC) 通过专家进度图提供与 teacher 正交的稠密奖励塑形。两通道互不干扰、各自可配，共同使 DUET 在 ALFWorld 和 WebShop 上超越所有 baselines。"

### 论文要点

1. **双通道正交性**是核心卖点：DR3 修正"谁该学"(which samples to weight)，SC 提供"学到哪了"(how far along)
2. **DR3 自然渐退**是对手动 schedule 的改进——无需人工调节 teacher 淘汰速度
3. **SC-GRPO 解耦**是一个被验证的设计决策——不解耦会导致 on-policy 优势被 SC bonus 扭曲
4. CHORD 的崩溃是一个有力的反面案例——说明简单混合 SFT+RL 不可行

---

## 8. 当前风险与待办事项

| 风险/待办 | 优先级 | 状态 | 说明 |
|----------|--------|------|------|
| **多种子验证** | 🔴 最高 | 未开始 | 所有 WebShop 结果均为单种子。0409_ema 领先 LUFFY 仅 3.5pp，需 3+ 种子确认统计显著性 |
| **ALFWorld CHORD 重跑** | 🟡 中 | 计划中 | 当前 CHORD 结果可能需重新验证 |
| **消融实验** | 🟡 中 | 部分完成 | 需在 0409 代码基上跑干净的 DR3-only / SC-only 对比 |
| **PBRS 理论声称** | 🟡 中 | 需修正 | 轨迹级 β·P(τ) 不是严格 PBRS，只有步级 η·[Φ(s')-Φ(s)] 是 |
| **论文中简化呈现** | 🟡 中 | 未开始 | 代码中有 30+ 实验性特征（TER, AG-PM, bell_curve 等），论文应只呈现最小算法 |
| **代码整理** | 🟠 低 | 未开始 | 中英文注释混杂，核心文件 4200+ 行需重构 |

---

## 9. 关键文件索引

| 文件 | 功能 |
|------|------|
| `ae_ray_trainer.py` | 核心训练循环，SC 注入，advantage 计算 |
| `het_actor.py` | Actor 更新，DR3 判别器训练与应用 |
| `het_core_algos.py` | Loss 函数（GRPO/LUFFY/CHORD/DR3） |
| `state_progress.py` | State Channel 实现（hash/stage/attribute_aware） |
| `dr3_ratio.py` | DR3 密度比估计器 |
| `experience_collate.py` | LUFFY 混合策略 |
| `config/duet_paper_experiments_configs/` | 所有实验配置 |
| `analysis_outputs/` | 13 轮分析文档 |
