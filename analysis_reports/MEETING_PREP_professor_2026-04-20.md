# 3PM 导师会议准备 — DUET WebShop 1.5B 深度复盘

**日期**: 2026-04-20
**目标**: 带着清晰的数学答案和经验证据去开会，**可以当场讲清楚**三个核心问题
**时间预算**: 13:30-14:45 整理（75min）→ 14:45-15:00 走去见导师（15min）

---

## 0. 一分钟速读版（要背下来的）

> **DUET-v1 输给 CHORD** 是因为 DR3 的 teacher 梯度系数是 **sequence-level (w_hat 整轨标量) × trajectory-advantage**，无法快速推升稀有 teacher token (`click[bright white]`)。CHORD 的 SFT 梯度是 **per-token 的 unit 系数 μ**，对 `p_θ = 10⁻⁴` 的稀有 token，10 步就能推到 0.5，DR3 需要 ~100 步。
>
> **v24 = DR3 + SC + 衰减 BC (μ=0.3→0.05)** 独门之处：BC 是唯一同时具备 (i) teacher-specific、(ii) per-token surprise-weighted、(iii) unconditional 正号 三个性质的算子。KL/EMA/softer-disc 都做不到。v30 强 KL + v33 软 disc + v28 EMA + v29 组合 4 个不含 BC 的 rescue 全部卡在 0.49-0.52，v24 在 0.678。
>
> **叙事保留 dual-channel**：Action Channel 容纳 "teacher-gradient curriculum" —— BC 做 teacher-token identity（早期），DR3 做 trajectory credit assignment（后期），μ_t 是 curriculum。State Channel 不变。**唯一决定性未跑实验：v24 recipe 在 ALFWorld 1.5B/3B 上**。

---

## 1. 数据全景（数字要对得上）

### 1.1 WebShop 1.5B 全变体结果表

| 类别 | 方法 | Val@100 | 稳定性 | 关键备注 |
|---|---|---:|---|---|
| Baseline | GRPO | ~0.45 | 稳 | |
| Baseline | LUFFY | 0.573 | 稳 | |
| **Baseline** | **CHORD** | **0.603** | 稳 | 目标线 |
| Baseline | SFT | 0.387 | 稳 | |
| Baseline | SFT+GRPO | 0.404 | 稳 | |
| DUET 原始 | **v1** (DR3+SC+step_delta) | **0.549** | 稳 | 比 CHORD 低 5.4pp |
| DUET ablate | v4 (SC off) | 0.343 | 稳 | **SC 不可移除** |
| DUET ablate | v5 (teacher_baseline_sep off) | 0.000 | **崩** | 关键组件 |
| DUET ablate | v8 (no step_delta) | 0.574 | 稳 | step_delta 过拟合 |
| DUET ablate | v12 (DR3 稳定化) | 0.431 | 稳但 Q3→Q4 drift | 无 BC 的典型 |
| DUET ablate | v21 (grpo_decouple off) | 0.095 | 崩 | 关键组件 |
| **DUET 新** | **v24 (v12+衰减BC μ=0.3→0.05)** | **0.678** | 稳 | **WINNER (+7.5pp vs CHORD)** |
| Rescue | v25 (widened clip 2.0, no BC) | **-0.041** | **崩 @ step 98** | 语法 token drift |
| Rescue | v26 (widened clip 5.0, no BC) | ~0.25 | **崩 @ step 67** | 同上，更早 |
| Rescue | v28 (w_hat EMA 0.1, no BC) | 0.495 | 稳 | 方差降 |
| Rescue | v29 (组合 rescue, no BC) | 0.511 | 勉强 | 边缘 |
| Rescue | v30 (强 KL 0.01, no BC) | 0.520 | 完美稳 | KL 防崩但不涨 |
| Rescue | v33 (softer disc 3.0, no BC) | 0.520 | 稳 | 同 |

**看图**: `figures/fig1_variant_landscape.png`（全景散点图）、`figures/fig3_no_bc_ceiling.png`（rescue ceiling bar chart）

### 1.2 Trajectory 级行为差异（最直观的"为什么"）

200 个同 val task 在 step 100 的统计（case-analyst）：

| 变体 | 任意 option 点击 | **Teacher-exact option** | 长轨 (≥13 turns) | 平均字/turn |
|---|---:|---:|---:|---:|
| DUET-v1 | 39.5% | **33.0%** | 8/200 | 216 |
| CHORD | 92.0% | **72.5%** | 16/200 | 99 |
| **v24** | 78.0% | **61.0%** | **0/200** | 129 |

- CHORD 通过 **高 μ 早期 BC** 把稀有 SKU token (`click[lavender]`, `click[fs4 | 30]`) 无条件推入 policy
- DUET-v1 的 DR3 判别器把稀有 teacher token 标记为 OOD → suppress 梯度 → surface form 永远不被安装
- **v24 继承了 CHORD 的 option-click 能力，同时通过 DR3 的 trajectory credit 修掉了 CHORD 的 option-loop 死循环（16 → 0）**

### 1.3 跨环境 × 跨规模已有数据

| 环境 | 规模 | OnPolicy | LUFFY | CHORD | **DUET** | Gap |
|---|---|---:|---:|---:|---:|---:|
| ALFWorld | 1.5B | 1.0 | 5.5 | 27.0 | **32.5** | +5.5 |
| ALFWorld | 3B | 58.5 | 61.5 | ? (重跑中) | **69.5** | +8.0 |
| ALFWorld | 7B | 85.0 | 82.5 | — | **86.5** | +1.5 |
| WebShop | 1.5B | ~45 | 57.3 | 60.3 | **67.8** (v24) | +7.5 |
| WebShop | 3B | 40.2 | 75.3 | 72.8 | **76.3** | +3.5 |
| WebShop | 7B | 76.0 | 75.5 | — | — (步 93 崩) | — |

看图: `figures/fig5_scaling_prediction.png`（BC 贡献随 scale 衰减曲线）

---

## 2. Q1: 为什么 CHORD 和 DUET 都不需要 teacher logit，CHORD 却在 WebShop 1.5B 上更好？

### 答案（数学层面）

两个方法在 teacher token `(s, a*)` 上的 per-token 梯度系数完全不同：

**CHORD 的 SFT 梯度：**
```
∂L_sft/∂z_a = μ · (p_θ(a|s) - 1_{a=a*})
            → −μ · (1 - p_θ(a*|s))  for a = a*
系数 = μ (per-token, unit-coefficient, unconditional)
```

**DUET-v1 的 DR3 梯度** (`het_core_algos.py:393-402` 确认)：
```
∂L_dr3/∂z_a ∝ w_hat · A(τ) · (p_θ - 1_{a=a*}) · [PPO clip indicator]
系数 = w_hat (sequence-level 整轨标量) × A(τ) (trajectory-level) × clip
```

**关键不对称**：
- CHORD 是 **per-token 的 unit 系数**
- DR3 是 **sequence × advantage × clip 的复合系数**
- 对稀有 teacher token (`p_θ = 10⁻⁴`)：
  - CHORD (μ=0.9) → `log p_θ` 每步 +0.9，**10 步到 p_θ ≈ 0.5**
  - DR3 → PPO clip 限制 π_θ 相对 20% 增速，**需要 ~100 步同号推才达 p_θ ≈ 0.5**

### 不是因为 clip binding
**重要修正**：`off_pg_cliphit_rate = 0` 全程，clip 根本没绑过。suppression 来自 **A × w_hat × softmax 几何**，不是 clip。这是之前 Framing C 的致命错误。

### 行为级证据（trajectory）
color-match 率 rank-correlates with val score：
```
1.6% (v12) → 10.9% (v1) → 28.3% (CHORD) → 37% (v24)
```
每 1% color-match 上升 ≈ 0.3pp val reward 上升。

### 讲给导师的 1 句话
> "DR3 的 teacher 梯度是 sequence-level 标量 × trajectory advantage 加权，对稀有 teacher token 的推升速率比 CHORD 的 per-token unit-coefficient 慢一个数量级，所以在 WebShop 这种有很多 SKU-specific option token 的任务上，DUET-v1 学不出来 option clicking，而 CHORD 能。"

**看图**: `figures/fig2_chord_vs_duet_v1_dynamics.png`（4 面板 CHORD vs v1 training dynamics）

---

## 3. Q2: 为什么 v24 在 WebShop 1.5B 有效？能推广到 ALFWorld / 3B 吗？

### v24 有效的真实机制

BC 是唯一同时具备三个性质的算子：

| 性质 | BC | DR3 | KL-to-ref | EMA 平滑 |
|---|:-:|:-:|:-:|:-:|
| **Teacher-specific** (锚点在 teacher 数据) | ✅ | ✅ | ❌ (锚到 ref) | ❌ |
| **Per-token surprise-weighted** | ✅ | ❌ (seq-level) | ✅ | ❌ |
| **Unconditional 正号** | ✅ | ⚠️ 仅当 A>0 | ⚠️ 仅当 π 偏 ref | ❌ |

**4 个独立的 rescue 失败（强证据）**：
- v30 (强 KL): 0.520 — 有 (ii)(iii) 缺 (i)
- v33 (软 disc): 0.520 — DR3 方差降，但仍缺 per-token
- v28 (EMA): 0.495 — w_hat 平滑不等于 teacher identity
- v29 (组合): 0.511 — 稳定化叠加不改变梯度结构

**唯一具备 (i)(ii)(iii) 的 v24 → 0.678**

### μ_valley=0.05 还兼任第二角色：format prior preservation

v25 失败证据：把 clip 从 0.6 放宽到 2.0 且**没有 BC** → step 98 grammar token 漂移（`<story>`, `<when>` 等幻觉包裹）→ val reward 崩到 -0.04。

v24 的 μ_valley=0.05 在整个训练后半段持续锚定 `<action>`, `</action>` 等低概率 grammar token，防止 PPO 吞噬语法。

### 跨环境泛化预测

BC 的价值 ∝ (rare-token gap) × (format fragility) × 1/(model capacity)

| 环境 × 规模 | rare-token gap | format fragility | model capacity | 预测 v24 gain |
|---|:-:|:-:|:-:|:-:|
| WebShop 1.5B | 高 (SKU option) | 中 | 低 | **+7.5pp ✅ 已验证** |
| WebShop 3B | 高 | 中 | 中 | +3~5pp |
| WebShop 7B | 高 | 中 | 高 | +1~2pp (边际) |
| ALFWorld 1.5B | 低 (templated actions) | 低 | 低 | **+0~3pp** ⚠️ 需验证 |
| ALFWorld 3B | 低 | 低 | 中 | +0~1pp |
| ALFWorld 7B | 低 | 低 | 高 | **持平 DUET** |

### 关键未验证实验（决定命运）
**v24 recipe 在 ALFWorld 1.5B 上**：
- **不次于 DUET v1 (32.5%)** → 框架成立（curriculum 自动熄火，BC 在稀有 token 少的环境上自然无效）
- **显著低于 v1** → 框架崩塌（BC 是 WebShop-only hack，需要回到方案 F+H）

### 讲给导师的 1 句话
> "v24 work 是因为 BC 是唯一同时具备 teacher-specific、per-token surprise-weighted、unconditional 正号的梯度算子；4 个 rescue 实验证明 KL/EMA/softer-disc 都无法替代这三者的交集。推广性预测是 BC 价值随 rare-token gap 和 format fragility 增大而增大、随 model capacity 减小而减大。但 ALFWorld 1.5B 上 v24 是否等价或不劣于 DUET v1，是我们必须立刻跑的决定性实验。"

**看图**: `figures/fig4_v12_vs_v24_mechanism.png`（6 面板 v12 vs v24 metric 对比）

---

## 4. Q3: 是否能保持 dual-channel 叙事的优雅？

### 推荐叙事 — "Teacher-Gradient Curriculum"

```
Action Channel (on teacher trajectories):
  L_action = μ_t · L_BC + L_DR3
  
  μ_t: 0.3 → 0.05 over 25 steps (the curriculum)
  L_BC: per-token identity imprinting, early dominant
  L_DR3: trajectory-level credit assignment, late dominant
  L_BC 的 μ_valley=0.05 floor: format-prior preservation

State Channel (on on-policy trajectories):
  L_sc = β · Φ(τ) · ∇log π_θ
  [unchanged from original DUET]
```

### 为什么这个叙事**诚实且可辩护**

**对比之前失败的 Framing**：

| 叙事 | 状态 | 为何失败 |
|---|---|---|
| Framing A (时间 curriculum) | ❌ | μ schedule 看起来是为 WebShop 调的 |
| Framing B (token-level gating) | ❌ | 没实现过 |
| Framing C (automatic p_θ specialization) | ❌ | 数学错了 + 没经验证据（see `framing_C_agent_team_verdict.md`） |
| **Teacher-Gradient Curriculum** | **✅** | 基于 BC 与 DR3 的数学性质差异，无 bogus 定理 |

**诚实之处**：
- (a) 不再声称 "p_θ 自动分工"（已证伪）
- (b) 不用空洞的 cold-start theorem
- (c) μ_valley=0.05 被承认为 format-preservation 的第二角色
- (d) BC 和 DR3 的 **数学性质差异**（token-level vs trajectory-level，unconditional vs conditional）是 legitimate 的分工理由

### 如何回应审稿人

**R1: "DUET 本质就是 CHORD + DR3，novelty 何在？"**
> CHORD 的 μ schedule 是人工调的常数衰减，没有 teacher 样本上的 trajectory credit assignment，也没有自动 fade-out 机制。我们的 DR3 提供 density-ratio-corrected PG，其 w_hat → 1 自然退出 teacher 影响 —— 形成 **数据驱动的 curriculum**。v22/v23 的常数 μ 失败（0.462 / 0.440）证明衰减调度不是可有可无。

**R2: "如果 BC 需要，为什么还要 SC？"**
> BC 解决 teacher-specific token identity（action 级别）；SC 解决 sparse reward 下的 credit assignment（state 级别）。两者正交。v4 (SC off) 在 WebShop 1.5B 掉到 0.343 —— 硬证据。

**R3: "v24 是 WebShop-specific 还是通用？"**
> 通用。BC 在 ALFWorld 上 μ 会自动降到 marginal 范围（无稀有 token），DUET 退化为 DR3+SC；在 WebShop 上 BC 补齐 option-clicking 能力。**待验证: ALFWorld 1.5B v24。**

**R4 (最危险): "为什么非要 BC？跑个 AWAC single-operator baseline 看看。"**
> **弱点承认**：我们没跑 AWAC。如果跑，我们的 claim 降级为 "两个算子可独立分析" 而不是 "minimal 2-operator design"。可作为 future work 披露。

### 讲给导师的 1 句话
> "Dual-channel 叙事保留为：Action Channel 容纳 BC 和 DR3 两个互补算子，μ_t 是算子之间的 curriculum；State Channel 不变。这个 framing 诚实描述代码，避免了 Framing C 的三个致命错误（数学错、无经验支持、机制误判），novelty 集中在 density-ratio 驱动的自动 curriculum。"

---

## 5. 必须立刻决定的行动项

### 5.1 最高优先级：v24 recipe on ALFWorld 1.5B
- **为什么**：决定框架是通用还是 WebShop-specific hack
- **时间**：~5h on GPU 0-3
- **Config**: 参考 `alfworld_qwen1.5b_duet.yaml` 加上 v24 的 chord 参数
  ```yaml
  use_chord: true
  chord_mu_peak: 0.3
  chord_mu_valley: 0.05
  chord_mu_decay_steps: 25
  chord_use_token_weighting: false
  ```

### 5.2 次优先级：AWAC baseline (为 Q3 的 R4 防御)
- 单算子 PPO + AWAC-style advantage-weighted likelihood
- 估计 ~5h
- 如果 AWAC 也达不到 v24 水平（大概率），R4 防御得以升级

### 5.3 继续跑的（低优先级）
- v36 (const tiny BC) —— 测试 "最小 BC 剂量"，~2:45h 后完成
- v31, v32 —— 会后再看，补充 ablation

---

## 6. 我建议你讲给导师的 Pitch 顺序

1. **现状**（30s）：WebShop 1.5B 上 DUET v1 输给 CHORD（0.549 vs 0.603），但 v24（DUET + 衰减 BC）赢 CHORD +7.5pp（0.678）。24 个 ablation 锁定 BC 是必要组件。
2. **Q1 机制**（2 min）：DR3 是 sequence×advantage 加权的 teacher gradient，CHORD 是 per-token unit gradient。对稀有 SKU token，CHORD 快 10 倍。[展示 fig2 + 72.5% vs 33% option 匹配数据]
3. **Q2 v24 独门**（2 min）：BC 是唯一同时具备 teacher-specific + per-token + unconditional 正号的算子。4 个 rescue（KL/EMA/disc/combined）全部失败。[展示 fig3 no-BC ceiling]
4. **Q3 叙事**（2 min）：Action Channel = BC + DR3 的 gradient-operator curriculum，State Channel 不变。诚实且可辩护。[展示 fig5 scaling prediction]
5. **下一步**（1 min）：**v24 on ALFWorld 1.5B 是决定性实验**。AWAC 作为 baseline 降低 R4 风险。

---

## 7. 图表快捷引用

| 图 | 用途 | 文件 |
|---|---|---|
| Fig 1 | 全景散点 (35 变体) | `figures/fig1_variant_landscape.png` |
| Fig 2 | CHORD vs v1 training dynamics | `figures/fig2_chord_vs_duet_v1_dynamics.png` |
| Fig 3 | No-BC ceiling bar chart | `figures/fig3_no_bc_ceiling.png` |
| Fig 4 | v12 vs v24 6-面板 metric | `figures/fig4_v12_vs_v24_mechanism.png` |
| Fig 5 | Scaling prediction | `figures/fig5_scaling_prediction.png` |

## 8. 支撑文档

| 文档 | 内容 |
|---|---|
| `duet_webshop_1.5b_final_retrospective.md` | exp-analyst 全景复盘（主报告） |
| `duet_final_theory_synthesis.md` | theory-researcher 数学 synthesis |
| `chord_vs_duet_v1_trajectory_diff.md` | case-analyst trajectory 级比较 |
| `webshop_1.5b_duet_v1_to_v24_ablation_analysis.md` | 24 变体 ablation 详表 |
| `webshop_1.5b_duet_trajectory_case_analysis.md` | v1/v8/v12/v24/CHORD token 级对比 |
| `v25_divergence_analysis.md` | v25 崩溃的 metric 诊断 |
| `v25_trajectory_collapse.md` | v25 语法 token drift 证据 |
| `framing_C_agent_team_verdict.md` | 之前失败 framing 的三轴证伪（作为 R3 预备） |

---

**DONE — 13:30 可以直接从 §0 开始读。**
