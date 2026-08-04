# DUET × Reasoning 模型：off-policyness 处理的算法与 infra 设计调研

日期：2026-08-03 · 状态：设计提案（待拍板项标 ▶）
范围：Qwen3.5-4B/2B 学生 × {DeepSeek-v4-flash（跨家族）, Qwen3.5-122B-A10B（同家族）} 教师
事实依据：`dr3_ratio.py` / `het_core_algos.py` / `het_actor.py` 逐行核对 + `EXPERIMENT_LOG.md` 新栈实测 + NeurIPS rebuttal 全档

---

## 0. 设定变了什么：三个 gap 的重新排序

NeurIPS 版（Qwen2.5-72B → Qwen2.5-1.5B/3B，同家族、非推理）的 off-policyness 主要是
**能力 gap**：教师会做、学生不会做，但双方的文本风格基本同分布。DUET 的全部机制
（序列级 ŵ、disc_acc 驱动 μ、全 token BC）都是在这个前提下调出来的。

新设定下 off-policyness 分裂成三个性质不同的 gap，且相对重要性反转：

| Gap | 旧设定 | 新设定 | 影响的组件 |
|---|---|---|---|
| **G1 风格/支撑集 gap**（think 文本的家族差异） | 几乎为零 | **主导**（教师 85.2% token 是 think；跨家族时学生对其 p≈0） | DR3 特征、teacher PG、BC |
| **G2 能力 gap**（教师会/学生不会） | 主导 | 收窄（4B GRPO 训练期 SR 已 50-64%）| BS、混合价值本身 |
| **G3 条件 gap**（rollout 与训练的上下文条件不一致） | 未被意识到 | 已由 exact-snapshot + 师生同上下文采集次序解决 | 管线（已定） |

**核心论断：DUET 现有机制把 G1 和 G2 混在一个标量 ŵ 和一个标量 μ 里。同家族时二者近似重合，
跨家族时二者背离——风格上极端 off-policy 的教师，能力上可能仍然极有价值。算法改进的主线就是
把这两件事拆开。**

---

## 1. 逐组件诊断（全部有代码/数据出处）

### 1.1 DR3：判别器的"双杀"失效模式，且特征设计使其必然发生

判别器特征（v3，`dr3_ratio.py:146-175`）：`lp_mean, lp_std, lp_min, lp_max,
lp_low_ratio(<−10), resp_len, kl_ref_mean` —— **全部是学生对文本的 NLL 统计量**。

- 同家族时这是正确设计：NLL 统计正是密度比的充分统计量，disc_acc 随学生逼近教师而缓慢爬升，
  给出论文引以为傲的 fade-out 曲线。
- 跨家族时它变成自证预言：DeepSeek think 文本的 `lp_low_ratio` 从第 0 步就显著高于学生
  rollout（长度捷径已被证伪，AUC=0.558，但 **NLL 捷径未测**，且从特征定义看几乎必然存在）。
  后果是**双杀**：
  - disc_acc→1 ⇒ r̂→0 ⇒ ŵ→w_min=0.01：teacher PG 通道静音；
  - disc_acc→1 ⇒ μ=valley（`μ ∝ 1−EMA(disc_acc)`）：BC 通道同时静音。
  - 一个判别器服务两个通道的优雅设计，在跨家族设定下成为**单点故障**：DUET 退化为 GRPO+SC，
    整个"教师利用"故事失效。这是 ICLR2027_PLAN §3.4 风险的机制化表述。

**关键洞察**：真实的密度比在 think 段上确实 ≈ 0（学生几乎不可能生成 DeepSeek 风格的 think），
所以判别器没"坏"——它如实报告了 think 段不可纠正。错的是**我们把只在 action 段有意义的修正，
用整条序列的统计量来估计**。

### 1.2 Teacher PG（Eq.9 通道）：对 think token 做策略梯度缺乏理论根据

Imputation 形式下 teacher 项 = ŵ_α·Â^β·∇log π（对全部 teacher token 求和）。两个问题：

1. **think token 上的 PG 没有 MDP 语义**：环境奖励只依赖抽取出的 action；think 是内部潜变量。
   对 think token 做 advantage 加权的密度比修正，既无必要（reward 不经过它）也不可能
   （跨家族时 q(think)>0 的支撑假设不成立）。
2. **梯度尺度失控**：∇log π 在 p≈e^{−10} 的 token 上是巨大的方向性推力，把学生往 DeepSeek
   风格推。PG 通道**没有** BC 通道的 φ(p) 阻尼（见 1.3），是裸露的。

注意 WebShop 的 hybrid 路径（ŵ × LUFFY 式 f(π)=π/(π+β) shaping，`het_actor.py:1925-1971`）
恰好有 per-token 阻尼：f 的梯度权重 ∝ π·β/(π+β)² 在 π→0 时消失。**这个当年为 WebShop 稳定性
打的补丁，其实就是跨家族设定需要的 token 级信任域**——只是当时没这样理解它。

### 1.3 BC 通道：φ(p) 加权意外地已经是正确设计，但有一个副作用

`compute_chord_sft_loss`（`het_core_algos.py:1767`）默认 `use_token_weighting=True`：
φ(p)=p(1−p)，对 p≈0（不可学）与 p≈1（已学会）的 token 都趋零，梯度幅值 ∝ p·|log p|。

- **正面**：跨家族 think token（p≈0）被自动静音，BC 不会发生梯度爆炸——这是现成的保护。
- **副作用**：**冷启动阶段恰恰学不到教师的推理**。85.2% 的教师 token 是 think，其中绝大部分
  初始 p≈0 → φ≈0。BC 实际只在"学生已经半会"的 token 带上学习（自步课程性质，可以正面写），
  但如果论文想主张"推理蒸馏"，φ 加权与该主张相互矛盾，必须做 φ on/off × think/action 的消融。
- **归一化问题**：`token-mean` 聚合下，85% 的 think token 稀释 action token 的有效学习率；
  μ 的有效作用强度隐式依赖教师 think 长度——不同教师（flash vs 122B，think 长度不同）下
  μ 的语义不可比。

### 1.4 BS 与 SC：受影响最小，且 SC 的地位反转

- **BS**：与 token 分布无关，只依赖组内奖励结构。强学生下教师成功不再稀有 → bias-1 减弱，
  弱学生列（2B/SciWorld）仍需要它。无需改动。
- **SC**：**跨家族设定下反而成为最可迁移的教师通道**——进度图 Φ 建立在环境观测上，观测文本
  与教师家族无关，think 完全不经过它。可预注册的预测：教师家族 gap 越大，SC 在总增益中的
  份额越大（flash vs 122B 直接可比）。这把 NeurIPS 版"SC 只是启发式"的防守叙事，翻转成
  "SC 是家族不变的状态通道"的进攻叙事。

### 1.5 汇总：每个通道对三个 gap 的暴露面

| 通道 | G1 风格 gap | G2 能力 gap | 现有保护 | 缺口 |
|---|---|---|---|---|
| DR3/ŵ | **致命**（NLL 特征全序列） | 正常工作 | 无 | 特征按段重设计 |
| teacher PG | **裸露**（无 token 阻尼，clip 形式） | 正常 | WS hybrid 有 f(π) | 统一到带阻尼形式 + think 退出 |
| BC | φ(p) 已挡住爆炸 | 正常 | φ(p) | 分段 μ + 归一化 |
| μ 调度 | **双杀共犯**（disc_acc 饱和→valley） | 正常 | 备选 driver 已在代码中 | driver 换成分段量 |
| BS | 无 | 减弱但仍需 | — | 无需改 |
| SC | **免疫** | 正常 | soft matcher | auto-β（已计划） |

---

## 2. 设计提案：DUET-R（segment-aware 化，四条改动）

**原则一（通道分配，2026-08-03 与用户对齐后确立）**：DUET 的初心是让学生快速适应并
吸收 off-policy 教师——包括教师怎么思考。因此 think **不是不学，而是换正确的工具学**：
**监督通道（BC）不需要密度比，跨支撑集依然无偏**（DeepSeek-R1 → Qwen 学生的纯 SFT
蒸馏是现成先例），教师 think 由它全额承担；**RL 通道（ŵ 加权 PG）需要可估的密度比**，
只在支撑重叠的 action 段工作。μ 的语义相应修正为"差距大 → 多模仿，学会了 → 退场"
——现行 disc_acc 驱动在跨家族下把这个方向弄反了（disc 一饱和 μ 即落 valley，把
"教师和我很不一样"解读成"少模仿"）。修正后的完整动力学：冷启动 think-BC 主导 →
学生分布向教师靠近 → 密度比变得可估 → 修正后的 action-PG 接棒 → 掌握后 μ、ŵ 自然
退场。注意这同时缓解 1.1 的双杀：双杀的前提是学生分布永不向教师靠近，think-BC 开启
后 disc_acc 不会钉死在 1，自适应机制整体复活。

**原则二**：不加新模块，把现有标量机制升级为"按段（think/action）分治"。全部改动落在
已有张量管线上（`teacher_mask`、`step_ids`、`extra_seq_features` hook 都已存在）。

### R1（P0）：判别器改为 action 段特征 ▶

`compute_sequence_features` 的 mask 从全 response 收窄到 **action 段 token**（think 段排除）。

- 语义：ŵ 估计的是"学生在该状态下会不会产生这个 action"的密度比——这才是 off-policy 修正
  在 MDP 里的对象。action 文本受环境语法约束，跨家族支撑重叠大，判别器有正常的学习曲线,
  fade-out 叙事在新栈存活。
- 实现：S 量级。特征 mask 换成 `teacher_action_mask`（think/action 段 mask 需在 tokenize 时
  产出，见 R4);`extra_seq_features` hook 可同时喂 think 段统计作为对照实验。
- 验收：离线 AUC 测试（见 §4-1）中 action-only 特征的 AUC 应显著低于全序列特征且 >0.5,
  disc_acc 轨迹恢复渐进爬升。

### R2（P0）：teacher PG 统一为"ŵ_act × f(π) shaping"形式，且只作用于 action 段 ▶

- 把 WebShop hybrid 路径升格为**唯一的** teacher surrogate（两环境同一形式），think token
  退出 PG（loss_mask 层面排除）：
  - 轨迹级：ŵ_act（R1 的 action 段密度比）修 G2/分布错配；
  - token 级：f(π)=π/(π+β) 修 G1/逐 token 支撑错配（梯度权重 π→0 时消失）;
  - clip/imputation 形式降为消融行。
- 这同时解决 C8（两环境两种算法的隐患）——ICLR 版方法定义即 hybrid，与 rebuttal 公开披露
  （"WebShop 用 shaping 变体，单一 ŵ 因子"）一致，不构成翻供。
- 论文形式：teacher token 的统一梯度权重写成一行
  **g_t = 1[action]·ŵ_act·Â^β·∂f + μ_seg·φ(p_t)**，所有机制在一个式子里可见。

### R3（P1）：μ 分段 + 语义修正（2026-08-03 按用户初心反转原稿方向）▶

> 原稿曾写"μ_think 默认小值"——**已否定**。DUET 的目标是快速吸收 off-policy 教师，
> 包括教师怎么思考；think 蒸馏是主通道之一，不是可有可无的附件。

- **μ_think 是冷启动推理蒸馏系数**：由学生对教师 think 段的 NLL 差距驱动——
  **差距大 → 多学，随掌握退火**。这与现行 disc_acc 驱动的方向相反（disc 饱和 → μ 落
  valley，把"教师和我很不一样"解读成"少模仿"，恰与初心相反；同家族时因 disc_acc 从
  0.5 缓慢爬升而被掩盖）。归一化建议：μ_think ∝ clamp(NLL_think 的 EMA / 基准值)，
  具体形式随 §4-2 的 φ(p) 画像定。
- **think 段 φ(p) 关闭或温度化**：φ 在 p≈0 处为零，推理 token 永远进不了可学区间，
  蒸馏无法启动。标准 SFT/蒸馏（R1-distill：DeepSeek-R1 → Qwen 纯 SFT，跨家族 think
  可学且高价值的先例）从来都是直接训 p≈0 token 的；φ 保留给 action 段（那里"已掌握"
  抑制仍有意义）。配套安全阀：per-token NLL clamp 防极端值。
- **μ_act**：保持 disc_acc 驱动（R1 后 disc_acc 恢复有信息量）；driver 容错保留——若
  R1 后 disc_acc 仍在 step≤5 内 >0.9，μ_act 预注册切换到 `chord_mu_adaptive_mode: nll`
  （代码已有），日志标记，避免事后调参嫌疑。
- **think-BC 消融**（计划 §2.3-f）升级为 μ_think 轴：0（纯动作克隆）/ NLL 驱动（主设定）/
  恒定大值（不退火的蒸馏，检验退火的价值）。
- BC 聚合改为**分段归一化**（action 段与 think 段各自 token-mean 再加权），消除教师 think
  长度对 μ 有效强度的隐式影响（1.3 的归一化问题）。

### R4（P0，管线前置）：think/action 段 mask 的一等公民化

- teacher 轨迹 tokenize 时产出 `segment_mask`（think=1/action=2/模板=0），与 `teacher_mask`
  同规格入 batch;学生 on-policy 样本同样打段标（自己的 think 有 advantage 信用,不受 R2 影响,
  照常训练——改动只针对 teacher 样本）。
- exact-snapshot 语义下的对齐：每条教师轨迹可采样 K>1 个 decision snapshot（教师回放不耗
  rollout 算力,只耗训练 token）,缓解"一轨迹一决策"导致的 BC 信号稀疏——K 作为超参默认 2-4。

### 不做的事（明确排除,防 scope creep）

- 不做 token 级判别器（样本效率差、训练不稳,f(π) shaping 已提供 token 级阻尼）;
- 不做对抗式特征去风格化（复杂度不可控）;
- 不做教师 logprob 蒸馏类方法（违背 text-only cache 的设定主张）;
- on-demand 任务级教师注入（difficulty-gated mixing）记为 P2,只在 P0/P1 全部落地后考虑。

---

## 3. 由设定产生的新叙事资产（写进论文的进攻点）

1. **师生家族轴 = 科学轴**：122B（同家族）保护机制、flash（跨家族）压力测试。判别器的
   disc_acc 轨迹本身就是"分布距离读数"——同一方法在两种教师下的 ŵ/μ/disc_acc 轨迹对比图,
   是任何 baseline 都给不出的机制证据。
2. **SC 的家族不变性**（1.4）：预注册"家族 gap ↑ ⇒ SC 份额 ↑"。
3. **φ(p) 的自步课程解读**：BC 在跨家族教师下自动从"近支撑 token"学起,随学生风格靠近逐步
   解锁 think 段——把 CHORD 的加权公式讲出新深度（需 φ×段消融支撑）。
4. **Reasoning-context budget B 轴**（已落盘）：B 同时控制学生信用分配与教师 think 覆盖率。
5. **API-teacher 属性表**（计划 §1.1）：全方法只需文本 cache;跨 tokenizer、跨家族是 DUET
   设定的自然延伸而非额外假设。

---

## 4. 本周可执行的验证（不占训练主力卡）

1. **判别器离线预测试（半天,决定 R1 是否必要）**：用现有 4B GRPO rollout dump ×
   `alfworld_dsv4pro_fullctx_4000.jsonl`,离线跑 `compute_sequence_features`（v3 全序列 vs
   action-only 两版）+ MLP,报 AUC 与 5-step disc_acc 轨迹。预期：全序列 AUC≈1（证实双杀）,
   action-only 显著更低。顺带复算长度 AUC 作 sanity（应 ≈0.558）。
2. **φ(p) 分段画像（2 小时）**：对教师轨迹逐 token 算学生 p 与 φ(p),按 think/action 分段
   出直方图——直接量化"BC 现在实际在学什么"。这张图无论结果如何都能进附录。
3. **段 mask 管线（R4）打通**：先在 tokenize 层产出 segment_mask 并落盘验证,不动 loss。

## 5. Infra 优化(独立于算法,直接买 seeds)

1. **Rollout 吞吐是最大杠杆**:gen 占步时 ~90%,根因 `MAX_NUM_SEQS=1-2`(FLA packed
   recurrent decode bug 规避)。行动:在验证 lane 上系统测 `MAX_NUM_SEQS ∈ {2,4,8}` 的
   正确性(logprob 一致性已有监控指标)与吞吐;若 kernel bug 可绕(升级 fla 或
   `GDN_PREFILL_BACKEND` 切换),预计 2-4× 吞吐 ≈ 同预算多 2-4× seeds。
2. **权重同步降频**:34-45s/步(~10%)。`behavior_logprobs_from_rollout=1` 意味着 staleness
   已被 rollout logprob 的 IS ratio 覆盖——同步周期从 1 提到 2-3 步理论上安全
   (`rollout_importance_ratio_outside_clip_fraction` 当前 0.003,有监控兜底)。先在 smoke
   上验证该指标不恶化再上生产。
3. **教师回放的 K-snapshot**(R4):教师 BC 信号密度 ×K,零 rollout 成本。
4. **122B 教师服务**:vllm2 env 可直接起(A10B 激活参数小,批量吞吐高);按计划预订硬窗口,
   与 flash 各采一套同上下文教师集,供家族轴对比。
5. **判别器训练本身**:buffer 1024、MLP 64 hidden,开销可忽略,无需优化。

---

## 6. 决策点汇总(需要拍板的)

| # | 决策 | 建议 | 依赖 |
|---|---|---|---|
| D1 | 判别器特征收窄到 action 段(R1) | 做,P0 | §4-1 离线测试确认 |
| D2 | teacher PG 统一为 hybrid 形式 + think 退出(R2) | 做,P0 | R4 段 mask |
| D3 | μ 分段与 nll-driver 容错(R3) | 做,P1 | R1 落地后 |
| D4 | 教师主次:122B 主 / flash 压力测试 | 建议如此 | 122B 采集窗口 |
| D5 | K-snapshot 教师回放,K 默认 2-4 | 做,随 R4 | exact-snapshot 管线 |
| D6 | MAX_NUM_SEQS 提升实验 | 立即排 | 验证 lane 空闲时 |
