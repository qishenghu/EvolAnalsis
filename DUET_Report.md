# DUET：面向Agent RLVR的双通道专家轨迹利用框架

## DUET: DUal Expert Trajectory Utilization for Agent RLVR

> **文档状态**：方法设计稿 v3（统一框架版）  
> **目标会议**：NeurIPS 2026  
> **最后更新**：2026-03-24  
> **核心主张**：专家轨迹中包含两类本质不同的信息——action-level（策略相关，需分布修正）和state-level（策略无关，可直接构造进度奖励）。DUET通过**Action Channel（密度比修正）**和**State Channel（进度奖励塑形）**分别处理两类信息，二者正交互补，共同解决agent RLVR中的分布修正和奖励稀疏问题。

---

## 目录

1. [问题背景与动机](#1-问题背景与动机)
2. [第一性原理分析：专家轨迹中的两类信息](#2-第一性原理分析专家轨迹中的两类信息)
3. [方法：DUET](#3-方法duet)
   - 3.1 设计哲学：两条正交通道
   - 3.2 Action Channel：密度比修正
   - 3.3 State Channel：进度奖励塑形
   - 3.4 DUET组合：完整的专家信息利用
   - 3.5 Step-Level Advantage扩展（可选增强）
   - 3.6 完整算法伪代码
4. [数学框架与理论保证](#4-数学框架与理论保证)
   - 4.1–4.2 State Channel理论（进度函数、非退化性、偏差有界性等）
   - 4.3 Action Channel理论（密度比估计、ESS-dual clipping）
   - 4.4 DUET组合理论（正交性、联合保证）
   - 4.5 Potential-Based Shaping的关系与澄清
   - 4.6 Step-Level扩展的数学基础
5. [与现有方法的系统性对比](#5-与现有方法的系统性对比)
6. [工程实现方案](#6-工程实现方案)
7. [实验设计](#7-实验设计)
8. [论文叙事与结构建议](#8-论文叙事与结构建议)
9. [风险评估与应对策略](#9-风险评估与应对策略)

---

## 1. 问题背景与动机

### 1.1 Agent RLVR中的奖励稀疏性

Group Relative Policy Optimization (GRPO) 已成为LLM后训练（post-training）的主流算法。其核心流程为：对每个prompt采样一组rollout，计算组内相对advantage，然后做policy gradient更新。

在multi-turn agentic环境（ALFWorld、WebShop等）中，GRPO面临**严重的奖励稀疏性问题**：

- **终端奖励**：只有任务完全成功时 $R=1$，否则 $R=0$（WebShop中为连续奖励但仍高度稀疏）
- **多步交互**：agent需要做5-20步决策，每一步都可能出错
- **探索困难**：随机策略完成任务的概率极低（ALFWorld中约10%，WebShop中获得高分概率更低）

**后果**：在GRPO group内，所有 $N$ 条rollout可能全部 $R=0$。此时：

$$A_i = \frac{R_i - \bar{R}}{\sigma_R} = \frac{0 - 0}{0} = \text{undefined}$$

advantage方差为零，**GRPO无法产生任何梯度信号**。训练停滞。

### 1.2 现有解决方案的局限

**LUFFY**（轨迹混入）：将expert trajectory直接混入GRPO group。当group内有1条expert（$R=1$）和7条on-policy（$R=0$）时，advantage非零。但信息量极低：所有on-policy获得均匀负advantage（$\approx -1/8$），没有区分哪条轨迹更好。且expert轨迹是off-policy数据，面临分布偏移问题。

**判别器修正**（如density ratio estimation）：用判别器估计密度比来修正off-policy expert trajectory的权重。缓解了分布偏移，但**不解决sparsity**——信号仍然只有1 bit（expert好 vs on-policy差）。且现有方法未给出完整的黑盒expert下的密度比估计方案。

**CHORD**（加权SFT组合）：expert data走加权SFT路径，再与on-policy RL做凸组合。回避了PG下的分布修正问题，但SFT本身有mode-covering局限，且不解决on-policy rollout的reward sparsity。

**TRAPO**（前缀引导）：将expert token作为前缀拼接，降低任务难度从而提高reward获取概率。但需要相同tokenizer，且机制上是token-level前缀拼接，不适用于multi-turn环境。

**R³**（反向课程）：从expert轨迹的中间状态启动rollout，逐步后退起点。需要环境支持从任意中间状态重置，且改变了rollout pipeline。

**GiGPO**（NeurIPS 2025，step-level credit）：在on-policy轨迹间识别共享anchor state，构建step-level advantage。但**不使用expert数据**——当所有on-policy轨迹都失败时，step-level advantage仍无法区分好坏action。

### 1.3 核心问题

> 在黑盒expert（如GPT-5、Claude Opus 4.6）只提供文本输出、没有logprob、tokenizer不同的约束下，如何**系统性地、完整地**利用expert trajectory中的所有可用信息，来**同时解决**分布修正和奖励稀疏两个问题？

我们的回答：通过**信息分解**——将expert trajectory分解为action-level和state-level两类信息，分别用最适合各自性质的方法处理。

---

## 2. 第一性原理分析：专家轨迹中的两类信息

### 2.1 专家轨迹的完整记录

一条expert trajectory的完整记录为：

$$\tau_E = (s_0^E, a_0^E, s_1^E, a_1^E, \ldots, s_T^E, R_E = 1)$$

其中 $s_t^E$ 是第 $t$ 步的环境状态（observation text），$a_t^E$ 是expert的action（text string）。

### 2.2 信息分类

**类型A — Action-level信息（"专家做了什么"）**

即 $\{a_0^E, a_1^E, \ldots, a_{T-1}^E\}$。

- 性质：**策略相关（policy-dependent）**。每个 $a_t^E$ 由 $\pi_{\text{expert}}(\cdot | s_t^E)$ 生成
- 要正确利用做policy gradient：**必须做分布修正**，即需要估计 $\frac{\pi_\theta(a_t^E | s_t^E)}{\pi_{\text{expert}}(a_t^E | s_t^E)}$
- 黑盒expert下的困难：$\pi_{\text{expert}}$ 完全不可获取，tokenizer不同导致概率空间不对齐

**类型B — State-level信息（"专家经过了哪里"）**

即 $\{s_0^E, s_1^E, \ldots, s_T^E\}$。

- 性质：**策略无关（policy-free）**。环境状态是环境的属性，不属于任何特定策略
- 利用方式：任何策略都可以访问这些状态。状态序列本身编码了"通向成功的路径信息"
- 黑盒约束下：**完全不受限制**。只需要observation text，无需logprob或tokenizer对齐

### 2.3 核心洞察

> **现有所有方法都在利用Type A信息（expert action），因此都面临分布修正的困难。**
>
> **Type B信息（expert state sequence）被完全忽略了，而它恰好是无需分布修正、且可以直接解决reward sparsity的信息来源。**

这个信息分解是DUET的**第一性原理基础**。基于此分解，DUET设计了两条正交通道：Action Channel利用Type A信息（配合分布修正），State Channel利用Type B信息（无需分布修正）。

---

## 3. 方法：DUET

### 3.1 设计哲学：两条正交通道

基于§2的信息分解，我们提出DUET框架，通过两条正交通道**完整利用**expert trajectory中的所有可用信息：

| | Action Channel | State Channel |
|---|---|---|
| **利用的信息类型** | Type A — Action-level $\{a_t^E\}$ | Type B — State-level $\{s_t^E\}$ |
| **核心机制** | 判别器密度比修正 | 状态进度奖励塑形 |
| **解决的问题** | 分布修正（off-policy bias） | 奖励稀疏（advantage退化） |
| **作用位置** | Policy gradient权重（修复 `old_log_prob`） | Reward信号（添加进度bonus） |
| **正交性** | 不改变reward → 不影响State Channel | 不改变PG权重 → 不影响Action Channel |

**关键洞察**：两条通道作用于GRPO训练流程的**不同位置**——Action Channel修改的是policy gradient中的importance weight，State Channel修改的是reward signal。它们互不干扰，可以独立使用，也可以组合使用获得更好效果。

---

### 3.2 Action Channel：密度比修正

#### 3.2.1 问题：黑盒Expert的Action信息利用

当将expert trajectory混入GRPO batch时（LUFFY方式），expert的action $a_t^E$ 是off-policy数据。标准policy gradient需要importance weight：

$$w(a_t^E | s_t^E) = \frac{\pi_\theta(a_t^E | s_t^E)}{\pi_{\text{expert}}(a_t^E | s_t^E)}$$

但黑盒expert下，$\pi_{\text{expert}}$ 完全不可获取。LUFFY直接忽略这个修正，导致off-policy bias。CHORD用加权SFT回避了PG，但SFT本身有mode-covering的局限。

**Action Channel的核心思路**：不直接估计 $\pi_\theta / \pi_{\text{expert}}$，而是用判别器在**序列级别**估计on-policy分布与teacher分布的密度比，然后通过修复 `old_log_prob` 将teacher数据送入标准PPO/RePO风格的token-level loss。

#### 3.2.2 判别器密度比估计

**定义（序列级密度比）**：设 $p(\tau)$ 为当前on-policy策略的轨迹分布，$q(\tau)$ 为teacher轨迹分布。训练一个二元分类器 $D_\phi(\mathbf{f}(\tau)) \in (0,1)$ 区分on-policy（标签1）和teacher（标签0）样本，其中 $\mathbf{f}(\tau)$ 是轨迹的序列级特征向量。

**判别器训练目标**：

$$\mathcal{L}_D = -\mathbb{E}_{\tau \sim p}\left[\log D_\phi(\mathbf{f}(\tau))\right] - \mathbb{E}_{\tau \sim q}\left[\log(1 - D_\phi(\mathbf{f}(\tau)))\right]$$

在Bayes最优解处：

$$D^*(\mathbf{f}) = \frac{(1-\alpha) \cdot p(\mathbf{f})}{(1-\alpha) \cdot p(\mathbf{f}) + \alpha \cdot q(\mathbf{f})}$$

其中 $\alpha$ 是混合batch中teacher样本的占比。

**从后验到密度比**：

$$\hat{r} = \frac{D_\phi}{1 - D_\phi} \approx \frac{(1-\alpha) \cdot p}{(1-\alpha) \cdot p + \alpha \cdot q} \cdot \frac{\alpha \cdot q + (1-\alpha) \cdot p}{q \cdot \alpha}$$

简化后得到 `p/q` 风格的odds ratio。

**Relative Density Ratio**（降低方差）：

$$\hat{w}_\alpha = \frac{\hat{r}}{(1-\alpha)\hat{r} + \alpha}$$

$\hat{w}_\alpha$ 的理论范围为 $[0, \frac{1}{1-\alpha}]$，比直接的 $p/q$ ratio（范围 $[0, \infty)$）有更小的方差。

#### 3.2.3 序列级特征设计

Action Channel不直接处理token序列，而是提取**轻量级序列级特征** $\mathbf{f}(\tau)$，包括：

- **Log-probability统计**：$\log \pi_\theta$ 沿response tokens的均值、标准差、最小值、尾部分位数
- **KL-to-reference proxy**：$\log \pi_\theta - \log \pi_{\text{ref}}$ 的统计量（当reference model可用时）
- **Response mask特征**：response长度（归一化）、有效token比例

当前主配置使用 `v3_aug` 特征模式（~10-15维），输入轻量MLP判别器（2-3层，隐藏维度64-128）。

**设计理由**：序列级特征+MLP的计算开销远低于token-level判别器，且避免了token-level判别器在不同tokenizer间的对齐问题。

#### 3.2.4 `old_log_prob` 修复机制

Action Channel的输出不是直接乘在loss上的权重，而是通过修复 `old_log_prob` 融入现有PPO/RePO框架：

$$\log \pi_{\text{beh}}(\tau) \leftarrow \text{sg}(\log \pi_\theta(\tau)) - \log \hat{w}_\alpha(\tau)$$

其中 $\text{sg}(\cdot)$ 表示stop-gradient。修复后的 $\log \pi_{\text{beh}}$ 作为标准RePO/PPO token-level loss的behavioral policy log-probability。

**直觉**：对于on-policy样本，$\hat{w}_\alpha \approx 1$，$\log \hat{w}_\alpha \approx 0$，修复无效果。对于teacher样本，$\hat{w}_\alpha$ 反映了当前策略与teacher的分布差异，修复后的importance ratio $\pi_\theta / \pi_{\text{beh}}$ 正确反映了off-policy correction。

#### 3.2.5 ESS-Dual Clipping（方差控制）

为防止密度比估计不准导致的高方差梯度，Action Channel使用Effective Sample Size (ESS) 驱动的自适应clipping：

**ESS定义**：

$$\text{ESS}(\{u_i\}) = \frac{(\sum_i u_i)^2}{\sum_i u_i^2}$$

其中 $\{u_i\}$ 是滑动窗口内off-policy样本的权重。ESS越低，表示少数样本主导了梯度，方差越大。

**Dual变量更新**：

$$\lambda \leftarrow \left[\lambda + \eta(\kappa N - \text{ESS})\right]_+$$

其中 $\kappa \in (0,1)$ 是目标ESS比例（如0.5），$N$ 是窗口内off-policy样本数。

**自适应Clip上界**：

$$\text{clip\_upper} = \frac{\min\left(\frac{1}{1-\alpha},\ \text{clip\_max}\right)}{1 + \lambda}$$

当ESS过低时，$\lambda$ 增大，clip上界收紧，限制极端权重的影响。当ESS恢复到目标水平，$\lambda$ 自动减小。

#### 3.2.6 Reward-Gap Teacher Gate

随着on-policy策略改善，teacher轨迹的信息价值递减。Action Channel通过reward-gap gate实现teacher影响的自适应衰减：

$$\text{teacher\_loss\_scale} = f(\Delta R)^{\gamma_{\text{gate}}}$$

其中 $\Delta R = \bar{R}_{\text{teacher}} - \bar{R}_{\text{on-policy}}$ 是teacher与on-policy的平均reward差距，$f$ 是归一化映射，$\gamma_{\text{gate}}$ 控制衰减速度。

**效果**：当on-policy reward接近teacher水平时，$\Delta R \to 0$，teacher_loss_scale → 0，teacher influence自动消失。

#### 3.2.7 训练稳定化机制

- **Warmup**：前 $K_{\text{warmup}}$ 步只observe不apply，让判别器积累足够样本
- **Rolling Buffer**：跨micro-batch累积样本，解决 `micro_batch_size=1` 下的单类样本问题
- **Class-Balanced Sampling**：从buffer中平衡采样on-policy/teacher样本训练判别器
- **Label Smoothing + Temperature**：防止判别器过度自信
- **Multi-GPU Sync**：`all_gather` 特征 + `broadcast` 参数

---

### 3.3 State Channel：进度奖励塑形

#### 3.3.1 设计哲学

**将expert trajectory的state信息角色从"被忽略"转变为"环境结构信息"。**

具体地：
- expert trajectory的**状态序列**用于构造一个进度度量函数
- 该进度度量函数为每条on-policy trajectory提供**密集的shaped reward信号**
- shaped reward使得GRPO group在reward sparsity下仍有非零advantage方差
- **无需将expert trajectory混入GRPO batch**，因此State Channel本身不产生off-policy问题

#### 3.3.2 进度函数（Progress Function）

**定义**：给定task $q$ 的一条expert trajectory $\tau_E$，其状态序列为 $(s_E^0, s_E^1, \ldots, s_E^T)$。定义**状态进度函数** $\Phi_q: \mathcal{S} \to [0, 1]$ 为：

$$\Phi_q(s) = \max_{j \in \{0, 1, \ldots, T\}} \left[\text{match}(s, s_E^j) \cdot \frac{j}{T}\right]$$

其中 $\text{match}(s, s') \in \{0, 1\}$ 是环境状态匹配函数：

- **结构化环境**（ALFWorld）：基于状态hash的精确匹配
- **半结构化环境**（WebShop）：基于页面内容hash或embedding相似度的匹配（阈值 $\delta$）

**直觉**：$\Phi_q(s) = k/T$ 表示"状态 $s$ 对应于expert在完成任务过程中 $k/T$ 的进度位置"。

**性质**：
- $\Phi_q(s) \in [0, 1]$ $\forall s$
- $\Phi_q(s_E^T) = 1$（expert终态的进度为1）
- $\Phi_q(s) = 0$ 若 $s$ 不匹配expert轨迹中的任何状态

### 3.3.3 轨迹进度度量（Trajectory Progress Measure）

**定义**：对一条on-policy trajectory $\tau_i = (s_0^i, a_0^i, \ldots, s_{T_i}^i)$，定义其**轨迹进度**为：

$$P(\tau_i) = \frac{1}{T_i + 1} \sum_{t=0}^{T_i} \Phi_q(s_t^i)$$

**为什么用平均累积而不用差分形式**：

差分形式 $\sum_t [\Phi(s_{t+1}) - \Phi(s_t)]$ 是经典的potential-based reward shaping，但在GRPO框架下（$\gamma = 1$，trajectory-level reward），差分会telescope为：

$$\sum_{t=0}^{T-1}[\Phi(s_{t+1}) - \Phi(s_t)] = \Phi(s_T) - \Phi(s_0)$$

所有中间步进度变化信息被压缩为"终态进度 - 初态进度"一个标量，丧失了dense signal的目的。

而平均累积 $P(\tau)$ 衡量的是"轨迹在expert路径上的**整体停留时间**"，能区分以下不同模式：

| 轨迹描述 | $\Phi(s_T)$ (终态进度) | $P(\tau)$ (累积进度) |
|---|---|---|
| 前3步对齐→偏离→终态差 | 0.1 | 0.25 |
| 一开始走错→最后碰巧到达好位置 | 0.6 | 0.15 |
| 前5步稳定对齐→第6步偏离 | 0.4 | 0.45 |
| 全程大部分对齐expert | 0.8 | 0.70 |

$P(\tau)$ 正确反映了轨迹的**整体质量**，而不仅仅是终态。

### 3.3.4 Shaped Reward与GRPO集成

**Shaped Reward**：

$$R'(\tau_i) = R(\tau_i) + \beta \cdot P(\tau_i)$$

其中 $\beta > 0$ 是进度奖励系数（超参数）。

**GRPO Advantage**（在shaped reward下）：

$$A_i' = \frac{R'(\tau_i) - \overline{R'}}{\max(\sigma_{R'}, \epsilon)}$$

其中 $\overline{R'} = \frac{1}{N}\sum_{i=1}^N R'(\tau_i)$，$\sigma_{R'} = \text{std}(\{R'(\tau_i)\}_{i=1}^N)$。

**Policy Gradient**（标准GRPO形式，不变）：

$$\nabla_\theta J = \frac{1}{N}\sum_{i=1}^N A_i' \sum_{t=0}^{T_i} \nabla_\theta \log \pi_\theta(a_t^i | s_t^i)$$

---

### 3.4 DUET组合：完整的专家信息利用

当同时启用Action Channel和State Channel时，两条通道在训练流程中的作用位置如下：

```
Expert Trajectory τ_E = (s_0^E, a_0^E, s_1^E, a_1^E, ..., s_T^E)
                    │                          │
                    │ Type A: actions           │ Type B: states
                    ▼                          ▼
         ┌──────────────────┐       ┌──────────────────┐
         │  Action Channel  │       │  State Channel   │
         │                  │       │                  │
         │  判别器估计       │       │  构造进度函数Φ    │
         │  密度比 ŵ_α      │       │  计算P(τ)        │
         └────────┬─────────┘       └────────┬─────────┘
                  │                          │
                  │ 修复old_log_prob         │ 修改reward
                  │ (PG权重修正)             │ R'=R+β·P(τ)
                  ▼                          ▼
         ┌──────────────────────────────────────────┐
         │           GRPO Training Loop             │
         │                                          │
         │  Advantage = (R'_i - mean(R')) / σ       │
         │  PG = Σ A_i · ∇log π_θ(a|s)            │
         │  (teacher样本用修复后的old_log_prob)       │
         └──────────────────────────────────────────┘
```

**组合时的数据流**：

1. **Rollout阶段**：生成 $N$ 条on-policy轨迹，同时将 $K$ 条teacher轨迹混入（LUFFY方式）
2. **State Channel（Reward层）**：对**所有**轨迹（包括teacher和on-policy），计算 $P(\tau_i)$ 并施加shaped reward $R'_i = R_i + \beta P_i$
3. **GRPO Advantage**：基于shaped reward $R'$ 计算group-relative advantage
4. **Action Channel（PG权重层）**：仅对**teacher轨迹**（无logprob的），估计密度比并修复 `old_log_prob`
5. **Token-level Loss**：所有样本统一送入RePO/PPO loss，teacher样本使用修复后的importance weight

**组合的正交性保证**：

State Channel只修改 $R'$（进而影响 $A_i'$），不触碰importance weight。Action Channel只修改teacher样本的 `old_log_prob`（进而影响importance ratio $\pi_\theta / \pi_{\text{beh}}$），不触碰reward。因此两者的效果严格叠加，不存在交互干扰。

形式化地，组合后的policy gradient对teacher样本 $\tau_j$ 为：

$$\nabla_\theta J_j = A_j'(\underbrace{R_j + \beta P_j}_{\text{State Channel: shaped reward}}) \cdot \sum_t \frac{\pi_\theta(a_t^j | s_t^j)}{\underbrace{\hat{\pi}_{\text{beh}}(a_t^j | s_t^j)}_{\text{Action Channel-repaired}}} \nabla_\theta \log \pi_\theta(a_t^j | s_t^j)$$

对on-policy样本 $\tau_i$：

$$\nabla_\theta J_i = A_i'(\underbrace{R_i + \beta P_i}_{\text{State Channel}}) \cdot \sum_t \nabla_\theta \log \pi_\theta(a_t^i | s_t^i)$$

（on-policy样本的importance ratio为1，Action Channel不介入。）

### 3.5 Step-Level Advantage扩展（可选增强）

作为论文的扩展实验，可以将进度函数 $\Phi$ 作为**近似value function**进行advantage decomposition：

$$A_{\text{final}}(i, t) = \underbrace{A'_i}_{\text{episode-level (GRPO)}} + \eta \cdot \underbrace{[\Phi(s_{t+1}^i) - \Phi(s_t^i)]}_{\text{step-level progress signal}}$$

含义：
- $\Phi(s_{t+1}) > \Phi(s_t)$（向expert路径靠近）→ 该步action获得额外正advantage
- $\Phi(s_{t+1}) < \Phi(s_t)$（偏离expert路径）→ 该步action获得额外负advantage
- $\Phi(s_{t+1}) = \Phi(s_t) = 0$（在expert路径之外）→ 无额外信号，退化为标准GRPO

注意：这个step-level项**不是**通过修改reward实现的（否则会telescope），而是**直接作用于advantage**。它等价于将 $\Phi$ 当作近似state-value function做TD-style advantage estimation。

### 3.6 完整算法伪代码

```
算法：DUET (DUal Expert Trajectory Utilization)
══════════════════════════════════════════════════════════

输入：
  - 任务集 T
  - 每个task的至少一条expert trajectory {τ_E}
  - 当前策略 π_θ
  - 超参数：β (进度权重), η (step-level系数, 可选)
  - Action Channel配置：判别器网络D_φ, ESS目标κ, α估计模式

═══ 离线预处理 ═══

// State Channel: 构造进度映射（每个task执行一次）
FOR each task q ∈ T:
  τ_E = expert_trajectories[q]
  ProgressMap_q = {}
  FOR j = 0, 1, ..., T_E:
    key = StateHash(normalize(observation_j^E))
    ProgressMap_q[key] = max(ProgressMap_q.get(key, 0), j / T_E)
  END FOR
END FOR

// Action Channel: 初始化Action Channel判别器和rolling buffer
初始化 D_φ (轻量MLP), RollingBuffer, λ_dual = 0

═══ 在线训练循环 ═══

FOR each training step:
  FOR each task q ∈ current_batch:
    
    // ─── 第1步：On-Policy Rollout + Teacher Mixing ───
    FOR i = 1, ..., N:
      τ_i = π_θ.rollout(q)
    END FOR
    // LUFFY-style: 将K条teacher trajectory替换部分on-policy rollout
    混合后的batch = {τ_1, ..., τ_{N-K}, τ_E^1, ..., τ_E^K}
    
    // ─── 第2步 [State Channel]：计算轨迹进度 ───
    FOR each trajectory τ_i (包括on-policy和teacher):
      progress_sum = 0
      FOR each step t in τ_i:
        key = StateHash(normalize(observation_t^i))
        progress_sum += ProgressMap_q.get(key, 0)
      END FOR
      P_i = progress_sum / len(τ_i)
    END FOR
    
    // ─── 第3步 [State Channel]：Shaped Reward ───
    FOR each trajectory τ_i:
      R'_i = R_i + β · P_i
    END FOR
    
    // ─── 第4步：GRPO Advantage（公式不变，输入改为 R'）───
    A'_i = (R'_i - mean({R'_j})) / max(std({R'_j}), ε)
    
    // ─── 第5步（可选）：Step-Level Advantage ───
    IF use_step_level:
      FOR each trajectory τ_i, each step t:
        Δ_t = Φ(s_{t+1}^i) - Φ(s_t^i)
        A_final(i, t) = A'_i + η · Δ_t
      END FOR
    ELSE:
      A_final(i, t) = A'_i
    END IF
    
    // ─── 第6步 [Action Channel]：密度比修正（仅对teacher样本）───
    FOR each micro_batch:
      // 6a: 提取序列级特征并推入buffer
      features = extract_sequence_features(log_probs, response_mask, ...)
      RollingBuffer.push(features, is_teacher_labels)
      
      // 6b: 训练判别器（class-balanced sampling from buffer）
      D_φ.train_step(RollingBuffer.balanced_sample())
      
      // 6c: 估计密度比
      r̂ = D_φ(features) / (1 - D_φ(features))
      ŵ_α = r̂ / ((1-α)·r̂ + α)
      
      // 6d: ESS-dual clipping
      ESS = compute_ESS(ŵ_α[teacher_indices])
      λ_dual = max(0, λ_dual + η_dual·(κ·N_off - ESS))
      clip_upper = min(1/(1-α), clip_max) / (1 + λ_dual)
      ŵ_α = clip(ŵ_α, 1.0, clip_upper)
      
      // 6e: 修复teacher样本的old_log_prob
      FOR each teacher sample j in micro_batch:
        old_log_prob_j = sg(log π_θ(τ_j)) - log ŵ_α(τ_j)
      END FOR
    END FOR
    
    // ─── 第7步：Policy Gradient（标准RePO/PPO token loss）───
    loss = Σ_i Σ_t A_final(i,t) · clip(π_θ/π_beh, 1-ε, 1+ε) · log π_θ(a_t^i | s_t^i)
    // 其中on-policy样本的π_beh = π_old（标准），teacher样本的π_beh = Action Channel修复值
    
  END FOR
  
  θ ← θ - lr · ∇_θ loss
  
END FOR
```

**简化配置**：如果只使用State Channel（不混入teacher trajectory），则跳过第1步的mixing和第6步的密度比修正，算法退化为纯State Channel模式——在纯on-policy rollout上施加进度shaped reward，无任何off-policy组件。

---

## 4. 数学框架与理论保证

### 4.1 基本定义

**定义 4.1（Expert Progress Function）**

给定task $q$ 的expert trajectory $\tau_E = (s_E^0, a_E^0, \ldots, s_E^T)$，定义状态进度函数 $\Phi_q: \mathcal{S} \to [0,1]$：

$$\Phi_q(s) = \max_{j \in \{0, \ldots, T\}} \left[\mathbb{1}[\text{match}(s, s_E^j)] \cdot \frac{j}{T}\right]$$

若 $s$ 不匹配任何expert state，则 $\Phi_q(s) = 0$。

**定义 4.2（Trajectory Progress Measure）**

对on-policy trajectory $\tau = (s_0, a_0, \ldots, s_{T_\tau})$：

$$P(\tau) = \frac{1}{T_\tau + 1}\sum_{t=0}^{T_\tau}\Phi_q(s_t)$$

**定义 4.3（Shaped Reward）**

$$R'(\tau) = R(\tau) + \beta \cdot P(\tau), \quad \beta > 0$$

**定义 4.4（GRPO Advantage under Shaped Reward）**

对GRPO group $\mathcal{G} = \{\tau_1, \ldots, \tau_N\}$：

$$A_i' = \frac{R'(\tau_i) - \overline{R'}}{\max(\sigma_{R'}, \epsilon)}$$

其中 $\overline{R'} = \frac{1}{N}\sum_{i=1}^N R'(\tau_i)$，$\sigma_{R'} = \sqrt{\frac{1}{N}\sum_{i=1}^N (R'(\tau_i) - \overline{R'})^2}$。

### 4.2 核心定理

#### 命题 1（Advantage非退化性 — 解决稀疏问题的核心保证）

> 设GRPO group $\mathcal{G} = \{\tau_1, \ldots, \tau_N\}$，所有terminal reward相同（$R(\tau_i) = c$ $\forall i$）。若至少存在 $i \neq j$ 使得 $P(\tau_i) \neq P(\tau_j)$，则：
>
> $$\sigma_{R'} = \beta \cdot \sigma_P > 0$$
>
> 即shaped reward下的advantage方差严格为正。

**证明**：

由 $R(\tau_i) = c$ $\forall i$，有 $R'(\tau_i) = c + \beta P(\tau_i)$。因此：

$$\sigma_{R'} = \text{std}(\{c + \beta P(\tau_i)\}_{i=1}^N) = \beta \cdot \text{std}(\{P(\tau_i)\}_{i=1}^N) = \beta \cdot \sigma_P$$

由 $P(\tau_i) \neq P(\tau_j)$，有 $\sigma_P > 0$，故 $\sigma_{R'} > 0$。$\square$

**推论 1.1**：在reward sparsity场景（$c = 0$，所有on-policy轨迹失败），只要不同轨迹的状态序列与expert trajectory有不同程度的重叠，GRPO就能产生非零梯度信号。

**推论 1.2**：shaped advantage简化为：

$$A_i' = \frac{P(\tau_i) - \bar{P}}{\max(\sigma_P, \epsilon/\beta)}$$

即advantage完全由轨迹进度的相对高低决定。

---

#### 命题 2（偏差有界性）

> shaped reward对advantage排序引入的偏差有界。设原始advantage为 $A_i$（当 $\sigma_R > 0$ 时有定义），shaped advantage为 $A_i'$。则：
>
> $$A_i' = \frac{(R_i - \bar{R}) + \beta(P_i - \bar{P})}{\sqrt{\sigma_R^2 + 2\beta\text{Cov}(R, P) + \beta^2\sigma_P^2}}$$

**证明**：

$$R'_i - \overline{R'} = (R_i + \beta P_i) - (\bar{R} + \beta\bar{P}) = (R_i - \bar{R}) + \beta(P_i - \bar{P})$$

$$\sigma_{R'}^2 = \text{Var}(R + \beta P) = \sigma_R^2 + 2\beta\text{Cov}(R, P) + \beta^2\sigma_P^2$$

代入 $A_i'$ 的定义即得。$\square$

**分析**：

当 $\beta \to 0$ 时，$A_i' \to A_i$（退化为标准GRPO）。

当 $\sigma_R = 0$（完全稀疏）时，$A_i' = \frac{P_i - \bar{P}}{\sigma_P}$（完全由进度决定）。

当 $\sigma_R > 0$ 时，shaped advantage是原始advantage与进度advantage的**加权混合**，权重由 $\beta$ 控制。

---

#### 命题 3（方向一致性 — 保证偏差"有益"）

> 设进度函数 $\Phi$ 从成功expert trajectory构造，且状态进度与任务完成概率正相关，即：
>
> $$\mathbb{E}[R(\tau) | P(\tau) = p_1] \geq \mathbb{E}[R(\tau) | P(\tau) = p_2] \quad \text{whenever } p_1 > p_2$$
>
> 则 $\text{Cov}(R, P) \geq 0$，且shaped reward的排序偏差方向与原始reward的期望排序一致。

**证明概要**：

由正相关假设，$R$ 和 $P$ 之间存在正协方差。shaped reward $R' = R + \beta P$ 是两个正相关变量的正系数线性组合，因此 $R'$ 的排序在期望上与 $R$ 一致。形式化地：

$$\text{Cov}(R, R') = \text{Cov}(R, R + \beta P) = \sigma_R^2 + \beta\text{Cov}(R, P) \geq \sigma_R^2 > 0$$

（当 $\sigma_R > 0$ 时。当 $\sigma_R = 0$ 时，原始advantage无定义，shaped advantage提供唯一信号。） $\square$

**直觉**：进度度量 $P$ 衡量的是"轨迹在多大程度上经过了expert曾经过的状态"。由于expert轨迹最终成功（$R_E = 1$），经过更多expert状态的轨迹在期望上更接近成功。因此shaped reward给"更接近成功"的轨迹更高奖励，与原始目标方向一致。

---

#### 命题 4（自然课程效应）

> 随着训练推进，若 $\pi_\theta$ 改善使得大部分on-policy轨迹的 $P(\tau)$ 趋向于1，则：
>
> (a) 在渐近阶段（$P(\tau_i) \to 1$ $\forall i$），$\sigma_P \to 0$
>
> (b) 进度项 $\beta(P_i - \bar{P})$ 对advantage的影响趋于零
>
> (c) shaped advantage退化为标准GRPO advantage

**注意**：在训练中间阶段，$\sigma_P$ 可能**不单调**——当部分轨迹已改善（$P \approx 0.6$）而另一些仍差（$P \approx 0.1$）时，$\sigma_P$ 可能暂时增大。这不影响方法的正确性：方差增大意味着更强的区分能力，此时shaped reward提供了更丰富的信号。课程效应体现在**渐近行为**上：当策略充分收敛后，$\sigma_P \to 0$，shaped reward自动退出。

**直觉**：方法在训练早期和中期提供主要信号（$\sigma_P > 0$时），在训练后期自动消退（$\sigma_P \to 0$时），遵循"bootstrap then let go"的课程效应。

---

#### 命题 5（信息分解的完备性与无分布修正利用的理论依据）

> Expert trajectory中的可利用信息可分解为：
>
> - **Action-level信息** $\{a_t^E\}$：由 $\pi_{\text{expert}}$ 生成，是off-policy数据，利用时需要分布修正
> - **State-level信息** $\{s_t^E\}$：是环境状态，与任何策略无关
>
> **State-level信息构造的进度函数 $\Phi_q(s)$ 是一个关于环境状态的确定性函数，其取值不依赖于任何策略。** 因此将 $\Phi_q$ 用于reward shaping不引入off-policy偏差。

**证明**：

$\Phi_q(s)$ 的定义只涉及 $s$ 与 $\{s_E^j\}$ 的匹配关系以及expert trajectory中的步序 $j/T$。这些都是确定性量——不涉及任何策略的概率分布。因此 $\Phi_q$ 是一个固定的状态函数（类似于手工设计的reward shaping），其使用等价于对环境reward function的修改，完全在on-policy RL的标准理论框架内。$\square$

---

### 4.3 Action Channel的数学框架

#### 定义 4.6（混合分布）

设on-policy策略 $\pi_\theta$ 的轨迹分布为 $p(\tau)$，teacher策略的轨迹分布为 $q(\tau)$。在LUFFY-style mixing下，GRPO batch中的样本来自混合分布：

$$m(\tau) = (1-\alpha) \cdot p(\tau) + \alpha \cdot q(\tau)$$

其中 $\alpha = K/N$ 是teacher轨迹占比。

#### 定义 4.7（Relative Density Ratio）

$$w_\alpha(\tau) = \frac{p(\tau)}{(1-\alpha) \cdot p(\tau) + \alpha \cdot q(\tau)} = \frac{p(\tau)}{m(\tau)}$$

**性质**：$w_\alpha(\tau) \in [0, \frac{1}{1-\alpha}]$。对on-policy样本，$w_\alpha \approx 1$；对远离on-policy分布的teacher样本，$w_\alpha \approx 0$。

#### 命题 6（判别器与Relative Density Ratio的关系）

> 设二元分类器 $D_\phi$ 在**class-balanced采样**下训练（50/50采样on-policy和teacher样本），标签1为on-policy，标签0为teacher。则Bayes最优判别器满足：
>
> $$D^*(\mathbf{f}) = \frac{p(\mathbf{f})}{p(\mathbf{f}) + q(\mathbf{f})}$$
>
> 对应的odds ratio为直接密度比：
>
> $$\hat{r} = \frac{D_\phi(\mathbf{f})}{1 - D_\phi(\mathbf{f})} \approx \frac{p(\mathbf{f})}{q(\mathbf{f})}$$
>
> 进而恢复relative density ratio：
>
> $$\hat{w}_\alpha = \frac{\hat{r}}{(1-\alpha)\hat{r} + \alpha} \approx \frac{p(\mathbf{f})}{(1-\alpha) p(\mathbf{f}) + \alpha q(\mathbf{f})} = w_\alpha(\mathbf{f})$$

**证明**：

Class-balanced BCE的最优解为 $D^*(x) = \frac{p(x)}{p(x) + q(x)}$（标准GAN判别器结果）。

取odds：$\hat{r} = \frac{D^*}{1 - D^*} = \frac{p}{q}$。

代入relative ratio公式：

$$\hat{w}_\alpha = \frac{p/q}{(1-\alpha)(p/q) + \alpha} = \frac{p}{(1-\alpha)p + \alpha q} = \frac{p}{m} = w_\alpha \quad \square$$

**设计选择的理由**：使用class-balanced采样（而非按真实混合比例 $(1-\alpha, \alpha)$ 采样）有两个好处：(1) 在 `micro_batch_size=1` 下，rolling buffer的class-balanced采样确保两类样本数目相当，判别器训练更稳定；(2) 通过odds → relative ratio的两步转换，将 $\alpha$（混合比例）的信息从判别器训练中解耦，使判别器只需学习 $p/q$，$\alpha$ 的估计可以独立进行（如通过batch统计或EMA）。

#### 命题 7（`old_log_prob` 修复的正确性）

> 设teacher轨迹 $\tau_E$ 的真实行为策略为 $\pi_{\text{expert}}$（不可观测）。Action Channel通过以下修复将teacher数据纳入标准PPO框架：
>
> $$\log \hat{\pi}_{\text{beh}}(\tau_E) = \text{sg}(\log \pi_\theta(\tau_E)) - \log \hat{w}_\alpha(\tau_E)$$
>
> 当判别器完美（$\hat{w}_\alpha = w_\alpha$）时，使用修复后的importance ratio等价于在混合分布 $m$ 上做重要性采样：
>
> $$\frac{\pi_\theta(\tau_E)}{\hat{\pi}_{\text{beh}}(\tau_E)} = \hat{w}_\alpha(\tau_E) \approx \frac{p(\tau_E)}{m(\tau_E)}$$
>
> 这恰好是从混合分布 $m$ 到目标分布 $p$ 的正确importance weight。

**证明概要**：

$$\frac{\pi_\theta(\tau_E)}{\hat{\pi}_{\text{beh}}(\tau_E)} = \frac{\pi_\theta(\tau_E)}{\pi_\theta(\tau_E) / \hat{w}_\alpha(\tau_E)} = \hat{w}_\alpha(\tau_E)$$

当 $\hat{w}_\alpha = w_\alpha = p/m$ 时，importance ratio恰好修正了从混合分布 $m$ 到on-policy分布 $p$ 的采样偏差。$\square$

**与直接 IS 的对比**：标准 IS 需要 $\pi_\theta / \pi_{\text{expert}}$（需要 $\pi_{\text{expert}}$ 的log-prob，黑盒下不可获取）。Action Channel转而估计 $p/m$（只需区分on-policy vs teacher的分布差异，不需要 $\pi_{\text{expert}}$ 本身），通过判别器实现。

#### 命题 8（ESS-Dual Clipping的方差控制）

> 设off-policy权重为 $\{u_i = \hat{w}_\alpha(\tau_i)\}_{i \in \text{teacher}}$，ESS定义为：
>
> $$\text{ESS}(\{u_i\}) = \frac{(\sum_i u_i)^2}{\sum_i u_i^2}$$
>
> Dual变量更新 $\lambda \leftarrow [\lambda + \eta(\kappa N - \text{ESS})]_+$ 配合clip上界 $C = \frac{C_{\max}}{1+\lambda}$ 保证：
>
> (a) 当 $\text{ESS} < \kappa N$（方差过大），$\lambda$ 增大，clip收紧，限制极端权重
>
> (b) 当 $\text{ESS} \geq \kappa N$（方差可控），$\lambda$ 减小，clip放松，保留更多信息
>
> (c) 在稳态下，ESS收敛到 $\kappa N$ 附近，实现自适应方差控制

**直觉**：ESS-dual机制自动在"信息保留"（保留off-policy权重的差异性）和"方差控制"（防止少数样本主导梯度）之间取平衡。

---

### 4.4 DUET组合的理论分析

#### 命题 9（正交性 — Action Channel和State Channel互不干扰）

> DUET的Action Channel和State Channel在GRPO训练流程中作用于**不同的量**：
>
> - State Channel修改 $R'(\tau) = R(\tau) + \beta P(\tau)$，进而影响advantage $A_i'$
> - Action Channel修改teacher样本的importance weight $w_\alpha$，即 `old_log_prob`
>
> 这两个修改**不存在循环依赖**：
> - $P(\tau)$ 的计算不依赖于importance weight
> - $\hat{w}_\alpha$ 的估计不依赖于reward值（判别器区分的是轨迹分布，不是reward）
>
> 因此组合的效果是**严格叠加**的。

**形式化**：组合后的policy gradient可分解为：

$$\nabla_\theta J = \underbrace{\frac{1}{N}\sum_{i \in \text{on-policy}} A_i'(R') \nabla_\theta \log \pi_\theta(\tau_i)}_{\text{on-policy项（State Channel提供shaped reward）}} + \underbrace{\frac{1}{N}\sum_{j \in \text{teacher}} A_j'(R') \cdot \hat{w}_\alpha(\tau_j) \cdot \nabla_\theta \log \pi_\theta(\tau_j)}_{\text{teacher项（Action Channel提供权重修正，State Channel提供shaped reward）}}$$

on-policy项仅受State Channel影响（shaped advantage），teacher项同时受Action Channel（权重修正）和State Channel（shaped advantage）影响，但两者的作用机制独立。

#### 命题 10（组合优于单一通道的条件）

> (a) **State Channel alone sufficient condition**：当on-policy轨迹之间的进度差异 $\sigma_P > 0$ 时，即使不混入teacher trajectory，GRPO也有非零advantage方差。此时Action Channel不是必需的。
>
> (b) **Action Channel provides additional value when**：teacher trajectory的action信息提供了"超出state coverage的指导"——即存在某些状态 $s$ 处，on-policy策略的action质量远低于expert，且state进度信号无法区分这些action之间的差异。此时Action Channel修正的teacher gradient提供了step-level的action-specific信号。
>
> (c) **Diminishing returns**：当State Channel的进度信号已经让on-policy策略充分改善后（$\sigma_P \to 0$，$\bar{R} \to$ target），Action Channel的teacher influence也应通过reward-gap gate自然消退。两条通道的消退机制互相一致。

**直觉**：State Channel回答"哪条路好"（trajectory-level信号），Action Channel回答"在同一个位置上哪个动作好"（action-level信号）。两者提供不同粒度的学习信号。

---

### 4.5 Potential-Based Shaping的关系与澄清

#### 为什么DUET不是标准的Potential-Based Reward Shaping

经典potential-based reward shaping（Ng et al., 1999）定义shaped reward为：

$$F(s, a, s') = \gamma\Phi(s') - \Phi(s)$$

在 $\gamma < 1$ 的无限horizon MDP中，这保证最优策略不变。

但在GRPO的episodic agent设定中：
- $\gamma = 1$
- GRPO使用trajectory-level total reward
- 差分形式 telescope为 $\Phi(s_T) - \Phi(s_0)$，丢失中间信息

因此DUET**有意选择**了非差分形式 $P(\tau) = \frac{1}{T}\sum_t \Phi(s_t)$，牺牲了严格的最优策略不变性保证，换取了更丰富的信号。

#### 偏差的可控性

由命题2，偏差大小由 $\beta$ 控制：
- $\beta \to 0$：无偏差，但也无进度信号（退化为标准GRPO）
- $\beta$ 适中：有偏差但有界，进度信号有效
- $\beta \to \infty$：进度信号主导，terminal reward被淹没

实践中，$\beta$ 可以通过验证集上的task success rate来调整，或使用动态衰减：

$$\beta_t = \beta_0 \cdot \max(0, 1 - \frac{\text{mean\_reward}_t}{\text{target\_reward}})$$

当on-policy的mean reward上升（sparsity缓解），$\beta$ 自动下降。

---

### 4.6 Step-Level扩展的数学基础

当使用step-level advantage decomposition：

$$A_{\text{final}}(i, t) = A_i' + \eta \cdot [\Phi(s_{t+1}^i) - \Phi(s_t^i)]$$

可以理解为将 $\Phi$ 作为**近似state-value function** $\hat{V}(s) \approx \beta \Phi(s)$，然后做类似TD(0)的advantage estimation：

$$\hat{A}(s_t, a_t) \approx r_t + \hat{V}(s_{t+1}) - \hat{V}(s_t) = r_t + \beta[\Phi(s_{t+1}) - \Phi(s_t)]$$

在terminal reward全为0时：
$$\hat{A}(s_t, a_t) \approx \beta[\Phi(s_{t+1}) - \Phi(s_t)]$$

这给出了有意义的step-level信号，且不依赖于学习一个value network。

---

## 5. 与现有方法的系统性对比

### 5.1 总览表

| 维度 | GRPO | LUFFY | TRAPO | R³ | GiGPO | **Action Channel** | **State Channel** | **DUET (Ours)** |
|---|---|---|---|---|---|---|---|---|
| **利用expert action** | ✗ | ✓ | ✓ | ✓ | ✗ | **✓** | ✗ | **✓** |
| **利用expert state** | ✗ | ✗ | ✗ | ✓(部分) | ✗ | ✗ | **✓** | **✓** |
| **分布修正** | — | 否(需要但未做) | 否(SFT) | 否 | — | **判别器** | **不需要** | **判别器(仅teacher)** |
| **需相同tokenizer** | — | 否 | 实际上是 | 否 | — | **否** | **否** | **否** |
| **需expert logprob** | — | 否 | 否 | 否 | — | **否** | **否** | **否** |
| **解决sparsity** | ✗ | 微弱 | ✓(prefix) | ✓(curriculum) | 部分 | **✗** | **✓** | **✓** |
| **改rollout pipeline** | — | ✓ | ✓ | ✓ | ✗ | **✓(混入)** | **✗** | **✓(混入)** |
| **额外模型** | — | ✗ | ✗ | ✗ | ✗ | **轻量MLP** | **✗** | **轻量MLP** |
| **适用多轮agent** | ✓ | ✓ | ✗ | 需env支持 | ✓ | **✓** | **✓** | **✓** |

**关键对比**：没有任何现有方法同时利用expert的action和state两类信息。LUFFY/CHORD只用action信息但无适当修正；R³部分使用state信息但通过curriculum而非reward shaping。我们的DUET框架是首个**完整利用expert trajectory全部可用信息**的方法。

### 5.2 逐一详细对比

#### vs. R³（ICML 2024）

| 维度 | R³ | DUET |
|---|---|---|
| 核心机制 | 改变rollout起始状态（从expert中间状态启动） | 改变reward信号（从expert状态序列构造进度度量） |
| 是否修改rollout过程 | **是** — agent从expert的中间状态开始生成 | **否** — 所有rollout从初始状态正常开始 |
| 是否需要环境重置到中间状态 | **是** — 环境需支持从任意状态初始化 | **否** — 只需比较observation text |
| 课程机制 | 手动设计的反向课程（从终点到起点分M阶段） | 自动产生的课程效果（$\sigma_P$ 随训练自然收敛） |
| 训练范式 | PPO | GRPO（直接兼容） |
| 分布修正需求 | 不需要（从中间状态on-policy生成） | 不需要（进度函数是状态函数，与策略无关） |

**本质区别**：R³改变了"agent从哪里出发"（rollout-level）。DUET的State Channel改变了"如何评价agent的表现"（reward-level）。机制完全不同。

#### vs. TRAPO

| 维度 | TRAPO | DUET |
|---|---|---|
| 核心机制 | Expert text作为token前缀拼接到prompt，SFT on prefix + RL on completion | Expert状态序列构造进度函数，shaped reward |
| 是否需要相同tokenizer | 实际上是（前缀拼接需要共享token空间） | **否**（只比较observation text） |
| SFT成分 | 有（TrSFT loss on prefix） | **无**（纯RL） |
| 适用场景 | 单轮推理（数学题） | **多轮agent交互** |
| 信息利用 | Expert的token序列（action-level） | Expert的state序列（state-level） |

**本质区别**：TRAPO把expert当作"生成内容的模板"。DUET的State Channel把expert当作"导航地图"。

#### vs. GiGPO（NeurIPS 2025）

| 维度 | GiGPO | DUET |
|---|---|---|
| 是否使用expert数据 | **否** | **是**（state序列） |
| 如何产生dense signal | Step-level credit assignment（anchor state grouping） | Progress-based reward shaping |
| 极端sparsity下表现 | 所有action都失败时，step-level advantage仍无法区分 | 不同轨迹进度不同，advantage方差 > 0 |
| 额外计算开销 | ~0（利用已有rollout中的重复状态） | ~0（hash查找） |
| 正交性 | — | **可与GiGPO组合使用** |

**本质区别**：GiGPO在"没有外部信息"的前提下做step-level credit。DUET的State Channel引入"外部状态信息"来打破sparsity死锁。两者正交互补。

#### vs. LUFFY（NeurIPS 2025）

| 维度 | LUFFY（仅action mixing） | **Ours (DUET)** |
|---|---|---|
| Expert的角色 | 训练数据（混入GRPO batch） | **双重角色**：action为训练数据 + state为信息源 |
| 利用的信息类型 | 仅Action-level | **Action + State双通道** |
| 分布修正 | 否（直接混入，存在off-policy bias） | **Action Channel判别器修正** |
| Reward sparsity | 微弱（仅多1条好轨迹的1-bit信号） | **State Channel: 每条轨迹有不同进度分** |
| 额外组件 | 无 | 轻量MLP + Hash table |

**本质区别**：LUFFY只利用action信息且不做分布修正。我们的框架通过信息分解，用最适合的方法分别处理两类信息——Action Channel修正action-level的分布偏差，State Channel从state-level构造进度信号解决sparsity。

---

## 6. 工程实现方案

### 6.0 代码库关键数据流梳理（基于codebase审查）

在给出实现方案前，先梳理实际代码库中与State Channel相关的数据流，确保设计与代码对齐：

**Rollout 数据流**：
```
ae_ray_trainer.fit()
  → env_manager.rollout(tasks, ...) 
    → env_worker.execute()
      → agent_flow.execute(context_manager=cmt, env=env, ...)
        → 循环: llm_chat_fn → env.step → cmt.save_env_output(state)
        → cmt.reward = Reward(outcome=score, ...)
  → return cmt_array (List[Linear_CMT])
```

**Reward → Advantage 数据流（ae_ray_trainer.fit()中）**：
```
trajectories = env_manager.rollout(...)           # List[Linear_CMT]
gen_batch_output = env_manager.to_dataproto(...)  # DataProto (含 reward_scores)
batch = union_gen_batch_via_task_id(...)
reward_tensor = parse_reward_from_dataproto(batch) # (bs, response_len), 标量放在最后一个有效token
batch.batch["token_level_scores"] = reward_tensor
# ... (DAPO overlong shaping, KL penalty 等)
batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
batch = compute_advantage(batch, ...)              # GRPO advantage计算
```

**关键数据结构**：
- **Linear_CMT (trajectories的元素)**：`full_context` 是 `List[ExtendedMessage]`，每个包含 `role`、`content`。环境observation在 `role="user"` 的消息中。
- **Teacher Trajectory (exp_manager中)**：`Trajectory.steps` 是 `List[dict]`，每个dict含 `role` 和 `content`。存储在 `exp_manager.teacher_task2trajectories: Dict[str, List[Trajectory]]`。
- **Reward**：`cmt.reward.outcome` 是标量（0或1）。在 `parse_reward_from_dataproto()` 中被放到 `reward_tensor[i, response_lengths[i]-1]`。

**⚠️ 关键发现：环境observation包含动态后缀**

审查 `alfworld_env.py` 发现：
```python
self.current_observation = reset_result["observation"] + "\nAVAILABLE ACTIONS: " + ", ".join(self.current_available_actions)
```

WebShop同理：
```python
content = f"{self.current_observation}\n\n{action_desc}"
```

即 **observation text 包含了 "AVAILABLE ACTIONS: ..." 后缀**，这些后缀在相同底层状态下也可能不同（因为可用动作会变化）。直接hash整个observation会导致匹配率过低。**需要在hash前做normalization（剥离动态后缀）。**

### 6.1 基于当前代码库的改动清单

基于 `agentevolver` 代码库的实际数据结构，DUET的实现改动极小：

#### 新增模块

**`agentevolver/module/exp_manager/state_progress.py`**（约180-220行）

```python
import re
from typing import Dict, List, Optional, Tuple
from loguru import logger


def normalize_observation(obs_text: str, env_type: str = "alfworld") -> str:
    """
    剥离 observation 中的动态后缀（AVAILABLE ACTIONS 等），保留核心状态描述。
    
    ⚠️ 关键：alfworld_env.py 在 observation 后拼接了 "\\nAVAILABLE ACTIONS: ..."，
    webshop_env.py 在 observation 后拼接了 "\\n\\n" + action_desc。
    这些后缀在相同底层状态下可能不同（可用动作变化），必须剥离后再 hash。
    
    Args:
        obs_text: 原始 observation text（来自 CMT 的 user message content）
        env_type: 环境类型（"alfworld" 或 "webshop"）
    
    Returns:
        归一化后的状态字符串
    """
    if not obs_text:
        return ""
    
    text = obs_text.strip()
    
    if env_type == "alfworld":
        # 剥离 "\nAVAILABLE ACTIONS: ..." 后缀
        idx = text.find("\nAVAILABLE ACTIONS:")
        if idx != -1:
            text = text[:idx]
    elif env_type == "webshop":
        # 剥离 "\n\nYou can use: search[...]" 和 "Clickable elements: [...]" 后缀
        # WebShop 的 action_desc 由 _format_available_actions() 生成
        patterns = [
            r'\n\nYou can use:.*$',
            r'\n\nClickable elements:.*$',
            r'\nClickable elements:.*$',
        ]
        for pat in patterns:
            text = re.sub(pat, '', text, flags=re.DOTALL)
    
    return text.strip()


def extract_observations_from_steps(
    steps: List[dict],
    env_type: str = "alfworld",
    skip_initial: int = 3,
) -> List[str]:
    """
    从 trajectory.steps（List[dict with role, content]）中提取环境 observation 序列。
    
    ⚠️ 代码库对齐：
    - Teacher trajectory: Trajectory.steps 是 List[dict]，role 为 "system"/"user"/"assistant"
    - 前 skip_initial 条消息是 system prompt + initial assistant ack + initial user query
    - 后续的 "user" role 消息是环境 observation（agent_flow.execute() 中 env.step 返回的 state）
    
    Args:
        steps: 轨迹的 message 列表
        env_type: 环境类型
        skip_initial: 跳过前 N 条初始化消息（默认3：system + assistant_ack + user_query）
    
    Returns:
        归一化后的 observation 列表（按时间顺序）
    """
    observations = []
    for i, msg in enumerate(steps):
        if i < skip_initial:
            continue
        if msg.get("role") == "user":
            raw_obs = msg.get("content", "")
            normalized = normalize_observation(raw_obs, env_type)
            if normalized:
                observations.append(normalized)
    return observations


def extract_observations_from_cmt(
    cmt,
    env_type: str = "alfworld",
    skip_initial: int = 3,
) -> List[str]:
    """
    从 Linear_CMT 对象中提取 observation 序列。
    
    ⚠️ 代码库对齐：
    - Linear_CMT.full_context 是 List[ExtendedMessage]
    - ExtendedMessage 有 .role 和 .content 属性（或通过 author 字段判断）
    - agent_flow.execute() 中，env output 通过 cmt.save_env_output(state) 存入，
      state 的 role 是 "user"（经 convert_tool_to_user_message 转换）
    
    Args:
        cmt: Linear_CMT 对象（rollout 产生的轨迹）
        env_type: 环境类型
        skip_initial: 跳过前 N 条初始化消息
    
    Returns:
        归一化后的 observation 列表
    """
    observations = []
    
    # Linear_CMT 有 full_context (List[ExtendedMessage]) 或可通过 steps 属性访问
    if hasattr(cmt, 'full_context'):
        context = cmt.full_context
        for i, ext_msg in enumerate(context):
            if i < skip_initial:
                continue
            role = getattr(ext_msg, 'role', None) or getattr(ext_msg, 'author', 'unknown')
            content = getattr(ext_msg, 'content', '')
            if role == "user":
                normalized = normalize_observation(content, env_type)
                if normalized:
                    observations.append(normalized)
    elif hasattr(cmt, 'steps') and isinstance(cmt.steps, list):
        # Fallback: 使用 steps（与 Trajectory 格式兼容）
        observations = extract_observations_from_steps(cmt.steps, env_type, skip_initial)
    
    return observations


class ExpertProgressMap:
    """
    从 expert trajectory 的 state 序列构建进度映射。
    
    ⚠️ 代码库对齐：
    - Expert trajectories 来自 exp_manager.teacher_task2trajectories: Dict[str, List[Trajectory]]
    - 每个 Trajectory 的 steps 是 List[dict]，包含 role="user" 的 environment observations
    - 构造时从 teacher_task2trajectories 读取，运行时对 on-policy CMT 做 hash 查找
    """
    
    def __init__(
        self,
        teacher_task2trajectories: Dict[str, list],
        env_type: str = "alfworld",
        match_mode: str = "hash",
        embed_model=None,
        similarity_threshold: float = 0.85,
    ):
        """
        Args:
            teacher_task2trajectories: exp_manager.teacher_task2trajectories 
                格式: Dict[task_id -> List[Trajectory]]，每个 Trajectory 有 .steps
            env_type: "alfworld" 或 "webshop"
            match_mode: "hash" (精确匹配) 或 "embedding" (模糊匹配)
        """
        self.env_type = env_type
        self.match_mode = match_mode
        self.embed_model = embed_model
        self.similarity_threshold = similarity_threshold
        self.progress_maps: Dict[str, Dict[str, float]] = {}
        
        # 统计信息
        total_states = 0
        total_tasks = 0
        
        for task_id, trajectories in teacher_task2trajectories.items():
            progress_map: Dict[str, float] = {}
            for traj in trajectories:
                # 从 Trajectory.steps 中提取 observation 序列
                steps = traj.steps if hasattr(traj, 'steps') else []
                obs_list = extract_observations_from_steps(steps, env_type)
                
                T = len(obs_list)
                if T == 0:
                    continue
                
                for j, obs in enumerate(obs_list):
                    key = self._state_key(obs)
                    progress = j / max(T - 1, 1)  # j/(T-1) 使得最后一个 obs 的进度为 1.0
                    progress_map[key] = max(
                        progress_map.get(key, 0.0), progress
                    )
                total_states += len(obs_list)
            
            if progress_map:
                self.progress_maps[task_id] = progress_map
                total_tasks += 1
        
        logger.info(
            f"[State Channel] Built ProgressMap: {total_tasks} tasks, "
            f"{total_states} total expert observations, "
            f"{sum(len(m) for m in self.progress_maps.values())} unique state keys"
        )
    
    def _state_key(self, observation: str) -> str:
        """将 observation 转为可 hash 的 key"""
        if self.match_mode == "hash":
            return observation  # 使用归一化后的 observation 本身作为 key（str 可 hash）
        else:
            # embedding 模式：后续扩展
            raise NotImplementedError("Embedding match mode not yet implemented")
    
    def get_potential(self, task_id: str, observation: str) -> float:
        """Φ(s): 返回状态 s 在 expert 轨迹中的进度值"""
        progress_map = self.progress_maps.get(task_id, {})
        if not progress_map:
            return 0.0
        key = self._state_key(observation)
        return progress_map.get(key, 0.0)
    
    def compute_trajectory_progress(self, task_id: str, observations: List[str]) -> float:
        """P(τ): 计算轨迹的平均进度"""
        if not observations:
            return 0.0
        total = sum(
            self.get_potential(task_id, obs) 
            for obs in observations
        )
        return total / len(observations)
    
    def compute_shaped_reward(
        self, task_id: str, observations: List[str],
        terminal_reward: float, beta: float = 0.5
    ) -> float:
        """R'(τ) = R(τ) + β·P(τ)"""
        progress = self.compute_trajectory_progress(task_id, observations)
        return terminal_reward + beta * progress
    
    def compute_step_deltas(
        self, task_id: str, observations: List[str]
    ) -> Tuple[List[float], List[float]]:
        """计算每步的进度变化 Φ(s_{t+1}) - Φ(s_t)"""
        potentials = [
            self.get_potential(task_id, obs) 
            for obs in observations
        ]
        deltas = [
            potentials[t+1] - potentials[t] 
            for t in range(len(potentials) - 1)
        ]
        return deltas, potentials
    
    def get_coverage_stats(
        self, task_id: str, observations: List[str]
    ) -> Dict[str, float]:
        """计算 on-policy 轨迹与 expert 状态的覆盖率统计（用于分析图表）"""
        if not observations:
            return {"coverage": 0.0, "matched": 0, "total": 0}
        matched = sum(
            1 for obs in observations
            if self.get_potential(task_id, obs) > 0.0
        )
        return {
            "coverage": matched / len(observations),
            "matched": matched,
            "total": len(observations),
        }
```

#### 修改 `AERayTrainer.fit()`

**精确注入点**：在 `ae_ray_trainer.py` 的 `fit()` 方法中，位于 `reward_tensor = self.reward_fn(batch)` 之后、`compute_advantage(batch, ...)` 之前的 reward 处理区域。具体在 `batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]` 之前。

```python
# ============================================================================
# ⭐ State Channel: Trajectory Progress Reward Shaping
# 在 reward_tensor 赋值后、advantage 计算前注入 shaped reward
# ============================================================================
tprs_config = self.config.exp_manager.get("tprs", {})
use_tprs = tprs_config.get("enable", False)

if use_tprs and trajectories:
    from agentevolver.module.exp_manager.state_progress import (
        ExpertProgressMap, extract_observations_from_cmt
    )
    
    # 懒初始化 ProgressMap（只在第一次调用时构建）
    if not hasattr(self, '_tprs_progress_map'):
        self._tprs_progress_map = ExpertProgressMap(
            teacher_task2trajectories=self.exp_manager.teacher_task2trajectories,
            env_type=str(self.config.env_service.env_type),
            match_mode=tprs_config.get("match_mode", "hash"),
        )
    
    tprs_beta = float(tprs_config.get("beta", 0.5))
    
    # 可选：动态 β 衰减
    if tprs_config.get("beta_decay", False):
        current_mean_reward = reward_tensor.sum(dim=-1).mean().item()
        target = float(tprs_config.get("beta_decay_target", 0.5))
        decay_factor = max(0.0, 1.0 - current_mean_reward / target)
        tprs_beta = tprs_beta * decay_factor
    
    # 为每条 on-policy 轨迹计算 P(τ) 并注入 shaped reward
    env_type = str(self.config.env_service.env_type)
    tprs_progress_values = []
    tprs_coverage_values = []
    
    for traj_idx, traj in enumerate(trajectories):
        if traj_idx >= reward_tensor.shape[0]:
            break
        
        task_id = traj.task_id if hasattr(traj, 'task_id') else ""
        
        # 从 CMT 对象提取归一化的 observation 序列
        obs_list = extract_observations_from_cmt(traj, env_type)
        
        # 计算 P(τ)
        P_i = self._tprs_progress_map.compute_trajectory_progress(task_id, obs_list)
        tprs_progress_values.append(P_i)
        
        # 覆盖率统计
        stats = self._tprs_progress_map.get_coverage_stats(task_id, obs_list)
        tprs_coverage_values.append(stats["coverage"])
        
        # 注入 shaped reward：找到 reward 所在位置，加上 β·P(τ)
        # reward_tensor[traj_idx] 中非零位置即为 reward 所在的 token
        nonzero_pos = (reward_tensor[traj_idx] != 0).nonzero(as_tuple=True)[0]
        if len(nonzero_pos) > 0:
            pos = nonzero_pos[-1]  # 最后一个非零位置
            reward_tensor[traj_idx, pos] += tprs_beta * P_i
        else:
            # 原始 reward 为 0，在最后一个有效 response token 放置 shaped reward
            response_mask_i = batch.batch["attention_mask"][traj_idx, -reward_tensor.shape[1]:]
            last_valid = response_mask_i.nonzero(as_tuple=True)[0]
            if len(last_valid) > 0:
                reward_tensor[traj_idx, last_valid[-1]] = tprs_beta * P_i
    
    # 更新 reward_tensor（已就地修改，但确保一致性）
    batch.batch["token_level_scores"] = reward_tensor
    
    # State Channel 诊断指标
    import numpy as _np
    if tprs_progress_values:
        metrics.update({
            "tprs/beta": tprs_beta,
            "tprs/progress_mean": float(_np.mean(tprs_progress_values)),
            "tprs/progress_std": float(_np.std(tprs_progress_values)),
            "tprs/progress_min": float(_np.min(tprs_progress_values)),
            "tprs/progress_max": float(_np.max(tprs_progress_values)),
            "tprs/progress_nonzero_ratio": float(
                sum(1 for p in tprs_progress_values if p > 0) / len(tprs_progress_values)
            ),
            "tprs/coverage_mean": float(_np.mean(tprs_coverage_values)),
            "tprs/coverage_std": float(_np.std(tprs_coverage_values)),
        })
```

**初始化**（在 `init_workers()` 中，`self.exp_manager = ExperienceManager(config=self.config)` 之后）：

```python
# ⭐ State Channel: 确保 teacher trajectories 已加载（State Channel 依赖 teacher state 序列）
tprs_config = self.config.exp_manager.get("tprs", {})
if tprs_config.get("enable", False) and not self.exp_manager.teacher_enabled:
    logger.warning(
        "[State Channel] State Channel is enabled but teacher_experience is not loaded. "
        "State Channel requires teacher trajectories for progress map construction. "
        "Please set exp_manager.teacher_experience.enable=true and provide data_path."
    )
```

### 6.2 不变的部分（经代码库审查确认）

| 组件 | 代码位置 | 是否改动 |
|---|---|---|
| Rollout生成 | `agent_flow.execute()` | ✗ 完全不变 |
| 环境交互 | `env_worker.execute()` + `env.step()` | ✗ 完全不变 |
| 轨迹→DataProto转换 | `env_manager.to_dataproto()` + `samples_to_dataproto()` | ✗ 不变 |
| GRPO advantage公式 | `compute_grpo_outcome_advantage()` | ✗ 不变（输入 reward 值改变） |
| Token-level PG | `het_core_algos.py` 中所有 loss 函数 | ✗ 不变 |
| 多卡同步 | `het_actor.py` 中的 FSDP/gradient sync | ✗ 不变 |
| Micro-batch逻辑 | `update_policy()` 中的 mini/micro batch 切分 | ✗ 不变 |
| LUFFY mixing | `LUFFYTeacherRolloutMixer.mix_trajectories()` | ✗ 不变（但 State Channel 可完全替代） |
| Action Channel 判别器 | `dr3_ratio.py` + `het_actor.py` 中的 Action Channel 路径 | ✗ 不变（但 State Channel 可完全替代） |

### 6.3 可以移除的部分（如果只使用State Channel，不启用Action Channel）

以下组件在纯DUET配置下**全部无需启用**（但建议保留代码以做消融对比实验A4）：

- `dr3_ratio.py` 中的 `DR3RatioEstimator` 和 `DR3Discriminator`（~500行）
- `het_actor.py` 中的 Action Channel observe/apply 路径（~200行）
- `experience_collate.py` 中的 `LUFFYTeacherRolloutMixer`（~300行）
- `het_core_algos.py` 中的 `het_compute_teacher_aware_loss` 的 teacher 分支
- `ae_ray_trainer.py` 中的 LUFFY rollout-level mixing 逻辑
- `env_manager.py` 中的 `_align_teacher_log_probs`（teacher log_prob 对齐）
- 所有 `teacher_loss_scale`、reward-gap gate、gap-beta scheduler 逻辑

**配置差异**：启用 State Channel 时只需：
```yaml
exp_manager:
  teacher_experience:
    enable: true        # 需要加载 teacher trajectories（但仅用于构建 ProgressMap）
    data_path: "..."    # teacher trajectories 文件路径
  tprs:
    enable: true
    beta: 0.5
    # ... 其他 State Channel 参数
```

不需要设置 `use_dr3`、`teacher_mix_mode`、`n_teacher_rollouts_per_task` 等。

### 6.4 计算开销分析

| 组件 | 开销 |
|---|---|
| 离线：构建ProgressMap（含 observation normalize） | 一次性，O(Σ T_E) across all tasks，约几秒 |
| 在线：每条轨迹提取 observations | O(T) per trajectory，string 操作 |
| 在线：每条轨迹的 hash 查找 | O(T) per trajectory，dict 查找 O(1) each |
| 在线：修改 reward_tensor | O(1) per trajectory |
| 总额外开销 | **< 0.1%** 相对于 rollout 生成的 LLM 推理 |

对比 Action Channel：Action Channel 需要判别器前向/反向传播、rolling buffer 维护、多卡同步，开销约 1-3%。State Channel 比 Action Channel 轻量约一个数量级。

### 6.5 配置文件示例

```yaml
# ═══ DUET完整配置（Action Channel + State Channel）═══
# config/paper_alfworld_duet.yaml
exp_manager:
  teacher_experience:
    enable: true
    data_path: "data/teacher_trajectories/alfworld_teacher.pkl"
    mix_mode: "rollout_level"        # LUFFY-style mixing（Action Channel需要）
    n_teacher_rollouts_per_task: 1
  use_dr3: true                      # Action Channel: 密度比修正
  dr3:
    enable: true
    feature_mode: "v3_aug"
    use_relative_ratio: true
    alpha_mode: "sync_batch_ema"
    ess_kappa: 0.5
    apply_warmup_steps: 5
  tprs:                              # State Channel: 进度奖励塑形
    enable: true
    beta: 0.5
    beta_decay: true
    beta_decay_target: 0.5
    match_mode: "hash"
    step_level: false

# ═══ State Channel Only配置（纯on-policy + 进度塑形）═══
# config/paper_alfworld_state_only.yaml
exp_manager:
  teacher_experience:
    enable: true
    data_path: "data/teacher_trajectories/alfworld_teacher.pkl"
    # 以下 LUFFY 相关配置在纯 State Channel 模式下不需要
    # mix_mode: "rollout_level"
    # n_teacher_rollouts_per_task: 1
  tprs:
    enable: true
    beta: 0.5                    # 进度奖励系数
    beta_decay: true             # 是否随训练衰减
    beta_decay_target: 0.5       # 当mean_reward达到target时beta→0
    match_mode: "hash"           # "hash" 或 "embedding"
    step_level: false            # 是否启用step-level advantage
    eta: 0.1                     # step-level系数（若启用）

# config/paper_webshop_tprs.yaml
exp_manager:
  teacher_experience:
    enable: true
    data_path: "data/teacher_trajectories/webshop_teacher.pkl"
  tprs:
    enable: true
    beta: 0.5
    match_mode: "hash"           # WebShop 也先用 hash，通过 normalize_observation 处理

# 消融配置
# config/paper_alfworld_tprs_no_step.yaml    → tprs.step_level: false
# config/paper_alfworld_tprs_high_beta.yaml  → tprs.beta: 1.0
# config/paper_alfworld_tprs_low_beta.yaml   → tprs.beta: 0.1
# config/paper_alfworld_tprs_no_decay.yaml   → tprs.beta_decay: false
# config/paper_alfworld_tprs_plus_luffy.yaml → tprs.enable: true + teacher_experience.mix_mode: "rollout_level"
# config/paper_webshop_tprs_embed.yaml       → tprs.match_mode: "embedding"
```

---

## 7. 实验设计

### 7.1 实验环境

| 环境 | 特点 | Reward Sparsity程度 | 状态匹配方式 |
|---|---|---|---|
| **ALFWorld** | 家庭任务模拟，10-20步，中等难度 | 中等（on-policy成功率~10%） | 房间+物品状态hash |
| **WebShop** | 网购任务模拟，5-15步，中等偏难 | 中高（部分奖励可获得，高分稀疏） | 页面URL+内容hash |

**环境特点分析**：

- **ALFWorld**：提供结构化的文本环境描述（房间、物品、物品状态），状态可以通过hash精确匹配。任务类型包括pick & place、clean、heat、cool等6类家庭任务。Reward sparsity程度中等——on-policy模型有一定概率完成简单任务，但复杂任务（如multi-step clean+place）成功率很低。
- **WebShop**：半结构化环境，agent浏览商品页面、搜索、点击按钮完成购物任务。状态由当前页面内容定义。Reward为[0,1]连续值（基于所购商品与目标的匹配度），但高reward（>0.5）稀疏。状态匹配可基于页面URL hash或页面内容embedding。

### 7.2 基线方法

| 基线 | 描述 |
|---|---|
| **GRPO** | 标准on-policy GRPO，无expert数据 |
| **LUFFY** | Expert trajectory混入GRPO batch（无分布修正） |
| **GiGPO** | Step-level anchor state credit（无expert数据） |
| **CHORD** | Expert data作为加权SFT + on-policy RL组合 |

### 7.3 主实验

**Ours (DUET)** vs. 所有基线的task success rate，across 2个环境（ALFWorld、WebShop） × 2-3个model scale。同时报告 Action Channel only (Action Channel) 和 State Channel only (State Channel) 作为方法分解。

### 7.4 消融实验

| 编号 | 配置 | 验证点 |
|---|---|---|
| **核心消融：A vs B vs DUET** | | |
| C1 | **DUET (full)** — Action Channel + State Channel | 完整方法（推荐配置） |
| C2 | **Action Channel only** — Action Channel（teacher mixing + 密度比修正） | Action Channel独立贡献 |
| C3 | **State Channel only** — State Channel（纯on-policy + 进度shaped reward） | State Channel独立贡献 |
| C4 | **LUFFY + State Channel** — teacher mixing无Action Channel修正 + State Channel | 验证Action Channel修正的额外价值 |
| **State Channel组件消融** | | |
| B1 | 仅shaped reward，无step-level | trajectory-level P(τ)的独立价值 |
| B2 | 仅step-level，无shaped reward | step-level signal的独立价值 |
| B3 | 仅终态进度 $\Phi(s_T)$ 替代 $P(\tau)$ | 累积进度 vs 终态进度（命题1实证） |
| B4 | 差分形式 $\Phi(s_T)-\Phi(s_0)$ 替代 $P(\tau)$ | telescope问题的实际影响 |
| B5 | 不同 $\beta$：{0.1, 0.3, 0.5, 1.0, 2.0} | 偏差-方差权衡（命题2实证） |
| B6 | $\beta$ 动态衰减 vs 固定 | 自然课程效应（命题4） |
| B7 | 随机progress map | expert state信息的必要性 |
| **Action Channel组件消融** | | |
| A1 | Action Channel without ESS-dual（固定clip） | ESS自适应clipping的贡献 |
| A2 | Action Channel without reward-gap gate | teacher fade-out的贡献 |
| A3 | Action Channel with direct p/q ratio（非relative ratio） | relative ratio的方差控制效果 |
| **正交性验证** | | |
| O1 | State Channel + GiGPO | 与step-level credit方法的正交互补性 |

### 7.5 关键分析图表

| 图表 | 内容 | 证明的论点 |
|---|---|---|
| **图1（最重要）** | Advantage方差 vs 训练步数（DUET vs GRPO vs LUFFY vs State Channel only） | 命题1：State Channel在sparsity下保持非零方差 |
| 图2 | DUET vs Action Channel vs State Channel的training curve对比 | DUET组合的互补效果 |
| 图3 | Shaped reward分布（同一batch内不同轨迹的R'对比） | 信号丰富度 |
| 图4 | $P(\tau)$ 分布随训练推进的变化 | 命题4：自然课程效应 |
| 图5 | Task success rate vs $\beta$ | 命题2：偏差-方差权衡 |
| 图6 | Action Channel的ESS和密度比分布随训练变化 | 命题8：ESS自适应控制 |
| 图7 | 进度函数 $\Phi$ 在ALFWorld状态空间上的可视化 | "导航地图"直觉 |
| 图8 | Anchor point覆盖率 | 进度信号密度 |
| 表1 | 主实验结果（2环境 × DUET vs baselines） | 方法有效性 |
| 表2 | 消融实验结果 | 各组件贡献 |

---

## 8. 论文叙事与结构建议

### 8.1 推荐标题方向

**主标题候选**：

1. *DUET: DUal Expert Trajectory Utilization via Information Decomposition for Agent RLVR*
2. *DUET: Decomposing Expert Trajectories into Action Correction and State Shaping for LLM Agent Training*
3. *Beyond Action Imitation: DUET — A Dual-Channel Framework for Expert Trajectory Utilization in Agent RLVR*

### 8.2 论文结构

**Abstract** (~200 words)

Agent RLVR with GRPO suffers from two intertwined challenges when leveraging black-box expert trajectories: distribution correction for off-policy action data and reward sparsity in multi-turn environments. Existing methods address these separately and incompletely. We observe that expert trajectories contain two fundamentally different types of information — action-level (policy-dependent, requiring distribution correction) and state-level (policy-free, correction-free). Based on this decomposition, we propose DUET (DUal Expert Trajectory utilization), a framework with two orthogonal channels: the **Action Channel** estimates trajectory-level density ratios via a lightweight discriminator to correct off-policy bias when mixing expert actions into GRPO batches, while the **State Channel** extracts expert state sequences to construct a progress function for reward shaping. We prove that the State Channel guarantees non-degenerate advantage under complete reward sparsity, the Action Channel provides correct importance weighting without expert log-probabilities, and the two channels are formally orthogonal. Experiments on ALFWorld and WebShop demonstrate that DUET significantly outperforms GRPO, LUFFY, GiGPO, and CHORD baselines, with ablations confirming the complementary value of both channels.

**§1 Introduction** (1.5 pages)
- Agent RLVR的两个核心挑战：分布修正 + 奖励稀疏
- 现有方法的局限（LUFFY无修正、CHORD用SFT回避、TRAPO需同tokenizer）
- 核心洞察：expert trajectory的信息分解（action-level vs state-level）
- 贡献列表（框架 + 两个方法 + 理论 + 实验）

**§2 Background** (1 page)
- GRPO公式
- Agent RLVR设定
- 黑盒expert约束

**§3 Method** (3.5 pages)
- §3.1 信息分解原理（核心framework）
- §3.2 Action Channel — 判别器密度比估计、old_log_prob修复、ESS-dual clipping
- §3.3 State Channel — 进度函数Φ、轨迹进度度量P(τ)、shaped reward
- §3.4 DUET组合：数据流、正交性、完整算法
- §3.5 讨论：与potential-based shaping的关系

**§4 Theoretical Analysis** (1.5 pages)
- State Channel: 命题1-5（非退化性、偏差有界性、方向一致性、课程效应、无需分布修正）
- Action Channel: 命题6-8（判别器最优解、old_log_prob修复正确性、ESS方差控制）
- DUET: 命题9-10（正交性、组合优势条件）

**§5 Experiments** (2.5 pages)
- 主实验：DUET vs baselines（2环境 × 2+ model scale）
- 消融：Action Channel only / State Channel only / DUET 对比
- 关键分析：advantage方差、进度分布演化、β敏感性

**§6 Related Work** (0.5 page)
- Off-policy methods: LUFFY, CHORD, OAPL
- Curriculum/prefix: R³, TRAPO
- Credit assignment: GiGPO
- Reward shaping: Ng et al., TIPS
- State-only imitation: POSG, Rank2Reward, SLOPE

**§7 Conclusion** (0.5 page)

### 8.3 核心贡献总结（供rebuttal参考）

1. **框架贡献**：首次提出将expert trajectory中的信息分解为action-level（策略相关，需要修正）和state-level（策略无关，无需修正），并据此设计两条正交的利用路径。这一分解为"如何利用黑盒expert数据"提供了系统性的思考框架。

2. **方法贡献 A (Action Channel)**：面向黑盒expert（无logprob、不同tokenizer）的判别器密度比修复层。通过序列级特征 + 轻量MLP判别器估计on-policy vs teacher的密度比，修复 `old_log_prob` 后复用标准PPO/RePO token loss。配合ESS-dual clipping和reward-gap gate实现自适应方差控制。

3. **方法贡献 B (State Channel)**：从expert状态序列构造进度函数，作为reward shaping注入GRPO。实现极其简洁（~150行核心代码），无需判别器、无需分布修正、无需修改rollout pipeline。

4. **理论贡献**：证明了State Channel的advantage非退化性、偏差有界性、方向一致性和自然课程效应；证明了Action Channel old_log_prob修复的正确性；证明了两条通道的形式正交性。共10个命题。

5. **实证贡献**：在ALFWorld和WebShop上验证框架有效性，DUET显著优于单一通道和所有baselines。消融实验清晰展示两条通道的独立价值和互补效果。

---

## 9. 风险评估与应对策略

### 9.1 潜在Reviewer质疑与应对

**Q1："这只是reward shaping，是well-studied的旧技术"**

**应对**：
- Reward shaping本身是旧技术，但**从黑盒expert trajectory的state序列构造势函数**是全新的应用方式
- 核心贡献不在于shaping机制本身，而在于**信息分解的洞察**和**state信息的独特利用方式**
- 类比：attention mechanism也是旧技术，但Transformer的贡献在于**如何应用**它

**Q2："进度函数的构造太简单了"**

**应对**：
- 简洁性是特征而非缺陷。Action Channel的判别器需要~500行代码 + 复杂的训练稳定化机制；State Channel的核心只需~50行
- 消融实验（A10）证明不是任意状态映射都有效——expert trajectory的结构是必要的
- 如果更复杂的构造（如embedding匹配、多条expert轨迹融合）能进一步提升效果，这说明方向正确且有扩展空间

**Q3："State匹配在复杂环境中可能不够精确"**

**应对**：
- ALFWorld提供结构化状态描述（房间、物品、物品属性），hash匹配足够精确
- WebShop的页面状态可基于URL hash或内容embedding匹配，提供两种变体
- 消融实验中报告不同匹配策略的对比（hash vs embedding）

**Q4："$\beta$ 的选择很sensitive"**

**应对**：
- 提供多个 $\beta$ 值的消融（A7），展示鲁棒范围
- 提供动态 $\beta$ 衰减方案及其效果对比（A8）
- 理论分析给出了 $\beta$ 的偏差上界（命题2），帮助指导选择

**Q5："为什么不直接结合TRAPO/R³/GiGPO？"**

**应对**：
- TRAPO需要相同tokenizer，不适用于黑盒expert
- R³需要环境支持从中间状态重置
- GiGPO与State Channel正交互补——消融A9展示组合效果
- State Channel的优势在于**约束条件最少**（仅需expert observation text）

**Q6："如果expert trajectory只有一条，且覆盖面不足怎么办？"**

**应对**：
- 多数task都有至少一条expert trajectory（与LUFFY/CHORD/TRAPO的数据假设相同）
- 报告anchor point覆盖率统计（图6）
- 当 $P(\tau_i) = 0$ $\forall i$（无匹配）时，State Channel退化为标准GRPO——不会harm

### 9.2 方法的局限性（论文中应诚实讨论）

1. **依赖expert trajectory的质量**：如果expert trajectory走了一条不必要的弯路，进度函数可能给出误导信号。多条expert trajectory的融合可以缓解。

2. **状态匹配的粒度问题**：过于粗糙的匹配可能给不相关的状态分配错误进度；过于精细的匹配可能导致覆盖率不足。需要环境特定的调整。

3. **理论保证的局限**：不保证最优策略完全不变（$\gamma=1$ 下非potential-based），只保证偏差有界且方向一致。

4. **单expert vs 多expert**：当前设计主要针对每个task一条expert trajectory。多条expert trajectory的最优融合策略是开放问题。

---

*文档结束*
