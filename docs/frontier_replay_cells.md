# 研究方向整理：如何把你的方法从“像 prefix guidance / replay 拼装”升级为一个更有新意的 ICML 级方案

> 文档性质：这是 FRC 的**研究定位 / 论文叙事笔记**，不是逐文件实现说明。
>
> 当前仓库里已经落地的是 `FRC-lite`：
>
> - 训练入口：`scripts/run_alfworld_frc_lite.sh`
> - 主配置：`config/paper_alfworld_frc_lite.yaml`
> - 设计与实现对齐说明：`docs/FRC_DESIGN.md`
> - 工程落地清单：`docs/FRC_LITE_IMPLEMENTATION_CHECKLIST.md`
>
> 为避免和当前实现混淆，阅读本文件时请注意：
>
> - 本文中关于 `Frontier Replay Cells` 的表述，主要服务于方法动机、论文定位与 reviewer 叙事；
> - 当前实现采用的是 memory-based `FRC-lite`，并带有明确的工程折中：
>   - frontier task 只在当前 mini-batch 内选择；
>   - 当前 on-policy trajectory 会被投影成 frontier-conditioned continuation，再与 replay continuation 一起训练；
>   - grouping 优先按 `cell_id`，必要时对 teacher-only unmatched cells 回退到 task-level group；
>   - `dr3_local` 当前是 lightweight local-ratio proxy，而不是单独训练的 local discriminator；
>   - `frontier_hash` 当前使用“observation 主体 + 最近 1-2 个 action”，不会把 `Thought` 直接纳入 hash，也还不是显式 object-state abstraction。

## 一、核心结论

基于目前对相关工作的梳理，我的判断是：

- 如果方法主线写成 **“expert prefix + suffix RL + token-level guidance”**，虽然很可能有效，但**很容易被审稿人质疑与已发表工作相似**。因为 2025–2026 已经出现了一簇非常接近的工作：BREAD、TRAPO、ICPO、Scaf-GRPO、DDE/DEEP-GRPO、CARL 等，它们共同关注 expert/prefix guidance、局部引导、learning cliff、关键状态或关键动作。
- 如果方法主线改写为 **“experience replay 的组织单位、修复目标和预算调度机制”**，尤其是把 replay 的对象从整条 trajectory 改成 **frontier-conditioned continuation cells**，那么新意会明显更强，也更符合你原本想坚持的 **experience replay + distribution repair + GRPO** 主线。

因此，最建议的方向不是继续强化“prefix guidance”，而是把方法中心改成：

> **Frontier Replay Cells for Sparse-Reward Agent RL**

也就是：

> 在 long-horizon、sparse-reward 的 agent RL 里，真正值得复用的经验单元，不是完整轨迹，而是“从某个 frontier 状态出发的一段 continuation 经验”。

---

## 二、为什么“expert prefix + suffix RL”容易被质疑撞车

### 1. 与 BREAD / TRAPO / ICPO 的相似性很高
这些工作已经把“从 expert prefix 出发，再让当前策略继续 rollout”的主线讲得很完整了。

- **BREAD**：在自采失败时插入一小段 expert prefix，再让模型从那里补全后续，并形成一种自然 curriculum。
- **TRAPO**：显式地让策略从 expert prefix 继续 rollout，而且 prefix 长度是自适应的，不是固定长度。
- **ICPO**：其最关键增益模块之一也是 expert-conditioned rollout generation。

所以如果你论文写成：

> “我们也用 expert 前缀把模型带到中间状态，然后让 GRPO 学 suffix。”

那么审稿人完全可能会说：

> 这属于已有的 prefix-conditioned continuation / expert-guided RL 范式，只是又加了一些新机制。

---

### 2. 与 Scaf-GRPO / DDE 这类“跨过 learning cliff”的方法也很近
另一类已发表工作不是直接做 prefix continuation，而是做最小必要指导、关键状态恢复、局部稠密重采样等，它们其实也在解决你 setup 里的核心困难：**奖励太稀疏，策略很难自己到达有学习信号的区域**。

所以如果你再强调：
- 局部 guidance
- sparse reward 下的辅助引导
- 关键步骤 supervision

这些点也会被认为只是落在同一大簇里。

---

### 3. “局部关键 token / action 更重要”这件事也开始拥挤
像 CARL 这一类工作已经开始强调：在 long-horizon agent RL 里，关键不是每一步都学，而是应该集中优化 critical actions。

因此，如果你把方法亮点写成：
- boundary-local guidance
- key token supervision
- local corrective training

这会有帮助，但不足以成为最核心的新意。

---

## 三、你真正应该占住的创新位

我认为你最应该占住的位置不是：

- “我们也做 prefix guidance”
- “我们也做 local hints”
- “我们也做稀疏奖励下的课程学习”

而是：

## **经验的基本组织单位是什么？**

这才是你和 ExGRPO / LUFFY / CHORD / DR3 / BREAD / TRAPO 真正拉开差距的地方。

---

### 1. 现有 replay / off-policy 方法大多复用的是“整条轨迹”
例如：

- **ExGRPO**：核心是 experience replay，提高样本效率，优先重放有效经验。
- **LUFFY**：强调 mixed-policy trajectory integration，把 demonstrations 和 on-policy rollouts 混合进 GRPO。
- **CHORD**：强调动态权衡 SFT 与 RL。

这些方法虽然各有不同，但多数还是把“trajectory / sample”作为 replay 或优化的基本单位。

---

### 2. 你的新意应该是：replay 的基本单位不是 trajectory，而是 frontier-conditioned cell
换句话说：

> **在 long-horizon sparse-reward agent RL 里，真正可复用的经验不是一整条过去轨迹，而是“从某个可恢复 frontier 出发的一段 continuation”。**

这可以定义成：

\[
c_k = (h_k,\; \mathcal{S}_k,\; m_k)
\]

其中：

- \(h_k\)：第 \(k\) 个 frontier 的历史 / prefix / state-like context
- \(\mathcal{S}_k\)：从这个 frontier 出发积累的 suffix 经验池
- \(m_k\)：这个 frontier 的统计量，例如当前成功率、recoverability、分歧度、学习进展等

这个表示方式会让你的方法从“再加一个 guidance 机制”升级成：

> **一种新的 experience representation and replay granularity**

我认为这是最可能被审稿人认可的新意点。

---

## 四、推荐的论文主线：Frontier Replay Cells

## 方法名称建议
可以考虑以下名字：

- **Frontier Replay Cells (FRC)**
- **Frontier-Conditioned Replay GRPO**
- **FORGE: Frontier Occupancy Replay for GRPO**
- **CellReplay-GRPO**
- **Frontier Memory Replay GRPO**

其中我最推荐的是：

## **Frontier Replay Cells for Sparse-Reward Agent RL**

因为它一下就把方法中心说清楚了。

---

## 五、主方法应该只保留三个核心模块

为了避免“机制堆叠感”，主文最好只保留 3 个核心模块。

---

### 模块 1：Frontier Cell Construction

#### 核心思想
从 expert 轨迹和自采经验中，自动构建 replay cells，而不是直接把 whole trajectory 丢进 buffer。

每个 cell 表示为：

\[
c_k=(h_k,\mathcal{S}_k,m_k)
\]

其中：

- \(h_k\)：frontier prefix / history
- \(\mathcal{S}_k\)：从该 frontier 出发的 suffix 轨迹集合
- \(m_k\)：该 frontier 的元信息，例如：
  - success rate
  - recoverability
  - frontier uncertainty
  - learning progress

#### 方法定位
你的 claim 应该是：

> **在 long-horizon sparse-reward 环境中，frontier-conditioned cells 比 whole trajectories 更适合作为 experience replay 的基本单位。**

#### 为什么重要
在你的 setup 里：
- 每 task 只有 1 条 expert
- 总交互步数只有 100
- ScienceWorld 比 ALFWorld 更稀疏

所以真正稀缺的不是“有没有 expert token”，而是：

> **能否把少量昂贵交互沉淀成可以持续复用的局部 continuation 单元。**

---

### 模块 2：Cell-Conditioned Replay Repair

#### 核心思想
分布修复不应该在全局 trajectory 分布上做，而应该在 **同一 frontier cell 内** 做局部 continuation 修复。

也就是说，不修：

\[
\pi(\tau)\quad \text{vs}\quad q(\tau)
\]

而修：

\[
\pi(\tau_{k:T}\mid h_k)\quad \text{vs}\quad q_k(\tau_{k:T}\mid h_k)
\]

其中：

- \(h_k\) 是 cell 的 frontier context
- \(q_k\) 是该 frontier 下历史 suffix 的行为分布

#### 为什么这比 DR3 更自然
DR3 的主线是全局 distribution repair。这个方向虽然数学上更硬，但容易被说成：
- 不是特别新
- 是若干稳定化机制的叠加
- 更像一般性的 off-policy correction

而你如果把 repair 明确限制在 cell 内，就可以强调：

> **我们不做一般性的全局修复，而是只对“同一 frontier 下的 continuation mismatch”做局部修复。**

这样：
- repair 更局部
- bias 更小
- 可解释性更强
- 更贴近 sparse-reward agent RL 的真实问题

#### 你论文里的 claim
> **在 frontier-conditioned continuation 上做 replay repair，比对全局 trajectory 分布做统一修复更适合长程 agent RL。**

---

### 模块 3：Progress-Driven Cell Scheduling

#### 核心思想
不是所有 frontier cells 都值得同样多的训练预算。  
在总训练仅 100 步的情况下，最关键的是：

> **把 rollout / replay 预算分给“当前最有学习价值的 frontier cells”。**

#### 具体做法
为每个 cell 维护一个 utility 分数，例如：

\[
u_k = \text{recoverability}_k \times \text{uncertainty}_k \times \text{learning-progress}_k
\]

或者更简单地，根据当前成功率估计：

\[
u_k \propto \hat p_k(1-\hat p_k)
\]

其中 \(\hat p_k\) 表示从 frontier \(h_k\) 出发，当前策略的成功率估计。

#### 直觉
- 如果 \(\hat p_k \approx 0\)：太难，短期学不到
- 如果 \(\hat p_k \approx 1\)：太容易，边际收益小
- 如果 \(\hat p_k \approx 0.5\)：最有学习价值

这和 ExGRPO 中“中等难度样本更有用”的精神一致，但你的粒度更细，是 **frontier-cell 级别的 progress scheduling**。

#### 你论文里的 claim
> **在极低交互预算下，真正决定样本效率的不是更复杂的全局 correction，而是 replay budget 的 frontier-aware scheduling。**

---

## 六、expert 在你的方法里应该扮演什么角色

这里非常关键。

### 不推荐的写法
不要把 expert 写成方法主角，不要强调：
- 训练主体由 expert prefix 驱动
- expert 负责给出主要 guidance
- 整体方法主要依赖 expert continuation

因为这样会很像 BREAD / TRAPO。

---

### 推荐的写法
把 expert 降级为：

## **Frontier Cell Initialization / Calibration Signal**

也就是说 expert 主要用于：

1. 初始化部分 frontier cells
2. 在少数极难 frontier 上做一次性校准
3. 提供 cell 的高质量 seed continuation

而不是主导所有训练过程。

#### 这样做的好处
- 你和 prefix-guidance 类方法区分更清楚
- 更符合“experience replay 为主”的主线
- 更适合你每 task 只有 1 条 expert 的设定

---

## 七、你和已有工作的差异应该怎么写

### 与 BREAD / TRAPO / ICPO 的差异
这些工作主要研究：

- 如何在 rollout 时提供 expert guidance
- 如何让策略从 expert prefix 开始继续探索
- 如何设计 prefix 长度或局部引导策略

而你的方法研究的是：

> **如何把 sparse-reward 环境中的历史交互组织成 frontier replay cells，并在 cell 内进行局部 continuation replay 与 repair。**

所以你的重点不是 rollout guidance，而是：

## **experience representation + local replay optimization**

---

### 与 Scaf-GRPO / DDE / critical-step 方法的差异
这些方法主要关注：
- minimal scaffolding
- learning cliff
- pivot states
- critical actions

而你的方法关注的是：

> **在 closed-loop 训练里，什么经验单元应该进入 memory，如何被重放，何时被调度。**

因此你不是“另一种局部提示策略”，而是：

## **一种 replay-centric sparse agent RL 框架**

---

### 与 ExGRPO / LUFFY / CHORD 的差异
这些方法主要分别侧重：
- replay 成功经验（ExGRPO）
- mixed-policy demonstration integration（LUFFY）
- SFT + RL 动态混合（CHORD）

而你的方法的重点是：

1. **replay 的对象是什么**  
2. **replay repair 在哪里发生**  
3. **训练预算如何在 replay cells 间动态分配**

这会让你和它们形成非常清楚的区分。

---

## 八、为什么这个方向更适合你的具体 setup

你的实验设定是：

- ALFWorld
- ScienceWorld
- 每个 task 仅 1 条 expert trajectory
- 总训练步数约 100
- ScienceWorld 更稀疏、更难

这个设定有几个非常重要的含义：

### 1. 全局 SFT guidance 不一定是最优利用方式
因为 expert 很少，每 task 只有 1 条。  
如果用 whole-trajectory imitation，很容易：
- 过拟合 single expert behavior
- 缺乏泛化
- 训练信号集中在整条路径，而非真正有价值的 frontier

---

### 2. 真正昂贵的是 environment interaction
既然总训练步数只有 100，那么每一次 rollout 都很贵。  
这意味着最重要的不是“再做一个更一般的 loss”，而是：

> **如何最大化每次交互沉淀出的可复用价值。**

Frontier cells 正是为这个目的服务的。

---

### 3. ScienceWorld 更 sparse，说明最关键的是 frontier reachability
在 ScienceWorld 里，失败往往不是因为模型不会某个 token，而是因为：
- 根本到不了有用状态
- 或到达后无法复用已有部分经验

因此，把经验组织成 frontier-conditioned continuation cells，会比整条 trajectory replay 更高效。

---

## 九、主文应避免的内容

为了避免“机制堆叠”印象，主文里尽量不要把以下内容作为核心模块：

- ESS
- dual control
- reflection memory
- boundary token guidance
- too many ratio tricks
- too many auxiliary losses

这些都可以存在，但最好放在：
- appendix
- ablation
- optional enhancement

主文一定要极简。  
我建议主文只保留：

1. Frontier Cell Construction  
2. Cell-Conditioned Replay Repair  
3. Progress-Driven Cell Scheduling  

这样审稿人才会觉得你有一个清晰的中心思想，而不是很多小技巧拼起来。

---

## 十、最终推荐的论文定位

我建议你把论文定位成：

## **A replay-centric framework for sparse-reward long-horizon agent RL**

更具体一点：

> We propose a frontier-cell-based closed-loop GRPO framework that represents experience at the level of frontier-conditioned continuations, repairs replay mismatch locally within cells, and allocates scarce rollout budget according to frontier learning progress.

中文可以表述为：

> 我们提出一种以 frontier replay cells 为核心的 closed-loop GRPO 框架，在 long-horizon sparse-reward agent 环境中，不再将完整轨迹作为经验复用单位，而是将 frontier-conditioned continuation 作为经验表示、局部修复与预算调度的基本对象。

---

## 十一、最精简的一句话总结

### 不要写成
> 我们也做 prefix guidance，只是再加了 replay 和 repair。

### 应该写成
> 我们提出一种以 frontier replay cells 为核心的 sparse agent RL 框架，将经验复用粒度从 whole trajectories 改为 frontier-conditioned continuations，并在 cell 内进行局部 replay repair 与进度驱动调度。

---

## 十二、最后的明确建议

### 当前最推荐的研究主线
**Frontier Replay Cells + Cell-Conditioned Replay Repair + Progress-Driven Scheduling**

### 不推荐继续当主线的方向
- 单纯 DR3 式全局分布修复
- 把 expert prefix 作为主角
- 把 token guidance 当作最核心卖点

### 最适合你的整体叙事
不是“我们提出一个更一般的 off-policy correction”，而是：

> **在极低交互预算、每任务单条 expert、奖励高度稀疏的 agent RL 设定下，真正决定样本效率的，是经验的表示粒度、局部修复位置以及 frontier-aware 的预算调度。**

---

## 参考工作定位说明
本文中的相关工作判断主要依据以下近邻研究方向：

- BREAD：partial expert guidance / branched rollout 
- TRAPO：adaptive expert prefix continuation 
- ICPO：expert-conditioned rollout generation 
- Scaf-GRPO：minimal scaffolding for learning cliffs 
- ExGRPO：experience replay for GRPO / sample efficiency 
- LUFFY：mixed-policy GRPO with demonstrations 
- CHORD：dynamic SFT + RL mixing 
- DDE / DEEP-GRPO：pivot/deep-state local resampling 
- CARL：critical-action learning in agent RL 

---


我换一种更直白的方式讲，不走论文口吻。

你现在困惑的核心，其实是这句话：

> **为什么“frontier replay cells”比“expert prefix / whole trajectory replay / SFT guidance”更值得做主线？**

我用三层来解释：
先讲**问题到底卡在哪**，再讲**我提议的方法在干什么**，最后讲**它和 CHORD / DR3 / ExGRPO 的区别**。

---

# 1. 你现在真正的问题，不是“不会模仿 expert token”

而是：

## 模型很难走到“有用的中间状态”

这在 ScienceWorld 特别明显。

比如一个任务要 8 步才能成功：

1. 找到柜子
2. 打开柜子
3. 拿烧杯
4. 去水槽
5. 装水
6. 去加热台
7. 打开加热
8. 检查温度

训练早期，模型经常在第 2、3 步前就乱掉了。
那会发生什么？

* 它根本到不了“装水”这个状态
* 所以后面关于加热、温度的奖励信号根本看不到
* 你即使有整条 expert，也只是知道“最终答案长什么样”
* 但模型缺的是：**怎么走到那个中间点**

所以问题不是：

> “模型已经到了关键状态，但不知道输出哪个 token。”

而是：

> **模型大多数时候根本到不了那个状态。**

这就是 sparse reward agent RL 最难的地方。

---

# 2. 为什么 CHORD 会让你感觉很强

因为 CHORD 的直觉很简单：

* expert 轨迹有 token-level supervision
* SFT 会直接拉模型输出 expert 风格
* 所以训练很稳、很有指导性

这个感觉没错。
它强在：

## 它给了非常直接的“你这一步该怎么说/怎么做”的信号

所以你会自然担心：

> “如果我们不做 SFT，只做 replay repair，会不会缺少 guidance？”

这是非常合理的担心。

---

# 3. 我为什么说“不要把主方法写成又一种 prefix guidance”

因为现在已经有很多工作在讲：

* 给一段 expert prefix
* 从那里继续 rollout
* 或者给 hint / scaffold
* 帮模型跨过 learning cliff

所以如果你也写成这个样子，审稿人会说：

> 你这个大方向别人已经做过了，只是又改了点实现。

所以我才建议你换一个角度来表述，不是说：

> “我们也给 prefix。”

而是说：

> **我们重新定义 replay 的基本单位。**

这就是“frontier replay cells”。

---

# 4. 什么叫 frontier replay cell

别把它想复杂。

它其实就是一句话：

## 不要把“整条轨迹”当经验单位

## 要把“从某个关键中间点开始的一段后续经验”当经验单位

比如上面那个任务里，假设模型已经能做到：

* 找到柜子
* 打开柜子
* 拿到烧杯

但后面不会了。

那“拿到烧杯”这个中间点，就是一个 **frontier**。

从这个点往后，可能有很多种历史经验：

* 一次失败：拿了烧杯但没去水槽
* 一次失败：去了水槽但没装水
* 一次成功：装水后去加热台，最后完成任务
* 一条 expert suffix：从这里开始的正确后续

这些“从同一个中间点往后”的经验，放在一起，就是一个 **cell**。

所以一个 cell 本质上是：

* 一个中间状态/前缀
* 从这里出发的若干后续尝试
* 以及这个中间点现在学得怎么样的统计信息

---

# 5. 它和 whole trajectory replay 的区别是什么

## whole trajectory replay 的想法

把一整条过去轨迹存起来，再拿出来训练。

问题是：

* 轨迹很长
* 很多前半段其实模型已经会了
* 真正难的只在某个局部
* 你每次都重放整条，会很浪费

## frontier cell 的想法

只盯住“模型目前还卡住的那个中间带”。

这样你学的不是：

> “整题怎么从头到尾做一遍”

而是：

> **“从这个关键中间点出发，后面怎么接下去”**

这更像人类学习复杂任务的方式。

---

# 6. 为什么这比“expert prefix 是主角”更好

如果你把 expert prefix 当主角，你的方法看起来就像：

* BREAD / TRAPO 这一类
* 核心是 expert 把你送到中间，再继续学

但如果你把 **cell** 当主角，expert 就只是：

## 用来初始化少量高质量的中间点经验

也就是说 expert 不是主方法，expert 只是帮助你建立 cell 的一个工具。

这样你的方法主线就变成：

* replay 的经验如何组织
* 哪些中间点值得反复学
* 在同一个中间点上如何复用历史经验

这就更像一个新的 replay 框架，而不是又一个 expert-guided RL。

---

# 7. 那“repair”到底在这里扮演什么角色

这里也容易混。

你原来想的是 DR3：

* 全局做 distribution repair
* 用 density ratio 修 teacher / replay 和 current policy 之间的偏差

这个思路没错，但会有两个问题：

1. 看起来像一个比较泛的 off-policy correction 技术
2. 容易被说成“修复 + clip + ESS + dual”很多东西堆在一起

所以我建议把 repair 缩小到更自然的位置：

## 只在同一个 frontier cell 里面做 repair

什么意思？

不是问：

> “整条 replay 轨迹和当前策略差多远？”

而是问：

> **“在这个中间点之后，这段 suffix replay 和当前策略差多远？”**

这就合理很多，因为：

* 条件都一样，都是从同一个 frontier 出发
* 差异更局部
* 修复更可信
* 更像在修“局部 continuation mismatch”

所以 repair 还在，但它不再是整个方法的主角。
它变成：

> **cell 内 replay 的一个局部矫正器**

---

# 8. 那 training dynamics 的创新点到底是什么

这是最重要的一步。

你问得很对：

> 如果没有 CHORD 那种强 SFT guidance，我们怎么在训练动力学上更强？

我的回答是：

## 不是靠更强的全局 guidance

## 而是靠“把训练预算始终打在最值得学的 frontier 上”

这就是 progress-driven scheduling。

直觉上：

* 有些 frontier 太难，现在学不到
* 有些 frontier 已经会了，再学没收益
* 最有价值的是“差一点就会”的 frontier

所以你每一步训练，不应该平均看所有经验。
而应该优先看：

> **当前最接近突破的那些中间点**

这和 ExGRPO 的“中等难度经验最有用”有精神上的相似之处。
但你的粒度更细：不是整道题，而是某个中间点。

所以真正的训练动力学创新不是：

* 再加一个 SFT
* 再加一个 token weight
* 再加一个 ESS

而是：

## 让 replay 成为“frontier 学习进度驱动的 closed-loop system”

策略变强
→ 某些 frontier 学会了
→ 预算自动移到新的 frontier
→ replay 和 repair 的重心也一起移动

这就是 closed-loop。

---

# 9. 用一句最简单的话概括整个方法

如果要极度直白地说：

## 旧思路

“把整条 expert / replay 轨迹拿来学”

## 新思路

“找到模型当前卡住的中间点，只反复学习从那个点往后的后续部分”

这就是核心。

---

# 10. 你可以把它和几种方法这样区分

## 和 CHORD 的区别

CHORD：
“给你整条 expert 的 token 级 guidance，让你整体更像 expert”

你这个方向：
“我不追求全局都像 expert，我只想在最关键的中间点之后，把 continuation 学会”

---

## 和 DR3 的区别

DR3：
“重点是全局 off-policy distribution repair”

你这个方向：
“重点是 replay 的经验粒度。repair 只是 cell 内局部使用的工具”

---

## 和 ExGRPO 的区别

ExGRPO：
“成功经验值得 replay，提高 sample efficiency”

你这个方向：
“不是所有成功经验都一样重要，真正该 replay 的是 frontier-conditioned continuation cells”

---

## 和 prefix-guidance 工作的区别

prefix-guidance：
“用 expert 把你送到中间，再继续学”

你这个方向：
“expert 不是主角，主角是中间点 replay memory。expert 只是 cell 的初始化来源之一”

---

# 11. 你现在最该记住的，不是数学，而是这个图景

想象你在爬山。

* CHORD：有人一直牵着你，告诉你整条路怎么走
* DR3：你在研究怎样修正“别人给你的路线”和“你自己走的路线”之间的分布偏差
* 我建议的新方向：
  **先找到你现在最容易卡住的半山腰平台，然后专门训练你从这个平台继续往上爬**

这个“半山腰平台”，就是 frontier。
围绕这个平台积累和复用经验，就是 replay cell。
决定今天练哪个平台，就是 progress scheduling。

---

# 12. 最后，我再帮你压缩成 3 句话

你可以先只理解这三句：

1. **Sparse reward 环境里，模型最大的困难不是不会 expert token，而是到不了有用的中间状态。**
2. **所以经验复用的基本单位不该是整条轨迹，而该是“从关键中间状态出发的一段 continuation”。**
3. **训练时应该始终把预算放在“当前最接近突破”的 frontier cells 上，而不是平均 replay 所有经验。**

你先消化这三句，就已经抓住 80% 了。

下一条我可以继续做一件更具体的事：
我直接用 **ALFWorld / ScienceWorld 的一个具体任务例子**，把“cell 是怎么建的、怎么 replay、怎么更新 GRPO”一步一步画出来。
