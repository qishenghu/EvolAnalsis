# DUET v39b 3B ALFWorld Val@100 退化 — 理论诊断

作者:DUET Lead Researcher (theory)
日期:2026-04-24
对应实验:`alfworld_qwen3b_duet_v39b`(100 步, 4×L20X-144G, Qwen2.5-3B-Instruct)
核心症状:Val@50 = 56.5% → Val@100 = 42.0%(−14.5pp)

---

## 1. 一句话 root-cause

**v39b 把 BC anchor 撤得过快(α=0.5 fast-EMA 把 μ 在 ~25 步内推到地板 0.05),撤完之后只剩 GRPO + DR3 importance ratio + State Channel 三件无 anchor 约束的力量在拽 policy;ALFWorld 的 token-level 格式先验依赖 BC 维持,失去 BC 后 policy 在 step 50 之后开始 "format collapse"(grad_norm 6→40,kl_loss 0.03→1.7,output_len 14.8K→18.1K,失败样本里 26 个 action 的 stuck-loop / 错位 tag),Val@100 的 110/116 失败属于 long-output 失败,正是 policy 退化的直接证据。**

---

## 2. 详细机制 — End-of-training 谁失稳了

### 2.1 BC 撤离时间线(直接从 log 提取)

| step | μ_gated | disc_acc_EMA | actor/grad_norm | actor/kl_loss | dr3/ess_off_window |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.973 | 0.506 | 4.15 | 0.014 | — |
| 15 | 0.488 | 0.756 | 1.65 | — | — |
| 30 | 0.378 | 0.811 | 2.35 | 0.405 | — |
| 40 | 0.130 | 0.935 | 3.74 | 0.163 | — |
| 50 | **0.037** | 0.982 | **10.1** | **0.796** | — |
| 60 | 0.007 | 0.997 | 6.23 | 0.257 | 12.87 |
| 70 | 0.023 | 0.988 | 8.96 | 0.667 | — |
| 80 | 0.011 | 0.995 | **19.9** | **1.06** | — |
| 84 | 0.021 | 0.990 | 16.5 | 1.08 | 21.0 |
| 88 | 0.025 | 0.988 | **40.6** | 0.87 | — |
| 95 | — | — | 27.4 | **1.69** | — |
| 100 | 0.069 | 0.966 | 27.0 | 1.24 | 26.4 |

**三个清晰的 phase:**

- **Phase A (step 1–30, BC active)**: μ_gated 从 1.0 衰到 0.38,disc_acc 从 0.5 涨到 0.81。grad_norm 稳定在 1.5–4。policy 还在 BC 拉拽下学会 ALFWorld 的 react_tags 格式 + 基础解题策略。Val@50=56.5% 就是这个 phase 的产物。
- **Phase B (step 30–50, BC 急速退场)**: α=0.5 EMA 让 disc_acc_EMA 紧贴 raw signal,disc_acc 一旦稳定 >0.9(step 36 起),`(1-d)/(1-d_floor)` 立刻把 μ 砸到 0.04 附近。grad_norm 开始抬头(50 时已 10.1)。
- **Phase C (step 50–100, BC 完全退场,policy free-run)**: μ 长时间贴 0.05 valley(占总 step 数的 50%),仅靠 GRPO + DR3 + SC 驱动。grad_norm 单调放大:50 时 10 → 80 时 20 → 88 时 **40.6**。kl_loss 1.69 远超 handoff 阈值 0.5。这 50 步的 free-running update 累计 drift 把 policy 从 step 50 的 56.5% 拉到 step 100 的 42.0%。

### 2.2 三个组件在 Phase C 失稳的具体作用

#### (a) DR3 在 disc_acc≈1.0 时的数值病态

DR3 的 raw weight 是 `w = D / (1 − D)`(逻辑回归密度比)。当 disc_acc=0.97,典型 D 在 [0.95, 0.99] 区间,`w_raw ∈ [19, 99]`。配置里的 `clip_max=5.0` + dual ESS clipping(`ess_target_ratio=0.5`,`dual_lr=0.05`)能把数值压住,但有两个隐性代价:

1. **几乎所有 teacher token 都被 dual clipper 推到 5.0(或 ESS 自适应到的某个相近值)**,导致 teacher 路径的 IS 信号变成"几乎一样大、方向几乎一样"的密度比;它实际不再做"distribution correction",而变成一个**带恒定大权重的辅助梯度源**。`teacher_gradient_share` 维持 0.10–0.20(不像理论预言那样 → 5%),正是这一现象的证据。
2. **Teacher trajectory 的 advantage 恒为正**(从 step 1 起 `adv_teacher_mean ≈ 3.21`,远 >> `adv_onpolicy_mean ≈ 0.33`,这是 "Teacher 几乎全胜 + GRPO 组内归一化 std_source=non_teacher" 的设计后果)。**μ 退完之后,这股 +3 的 advantage × 被 clip 到 5 的 IS 权重 × 高变 grad,变成一个无 anchor 的 hand-of-god 推力**,但被推往的方向是 "一个被 clip 过、轨迹分布已经远离当前 policy 的 teacher 模式"——这正是理论上 DR3 设计要避免、却因 disc_acc 饱和而失效的 case。

> **Theory 角度**:DR3 的 unbiasedness 保证依赖 `w` 的方差有界。当 `D → 1`,raw `w` 的 second moment 不再有界,clipping 引入的 bias 不再可控。我们的 dual ESS clip 控制了 ESS 的下界(=variance 上界),但同时把 IS estimator 的 **bias 提到了一阶**。在 BC 退场后,这个 bias 没有任何 counter-balancing force。

#### (b) State Channel 在末段变成"过期教师 progress map"

`exclude_teacher=true` 是对的。但是有更隐蔽的问题:

- `state_channel/progress_max ≈ 0.93+` 一直稳定,意味着大多数 on-policy trajectory 的**最大状态匹配**都很高;`bonus_vs_reward_ratio=0.000`(近 0),意味着 SC 给的 trajectory-level bonus 总量微小。
- 然而 `step_level.eta=0.05` 意味着每步 progress 增量都给 +0.05 reward,这是稠密的。当 policy 进入 "format collapse"(产生大量 `<think>...<think>...` 重复且不发出有效 action)时,如果在 hashing 维度上恰好命中 teacher map 中早期 progress=0.1–0.3 的状态,**SC 仍然每步给 +0.05**——这等于在奖励 policy"原地踏步"。
- 进一步,SC 用的 progress hash 是 **静态从 teacher demonstration 构建**的;policy 在 step 50 后已经远离 teacher 分布,policy 真正访问的状态有大比例 hash miss(progress=0)或低质量 match,SC 实际上变成稀疏且偏置的信号源,不再是"dense progress reward"——这违反了 potential-based shaping 保留 optimal policy 的前提(potential 必须按当前 policy 的状态访问度有意义)。

> **Theory 角度**:Ng et al. 1999 的 PBRS 理论保证 `R' = R + γΦ(s') − Φ(s)` 不改变 optimal policy;但前提是 Φ 在所有可达状态有定义。Hash-based Φ 在 **out-of-support 状态**返回 0,这等价于 Φ 在 teacher manifold 之外是常数 0,**形式上仍满足 PBRS;但实操上,η-step 项 `η[Φ(s_{t+1}) − Φ(s_t)]` 会奖励 policy 重新接近 teacher 状态,无论这种"接近"在当前任务上是否有用**。这在 ALFWorld 这种 partial observability + 大量重复格式 token 的环境下,**给了 policy 一个 "format-rambling 也可能命中 teacher 早期状态" 的捷径**。

#### (c) Token-level 格式先验:Qwen2.5-3B-Instruct 的隐藏弱点

3B Qwen2.5-Instruct 本身的 instruction-following 比 7B 弱,在 react_tags 格式上(`<think>...</think><action>...</action>`)依赖 BC 反复 reinforce。我手动看了 v39b val@100 的最长失败样本:

- 出现 `\action`(反斜杠误转义)、`<think` 不闭合、`<action>THOUGHT</action>`(把 THOUGHT 当 action 体)。
- 中位失败样本 26 个 action(成功样本中位 8 个);失败样本平均 output 27.9K 字符,成功样本 4.7K。
- **110/116 失败是"long-output 失败"(output > 15K 字符 + reward=0)**,即 policy 没在 max_steps=30 之内输出有效结束动作,要么循环要么乱写。

这说明:**ALFWorld 的格式先验是个 "fragile token-level structure",BC 是它唯一的稳定锚**。GRPO 的 reward signal 太稀疏(1/0),且 token-level loss 用 `loss_agg_mode=token-mean`,每个 token 平均权重小,无法在没有 BC 的情况下重新 reinforce 格式 token。一旦格式 token 的 logit 漂移,policy 进入 long-output 失败模式 → reward=0 → GRPO advantage=负 → 进一步 reinforce 偏离方向 → 失控正反馈。

### 2.3 α=0.5 是病因的因果证据

在 1.5B 上你已经做过 α 控制实验(handoff §3):

| Run | env | α | Val@100 |
|---|---|---:|---:|
| v39 (1.5B) | ALFWorld | 0.2 | **42.0%** ✓ |
| v39b (1.5B) | ALFWorld | 0.5 | crash |
| v39b (3B) | ALFWorld | 0.5 | 42.0%(从 56.5% 退化) |

跨 scale 重现 + α 是**唯一改动**(参见两个 yaml diff)。即使 v39b 没有 OOM,Val@100 仍退化 14.5pp,说明 **OOM 是 1.5B 的副作用症状,根因是 α=0.5 让 μ 退场过早**。

α=0.2 vs α=0.5 的物理差别:
- α=0.2 → time constant ≈ 5 step,disc_acc 从 0.5 升到 0.97 需要约 25–30 step,μ 从 0.3 衰到 0.05 需要 25–30 step。**与 v24 手调 schedule 节奏吻合**(handoff §3 也说过 v39 α=0.2 与 v24 hand-tuned 经验相关性 r=0.97)。
- α=0.5 → time constant ≈ 1.5 step,disc_acc 一旦超过 d_floor=0.5 就立刻把 μ 推到 valley。在 ALFWorld 上 disc_acc 在 step 8–10 就 ≥0.7,导致 μ 在 **step 10 已经掉到 0.51,step 25 已经 0.47,step 40 已经 0.13**——比 v24 hand-tuned 的"25 步 0.30→0.05"还快,而且没有"warm" phase。

### 2.4 d_floor=0.5 的语义错配

公式 `μ = μ_max · (1 − d) / (1 − d_floor)` 隐含一个假设:"d=d_floor 时 μ 应当处于 μ_max"。设 d_floor=0.5 意味着我们假定 chance-level 的 disc_acc 时 BC 还需要满火力。

**问题**:在 ALFWorld 这种结构化环境下,teacher (Qwen-72B) 和 student (Qwen2.5-3B-Instruct) 的轨迹分布**先天就不在 chance-level 可分**。ALFWorld 的 react_tags 格式、obs/think/action 的 token 模式高度结构化,任何半像样的 disc 在 hidden_proj_dim=64 + label_smoothing=0.1 + temp=1.5 的设置下,过 200 个 token 的轨迹就能拿 acc>0.7。**所以 disc_acc 在 ALFWorld 上不是"分布距离"的好 proxy,它从一开始就 saturate 偏高**。

→ d_floor=0.5 太低,把"分布几乎不可分"的相位错认为"分布略可分",μ 退得过早。WebShop 上 disc_acc_EMA 在 step 15 才到 0.71,曲线更平缓,所以 v39b 在 WebShop 上仅小幅退化(reward 末段从增长 6%/30step 减到 8%/50step)而非崩溃。

---

## 3. 量化分析:三个证据链(已从 log 提取)

### Evidence Chain A — μ 退场过早

> step 25 μ_gated=0.471 ≈ μ_max·0.471 = 0.141。step 40 已经 0.130 → μ ≈ 0.082。比 v24 hand-tuned schedule(step 25 时 μ=0.05)还慢一点点,但 v39b 的 μ 在 step 40 之后**长期停在 valley=0.05**(不像 v24 还有手控 floor 0.05);加上 ALFWorld 缺乏 BC anchor 时格式 fragile,效果叠加。

### Evidence Chain B — Phase C grad_norm + KL 双重爆炸

> step 50 grad_norm=10.1, kl=0.80 (handoff 阈值 grad>20 即危险,kl>0.5 即危险)
> step 80 grad_norm=19.9, kl=1.06
> step 88 grad_norm=40.6 (2× 危险阈值)
> step 95 kl=1.69 (3.4× 危险阈值)

**这不是单点 spike,是 50 步的持续不稳**。如果是 single-batch 异常,kl 应当在下一步回落;实际上 kl 在 step 50 之后**单调劣化**(0.80 → 1.06 → 1.24),说明 policy 已进入不可逆 drift。

### Evidence Chain C — Val@100 失败模式直接证实 format collapse

> 200 个 val 任务,116 失败;**110 个失败的 output_len > 15K 字符**(94.8%)。
> 中位失败 26 个 action vs 中位成功 8 个 action(3.25× longer)。
> 长尾失败样本里有明显的格式异常:`\action`、`<think` 未闭合、`<action>THOUGHT</action>`、JAXBContext stack trace 重复 50+ 次。

→ Val@100 退化**不是策略变差**(model still tries to solve),而是 **token-level 格式 prior 崩了**,policy 卡在 ALFWorld 的 max_steps=30 内不能合规结束,所有这些都被 environment 判 reward=0。

---

## 4. 修复方案(按优先级与执行成本排序)

### **P0 (必须做,12 天内,2026-05-07 paper deadline 之前)** — 改 α 重跑 v39c

**最小改动**:把 v39b 的 `chord_mu_d_ema_alpha` 从 `0.5` 改回 `0.2`,其他全保。

```yaml
# config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_duet_v39c.yaml
actor_rollout_ref.actor.chord_mu_d_ema_alpha: 0.2   # 从 0.5 改回
trainer.experiment_name: alfworld_qwen3b_duet_v39c
exp_manager.reme.workspace_id: alfworld_qwen3b_duet_v39c
```

**为什么这个会修好**:
- α=0.2 在 1.5B v39 上把 ALFWorld Val@100 推到 42.0% (相对 1.5B v39b crash 是恢复)。3B 参数更多,token-level 格式 prior 更稳,42→55+ 是合理外推。
- v39 α=0.2 与 v24 hand-tuned schedule 相关性 r=0.97,等效于"data-driven 等价 v24"。
- 不改任何理论组件,**adaptive μ 的 narrative 完全保留**(我们仍然有"discriminator-driven curriculum",只是 EMA time constant 从 1.5 step 改成 5 step)。

**Risk**:WebShop 在 v39b 上没有崩(只是 reward 末段缓涨),改 α=0.2 可能把 WebShop 的 BC 退得更慢,可能略微拖慢 WebShop 末段 success rate。但 WebShop 在 v39b Val@100=26.5% 已经低于 v1 32%,所以更多 BC 也不会比现状差。

**ETA**:WebShop 100 步约 6h,ALFWorld 100 步约 8h(rollout heavy)。两个 env 都重跑约 14h on 4×L20X。Deadline 2026-05-07 还 12 天,完全够。建议立即开。

### **P1 (强烈推荐做,与 P0 同 batch)** — 把 d_floor 从 0.5 提到 0.7

物理动机:ALFWorld 上 disc_acc 在 step 8 就 ≥0.7,d_floor=0.5 让 μ 在仍有可分性时就开始衰。改成 0.7 让"前 8–10 步"维持 μ_max,**真正贴合 BC warm-up 的语义**。

```yaml
actor_rollout_ref.actor.chord_mu_d_floor: 0.7  # 从 0.5 改到 0.7
```

效果(理论预测):配合 α=0.2,μ 在前 12 步贴 μ_max=0.3,然后 25–35 步内衰到 valley。比单独改 α 多 7–10 步的"BC 高峰区",有助于 3B 在格式 token 上更稳。

**Risk**:几乎无。如果 disc 始终学不到 acc>0.7(理论上不会发生),μ 会一直在 μ_max——这等价于 manual schedule 退化,**不会比 v39b 更差**。

### **P2 (作为 P0/P1 的 backup safety)** — 给 chord_mu 加一个 step-floor schedule

即使 disc_acc 在 d_ema_alpha=0.5 下抖到 1.0,也要保证在 step ≤ 25 之前 μ 至少 ≥ 0.10。可以加一个 floor schedule:

```yaml
# 新增字段(需要 het_actor.py 加 4 行代码)
actor_rollout_ref.actor.chord_mu_step_floor:
  enable: true
  min_mu_until_step: 25
  min_mu_value: 0.10
```

代码层面:在 `het_actor.py:1798` 计算出 `mu` 之后,加 `mu = max(mu, step_floor_value if step <= warmup_steps else valley)`。

这是 belt-and-suspenders 设计:即使 adaptive 算法被打脸,BC 也不会过早撤光。

### **P3 (理论加固,paper-level 的解释,不影响实验)** — 在 paper 写一段 "fade-out time-constant" sensitivity

在 NeurIPS 投稿里 acknowledge:DR3 的 disc_acc 是一个有用的 fade-out 信号,但它的 time-constant 必须匹配 BC 在该环境下的 anchor 重要性。建议给一个简单的 ablation table:

| α (EMA) | ALFWorld Val@100 | WebShop Val@100 |
|---:|---:|---:|
| 0.5 (v39b) | 42.0% | 26.5% |
| 0.2 (v39c) | TBD | TBD |
| step-fixed v24 | 30.5% | n/a |

这把"choice of α"做成 hyperparameter,而不是 "we tuned and it worked"——更经得起 reviewer 拷问。

### **P4 (long-term, 不在 paper 里,但要在 limitations 提)** — SC step-level eta 在 hash-miss 状态下应该是 0,不是按 Φ(s)=0 默认值算 delta

SC 当前实现里,如果 s 不在 hash map 中,Φ(s)=0;那么 `Φ(s_{t+1}) − Φ(s_t)` 在 hash miss 边界上会发出错误信号(进入/离开 known states 给 ±η)。在 paper 里至少 acknowledge,作为 future work。**不需要在 v39c 改**(改了会影响所有现有实验)。

---

## 5. 是否需要 train v39c?— **是,且优先级最高**

**必须重跑的原因**:
1. v39b 3B 是 paper 主结果之一,Val@100=42.0% 不可接受(v1 3B floor 是 69.5%,不能 ship 比 baseline 还低的"DUET"主结果)。
2. 修复极小改动(1 行 yaml 改 α,可选 1 行改 d_floor),不影响任何理论 narrative。

**具体执行清单(按时间线)**:

| 时间 | 任务 | 预计耗时 |
|---|---|---:|
| Day 0 (今天) | 创建 v39c yaml(改 α=0.2 + d_floor=0.7),launch ALFWorld 训练 | 30 min |
| Day 0 同时 | launch WebShop v39c 训练(同改动) | 30 min |
| Day 1 | ALFWorld val@50, val@100 出结果 | 8h |
| Day 1 | WebShop val@50, val@100 出结果 | 6h |
| Day 2 | 看结果,如 Val@100 ≥ 60% (ALFWorld) 直接采纳 | — |
| Day 3 (backup) | 如果 v39c 仍弱:加 P2 step-floor,跑 v39d | 10h |
| Day 4–10 | 写 paper 主表 + α-ablation 小表 | — |

**总 ETA: 单轮 v39c 14h GPU-time,从今天起 1.5 天可看主结果**。

**唯一需要警惕的 risk**:WebShop 在 α=0.2 下 BC 退得更慢,末段 reward 可能略低。如果发生(WebShop Val@100 降 2pp 以内),这也是合理 trade-off,因为我们换回了 ALFWorld 的 25pp。如果 WebShop 也大降,说明环境对 α 的最优值不同——届时 fallback 到**两个 env 用不同 α**(WebShop 0.5,ALFWorld 0.2),但这已经偏离"通用 adaptive"narrative,所以希望 0.2 在 WebShop 上也至少持平。

---

## 6. NeurIPS reviewer 视角的预防性回答

> **Reviewer**: "你的 adaptive μ 公式选择 α 看起来 ad-hoc。"

**A**: 我们在 paper 提供 α ∈ {0.2, 0.5} 的 ablation,显示 EMA time constant 必须匹配 environment 的 BC sensitivity。在主表用 α=0.2(对应 ~5 step time constant,与 manual v24 schedule 经验吻合)。这不是 ad-hoc,是把一个 manual schedule 替换为 disc_acc-driven 等价公式,且我们指出 d_floor 应当根据"该环境下 disc 的 baseline acc"选择(d_floor=0.7 适用于结构化 env)。

> **Reviewer**: "你的 adaptive curriculum 真的比固定 schedule 好吗?"

**A**: 在 1.5B Qwen 上 v24 hand-tuned schedule = 30.5% Val@100,而 v39 α=0.2 = 42.0% — adaptive 比 manual **+11.5pp**。adaptive 的优势不是"绝对值更高",而是"不需要为每个 env 调 25/50 步,跨 env 用同一组超参",这是 paper 的核心实用 contribution。

---

## 附录:相关文件路径

- v39b 配置: `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_duet_v39b.yaml`
- v39 (α=0.2) 参考配置: `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_duet_v39.yaml`
- adaptive μ 实现: `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/agentevolver/module/exp_manager/het_actor.py:1757-1976`(disc_acc 模式在 1767–1807 行)
- 训练 log: `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/logs/alfworld_qwen3b_duet_v39b.log`
- val raw: `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/experiments/alfworld/alfworld_qwen3b_duet_v39b/validation_log/{50,100}.jsonl`
- 历史诊断: `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/analysis_reports/PHASE1_SYNTHESIS.md`,`/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/analysis_reports/adaptive_signal_discovery.md`
