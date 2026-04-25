# DUET v39b 3B (ALFWorld) Case Study: Val@50→Val@100 退化诊断

**Author**: case-analyst (Trajectory & Behavior)  
**Date**: 2026-04-24  
**Scope**: 解释 ALFWorld 上 3B v39b 在 step 50 时 Val=56.5%、step 100 时 Val=42.0%(净退化 −14.5pp)的行为根因。  
**Data**: `experiments/alfworld/alfworld_qwen3b_duet_v39b/{validation_log,rollout_log}/*` + `checkpoints/agentevolver/.../Trajectory/batch_diag_step_*.json`

> 结论先行:**退化不是来自 reward 信号失真,而是 policy formatter 在 step 78–95 之间崩塌**。失败模式有两条主线 ——(A)agent 把字面字符串 `THOUGHT` 当作 action 输出导致环境拒绝(invalid format),后续陷入循环;(B)中文/CJK 思考混入输出。两者在 rollout step 88 后呈指数增长,且与 grad_norm 在 step 79、88 的两次 spike(20.5 / 40.6)在 timing 上吻合(spike 后 6–10 步开始扩散)。同时 GRPO advantage 自我反馈环路(on-policy reward 暴跌 → teacher–on gap 拉开 → teacher token-advantage 飙到 2.5 → DR3 fade-out 反向)放大了崩塌速度。

---

## A. 退化定量

### A.1 Transition matrix(同 200 个 val task,index 对齐)

| 转移类型 | 计数 | 占比 |
|---|---:|---:|
| S→S(始终成功) | **66** | 33.0% |
| **S→F(regression)** | **47** | **23.5%** |
| F→S(improvement) | 18 | 9.0% |
| F→F(始终失败) | 69 | 34.5% |

**净 Δ = 18 − 47 = −29 / 200 = −14.5pp**(对上 56.5% → 42.0%)。**regression 数量是 improvement 的 2.6 倍**;policy 不是"换了一批任务擅长",而是**单向崩坏**。

### A.2 任务类型分布(启发式分类:`use desklamp`/`microwave`/`fridge`/`sinkbasin`/多 put 关键词)

| 类型 | n | V@50 | V@100 | Δ | regression / improvement |
|---|---:|---:|---:|---:|---:|
| pick_and_place | 26 | 19.2% | 15.4% | −3.8 | 5 / 4(净 −1) |
| pick_two_obj | 39 | 69.2% | 53.8% | **−15.4** | 7 / 1(净 **−6**) |
| pick_clean_then_place | 28 | 67.9% | 46.4% | **−21.4** | 8 / 2(净 **−6**) |
| pick_heat_then_place | 43 | 65.1% | 39.5% | **−25.6** | **14 / 3**(净 **−11**) |
| pick_cool_then_place | 26 | 53.8% | 50.0% | −3.8 | 5 / 4(净 −1) |
| look_at_obj_in_light | 38 | 52.6% | 42.1% | −10.5 | 8 / 4(净 −4) |

**关键发现**:`pick_heat_then_place` 一类贡献了 **38%** 的净退化(−11/−29),`pick_clean_then_place` 与 `pick_two_obj` 各贡献 ~20%。这三类的共同点是:**任务 horizon 长、需要器具二段交互(microwave/sinkbasin/重复 put)**。简单 `pick_and_place` 与 `pick_cool_then_place` 几乎不退化。退化高度集中在长 horizon 任务上 —— 与 "policy 在末段步数耗光 / 进入循环" 的 behavioral 假说一致。

### A.3 行为统计:全集 vs regression 子集

| 指标 | V@50 | V@100 | Δ | regression V@50→V@100 |
|---|---:|---:|---:|---:|
| 平均 action 步数 | 14.32 | 16.80 | +2.48 | **12.64 → 23.53(+10.89)** |
| 最长连续重复 action | 1.51 | 2.07 | +0.55 | **1.11 → 3.17(+2.06)** |
| 重复 ≥4 次轨迹比例 | 5.0% | 16.0% | +11.0pp | **2.13% → 31.91%(+29.8pp)** |
| CJK 输出轨迹数 | 1 / 200 | **24 / 200** | +23 | 0 → **9 / 47**(19.1%) |
| 步数 ≥10 增加(同 idx) | — | — | — | **30 / 47(63.8%)** |

**最有信息的两个数字**:regression 子集在 step 100 时,**32% 的轨迹连续重复 ≥4 次同一 action**(基线只有 2%);**64% 的轨迹比 step 50 时多用 ≥10 步**。这是循环耗预算的直接证据,不是任务变难。

---

## B. 失败模式 Taxonomy(基于 47 个 regression case 的人工聚类)

> 每条都给出代表 idx 与 trajectory 片段。原文 quote 控制在 ≤25 行。

### B.1 `THOUGHT-as-action` 关键词污染(11 / 47 = 23%,主导失败模式)

**特征**:agent 输出像 `<action>THOUGHT</action>`、`<action>thouht...</action>`、`<action>thougt...</action>`,即把"思考"这个角色字面塞到了 action 槽位。环境收到的字符串不在 available actions 里,产生 "Nothing happened" 但 agent 不退出循环,继续重复输出同款。这是 **format collapse**,不是 reasoning failure。

**示例 idx=15**(pick_clean_then_place lettuce;V@50 7 步 SUCCESS,V@100 14 步 FAIL)

V@50 干净轨迹(7 步即完成):
```
[0] go to fridge 1   [1] open fridge 1   [2] take lettuce 4 from fridge 1
[3] go to sinkbasin 1   [4] clean lettuce 4 with sinkbasin 1
[5] go to fridge 1   [6] put lettuce 4 in/on fridge 1   ✓
```
V@100 同任务,前 6 步与 V@50 完全一致,然后崩塌:
```
[5] go to fridge 1
[6] go to fridge 1                ← 重复
[7]  THINK
[8]  THINK
[9]  THINK    ... [13] THINK     ← 7 步 THINK 死循环,episode 截断
```
agent 已经走到正确容器前(fridge 已开,lettuce 在手),只差一个 `put lettuce 4 in/on fridge 1` —— 但 policy 抽到了 `THINK` 这个字符串而非合法动作。

**示例 idx=51**(pick_two_obj remotecontrol;V@50 8 步,V@100 28 步): V@50 8 步教科书式完成 take→put×2;V@100 拿到 remotecontrol 3 后**走到 shelf 1 而不是 sofa 1** 放下(目标位置错),然后从 step 14 起连续 14 个 `<action>THOUGHT</action>` 直到 step 27。

**示例 idx=3**(pick_clean_then_place plate):V@100 step 6 输出 `clean plate 1 with fridge 1`(语义错配,fridge 不能洗东西),环境拒绝后 agent 进入 `[8] thouht / [9..11] THOUGHT / [12] go to countertop 1 / [13..15] THOUGHT / ...` 共 11 个 THOUGHT/think 步。

**为何是 11/47 而不是更多**:这只是按"action 起始 token = THOUGHT 关键词"匹配的最严格集合;若把更模糊的"action 内容含 `<think>` 标签"也算上则达 ~17/47。

### B.2 CJK 语言塌陷(9 / 47 = 19%)

**特征**:agent 在 think 块或 action 文本里混入中文(偶尔日文),整段输出从英文 ALFWorld instruction-following 状态切换到中文自我反思状态。属于 LLM 的语言一致性崩坏。

**示例 idx=13**(pick_and_place pen,V@50 11 步成功,V@100 30 步达上限失败):
```
[20]  thouht
      在如此全面的搜索和如此仔细的检查下，我应该重新审视为何不能找到这个笔。
      我之前可能在某些细节上忽略了什么？或者这个笔的位置就在某个我忽略的细节处。
[21]  THOUGHT
      Nothing happened，之前的方法可能没有考虑到一些细节...
[22]  THOUGHT
      我需要更细致地思考一下其他可能的地方，比如床底下，或者沙发上...
[27]  go to sofa\ postseason
      <think>既然已经检查了沙发，但仍然没有找到笔...
[29]  THOUGHT
      目前看来，我需要更细致地审视之前可能忽略的细节，与主人合作回忆可能的地点...
```
注意 (a) `THOUGHT` 关键词污染与 CJK 通常**共生**;(b) `go to sofa\ postseason` 显示生成器同时还吐出了 garbage token(英文残片 + LaTeX 转义)。这指向 **policy distribution 末段已经显著偏离 instruction-tuned manifold**。

### B.3 截断 / 长度爆炸(10 / 47 = 21%,常与 B.1/B.2 共生)

**特征**:V@100 轨迹达到 max_steps=30 上限被强制截断,V@50 同任务平均只用 8–15 步。30 步轨迹中通常 50% 是 [B.1] thought-action 重复,与 B.1 高度重叠。

**示例 idx=26**(V@50 4 步成功,V@100 30 步截断):V@100 没有进入 thought 模式但反复 search 错误位置 → 产生 search loop。

**示例 idx=5**:V@50 14 步,V@100 30 步 — 中间出现把目标 take→put→take→put 同一容器的 oscillation。

### B.4 错误 sub-goal / 执行顺序错乱(17 / 47 = 36%,heuristic"其他类")

**特征**:agent 没有进入语言/格式塌陷,但 sub-goal 选择错误。例:idx=1(book look-at-light)在 V@100 时不去找 desklamp,反而连开多个 drawer 直到第 14 步才输出 `thougt I should choose an action from the available actions.`。**这一类有相当一部分仍最终塌陷为 thought-action**(idx=1 step 14–18 即如此),所以与 B.1 的真实重叠 > 单纯计数。

### B.5 Subset 共生关系(对 47 case 做交叉)

| 同时具有的失败信号 | 计数 | 占 regression |
|---|---:|---:|
| 至少一项 (B.1∨B.2∨B.3) | **30 / 47** | **63.8%** |
| B.1 thought-action | 17 / 47 (含模糊匹配) | 36% |
| B.2 CJK | 9 / 47 | 19% |
| B.1 ∧ B.2 | 6 / 47 | 13%(共生) |
| 仅 B.4(纯 sub-goal 错) | 17 / 47 | 36% |

**60% 以上的 regression 是 generation-level format/language 崩塌,而非 reasoning 错误**。这与 reward gap、advantage 等 GRPO 诊断信号是一致的:崩塌发生在 generation 端,不是 task understanding 端。

---

## C. 与训练动力学的关联

### C.1 Rollout 行为指标的演化(每 step 64 trajectories,score≥1 视为 success)

| step | succ@64 | thought_action 轨迹数 | CJK 轨迹数 | 备注 |
|---:|---:|---:|---:|---|
| 50 | 39 (60.9%) | 0 | 1 | 健康 |
| 60 | 51 (79.7%) | 0 | 1 | **峰值** |
| 70 | 51 (79.7%) | 0 | 0 | 仍维持 |
| 78 | 36 (56.2%) | 5 | 5 | **首次抬头** |
| 79 | 36 | 5 | 5 | grad_norm spike=20.5 |
| 84 | 29 | **16** | 9 | 加速 |
| 88 | 46 | 14 | 1 | grad_norm spike=40.6 |
| 90 | 37 | **25** | 5 | 指数化 |
| 95 | 24 (37.5%) | 31 | 13 | 半数轨迹有 thought_kw |
| 98 | 17 (26.6%) | **41** | 16 | **谷底** |
| 100 | 28 (43.8%) | 34 | 16 | 部分恢复 |

**两个 grad_norm spike 后的扩散滞后约 6–10 步**:step 79 spike → step 84 起首批 16/64 thought_action;step 88 spike → step 90 起 25/64,step 94 起 36/64。这与 RL 中 catastrophic update 后的"格式偏好"扩散到整个 sample distribution 是匹配的。

### C.2 GRPO advantage 在崩塌期的反馈

来自 `batch_diag_step_*.json`:

| step | reward_onpolicy | reward_teacher | adv_teacher_token | adv_onpolicy_token | teacher_minus_on_gap |
|---:|---:|---:|---:|---:|---:|
| 50 | 0.575 | 1.0 | +1.93 | −0.22 | 0.42 |
| 70 | **0.768** | 1.0 | +0.56 | −0.06 | 0.23(最小) |
| 95 | 0.297 | 1.0 | **+2.41** | −0.11 | 0.62 |
| 100 | 0.358 | 1.0 | **+2.50** | −0.21 | **0.64** |

**自反馈循环很清楚**:
- step 70 时 on-policy 已逼近 teacher(0.77 vs 1.0),teacher–on gap 收到 0.23,DR3 把 teacher token-advantage 压到 +0.56(相对值)。
- step 95 onwards on-policy reward 暴跌到 0.30,gap 反弹到 0.64 —— **GRPO 的 group-relative normalization 把 teacher 的 token advantage 反推到 +2.41,几乎是 step 50 训练初期的 1.25 倍**。teacher_adv_pos_ratio 也一路涨到 1.0(每个 teacher sample 都拿正梯度)。
- 此时 DR3 的 fade-out 设计被反向激活:teacher gradient share 不再随训练进度递减,而是因为 reward gap 拉开重新主导;同时 on-policy entropy 收到 0.07(比 step 50 的 0.083 还低)—— **policy mode-collapse 同时在 generation 端(format)和优化端(advantage)发生**。

### C.3 与 grad_norm spike(step 79=20.5、step 88=40.6)的因果关系

按上面的滞后模式,我倾向解读为 **spike 是结果而非原因**。step 78 之前 thought_action 已有 ≥5 个轨迹埋下隐患(step 75 已 7/64);spike 反映的是 GRPO 在 reward 突然下降后给 teacher 高 advantage、给 on-policy 多个 token 反向梯度,grad 单步爆炸。这与 step 95+ 的"reward 二次崩盘 vs adv_teacher_token=2.5" 相符 —— 第三次 spike 没有出现,可能是 `grad_norm_clip` 截了下来,但 update direction 已被偏置主导。

---

## D. 给 algo / theory team 的 actionable insights

### D.1 行为现象 → 算法层假设

1. **核心崩塌信号是 format-level**(B.1+B.2+B.3 共占 64%),不是 reasoning 失误。这意味着 KL-to-reference 或 entropy bonus **正是当前 missing 的稳定项**。on-policy entropy 从 0.083(step 50)单调收到 0.06–0.07(step 95+),`actor.kl_loss` 显然没扛住。建议:

   - 提高 `actor.kl_loss_coef`(目前估计 ≤0.001 量级),做 step-level KL anchoring;**至少在 grad_norm > 阈值时短期 ramp up KL**。
   - 在 generation 阶段加 format reward shaping:`<action>` 内容必须从 available actions 里精确匹配,不匹配直接给 −0.05~−0.1 negative shaping(目前 invalid 是软返回 "Nothing happened",rollout 数据显示这并不能阻止 thought_action 持续输出)。

2. **GRPO advantage 反馈循环是放大器**。当 on-policy reward 跌、gap 拉开,teacher token advantage 飙到 2.5,DR3 的 density ratio 还要乘 w_hat,即便 w_hat → 0.01 floor,**teacher loss term 的有效幅度仍 = 2.5 × 0.01 = 0.025**,而 on-policy 端是 −0.21 × 1.0 = −0.21,**比例上 teacher 仍占主导**(尤其考虑 token-level 加权)。建议:

   - 使 `teacher_baseline_separation` 不仅做 mean/std 分离,**还要 cap teacher advantage 上限**(比如 |adv_teacher| ≤ percentile_95(adv_onpolicy)),否则 reward gap 自己会把 teacher gradient 放大到失控。
   - DR3 `w_min` 在 reward 暴跌阶段应**进一步压缩**(动态降到 1e-3),用 on-policy reward trend 触发,而不是固定 0.01。

3. **退化集中在 long-horizon 任务**(heat/clean/two_obj 占净退化 76%)说明 SC 的 per-step delta η·[Φ(s_{t+1})−Φ(s_t)] **在末段并没起到稳定长 horizon 的作用**,可能因为 SC `exclude_teacher=true` 导致 teacher 无 SC bonus,反而让 reward gap 在 long-horizon 更难收窄。建议:

   - 验证 `state_channel.bonus_vs_reward_ratio` 在 step 90+ 是否 >0.15(报警阈)。
   - 考虑**只对 long-horizon 任务 ramp η**(η_long > η_short),把 SC 的稳定信号集中给那些容易循环的任务类型。

4. **α=0.5(v39b)对 3B 偏激进**。1.5B v39 用 α=0.2 在 step 100 也是 42%(成功),但 3B v39b 用 α=0.5 也是 42%(刚好 collapse 边缘);1.5B v39b α=0.5 直接 crashed。**3B 在更大模型容量下并没获得 α 容差** —— 反而因为模型 expressivity 更大,format collapse 一旦发生,扩散更快。
建议:**3B 应回退到 α=0.2~0.3**;若坚持 α=0.5,必须配 KL ramp 与 advantage cap。

### D.2 优先级建议(为 5/7 ddl)

| 等级 | 行动 | 预期收益 |
|---:|---|---|
| **P0** | 用现有 ckpt 重跑 step 50–100 区间,**仅打开 format reward shaping**(invalid action: −0.05) | 直接抑制 thought-action 失败模式,理论上可挽回 ~15% × −14.5pp 的退化 |
| **P0** | 在 wandb 加 `format_collapse_rate`、`cjk_rate` 实时监控(rollout 端 trivial 计算) | 让训练崩塌可被观测,不用等 val |
| P1 | α 退到 0.3 + KL coef ×3 | 防止 step 90 后 advantage 反馈循环 |
| P1 | teacher advantage cap = percentile_95(on advantage) | 切断 vicious cycle |
| P2 | `w_min` 自适应:on-policy reward 5-step EMA 下降 >20% 时降到 1e-3 | 让 DR3 fade-out 能"再加速" |

---

## 附:数据出处

- 47 regression idx, 18 improvement idx:`analysis_reports/_v39b_3b_workdir/regression_idx.json`
- 失败模式分类(cjk/long_repeat/truncated/wrong_subgoal):`analysis_reports/_v39b_3b_workdir/reg_categories.json`
- 任务类型:`analysis_reports/_v39b_3b_workdir/types.json`
- 全部分析脚本(可复跑):`analysis_reports/_v39b_3b_workdir/`
