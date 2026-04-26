# 3B WebShop Post-Truth 分析:为什么 α=0.5 (v39b) 反而赢 α=0.2 (v39),还差 LUFFY 4pp 在哪里

**日期**: 2026-04-25
**触发**: bug-fix (commit `eabd1019`) 后,WebShop 3B 上 v39b (α=0.5) Val@100 = 45.5% success,而 v39 (α=0.2) 仅 32.0%。这彻底反转了 1.5B 时代得到的"α=0.2 更稳"结论。
**目标**: 解释机制 + 提出 v41 系列变体 + 决定 v40_psh / v40_strong_bc 是否取消。

---

## A. v39b vs v39 关键 metric 对比(分窗口聚合)

把 100 步切成 6 个窗口,每个窗口取算术平均(避免单步噪声):

| Window | metric | **v39b (α=0.5)** | **v39 (α=0.2)** | Δ (v39b − v39) |
|---|---|---:|---:|---:|
| 1–5 | chord/mu | 0.295 | 0.297 | ≈0 |
| 1–5 | tg_share | 0.252 | 0.288 | −0.036 |
| 1–5 | reward_on | 0.155 | 0.130 | +0.025 |
| 1–5 | succ_on | 0.014 | 0.007 | +0.007 |
| 6–15 | chord/mu | 0.194 | 0.229 | **−0.035** (v39b BC 更小) |
| 6–15 | reward_on | 0.273 | 0.262 | +0.011 |
| 16–30 | chord/mu | 0.196 | 0.190 | ≈0 |
| 16–30 | reward_on | 0.468 | 0.496 | −0.028 |
| 16–30 | succ_on | 0.028 | 0.028 | 0 |
| 31–50 | chord/mu | 0.168 | 0.146 | +0.022 (v39b BC 更大) |
| 31–50 | reward_on | 0.623 | 0.650 | −0.027 |
| 31–50 | succ_on | 0.082 | 0.126 | **−0.044** (v39 中段领先) |
| 51–75 | chord/mu | 0.113 | 0.089 | +0.024 |
| 51–75 | reward_on | 0.689 | 0.742 | **−0.053** (v39 此时仍领先) |
| 51–75 | succ_on | 0.197 | 0.295 | **−0.098** |
| 51–75 | kl_loss | 0.969 | 1.223 | −0.254 (v39b kl 更小) |
| **76–100** | **chord/mu** | **0.100** | **0.080** | **+0.020** |
| **76–100** | **reward_on** | **0.818** | **0.792** | **+0.026** |
| **76–100** | **succ_on** | **0.404** | **0.281** | **+0.123 (v39b 反超)** |
| 76–100 | kl_loss | 1.039 | 1.257 | −0.218 |
| 76–100 | grad_norm | 16.27 | 16.44 | ≈0 |

**关键观察 1**:v39b 在前 75 步(尤其 31–75 中段)**实际上落后** v39 5–10pp success。v39 中段曲线明显更陡(succ 6%→13%→30%),v39b 反而是慢热(3%→8%→20%)。这与 1.5B 上"快 EMA 噪声大不利收敛"的直觉一致。

**关键观察 2**:Val@50 的快照(v39b 16.5% vs v39 33.0%)与 train succ_on@31–50 (v39b 8.2% vs v39 12.6%) 一致——v39b 上半段确实是输的。

**关键观察 3**:**翻盘发生在 76–100 窗口**。v39b 在最后 25 步从 succ ≈ 0.20 跳到 0.40+(end-of-run 单步 0.60),而 v39 从 0.295 滑落到 0.281,**完全失去动量**。这就是 +29pp val@50→val@100 vs −1pp 的曲线差异。

**关键观察 4**:v39b 末段 μ ≈ 0.10,v39 末段 μ ≈ 0.075。**v39b 反而是 BC 更厚的那一个**——这与 task 文档里"α=0.5 退得快"的直觉相反。

---

## B. α=0.5 赢 α=0.2 的真实机制论证

我重新对比了 μ 轨迹的**响应特性**而不是"快慢"。两条曲线起点都是 0.30,floor 都是 0.05,disc_floor 都是 0.5。差异在 EMA 系数:

| Phase | v39b (α=0.5) μ 行为 | v39 (α=0.2) μ 行为 |
|---|---|---|
| step 1–10 | 快速跟踪 disc_acc ↑ 衰减到 ~0.20 | 缓慢衰减到 ~0.22 |
| step 10–30 | 进入"震荡平台"在 0.17–0.21 间反复(disc_acc 微波动) | 单调下降到 0.18,几乎无震荡 |
| step 50 后 | μ 在 0.07–0.13 间持续震荡(一旦 disc_acc 抖一下就把 μ 弹回 0.10+) | μ 单调收敛到 0.08 不再回弹 |

**真实机制不是"α=0.5 退得快",而是"α=0.5 让 μ 永远不死透"**:fast EMA 对 disc_acc 的 sub-batch 噪声敏感,任何一次 disc_acc 从 0.95 抖到 0.85,μ 就立刻从 valley 弹回 0.10–0.13,导致 BC gradient 阶段性回归,**不断校准 policy 不让其偏离教师 boilerplate**。slow EMA (α=0.2) 一旦决定 disc_acc 大致 ≥0.9 就把 μ 锁死在 valley,policy 没有 BC 校准回路,慢慢 drift。

这与"BC 越早退出 ⇒ GRPO 越自由"的简单论断相反。**WebShop 真正的机制是**:

1. **Policy 早期 KL 上升不可避免**(看 v39 kl_loss: 0.94→1.22→1.26 单调上升 vs v39b 0.72→0.97→1.04 显著低 0.2)。GRPO 在 webshop 上倾向"投机性长 query"(看 response_len 趋势),而 BC 是把它拉回正确 boilerplate (`search[X]`/`click[Y]` 的精确格式)的唯一锚。
2. **v39 锁死 μ 的代价**:policy KL 在 step 50 后失去外部约束,缓慢漂移到次优 mode (response_length 从 2400 漂到 2070,内容变短但 reward 不涨——典型的"模式坍缩到不需要长推理的 trick"),曲线就此卡住。
3. **v39b 的震荡 μ 是隐式的"BC 校准回路"**:每次 policy KL 微涨触发 disc_acc 微跌,fast EMA 立刻给一发 BC gradient 把 policy 拉回。这是**自适应 KL 控制**的副产品,而不是 BC schedule 本身的设计意图。

**所以 v39b 的赢面来自"fast EMA 把 BC 当 KL regularizer 用"**,这是 1.5B → 3B 的 scale-up 中没人预料的涌现行为。1.5B 时 policy capacity 小,KL drift 不严重,慢 EMA 就够;3B policy capacity 大、drift 倾向更强,需要 BC 校准回路。这条机制论证既能解释为什么 1.5B 时 α=0.2 赢、3B 时 α=0.5 赢,也能预测 7B 时 α=0.5 仍可能不够、需要更高 floor。

**支持证据**:
- v39b kl_loss 在 51–100 步比 v39 低 0.22(absolute),pg_loss 也更稳定。
- v39b grad_norm 末段不暴增(16.27 vs v39 16.44 持平,但 v39 中段 17.20 更高)——KL 校准让 grad 更平稳。
- v39b succ_on 末 5 步序列:0.37 / 0.54 / 0.25 / 0.54 / 0.60(单调向上但伴随震荡);v39 末 5 步:0.39 / 0.54 / 0.11 / 0.37 / 0.31(末步反而崩溃)。

---

## C. v39b 还差 LUFFY 4pp 的归因

LUFFY config 关键差异(`webshop_qwen3b_luffy.yaml` vs v39b):

| Knob | LUFFY | v39b | 影响 |
|---|---|---|---|
| `actor.teacher_policy_shaping_enable` | **true** | false | LUFFY 在 actor loss 内对教师 token 做 p/p_β shaping(常驻教师"地板") |
| `actor.teacher_policy_shaping_mode` | `p_div_p_beta` | n/a | 同上 |
| `actor.teacher_policy_shaping_beta` | 0.1 | n/a | shaping 强度 |
| `exp_manager.teacher_experience.policy_shaping.enable` | **true** | false | exp_manager 层面也开了 shaping |
| `actor.use_dr3` | false | true | LUFFY 没有 DR3(教师贡献"恒定地板") |
| `actor.use_chord` | false | true | LUFFY 没有 BC(完全靠 mix + shaping) |
| `n_teacher_rollouts_per_task` | 1 | 1 | **相同** |
| `max_trajectories_per_task` | 6 | 6 | **相同** |
| `algorithm.grpo.teacher_baseline_separation.enable` | true | true | **相同** |

**4 pp 差距的最可能归因(按概率排序)**:

1. **(60%) LUFFY 的 p/p_β shaping 提供了 v39b 缺失的"恒定教师地板"**。
   v39b 的 DR3 fade-out 把 teacher_grad_share 在 step 50 后压到 5%(实测末段 0.092);LUFFY 没有 fade-out 机制,教师梯度始终通过 shaping 公式作用,**等效于一个不衰减的强 prior**。WebShop 上 click[xxx] / search[term] 这类 rare action token 的 imitation 似乎需要长期注入,DR3 切得太早。

2. **(25%) v39b 的"间歇 BC"(μ 在 0.07–0.13 震荡)效果不如"平滑 shaping"**。
   p/p_β 是 token 级、连续、可微的修正;BC 是 sample 级、离散开关式的二次损失。在 webshop 这类 token 分布相对窄的任务里,token-level 修正信噪比更高。

3. **(10%) `chord_use_token_weighting=false` 是均匀 BC**。
   CHORD baseline (0410) 用 token_weighting=true 拿到 39% success,接近 v39b 的 45.5%。如果 v39b 也开 token_weighting,理论上能再 +2–3pp。

4. **(5%) 其他**:teacher_baseline_separation 都开了,n_teacher 都是 1,排除这些原因。

**关键否定**:**不是 n_teacher_rollouts 太少**——LUFFY 也是 1。**不是 mix_mode**——都是 rollout_level。**不是 max_trajectories**——都是 6。所以"v41_strong_teacher_mix"(把 n_teacher 从 1 提到 2)在不改 shaping 的前提下大概率不会过 LUFFY。

---

## D. v41 系列变体提议(2 个高信心 + 1 个低成本探索)

每个变体都基于 v39b (α=0.5),只动 1–2 knob,11 天预算允许 4–5 次试验。

### v41_psh(强烈推荐,高信心)

**机制论证**:在 v39b 基础上加 LUFFY 风格的 actor-side teacher policy shaping。这正是归因 (1) 的直接处方:DR3 提供"自适应衰减的额外贡献",p/p_β 提供"恒定地板"——两者叠加而非互斥。这才是真正的 DUET 全栈版本(Action Channel = DR3 + LUFFY shaping,State Channel 不变)。

**yaml diff(基于 `webshop_qwen3b_duet_v39b.yaml`)**:
```yaml
actor_rollout_ref:
  actor:
    teacher_policy_shaping_enable: true       # was false
    teacher_policy_shaping_mode: p_div_p_beta  # new
    teacher_policy_shaping_beta: 0.1           # new
exp_manager:
  teacher_experience:
    policy_shaping:
      enable: true                              # was false
      mode: p_div_p_beta
      beta: 0.1
```

**预测**: success@100 = **51 ± 3 %**(信心 H)。理由:LUFFY 49.5% 是"shaping only";v39b 45.5% 是"DR3+BC+SC only";两者贡献近似正交,叠加应该 +3–5 pp。

### v41_token_weighting(低风险增量,中等信心)

**机制论证**:CHORD 0410 baseline (39%) 用 token_weighting=true,而 v39b 用 false。token_weighting 让 BC 在 high-uncertainty token (低教师 prob) 上权重更大,与 fast EMA 的"震荡 BC 当 KL regularizer"逻辑天然契合——如果 BC 阶段性激活,那就让它打在最该被校准的 token 上。

**yaml diff**:
```yaml
actor_rollout_ref:
  actor:
    chord_use_token_weighting: true           # was false
```

**预测**: success@100 = **47 ± 3 %**(信心 M)。理由:CHORD 0410 (39%) → CHORD with adaptive μ (45.5%) 已经吃掉大部分 gain,token_weighting 边际可能只剩 +1–3 pp。

### v41_psh_high_floor(融合两大 trick,中等信心)

**机制论证**:同时开 v41_psh 的 shaping,把 `chord_mu_d_floor` 从 0.5 提到 0.6,让 μ 不会过早在 disc_acc=0.5 时切到 valley(给 BC 更多生命周期),同时 shaping 提供平滑地板。是 v41_psh 的"扩展版"。

**yaml diff(在 v41_psh 基础上加)**:
```yaml
actor_rollout_ref:
  actor:
    chord_mu_d_floor: 0.6   # was 0.5
```

**预测**: success@100 = **50 ± 4 %**(信心 M)。如果 v41_psh 已经吃掉地板增益,这个变体可能 wash;但如果 v41_psh 显示 BC 仍偏少,这个能补上 1–2 pp。**只在 v41_psh 跑完且 succ < 50 % 时启动**。

### 不推荐的变体

- **v41_strong_teacher_mix (n_teacher 1→2)**:LUFFY 也是 n=1,所以这不是瓶颈。Cost 高(1.5x rollout time)收益不明。
- **v41_higher_peak (μ_peak 0.3→0.5)**:v39b 早段 μ 已 0.30,实测 succ_on 1–5 仅 1.4%——不是 BC 不够强,是早期 RL signal 太弱。提 peak 只会让早期更慢。
- **v41_grpo_teacher_baseline_off**:teacher_baseline_separation 是 DUET 的核心稳定性设计,关掉它意味着教师 reward 主导 advantage normalization。失败概率高,不值得花 3.5h。

**优先级**: v41_psh (must-run) → v41_token_weighting (parallel if GPU 空闲) → v41_psh_high_floor (conditional)。

---

## E. v40_psh / v40_strong_bc 是否 abort?

**结论:abort 两者,改跑 v41_psh + v41_token_weighting**。

**论证**:
- v40_psh 的 base 是 v39 (α=0.2),v39 在新真相下被证明 succ@100 = 32% 卡死(76–100 窗口已 stagnant、kl_loss 1.26 还在涨)。在这个 base 上加 shaping 大概率最多到 38–42 %,**不会过 v39b 的 45.5%**,**几乎肯定不过 LUFFY 49.5%**。Cost 3.5h 换不到 paper 价值。
- v40_strong_bc 同理:α=0.2 已经把 μ 锁死在 0.075,再加强 BC 也是在死掉的回路上加压。机制上不可能比 v39b 更好。
- v41_psh 是 v40_psh 的"正确 base"版本,跑它直接拿到我们想验证的"DR3 + LUFFY shaping"叠加效应,paper story 也更干净。

**唯一保留 v40_psh 的理由**:如果实验调度系统已经 queue 且改起来麻烦,可以让它跑——把它当作"v39 base 上的 ablation,证明 base 选错"——但**优先 GPU 资源给 v41_psh**。如果只能二选一,**砍 v40 系列,跑 v41 系列**。

---

## 总结(给项目 lead 的一句话)

> 3B WebShop 上 α=0.5 赢 α=0.2 不是因为 BC 退得快,而是因为 fast EMA 让 μ 在 valley 附近震荡,等效于一个隐式 KL regularizer,防止 policy 在 step 50 后 drift——这是 1.5B 没观察到的 scale-up 涌现现象。还差 LUFFY 4pp 主要是因为我们没开 actor-side `teacher_policy_shaping`(常驻教师地板),v41_psh 把它叠回去预计能跨过 50%。建议 abort v40_psh / v40_strong_bc(base 选错),改跑 v41_psh + v41_token_weighting。

---

**附录**:本报告用到的原始数据来自:
- `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/logs/webshop_qwen3b_duet_v39b.log` (100 steps)
- `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/logs/webshop_qwen3b_duet_v39.log` (100 steps)
- `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v39b.yaml`
- `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v39.yaml`
- `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/config/duet_paper_experiments_configs/webshop/webshop_qwen3b_luffy.yaml`
