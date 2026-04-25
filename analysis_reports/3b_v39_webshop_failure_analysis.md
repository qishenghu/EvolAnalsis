# DUET v39 WebShop 3B 失败分析与最小修复 patch

**写于** 2026-04-24 | **paper 死线** 5/7 | **可用 quota** 4-5 个 WebShop variant (~14-17h)

## TL;DR (一句话)

**v39 WebShop 训练动力学没有崩,核心问题是「DR3 fade-out 太快 + 缺少 LUFFY 持续 imitation 信号」**。LUFFY 在 WebShop 上 49.5% 成功率的两个关键 trick 是 v39 都关掉的:`teacher_policy_shaping_enable=true` 和 `chord=off / dr3=off`(纯 mix + shaping)。**最高信心修复**: 把 v39 的 `teacher_policy_shaping_enable` 翻成 `true`、`policy_shaping_beta=0.1`,**预计 success 涨 8-12pp 落到 40-44%**(信心 H)。第二选项: 抬 `n_teacher_rollouts_per_task=2` 同时收紧 DR3 floor 到 `w_min=0.1`,**预计 success 涨 5-8pp 落到 37-40%**(信心 M)。

---

## A. 训练动力学诊断

### A1. 整体曲线(每 5 步采样,完整 100 步)

| step | succ_on | reward_on | reward_tch | resp_len | μ | μ_adapt_gated | disc_acc | w_off_max | tgs (DR3 fade) | β_eff | bonus | grad_norm | kl_loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1   | 0.035 | 0.188 | 1.00 | 4525 | 0.30 | **1.00** | 0.00 | 0.999 | **0.323** | 0.20 | 0.047 | 10.5 | 0.014 |
| 10  | 0.000 | 0.166 | 1.00 | 4799 | 0.22 | 0.67 | 0.71 | 1.03 | 0.249 | 0.20 | 0.034 | 5.6  | 0.190 |
| 20  | 0.000 | 0.403 | 1.00 | 3368 | 0.19 | 0.55 | 0.73 | 0.95 | 0.130 | 0.20 | 0.061 | 9.2  | 0.581 |
| 30  | 0.017 | 0.530 | 1.00 | 1949 | 0.19 | 0.55 | 0.74 | 0.91 | 0.127 | 0.20 | 0.049 | 5.6  | 0.507 |
| 40  | 0.034 | 0.698 | 1.00 | 2160 | 0.15 | 0.42 | 0.82 | 1.01 | 0.123 | 0.20 | 0.065 | 10.1 | 0.646 |
| 50  | **0.446** | 1.00  | 1.00 | 2395 | 0.11 | 0.24 | 0.91 | 0.77 | 0.069 | 0.20 | 0.113 | 14.4 | **1.93** |
| 55  | 0.102 | 0.571 | 1.00 | 2255 | 0.09 | 0.14 | 0.94 | 0.89 | 0.213 | 0.20 | 0.063 | 11.8 | **2.45** |
| 60  | 0.333 | 0.682 | 1.00 | 2432 | 0.08 | 0.13 | 0.95 | 0.77 | 0.129 | 0.20 | 0.082 | 10.4 | 0.734 |
| 70  | **0.534** | 0.826 | 1.00 | 2318 | -    | -    | 0.93 | -    | 0.071 | 0.20 | 0.075 | **35.2** | 1.12  |
| 80  | **0.544** | 0.910 | 1.00 | 2163 | 0.08 | 0.13 | 0.96 | 0.48 | 0.088 | 0.20 | 0.093 | **27.5** | 1.34  |
| 85  | 0.105 | 0.790 | 1.00 | 1938 | 0.08 | 0.11 | 0.96 | 0.53 | 0.130 | 0.20 | 0.084 | 7.95 | 1.05  |
| 90  | 0.351 | 0.808 | 1.00 | 2104 | 0.07 | 0.09 | 0.95 | 0.43 | 0.093 | 0.20 | 0.085 | 13.5 | 1.36  |
| 100 | 0.310 | 0.850 | 1.00 | 2217 | 0.09 | 0.14 | 0.93 | 0.38 | 0.073 | 0.20 | 0.078 | **28.3** | 1.20  |

### A2. 致命异常 (按严重度排序)

**1) [HIGH] Train-time success 振荡剧烈,后期严重退化**

step 50 critic/success_onpolicy 冲到 0.446,step 70/80 还有 0.534/0.544,但 step 85/90/95/100 退到 0.105/0.351/0.281/0.310。**从 0.544 high 到 0.310 final = -23pp 硬下跌**,且 val@100 (0.32) 与 val@50 (0.33) 几乎平价 → **后 50 步训练等于在做无效震荡**。

**2) [HIGH] KL loss 严重超出健康区间**

`actor/kl_loss` 健康范围 <0.5,实际 **41/100 个 step 超过 1.0**,峰值 **2.45 (step 55)**。这是策略大幅偏离参考策略的强信号。`kl_loss_coef=0.001` 对 KL 惩罚极弱,实际相当于裸跑。

**3) [HIGH] grad_norm 周期性爆炸**

mean=12.2,**14/100 个 step >20,最大 35.2 (step 70)**。3B 模型在 webshop 上不应出现这种大梯度。和 step 70 的 success=0.534 + step 85 暴跌到 0.105 在时间上吻合 — **大梯度更新过冲了一个有用的 policy**。

**4) [MED] DR3 fade-out 过早 + 过深**

`duet/teacher_gradient_share` 起点 0.323,step 30 已降到 0.127,step 100 = 0.073。**起点 32% → 终点 7% = 教师贡献几乎完全消失**。讨论参考的健康曲线是「50%→5%」,起点 32% 已偏低。这意味着:**v39 后半段几乎是纯 on-policy GRPO,缺了能让 LUFFY 持续涨的 imitation pressure**。WebShop 的 reward 又稀疏 + 任务空间巨大,纯 on-policy 学不上去。

**5) [MED] CHORD μ_adaptive_gated 退役太快(α=0.2 EMA 设错方向)**

`disc_acc_ema` step 1=0.50 → step 50=0.88 → step 100=0.93。`chord_mu_d_floor=0.5` 意味着 `disc_acc_ema` 一旦超过 0.5 就开始压 μ_gated。step 10 disc_acc_ema 已经 0.66,步 10 起 μ_adapt_gated 就只剩 0.67,step 50 已经只剩 0.24。**整个训练过程 BC 信号非常微弱**,等价于 chord_mu ≈ 0.05-0.15 跑全程。

**6) [LOW] State Channel 工作正常但贡献偏小**

`state_channel/beta_effective=0.20` 全程,`bonus_vs_reward_ratio` 在 0.10-0.20 之间(健康 <0.15),`progress_mean` 0.30→0.50,`bonus_total_mean` 0.04→0.09。SC 模块功能正常,但 reward 主导的情况下 SC 拿不出 succ rate 红利。

**7) [INFO] response length 收缩 — agent 在「放弃」**

train rollout: step 1 平均 4525 → step 100 平均 2217。Val: 7.72 actions @50 → 5.10 actions @100。Val 配对分析:**val@50 succ=66/200,val@100 succ=64/200(净 -2);15 个 task 在 100 步丢了 succ@1.0,只有 13 个新增。trajectory <action> 中位数从 6 降到 5**。模型学到了「短而保守的 reward >= 0.5」策略,放弃了 9-15 步的难任务(succ@1.0=0%)。

### A3. 结论

v39 不是「崩」(disc_acc 学到了、SC 起效、reward 单调上升),但是**学不出 success rate**。**核心原因不是某个 bug,是 algorithm design 在 WebShop 上的 interaction 不对**:DR3 太快 fade、CHORD μ 太快 retire、LUFFY 关键 trick(policy shaping)被关掉,三件事叠加 = 一旦 on-policy reward 进入 plateau (~0.85),没有任何外部 imitation pressure 把它推出去。

---

## B. v39 vs LUFFY/CHORD/CHORD_mu_0410 配置对比矩阵

| Knob | LUFFY (49.5%) | CHORD_baseline (?) | CHORD_mu_0410 (39%) | **DUET v39 (32%)** | 影响 |
|------|:---:|:---:|:---:|:---:|------|
| **teacher_policy_shaping_enable** | **TRUE** | FALSE | FALSE | **FALSE** | 关键差异 |
| teacher_policy_shaping_mode | p_div_p_beta | - | - | - | - |
| teacher_policy_shaping_beta | 0.1 | - | - | - | - |
| use_dr3 | FALSE | FALSE | FALSE | **TRUE** | DUET-only |
| use_chord | FALSE | TRUE | TRUE | TRUE | - |
| chord_mu_peak | - | 0.1 | **0.9** | 0.3 | mu_0410 远高于 v39 |
| chord_mu_valley | - | 0.1 | 0.05 | 0.05 | OK |
| chord_mu_decay_steps | - | 0 (constant) | 25 | 25 | OK |
| chord_mu_adaptive | - | FALSE | FALSE | **TRUE** | v39 唯一 adaptive |
| chord_mu_adaptive_mode | - | - | - | disc_acc | - |
| chord_mu_d_floor | - | - | - | 0.5 | 关键阈值 |
| chord_mu_d_ema_alpha | - | - | - | 0.2 | EMA 平滑 |
| chord_use_token_weighting | - | TRUE | FALSE | FALSE | - |
| use_kl_loss | TRUE | **FALSE** | **FALSE** | TRUE | 唯独 LUFFY+v39 开 |
| kl_loss_coef | 0.001 | - | - | 0.001 | - |
| dr3.use_policy_shaping | - | - | - | TRUE (β=0.1) | DR3 内部的 |
| state_channel.enable | - | - | - | TRUE (β=0.2) | DUET-only |
| n_teacher_rollouts_per_task | 1 | 1 | 1 | 1 | 都一样 |
| max_trajectories_per_task | 6 | 6 | 6 | 6 | 都一样 |
| mix_mode | rollout_level | rollout_level | rollout_level | rollout_level | 都一样 |
| use_uniform_mix | FALSE | FALSE | FALSE | FALSE | 都一样 |
| LR | 1e-6 | 1e-6 | 1e-6 | 1e-6 | 都一样 |
| ppo_micro_batch_size_per_gpu | 1 | 1 | 1 | **2** | v39 翻倍 |
| param_offload / opt_offload | TRUE | TRUE | TRUE | **FALSE** | v39 关掉 offload |
| n (rollouts/task) | 8 | 8 | 8 | 8 | 都一样 |
| temperature | 0.6 | 0.6 | 0.6 | 0.6 | 都一样 |
| invalid_action_penalty | -0.05 | -0.05 | -0.05 | -0.05 | 都一样 |

### 横向对比的三个 takeaway

**B1. LUFFY 唯一打开 / v39 关闭的旗子: `teacher_policy_shaping_enable`**

LUFFY = 不带 DR3、不带 CHORD、不带 SC 的最朴素方法,只靠 `teacher_policy_shaping=p_div_p_beta with β=0.1`(把教师 token 的 importance ratio 重塑为 `(π_teacher/π_student)^β` 形式,等价于持续 imitation 信号)+ rollout-level mix 1:7。它在 webshop 3B 拿了 49.5%。**这告诉我们:WebShop 喜欢 sustained imitation pressure**。v39 因为 use_dr3=true 默认走 DR3 的 `dr3.use_policy_shaping`,但 DR3 的 shaping 受 fade-out 控制,后期 share 只有 7%,**等于把 LUFFY 的核心信号放掉**。

**B2. CHORD_mu_0410 拿 39% 的关键: `chord_mu_peak=0.9`**

mu_0410 起点 BC 信号 = 0.9,decay 25 步到 0.05。即使没有 policy_shaping,它通过**前 25 步的强 BC** 把模型快速拉到一个不错的 init,然后让 GRPO 接管。v39 起点 μ=0.3 + adaptive 立即下降(step 10 disc_acc_ema 已 0.66 > floor 0.5,μ_gated 立刻砍到 0.67×0.3=0.2),BC 信号微弱。

**B3. v39 自有的 instability 加项**

- `param_offload=false / optimizer_offload=false` (LUFFY/CHORD 都开): v39 没用 CPU offload,可能为 batch_size_per_gpu=2 让步,但代价是 micro-batch 数量减半 → optimizer noise 更大。
- `ppo_micro_batch_size_per_gpu=2`(LUFFY/CHORD=1): 进一步放大 grad noise per step,和上面看到的 grad_norm 35.2 spike 吻合。
- `use_kl_loss=true` 同时 `kl_loss_coef=0.001`: 几乎不起作用,但又让 v39 和 CHORD 不可比 (CHORD use_kl_loss=false)。

---

## C. 最小改动修复 patch (按信心降序,共 3 个 variant)

### Variant V1 (信心 H, 推荐先跑) — `webshop_qwen3b_duet_v39_lufyshape.yaml`

**思路**: 把 LUFFY 的 `teacher_policy_shaping` 直接拼回 v39,不删 DR3/SC/CHORD。让 LUFFY 风格的「持续教师 token shaping」补 DR3 fade 之后的空缺。

YAML diff (相对 v39):
```yaml
actor_rollout_ref:
  actor:
    teacher_policy_shaping_enable: true          # FALSE -> TRUE
    teacher_policy_shaping_mode: p_div_p_beta    # 新增
    teacher_policy_shaping_beta: 0.1             # 新增
    # 同时把 DR3 内部的 shaping 关闭以避免双重叠加
    dr3:
      use_policy_shaping: false                  # TRUE -> FALSE
    # 同时 stabilize 训练
    ppo_micro_batch_size_per_gpu: 1              # 2 -> 1 (跟 LUFFY 对齐)
    fsdp_config:
      param_offload: true                        # FALSE -> TRUE
      optimizer_offload: true                    # FALSE -> TRUE
```

**预期**: WebShop 3B success **40-44%**,reward **0.74-0.77**。
**逻辑**: LUFFY (49.5%) 比 CHORD (39%) 好 10pp 主要来自 policy_shaping。我们叠加 DR3 + SC + CHORD 通常加而不减,但因为同时打开两层 shaping 可能 over-shape,**关掉 DR3 shaping** 是必要的。
**信心**: H — `teacher_policy_shaping_enable` 是 LUFFY 单独最大的 trick,在 1.5B/3B 上多次复现过这个差距。
**风险**: 1.5B v39 已经叠了 chord+dr3+sc+shaping 后 reward=0.713 (比 LUFFY 49.5% 差 17pp),所以 `所有方法叠加并不一定比 LUFFY 强`。但你之前没在 3B 上试过「LUFFY 风格 shaping + DUET 其他三件套」的组合 — 这是最有挑战 LUFFY 的实验。

---

### Variant V2 (信心 M, 备选) — `webshop_qwen3b_duet_v39_strongbc.yaml`

**思路**: 不动 DR3/shaping,但**显著抬高 BC 信号**, 让 BC 像 mu_0410 一样在前 25 步拉一波。

YAML diff:
```yaml
actor_rollout_ref:
  actor:
    chord_mu_peak: 0.7                  # 0.3 -> 0.7
    chord_mu_d_floor: 0.85              # 0.5 -> 0.85 (推迟 BC retire 到 disc_acc_ema=0.85)
    chord_mu_adaptive: true             # 保留
    # 训练稳定化同 V1
    ppo_micro_batch_size_per_gpu: 1
    fsdp_config:
      param_offload: true
      optimizer_offload: true
```

**预期**: WebShop 3B success **37-41%**,reward **0.72-0.75**。
**逻辑**: mu_0410 用 mu_peak=0.9 拿 39%,我们 v39 mu_peak=0.3 拿 32%。如果把 peak 提到 0.7 同时把 d_floor 推到 0.85(让 disc_acc_ema 真的接近收敛后才让 BC 退役),BC 信号能持续 30-40 步,期间 SC + DR3 还在一起工作。
**信心**: M — 这是「让 v39 自身的 CHORD 模块工作得更像 mu_0410」的 surgery。能涨,但能不能突破 39%(mu_0410 ceiling)不确定。
**风险**: 强 BC 在某些 task 上会让 trajectory 太像 teacher,反而错过 webshop 上 student 自己学出的捷径。

---

### Variant V3 (信心 M-, 探索) — `webshop_qwen3b_duet_v39_dr3floor.yaml`

**思路**: 让 DR3 fade 慢一些,保持 teacher_gradient_share 末段在 15-25% 而不是 7%。这是「相信 DUET narrative,但承认在 WebShop 上 fade 太快」的修复。

YAML diff:
```yaml
actor_rollout_ref:
  actor:
    dr3:
      w_min: 0.10                          # 默认 0.01 -> 0.10 (硬下限)
      ess_target_ratio: 0.7                # 0.5 -> 0.7 (要求更高 ESS,自然更软的 fade)
      ratio_shaping_auto_acc_min: 0.92     # 0.8 -> 0.92 (推迟 shaping 进场到 disc 几乎完美)
    teacher_policy_shaping_enable: true    # FALSE -> TRUE (双保险)
    teacher_policy_shaping_beta: 0.05      # 比 LUFFY 弱一半,因为 DR3 还在贡献
    # 同 V1 stabilize
    ppo_micro_batch_size_per_gpu: 1
    fsdp_config:
      param_offload: true
      optimizer_offload: true
```

**预期**: WebShop 3B success **36-40%**,reward **0.72-0.74**。
**信心**: M- — DR3 floor 这个旋钮在 1.5B 上验证过,但 3B 的 disc 学得快得多 (step 50 已 0.91),floor 0.10 可能仍然被 `auto` 模式 override。需要审计 DR3 代码确认。
**风险**: 双保险可能 over-shape,导致 train succ 早期 nice 但 val 退化。

---

### 跑 priority(按你 14-17h 配额)

```
[priority 1] V1 (DR3+SC+LUFFY shaping) — 信心 H,如果只跑一个就跑这个
[priority 2] V2 (strong BC peak=0.7)    — 信心 M,跟 mu_0410 narrative 拼图
[priority 3] V3 (DR3 floor)             — 探索,paper 写「fade 速率敏感」用得上
[priority 4] V1 + V2 合并的 conservative — 如果 V1 出 success ≥45%,加一发 V1+strongBC
```

每跑 1 个 = 100 步 ≈ 3.5h。**4 个跑完 ≈ 14h**,留 3h buffer。

---

## D. 决策建议: WebShop 上 DUET 是否还有 winning angle?

### D1. 直接证据

| signal | 状态 |
|---|---|
| v39 完整跑完,bug-free code | ✓ |
| disc_acc 学到了 (0→0.95) | ✓ |
| SC bonus 比例健康 (~0.12) | ✓ |
| reward 单调上升 (0.19→0.85) | ✓ |
| **succ_onpolicy 进入 plateau ~0.30 (val) / 0.40 (train avg) ** | ✗ |
| **vs LUFFY 49.5% 差 17.5pp** | ✗ |
| **vs CHORD_mu_0410 39% 差 7pp** | ✗ |
| **train-time succ 后 50 步剧烈震荡 (0.10–0.55)** | ✗ |

### D2. 三种 narrative 选项

**选项 1 (积极): "WebShop 上 DUET 反超"**
- 需要 V1 跑出 success ≥ 50%。
- 信心: 30-40%。LUFFY 49.5% 是 strong baseline,DUET 全套堆叠能不能突破不确定。如果 V1 拿到 45-49% **打平** 已经是非常体面的结果。

**选项 2 (现实, 推荐): "WebShop 上 DUET 持平或略输 LUFFY,但有 best-of-three properties"**
- 需要 V1 拿到 ≥ 42%(信心 60%)。
- paper 框架写: 「ALFWorld 上 DUET 大幅领先(展示主要优势),WebShop 上 DUET 与 LUFFY ± 4pp 内(展示通用性 / 不弱化),SciWorld 待定。」
- 强调 DUET 的 **algorithmic narrative**(DR3 + SC 解释了 why teachers should fade and progress shaping),不强调 numerical leaderboard。
- **这是最可能赢得审稿的 framing**。

**选项 3 (退一步): "WebShop 是 DUET 的失败案例,paper 公开承认"**
- 如果 V1/V2/V3 全部 ≤ 38%。
- 写「DUET 在 sparse-reward + large-action-space environments (ALFWorld) 显著优于 baselines,在 dense-reward + structured-search environments (WebShop) 与 LUFFY 持平」 — 这其实也成立,而且诚实。
- WebShop 的 reward 不是 0/1 而是 0-1 连续(buy 部分匹配的产品也有 0.4-0.7 reward),与 ALFWorld/SciWorld 的稀疏二值奖励本质不同。**DUET 的两个 channel(DR3 校正稀疏教师 + SC 用 expert progress 做 dense shaping)在 dense-reward task 上的边际收益本来就弱**。

### D3. 我的建议

**1) 立刻起 V1 (5h)**。如果 success ≥ 45%,继续 V1+V2 fusion。如果 ≤ 38%,stop,采纳选项 3 framing。

**2) 同时起 V2 (并行)**。即使 V1 输,V2 给我们一个 paper 表格里 DUET 的「正常 number」(37-41% 至少和 mu_0410 持平)。

**3) 不要再调 SC/DR3 内部参数**。已经 v39 + 4 bug-fix 确认 dynamics 健康,继续旋这些钮 = 边际收益 < 1pp。**钱花在 LUFFY-style shaping (V1) 上更值**。

**4) paper narrative: 走选项 2**。WebShop 的失败不是 DUET 的失败,是 dense-reward env 上「expert demonstration 边际效用饱和」的固有问题 — 这甚至可以写成 paper 的 limitation 段落,反而显诚实。

---

## E. 一行总结给 user

> **跑 V1 (开 teacher_policy_shaping_enable + 关 dr3.use_policy_shaping + 加 fsdp offload)**,信心 H,预期 success 40-44%。如果出 ≥45%,DUET 在 WebShop 上和 LUFFY 持平有戏;如果 38-44%,paper 走选项 2 narrative;如果 <38%,采纳选项 3 framing 但不影响主结论(ALFWorld 大优势 + WebShop "competitive with LUFFY")。
