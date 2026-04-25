# DUET WebShop 3B Variant 设计 — v39 后续

**作者**: 算法工程师
**日期**: 2026-04-24
**截稿背景**: paper 死线 5/7,WebShop 3B 上 DUET v39 输 LUFFY 17.5pp。还能跑 4-5 个 variant(总 14-17h)。

---

## 0. TL;DR(给赶时间的人)

按"信心 × 影响 / 时间成本"排序,推荐 **5 个** WebShop 3B variant,优先级从高到低:

| 优先级 | 名字 | 核心改动 | 假设 | 预测 success | 信心 |
|--------|------|---------|------|-------------|------|
| **P0** | `v40_psh` | 关 DR3 ratio_shaping,回归 LUFFY 风格的 actor-level `teacher_policy_shaping_enable: true` + DR3 仅做 BC 调度 | DUET 在 WebShop 上把 LUFFY 的关键武器(p/p_β)关了 | 42-48% | **High** |
| **P0** | `v40_strong_bc` | μ_peak 0.3→0.6,μ_valley 0.05→0.15,decay_steps 25→60,关 DR3 ratio shaping | BC 在 WebShop 退得太早 + DR3 提前削弱 teacher | 38-44% | **High** |
| P1 | `v40_uniform_mix` | `use_uniform_mix: true` + `n_teacher_rollouts_per_task: 2` | LUFFY 实际给 teacher 的 effective weight 高于 1/8 | 36-42% | Med |
| P1 | `v40_gap_gate` | `dr3.gap_gate_enable: true`(ALFWorld 已开,WebShop 没开) | gap-gate 防止 DR3 早期错杀 teacher | 35-40% | Med |
| P2 | `v40_dapo` | 开 DAPO dynamic_sampling + clip-higher 异步,n=12 | rollout n=8 太少 → all-zero group 占比高 → 梯度信号差 | 35-41% | Med-Low |

**强烈推荐先跑 P0 两条**(7h),其中 `v40_psh` 是机制论证最硬的一个。如果二者都失败再考虑 P1/P2。

---

## 1. v39 完整参数空间审计

读 `config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v39.yaml` (`config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v39.yaml`),v39/v39b/v39c 已扫的轴 vs 没动过的轴一目了然:

### 1.1 已扫轴(v39/v39b/v39c 都在动的)

| Knob | v39 值 | v39b 改动 | v39c 改动 |
|------|--------|-----------|-----------|
| `chord_mu_d_ema_alpha` | 0.2 | **0.5** | 0.2 (保留) |
| `chord_mu_valley` | 0.05 | 0.05 | **0.10** |
| `kl_loss_coef` | 0.001 | 0.001 | **0.003** |

也就是说, **过去的迭代只在三轴(EMA 速度 / BC 下界 / KL 强度)做过微调**。

### 1.2 没动过的高杠杆轴(完整清单)

下表用 v39 的当前值。列出"机制作用"和"在 WebShop 上的理论敏感性"。**WebShop sensitivity** 是我对该 knob 在 WebShop 任务结构(query→search→item→buy 4 阶段、稀疏 reward、轨迹中等长度)下的影响判断。

#### A. BC channel (`chord_mu_*`)

| Knob | v39 | 已搜索? | WebShop 敏感性 |
|------|-----|--------|---------------|
| `chord_mu_peak` | 0.3 | 没搜 | **高**:CHORD baseline 是 μ=0.1 常量+token weighting=true 拿到 39%;`chord_mu_0410` 用 μ=0.9 + token weighting=false 也拿到 39%。说明 **WebShop 对 μ 上界容忍度极宽**,我们 0.3 偏小 |
| `chord_mu_valley` | 0.05 | v39c 试 0.10 | **中**:决定后期还有多少 BC 残余 |
| `chord_mu_decay_steps` | 25 | 没搜 | **高**:WebShop 800 task / batch 8 / n=8 → 总 step 数 ~100。25 步 decay 意味着 **第 25% 训练就退到 valley**。LUFFY 没有 decay,CHORD 也没有 decay |
| `chord_mu_d_floor` | 0.5 | 1.5B v39d 试 0.3 | 中 |
| `chord_use_token_weighting` | false | 没搜 | **中**:CHORD WebShop 默认 true |
| `chord_mu_warmup_steps` | 0 | 没搜 | 低 |

#### B. DR3 channel (`dr3.*`)

| Knob | v39 | 已搜索? | WebShop 敏感性 |
|------|-----|--------|---------------|
| `dr3.ratio_shaping_mode` | auto | 没搜 | **极高**:auto 在 acc>0.8 / buf>=512 / ESS>=16 触发 ratio shaping。**触发后会给 teacher 样本应用一个 β=0.1 的 p/p_β 比率 shaping,同时 DR3 已经在做 closed-form fade-out**。两条腿都在削 teacher → "double fade-out" |
| `dr3.policy_shaping_beta` | 0.1 | 没搜 | 高 |
| `dr3.gap_gate_enable` | **false** | 没搜 | **极高**:ALFWorld DUET 把这个开了,WebShop 没开。gap_gate 是"在 disc_acc 没起来前不要 shaping"的安全网 |
| `dr3.use_policy_shaping` | true | 没搜 | 高:这个开启了 DR3 的 hybrid 路径,把 actor-level `teacher_policy_shaping_enable: false` 给 override 了 |
| `dr3.disc_temperature` | 1.0 | 没搜 | 中:温度高 → w_hat 平坦 → 更慢的 fade-out |
| `dr3.w_hat_ema_alpha` | 0.3 | 没搜 | 中 |
| `dr3.clip_max` | 5.0 | 没搜 | 低 |
| `dr3.dual_enable` | true | 没搜 | 中 |
| `dr3.ess_target_ratio` | 0.5 | 没搜 | 中 |
| `dr3.feature_mode` | v3_aug | 没搜 | 低(大改动) |
| `dr3.disc_steps_per_call` | 2 | 没搜 | 低 |

#### C. State Channel (`state_channel.*`)

| Knob | v39 | 已搜索? | WebShop 敏感性 |
|------|-----|--------|---------------|
| `state_channel.beta` | 0.2 | 没搜 | 中 |
| `state_channel.beta_decay` | false | 没搜 | 低 |
| `state_channel.match_mode` | attribute_aware | 没搜 | **中**:WebShop attribute_aware 是已实现的精确路径,改成 hash 会丢精度。但 `step_level.eta` 没搜过 |
| `state_channel.step_level.enable` | true | 没搜 | 中 |
| `state_channel.step_level.eta` | 0.05 | 没搜 | 中 |
| `state_channel.exclude_teacher` | true | invariant | (设计不变量,不动) |

#### D. Teacher mix (`teacher_experience.*`)

| Knob | v39 | 已搜索? | WebShop 敏感性 |
|------|-----|--------|---------------|
| `mix_mode` | rollout_level | 没搜 | **高**:可选 step_level / interleave / hybrid |
| `n_teacher_rollouts_per_task` | 1 | 没搜 | **极高**:n=8 时 teacher_ratio=1/8。LUFFY 论文是同样设置,但 LUFFY 用 actor-level p/p_β shaping 会**自适应 up-weight teacher 上有用的 token**,DUET 用 DR3 反而会 down-weight。如果 DR3 失效 → DUET 实际 teacher 信号 < LUFFY |
| `max_trajectories_per_task` | 6 | 没搜 | 低 |
| `select_mode` | random | 没搜 | 低 |
| `policy_shaping.enable` (teacher 那一套,actor 顶层是 false) | false | 没搜 | **极高**:LUFFY 这个是 true,DUET 这个是 false |
| `adaptive_weight.enable` | false | 没搜 | 中 |

#### E. Actor / GRPO

| Knob | v39 | 已搜索? | WebShop 敏感性 |
|------|-----|--------|---------------|
| `clip_ratio_high` | 0.28 | 没搜 | 中:DAPO 提倡 0.28(已设),想要更激进 0.32 |
| `clip_ratio_low` | 0.2 (default) | 没搜 | 低 |
| `entropy_coeff` | 0 | 没搜 | 中:WebShop 探索性低,加一点 1e-3 可能有用 |
| `kl_loss_type` | low_var_kl | 没搜 | 低 |
| `use_dapo` | false | 没搜 | **中**:DAPO 的 dynamic_sampling 对 WebShop 这种稀疏 reward 任务非常对症(过滤 all-zero group) |
| `use_uniform_mix` | false | 没搜 | **高**:LUFFY 论文也是 false,但 1.5B 上 v23/v25 都试过 false。3B 上没试过 true |

#### F. Rollout

| Knob | v39 | 已搜索? | WebShop 敏感性 |
|------|-----|--------|---------------|
| `rollout.n` | 8 | 没搜 | **高**:WebShop reward 稀疏,n=8 + teacher=1 意味着 **on-policy = 7**,如果 7 个全 0 → group baseline=0,advantage=0,GRPO 死信号。提到 12 可能差异大 |
| `rollout.temperature` | 0.6 | 没搜 | 中 |

---

## 2. 横向对照 LUFFY / CHORD

读了 `webshop_3b_luffy.yaml`、`webshop_3b_chord.yaml`、`webshop_3b_chord_mu_0410.yaml`(用户验证 39%)。**关键差异**:

### 2.1 LUFFY (49.5%) vs DUET v39 (32%) 的 7 处差异

| 项目 | LUFFY | DUET v39 | 影响 |
|------|-------|----------|------|
| `actor.use_dr3` | **false** | true | DUET 多了 DR3 通道 |
| `actor.use_chord` | **false** | true | DUET 多了 BC 通道 |
| `actor.teacher_policy_shaping_enable` | **true** | false | **LUFFY 的核心武器** |
| `actor.teacher_policy_shaping_mode` | p_div_p_beta | (off) | LUFFY 用 p/(p+β) 蒸馏 |
| `actor.teacher_policy_shaping_beta` | 0.1 | (off) | |
| `teacher_experience.policy_shaping.enable` | true | false | LUFFY 双开(actor + teacher_exp) |
| `state_channel` | (无) | enable=true, β=0.2 | DUET 多 SC |

**机制论证**:LUFFY 在 WebShop 49.5% 几乎完全靠 `p/(p+β)` 这个 importance-weight 重塑。它的效果是: teacher 上模型已经学会的 token (高 p) 的梯度被压平,模型不会的 token (低 p) 的梯度被放大,**自适应让 teacher 集中在"教学价值高"的 token 上**。

DUET v39 的设计哲学是"用 DR3 替代 p/p_β 做更精细的 importance reweighting":
- DR3 通过判别器学 ρ̂ ≈ p_θ / p_β 然后乘到 importance ratio 上
- 理论上等价但更准

**问题**:DR3 fade-out 在 WebShop 实证太激进。配置里 `disc_acc → mu` 把 BC 一起拖下来,`ratio_shaping_auto_acc_min: 0.8` 触发后再加一层 shaping。**两层 shaping 叠加在 disc_acc 起来时同时削弱 teacher**,让 teacher signal 在中后期(disc_acc>0.8 时)被双重压制。LUFFY 没有这个问题——它的 p/p_β 是"恒定逻辑",不随训练进度衰减。

### 2.2 CHORD (39%) vs DUET v39 (32%) 的差异

| 项目 | CHORD baseline | CHORD `chord_mu_0410` | DUET v39 |
|------|---------------|------------------------|----------|
| `chord_mu_peak` | 0.1 | **0.9** | 0.3 |
| `chord_mu_valley` | 0.1 | 0.05 | 0.05 |
| `chord_mu_decay_steps` | 0 (恒定) | 25 | 25 |
| `chord_use_token_weighting` | **true** | false | false |
| `use_chord` | true | true | true |
| `use_dr3` | false | false | **true** |

**机制论证**:CHORD 39% 有两条不同路径:
- **路径 A**(原始 baseline):μ=0.1 恒定 + token_weighting=true。这是"小但持续的 BC + token-level reweighting"
- **路径 B**(`chord_mu_0410`):μ=0.9 → 0.05 衰减 + token_weighting=false。这是"超强初始 BC,然后退化为纯 RL"

**两条路径都 work**,说明 WebShop 对 μ 范围非常宽容。DUET v39 用 μ_peak=0.3 是中庸值,既没有 token_weighting 救场,又没有足够的初期 BC 强度。**这是潜在的 dead zone**。

### 2.3 LUFFY 的 effective teacher weight 真的是 1/8 吗?

仔细看:LUFFY 的 `n_teacher_rollouts_per_task: 1`(同 DUET),但是它 `teacher_policy_shaping_enable: true` —— 在 actor loss 上,teacher token 的 importance weight 被 p/(p+β) 重塑后,如果模型对 teacher token 的预测概率低(p << β),ratio 接近 p/β >> 1 实际是会**放大** teacher token 的梯度的(直到 clip)。

而 DUET v39 的 DR3 ratio_shaping 是: `ratio = old_ratio · ρ̂_clipped`,其中 ρ̂ 越接近 1 表示老师和学生越像。在训练初期 ρ̂ ≈ 0.01~0.5(被 dual ESS clipping 进一步压低),teacher 的 effective gradient ≈ 0.1 × LUFFY 的 effective gradient。

**结论**:DUET 实际给 teacher 的有效 weight 比 LUFFY 低 5-10x。这就是 17.5pp 差距的最直接来源。

---

## 3. 假设 - Variant 设计(机制论证)

下面 5 个 variant **互相正交**(不同 hypothesis,不重复),按推荐优先级排:

---

### P0 / Variant 1: `webshop_qwen3b_duet_v40_psh` — 给 DUET 装回 LUFFY 武器

**假设 H1**:DUET 在 WebShop 上失败的根本原因是**用 DR3 替代了 p/p_β,但 DR3 在 WebShop 上 fade-out 太快,导致 teacher 信号在中后期形同虚设**。LUFFY 的 p/p_β 是恒定逻辑,DUET 应该把它作为"地板"而不是"替代品"。

**机制**:同时启用 LUFFY 的 actor-level p/p_β shaping **和** DR3 fade-out。让 p/p_β 提供基础 teacher utilization,DR3 只在 disc_acc 高、ESS 充裕时**加强**teacher 重要性,而不是替代。

**Yaml diff** (从 v39 出发):
```yaml
# diff vs webshop_qwen3b_duet_v39.yaml
trainer:
  experiment_name: webshop_qwen3b_duet_v40_psh

actor_rollout_ref:
  actor:
    # 关键:开启 actor-level policy shaping (LUFFY 的核心武器)
    teacher_policy_shaping_enable: true
    teacher_policy_shaping_mode: p_div_p_beta
    teacher_policy_shaping_beta: 0.1

    # 关闭 DR3 内部的 ratio_shaping(避免和 actor-level shaping 双重作用)
    dr3:
      ratio_shaping_mode: off  # 原 auto
      use_policy_shaping: false  # 原 true,这个会触发 DR3 hybrid 路径
      gap_gate_enable: true     # 原 false,保险栓:disc_acc 不到 0.6 时不强制 fade
      # 保留 DR3 disc 训练用于 BC 调度信号(chord_mu_adaptive 还要用 disc_acc)

    # 同时也在 teacher_experience 那侧开 policy_shaping
exp_manager:
  teacher_experience:
    policy_shaping:
      enable: true
      mode: p_div_p_beta
      beta: 0.1
```

**预测 success**:**42-48%**(信心 **High**)。理由:这是"LUFFY + 弱化的 DR3 + SC"。BC 通道还在,但不再被 DR3 双重压制。LUFFY 49.5% 是基线;BC 和 SC 加在 LUFFY 上理论上小幅度增益,worst case 退化到 LUFFY 49%(因为我们关了 DR3 ratio shaping,DR3 几乎只剩 disc 训练给 chord_mu_adaptive 提供信号)。

**风险**:DR3 学到的 ρ̂ 这版本不直接用了,paper 故事会受影响。但**让 DUET 先赢,再考虑故事**。

**ETA**:3.5h

---

### P0 / Variant 2: `webshop_qwen3b_duet_v40_strong_bc` — 强 BC,DR3 退居后台

**假设 H2**:DUET v39 BC 退得太早(decay_steps=25 / 总 ~100 步),DR3 又同时削 teacher。WebShop 实测两条 work 路径都强 BC: chord_mu=0.1 恒定 + token_weighting,或 μ=0.9→0.05 衰减。v39 在中间 dead zone (μ=0.3 quick decay)。

**机制**:大幅提高 BC 强度和持续时间,关 DR3 ratio_shaping(避免它干扰 BC)。这个 variant 本质上是"DUET 退化成 CHORD-mu-0410 + SC + DR3 fade-out 信号"。

**Yaml diff**:
```yaml
trainer:
  experiment_name: webshop_qwen3b_duet_v40_strong_bc

actor_rollout_ref:
  actor:
    chord_mu_peak: 0.6        # 0.3 → 0.6
    chord_mu_valley: 0.15     # 0.05 → 0.15
    chord_mu_decay_steps: 60  # 25 → 60(覆盖训练 60% 而不是 25%)
    chord_use_token_weighting: true  # false → true(CHORD baseline 风格)
    chord_mu_d_floor: 0.6     # 0.5 → 0.6(disc_acc 较高才开始退 BC)

    dr3:
      ratio_shaping_mode: off
      use_policy_shaping: false
      gap_gate_enable: true
```

**预测 success**:**38-44%**(信心 **High**)。这本质是"CHORD-mu-0410 (39%) + token_weighting (CHORD 39%) + SC + DR3 信号"。两条 39% 路径的并集,加 SC bonus,worst case ≈ 39%,best case 上探到 44%。**触底 = CHORD baseline**,这是非常稳的下保。

**风险**:如果 SC 和强 BC 冲突(SC 给 on-policy reward,BC 拉 teacher),可能内耗。可加 `state_channel.beta: 0.1` 减半作为保险。

**ETA**:3.5h

---

### P1 / Variant 3: `webshop_qwen3b_duet_v40_uniform_mix` — 提高 effective teacher weight

**假设 H3**:即便 LUFFY/DUET 字面上都是 `n_teacher=1`,DUET 没有 p/p_β 让 teacher 信号被 DR3 削弱;另一种补救是**增加 teacher 样本数量** + 启用 `use_uniform_mix` 让 GRPO group 看到更多 teacher。

**机制**:
- `n_teacher_rollouts_per_task: 1 → 2`,teacher_ratio 从 1/8 = 12.5% 升到 2/8 = 25%
- `use_uniform_mix: true`(LUFFY 风格的均匀 group sampling)

**Yaml diff**:
```yaml
trainer:
  experiment_name: webshop_qwen3b_duet_v40_uniform_mix

actor_rollout_ref:
  actor:
    use_uniform_mix: true   # false → true
    dr3:
      gap_gate_enable: true  # 顺手把这个安全栓打开

exp_manager:
  teacher_experience:
    n_teacher_rollouts_per_task: 2  # 1 → 2
    max_trajectories_per_task: 8     # 6 → 8(给 random select 更多池子)
```

**预测 success**:**36-42%**(信心 **Med**)。

**风险**:
- teacher_ratio 翻倍意味着 on-policy 从 7 降到 6,GRPO 多样性损失。如果 baseline separation 起作用还好。
- `use_uniform_mix` 在 1.5B 上没试过 true,3B 也没,完全未知。

**ETA**:3.5h

---

### P1 / Variant 4: `webshop_qwen3b_duet_v40_gap_gate` — 最小改动安全栓

**假设 H4**:ALFWorld DUET 把 `dr3.gap_gate_enable: true` 开了(DUET 在 ALFWorld 上比 LUFFY 强 8pp),WebShop 没开。gap_gate 的作用是"disc_acc 没到阈值前不让 DR3 干扰 ratio,避免 cold-start 期错误 fade"。WebShop 任务数量更少(800 vs ALFWorld 通常 3K+),disc 学习速度更慢,**没开 gap_gate 等于让 DR3 在 disc 还没学好就开始削 teacher**。

**机制**:仅一个开关,最小风险测试。

**Yaml diff**:
```yaml
trainer:
  experiment_name: webshop_qwen3b_duet_v40_gap_gate

actor_rollout_ref:
  actor:
    dr3:
      gap_gate_enable: true  # false → true
```

**预测 success**:**35-40%**(信心 **Med**)。

**风险**:gap_gate 阈值如果设错(默认在 het_actor.py 里写死),可能毫无效果。需要看代码确认默认 gap_threshold。

**ETA**:3.5h

**为什么排 P1**:改动最小,但增益也最小。如果 H1/H2 假设对,这个改动单独不足以补 17.5pp。但作为 P0 的一部分(P0 都顺手开了)被吸收。**单独 ROI 不高,可省略**。

---

### P2 / Variant 5: `webshop_qwen3b_duet_v40_dapo` — 解决稀疏 reward group

**假设 H5**:WebShop reward 稀疏,n=8 + 1 teacher → 7 on-policy 全 0 概率高。GRPO 在 all-zero group 上 advantage = 0,**整个 group 不贡献梯度**。LUFFY 通过 p/p_β 至少让 teacher token 有蒸馏信号(不依赖 group reward);DUET 没有这个 fallback。DAPO dynamic_sampling 主动过滤 all-zero group,把 batch 资源放在有信号的 group 上。

**机制**:开 DAPO,过滤 acc=0 和 acc=1 的退化 group。

**Yaml diff**:
```yaml
trainer:
  experiment_name: webshop_qwen3b_duet_v40_dapo

algorithm:
  dapo:
    enable: true
    dynamic_sampling:
      enable: true
      filter_mode: remove_all_incorrect  # 只过滤全错;全对(WebShop 几乎不会全对)保留

actor_rollout_ref:
  actor:
    use_dapo: true  # 开 DAPO clip-higher
    dr3:
      gap_gate_enable: true
```

**预测 success**:**35-41%**(信心 **Med-Low**)。

**风险**:
- DAPO dynamic sampling 会"重新生成"被过滤的 group,实际 wallclock 会变长 1.5-2x,**ETA 可能变 5-6h 而不是 3.5h**。死线紧的话谨慎。
- `filter_mode: strict` 会同时过滤全对,WebShop 任务不像数学题不太会全对,但 reward 可能一致(都 buy 错了同个 wrong item),误过滤风险小。

**ETA**:**5-6h**(注意比其他 variant 慢)

**为什么排 P2**:运行时长不可控,ROI 中等。仅当 P0 都失败时再考虑。

---

## 4. 不推荐 / 可省略的 variant

| 不推荐 | 原因 |
|--------|------|
| 关 DR3 完全退化为 CHORD | 等价于已知 39%,无新信息 |
| 改 SC `match_mode` 从 attribute_aware 到 hash | hash 在 WebShop 精度低,纯回退 |
| 单独提高 SC `beta` 或 `step_level.eta` | 1.5B v39_alpha/sc 已扫,WebShop SC 增益有限 |
| 改 DR3 `feature_mode` 或 `disc_hidden` | 改判别器架构,见效慢且 risk 大 |
| 加 entropy_coeff | 没硬证据 WebShop 是探索受限的 |
| 改 LR / batch | 用户禁了 |

---

## 5. 推荐执行顺序

按 14-17h 总预算分配:

```
T0: 启动 v40_psh (P0, 3.5h)            [parallel slot 1]
T0: 启动 v40_strong_bc (P0, 3.5h)      [parallel slot 2 — 如果有 8 GPU 用 4+4 拆]
T+3.5h: 看 v40_psh / v40_strong_bc 中间 metrics(success@50)
        - 任一 ≥ 38%: 继续跑完
        - 都 < 32%: kill,启动 v40_uniform_mix
T+7h: 收 v40_psh / v40_strong_bc 结果
T+7h: 启动 P1 (v40_uniform_mix, 3.5h)  — 单独跑
T+10.5h: 收 P1 结果
T+10.5h: 如果 P0 + P1 最高仍 < LUFFY 49.5%, 启动 v40_dapo (P2, 5h)
T+15.5h: 总结
```

**4 GPU 单 slot 限制下**,串行 4 个 variant = 14h(`v40_psh → v40_strong_bc → v40_uniform_mix → v40_gap_gate`)。

---

## 6. 监控指标(每个 variant 必看)

run 中如果以下任一信号出现 **kill 重新决策**:

| 信号 | 阈值 | 含义 |
|------|------|------|
| `critic/success_onpolicy/mean@50` | < 0.20 | 训崩 |
| `dr3/disc_acc@50` | > 0.95 | disc 过强,teacher 已被严重削弱 |
| `actor/kl_loss` | > 1.0 | 策略炸 |
| `duet/teacher_gradient_share@50` | < 0.02 | teacher 实际无贡献 |
| `state_channel/bonus_vs_reward_ratio` | > 0.3 | SC 反客为主 |

---

## 7. 给 paper 故事的备选叙事

如果 P0 (`v40_psh`) 赢 LUFFY:**"DUET = LUFFY-style policy shaping + BC scaffolding + state channel"**,DR3 仍然在(用于 disc 信号驱动 BC 调度),但不再做 ratio shaping。这个故事弱化了 DR3 论点,但**不丢 DR3**。

如果 P0 (`v40_strong_bc`) 赢 LUFFY:**"DUET 的 BC 通道 + SC 是关键,DR3 提供自适应调度信号"**。同样保 DR3 但定位变。

如果两个 P0 都没赢:意味着 v39 family 的失败不是单 knob 问题,需要更深的诊断。这种情况下最现实的应急方案是 **paper 承认 WebShop 是 DUET 弱项,主推 ALFWorld 强项**(用户提到 v1 ALFWorld 比 LUFFY 强 8pp)。

---

## 8. 附:Quick start 命令

```bash
# Variant 1 (P0, 高优先)
python launcher.py --conf config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v40_psh.yaml

# Variant 2 (P0, 高优先)
python launcher.py --conf config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v40_strong_bc.yaml

# Variant 3 (P1)
python launcher.py --conf config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v40_uniform_mix.yaml

# Variant 4 (P1, 可选)
python launcher.py --conf config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v40_gap_gate.yaml

# Variant 5 (P2, 仅 P0 都败时)
python launcher.py --conf config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v40_dapo.yaml
```

---

**关键判断重申**(给作者): v39 在 WebShop 输 17.5pp 不是因为 BC 或 EMA 调小细节,而是因为 **DUET 在 WebShop 上事实上把 LUFFY 的 p/p_β 武器关了**,期待 DR3 的 closed-form 重要性修正能替代,但 DR3 在 WebShop disc 学习速度跟不上,中后期 teacher 效应 ≈ 0。 P0 的 `v40_psh` 是最直接的修复,把 LUFFY 武器装回来,DUET 退居增量优化。这个 variant 是 paper 死线下信心最高的赌注。
