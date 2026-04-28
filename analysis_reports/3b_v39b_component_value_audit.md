# 3B WebShop v39b 组件价值审计:每个机制到底在做什么实质性工作?

**日期**: 2026-04-24  
**触发**: v39b 3B WebShop success=45.5%,输 LUFFY 49.5% 4pp。质疑 DUET 5+ 机制 / 20+ knobs 是否过度复杂。  
**输入**: `logs/webshop_qwen3b_duet_v39b.log`(100 步)、v39b yaml、`het_actor.py`、`het_core_algos.py`、`dr3_ratio.py`、`ae_ray_trainer.py`。  
**结论(剧透)**: 实际起作用的核心是 **adaptive μ via disc_acc + LUFFY p/(p+β) shaping (隐藏在 dr3.use_policy_shaping 里) + GRPO baseline-sep**;DR3 的 importance-weight reweighting 在前 30 步几乎是恒等(w≈1),`ratio_shaping_mode=auto` 在 step 40 就关掉了,SC step-level deltas 在 step 17 后再没触发,dual_lambda 全程 0,clipfrac 全程 0。多个 shaping 路径相互冗余、命名混淆。

---

## A. 每个机制的实际贡献评分(0=死,1=轻微,2=有效,3=核心)

| # | 机制 | 评分 | 数据证据 |
|---|---|:-:|---|
| 1 | DR3 density-ratio reweight (`old_lp ← log_p − log w_hat`) | **1.5** | step 1–40: `logw_applied_abs_mean ∈ [0.01, 0.07]` → `exp(logw)` 仅 1.01–1.07,几乎是恒等;step 50+ 才达 `0.4–0.7` (exp 1.5–2.0×),但此时 success 已开始爬升,因果模糊 |
| 2 | DR3 dual ascent ESS clipping (`dr3.dual_enable=true, ess_target_ratio=0.5`) | **0** | `dr3/dual_lambda` 全程 100/100 步 = 0;ESS 从 step 5 起就稳定在 31.5/32 ≈ ratio 0.98,从未触发紧绷条件 |
| 3 | DR3 hard clip (`clip_max=5.0, ppo_clip_eps=0.2`) | **0** | `dr3/w_clipfrac_off` 全程 100/100 步 = 0;`w_off_max` 永远 ≤ 1.09,clip 上限 5.0 / ppo_eps 0.2 都从未触发 |
| 4 | DR3 ratio_shaping (`ratio_shaping_mode=auto`,在 off-policy ratio 上 `r/(r+β)`) | **0.5** | step 10–25 enabled=1(disc_acc<0.8 时),step 30–40 间歇,step 40 后永久关闭(disc_acc≥0.8∧buf≥512∧ess≥16 全满足)。但前期 `dr3/logw_applied_abs_mean<0.07`,policy ratio≈1,shaping 实际没改变什么 |
| 5 | DR3 policy shaping (`dr3.use_policy_shaping=true` → `het_compute_teacher_aware_loss(teacher_policy_shaping_enable=True, mode=p_div_p_beta, β=0.1)`) | **3** | 这才是 v39b 真正用的"老师梯度地板"。`dr3/hybrid_policy_shaping=1.0` 全程,其实质等同于 LUFFY 的 p/(p+β) shaping 应用到 teacher token,只不过在 ratio 上又乘了个 w_hat(`teacher_loss_scale = w_hat`)。**这条路径在 het_actor.py:1545–1590 被激活,与 top-level `teacher_policy_shaping_enable=false` 冲突命名**。 |
| 6 | DR3 disc_acc 信号 (作为 adaptive μ 的输入) | **3** | `chord/disc_acc_ema: 0.5→0.93`,直接驱动 μ。这是 DR3 最有价值的副产物,而不是 importance reweighting 本身 |
| 7 | SC progress hash + bonus β·P(τ) (`exp_manager.state_channel`) | **1** | `bonus_vs_reward_ratio` 均值 0.121(范围 0.074–0.183),即 SC 把 reward 平均拉高 12%,数量级正确但占比不大;teacher 排除生效(`teacher_excluded_count` 与 `total_teacher_rollouts` 完全对齐) |
| 8 | SC step-level deltas η·[Φ(s_{t+1})−Φ(s_t)] | **0** | 100 步只触发 8 次,全部在 step 2–17(coverage 还没建立稳定 hash 之前的边角 case);step 18 后 `step_level_delta_count=0` 持续到 100。**实质死代码** |
| 9 | Adaptive μ via disc_acc EMA (`chord_mu_adaptive=true, mode=disc_acc, α=0.5, peak=0.3, valley=0.05, d_floor=0.5`) | **3** | μ 公式 100% 复刻 (验证 `μ = 0.05 + 0.25·max(0, min(1, (1−EMA(d))/0.5))`,误差 < 0.001)。μ 范围 0.077–0.3,**从未达到 valley**;末段 25 步均值 0.098,等同于"BC 长期保持 ~10% 权重"。这是真正的核心 schedule。 |
| 10 | LUFFY mix (`teacher_experience.mix_mode=rollout_level, n_teacher=1, max_traj=6`) | **3** | `luffy/total_teacher_rollouts` 稳定 5–8 / 64,`teacher_sample_ratio` 紧守 0.094–0.125;teacher 数据流持续注入梯度(`teacher_gradient_share` 范围 0.02–0.32) |
| 11 | GRPO teacher baseline separation (`teacher_baseline_separation.enable=true`) | **2** | `duet/adv_teacher_effective_abs_mean` 均值 0.176 vs `duet/adv_onpolicy_effective_abs_mean` 均值 0.156。绝对值相近但**分布不同**(teacher 早期高、后期降到 0.04;onpolicy 持续在 0.10–0.35 间)。如果不分离,teacher 几乎全部 reward=1(success)而 onpolicy 早期几乎全 0,合并均值会让 onpolicy 正样本被压成负 advantage(zero-pivot 漂移)。**这是个被忽视但真有作用的稳定器** |
| 12 | actor-level policy_shaping (`actor.teacher_policy_shaping_enable=false`) | **0** | 字面被关。但功能被 `dr3.use_policy_shaping=true` 在内部以同一函数 `het_compute_teacher_aware_loss(teacher_policy_shaping_enable=True)` 重新打开了。**纯命名冗余** |

**结论**: 真正核心(3 分)的只有 4 个:
- adaptive μ via disc_acc(BC 调度)
- LUFFY p/(p+β) shaping(教师梯度地板,藏在 dr3.use_policy_shaping 里)
- LUFFY mix(1/8 teacher 注入)
- DR3 disc_acc 信号(给 adaptive μ 做输入)

**装饰性(0–1 分)的**: dual ascent、hard clip、SC step-level、ratio_shaping (前期间歇)、SC β·P(τ) (12% 噪声)、actor 顶层 policy_shaping 开关。

---

## B. 死代码 / 冗余路径列表

### B1. 完全死的代码路径(0 / 100 步触发)

| 路径 | 文件:行 | 死因 |
|---|---|---|
| `dr3.dual_enable=true` 的 dual ascent | `dr3_ratio.py` | ESS 从未紧绷:`ess_off_window` 从 step 5 起即 ≥31.5,target ratio 0.5 → target 16,从未跌破 |
| `dr3.clip_max=5.0` 的硬 clip | `dr3_ratio.py` | `w_off_max` 全程 ≤1.09,clip 5.0 完全冗余;`w_clipfrac_off=0` 全程 |
| `dr3.ppo_clip_eps=0.2` 在 hybrid path | `het_actor.py:1576` 的 `clip_ratio_c` | hybrid path 用的是 `het_compute_teacher_aware_loss` 不是 `repo_compute_token_loss`,`dr3_clip_eps` 在 hybrid 分支被绕过 |
| `state_channel.step_level.enable=true` | `ae_ray_trainer.py:3513–3609` | 100 步只触发 8 次(全在 step 2–17);step 18 之后 `compute_step_deltas` 始终返回空 list。SC 的"dense reward shaping"在跑 v39b 的窗口内**事实上没有 dense 化任何东西** |

### B2. 冗余命名 / 命名冲突

1. **两个 policy_shaping 开关并存且功能重叠**:
   - `actor.teacher_policy_shaping_enable` (顶层) — v39b: `false`
   - `actor.dr3.use_policy_shaping` (DR3 子节点) — v39b: `true`
   - 后者在 het_actor.py:1580 内部强制 `teacher_policy_shaping_enable=True` 调用同一个 loss 函数,**用户读 yaml 会以为关了 LUFFY shaping,实际并没有**。Paper 框架"DR3 ⊥ LUFFY"在代码里其实是"DR3 调用 LUFFY"。

2. **chord_global_step / dr3 warmup / apply_min_buf_size 三套阈值**:
   - `dr3.apply_warmup_steps=10` 让 DR3 apply 在 step 10 才 ready
   - `dr3.apply_min_buf_size=512` 让 buffer 在 step 17 才满足
   - `dr3.disc_train_min_buf_size=256` 让 disc 训练在 step 5 就 ready
   - `chord_mu_warmup_steps=0` 让 BC 第 1 步就开
   实际生效顺序:CHORD μ 从 step 1(BC 立刻拉满 0.3)→ disc_train step 5 → apply ready step 10 → buf 满 step 17 → ratio_shaping 自动关 step 40。**前 10 步 DR3 完全不工作,BC 和 LUFFY shaping 在裸跑。前 17 步 DR3 也几乎不工作(buffer 还没满,disc 训练样本不足)**。

3. **ratio_shaping 三模式 (`step / always / off / auto`) 在 v39b 实际只用 `auto`**,且 auto 在 step 40 后永久关闭。这条路径只在 step 10–40 短暂活跃,**贡献区间窄到难以与 BC 区分**。

### B3. v39b 实际生效的 PG loss 路径(代码追踪)

```
het_actor.update_actor → use_dr3=true → dr3_enable=true →
  use_chord=true → has_teacher_data=true →
    dr3_use_policy_shaping=true (line 1544) →
      het_compute_teacher_aware_loss(
          teacher_policy_shaping_enable=True,         # 强制 True!
          teacher_policy_shaping_mode='p_div_p_beta',
          teacher_policy_shaping_beta=0.1,
          teacher_loss_scale=w_hat (DR3 importance weight),
          teacher_use_clip=False,
          ...
      )
    + chord SFT loss (μ=adaptive, weighted_sft_loss)
    + KL loss (kl_loss_coef=0.001)
```

**这等价于**: GRPO + LUFFY p/(p+β) shaping(教师 token)+ w_hat 缩放(几乎是 1) + adaptive μ BC + GRPO baseline-sep + LUFFY mix。

去掉 w_hat 几乎不变("DR3 importance weight" 在前 50 步= 1.0;后 50 步约 0.55–0.7,效果 = 把 teacher 梯度地板降低 30–40%)。

---

## C. config 协调时间线表(v39b 实际)

| step | DR3 buf | DR3 disc | DR3 apply | DR3 logw_applied | DR3 ratio_shaping | SC β·P bonus | SC step_delta | adaptive μ | μ 数值 | LUFFY mix | success_on |
|---:|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 36 / 1024 | **off** (warmup) | **off** | n/a | n/a | active(β_eff=0.2) | active(count=23 only at step 2) | active(d=0.5) | 0.30 | active(7/8) | 0.018 |
| 5 | 292 | **first ready** (acc=0.612) | **off** (still <10) | n/a | n/a | active | active | active | 0.279 | active | 0.018 |
| 10 | 612 | acc=0.679 | **first ready** | 0.01 (≈0 effect) | enabled | active | dead from now | active | 0.217 | active | 0 |
| 17 | **first 1024** | acc=0.766 | ready | 0.05 | enabled | active | dead | active | 0.16 | active | 0.143 |
| 25 | 1024 | acc=0.682 | ready | 0.07 | enabled | active | dead | active | 0.198 | active | 0.018 |
| 40 | 1024 | acc=0.773 | ready | 0.066 | **first OFF** (auto trips) | active | dead | active | 0.162 | active | 0 |
| 50 | 1024 | acc=0.789 | ready | 0.41 (**first meaningful**) | mostly off | active | dead | active | 0.155 | active | 0.196 |
| 70 | 1024 | acc=0.909 | ready | 0.60 | off | active | dead | active | 0.088 | active | 0.31 |
| 100 | 1024 | acc=0.902 | ready | 0.47 | off | active | dead | active | 0.094 | active | 0.603 |

**Phase 划分**:

- **Phase A (step 1–4): GRPO + BC(μ=0.30 满)+ LUFFY mix + SC β·P bonus**。DR3 完全不参与(disc 还没训出来,apply_warmup_steps=10)。**SC step-level 在 step 2 触发 23 次 — 唯一一次像样工作**。
- **Phase B (step 5–17): + DR3 disc 启动训练 (但 apply 还没 ready 直到 10 / 17)**。disc_acc 0.6→0.77,主要给 adaptive μ 喂信号让 μ 从 0.30 跌到 0.16。LUFFY shaping 一直在用。
- **Phase C (step 18–40): DR3 全 ready,但 logw≈0.05(几乎不做 importance reweighting),ratio_shaping 间歇 enable**。这一段是 BC 主导(μ ~0.16)+ LUFFY shaping。success 在 0–0.14 反复,无明显爬升。
- **Phase D (step 41–60): success 开始起跳(0.20→0.30)**。DR3 logw_applied 从 0.07 跃到 0.4–0.7 (因为 disc_acc 上 0.85+, w_off_mean 跌到 0.6),DR3 importance reweight **首次实质生效**。但同时 μ 下到 0.10 附近,BC 也在退场。**因果绑定:DR3 的"实质工作期"和 BC 退出期 重叠,success 起跳无法归因到 DR3 单独**。
- **Phase E (step 61–100): success 0.31→0.60,μ 在 0.08–0.13 震荡,DR3 logw 维持 0.4–0.7**。这是 v39b 的"赢面"区间,机制可能是 μ 永远不死透 + LUFFY shaping 持续工作 + 偶尔的 DR3 importance reweight,**但 paper 写"DR3 数据驱动 fade-out"在数据上看其实是 disc_acc≈0.9 时 w_hat≈0.6 (而不是趋于 0)。"自然 fade-out" 只到 70%,远未消失**。

**关键观察**:`dual_lambda=0`、`clipfrac=0`、`step_level_delta_count=0` (after step 17)、`ratio_shaping=0` (after step 40) 这 4 条全程 / 大段死。DR3 自身真正的 importance reweighting 在前 30 步是恒等映射(后 50 步才有意义)。

---

## D. 简化版本建议(只保留 ≥2 分机制)

如果把 v39b 砍到只剩起作用的部分,理论 yaml 应是:

```yaml
# 核心 4 件套: GRPO + adaptive μ + LUFFY shaping + LUFFY mix
algorithm:
  adv_estimator: grpo
  grpo:
    teacher_baseline_separation:
      enable: true   # 2 分: 防 teacher reward dominance

actor_rollout_ref:
  actor:
    use_chord: true                          # 3 分: BC 主调度
    chord_mu_warmup_steps: 0
    chord_mu_decay_steps: 25                 # 仅做 fallback,实际不用
    chord_mu_peak: 0.3
    chord_mu_valley: 0.05
    chord_mu_adaptive: true                  # 3 分: 关键
    chord_mu_adaptive_mode: "disc_acc"
    chord_mu_d_floor: 0.5
    chord_mu_d_ema_alpha: 0.5

    use_dr3: true                            # 但只用作 disc_acc 信号源 + p/(p+β) 通道
    teacher_use_log_prob: false
    teacher_policy_shaping_enable: true      # 3 分: 移到顶层,移除 dr3 内部冗余
    teacher_policy_shaping_mode: p_div_p_beta
    teacher_policy_shaping_beta: 0.1
    dr3:
      enable: true
      apply_to: teacher_no_logprob
      use_policy_shaping: false              # 关掉 hybrid path,因为 LUFFY shaping 已经在顶层开了
      gap_gate_enable: false
      # 以下保留只为给 disc_acc 提供信号
      disc_train_min_buf_size: 256
      buffer_size: 1024
      train_batch_size: 128
      disc_steps_per_call: 2
      disc_lr: 0.0003
      hidden_proj_dim: 64
      ess_window: 32
      sync_across_ranks: true
      broadcast_params: true
      # 以下全部删除/默认: dual_enable, clip_max, ppo_clip_eps, ratio_shaping_*,
      #                    apply_warmup_steps, apply_min_buf_size, w_min, alpha_mode,
      #                    feature_mode, gap_gate_*, w_hat_ema_alpha
      # 因为它们要么从未触发,要么对 loss 无可观察影响

exp_manager:
  teacher_experience:
    enable: true                             # 3 分
    mix_mode: rollout_level
    n_teacher_rollouts_per_task: 1
    max_trajectories_per_task: 6
    use_log_prob: false
  state_channel:
    enable: false                            # 1 分: bonus_vs_reward_ratio=12% 但与 reward 高度共线,
                                             # 在小批 batch 下与 GRPO baseline-sep 效果重叠;
                                             # step_level 已死,留着只是噪声源
```

**这个简化版的预期**:
- 砍掉的 knob 数: ~12 个(dual ascent / hard clip / ratio_shaping (3-mode + 4 阈值) / apply_warmup / apply_min_buf / step_level / w_min / SC bonus)。
- **理论 success ≈ v39b 的 45.5% ± 噪声**: 因为去掉的全是事实上没在工作的机制。
- **实际跑出来 vs LUFFY (49.5%) 的 4pp 差距**: 这个简化版预计补不上 4pp,因为 4pp 的来源不在那些被砍的机制里(它们已经没在做事)。**真正的 4pp 来源最可能是 LUFFY 的 `mix_mode` 或 ref-policy KL 设置或 chord_use_token_weighting=true** — 这是另一个调研方向,不在本次审计范围。

**能跑吗**: 能。所有保留的 yaml 字段都对应真实使用的代码路径。`dr3.use_policy_shaping=false` 时,het_actor.py:1545 走的是 `repo_compute_token_loss` + 顶层 `teacher_policy_shaping_enable=true` 应用到 het_compute_teacher_aware_loss,行为等价于 v39b 但调用栈一致性更好。

---

## E. paper 重新定位建议

DUET 当前框架"3 通道协同 (Action Channel + State Channel + 适应调度)"在 v39b 数据上撑不住:
- **State Channel** (β·P(τ) bonus + step-level deltas):step_level 死,bonus 占 12% reward 但贡献模糊。本质是个 12% 的奖励倍数器,与"通道"概念严重不匹配。
- **Action Channel (DR3 importance reweighting)**: 前 30 步 w≈1 (恒等),step 50+ 才有 0.5–0.7 缩放,但 dual ascent / clip 都没触发。"DR3 修正了 importance weight 让 off-policy 收敛" 在 webshop 3B 数据上**没有强证据**。
- **Adaptive μ**: 这个**在数据上确实是核心**,r=0.97 与 v24 schedule 的相关性是真的。

**诚实重定位**:DUET 真正起作用的是"用 DR3 discriminator 的 acc 当 BC 强度的自适应信号 + LUFFY shaping 提供教师梯度地板",不是"3 通道协同"。把 paper 重写为:

> "We propose **DUET (Discriminator-driven adaptive BC)**: a single-knob simplification of LUFFY/CHORD where a lightweight discriminator's training accuracy becomes a free, online schedule signal for BC strength. Unlike CHORD's hand-tuned cosine, DUET tracks policy-teacher distinguishability in real time and re-injects BC pressure whenever the policy drifts. We additionally retain LUFFY's p/(p+β) policy shaping for stable token-level credit assignment on rare action tokens."

这个 framing 在数据上 100% 站得住,而且能解释为什么 v39b α=0.5 (fast EMA) 反超 v39 α=0.2:fast EMA 让 μ 永远不死透,等价于一个**自适应的 KL regularizer**(post_truth doc 已经讲到这个机制,但用了"valley 震荡"描述。**修正**:μ 在 v39b 末段振幅是 0.077–0.155,从未达到 valley=0.05,叙事应改为"μ 在 0.08–0.13 持续震荡,从未沉到 valley")。

---

## F. 五个最强证据的一览

1. **`dr3/dual_lambda` 100/100 步 = 0**: dual ascent 是个安全网,从未启动。Paper 不应吹这个机制。
2. **`dr3/w_clipfrac_off` 100/100 步 = 0**: w_off_max 从未超过 1.09。clip_max=5.0 完全冗余。
3. **`dr3/logw_applied_abs_mean` 在 step 1–40 ∈ [0.01, 0.07]**: DR3 在前 40 步对 old_log_prob 的修正小于 7%,事实上等同于"把 teacher off-policy 当 on-policy 用"。
4. **`state_channel/step_level_delta_count` step 18+ = 0** (持续 83 步): SC step-level 这一整套(extract_observations、compute_step_deltas、token-level 注入)是**死代码**。
5. **`chord/mu` 范围 0.077–0.300, 末 25 步均值 0.098**: μ 从未达到 valley 0.05。"自然退化到 valley" 的叙事错误,真实是"BC 长期保持 ~10% 权重,与 fast EMA 噪声共振保持活性"。

---

**核心建议**:

1. **删掉 dual ascent / hard clip / ratio_shaping_mode / SC step_level / w_min** 这 5 组 knob,实测应不影响 success。
2. **合并两个 policy_shaping 开关**(顶层 + dr3 子节点)成一个,避免命名冲突。
3. **paper 重新定位为"discriminator-driven adaptive BC"**,DR3 的角色从"importance reweighting" 改为 "BC schedule signal source"。这与数据 100% 一致。
4. **SC 在 webshop 3B 上贡献接近噪声**(bonus_vs_reward_ratio 均值 12% 但与 reward 高度相关 → 进 GRPO 后被 baseline 减掉),建议 ablation 关闭 SC 看是否真的下降。如果不下降,SC 可以从核心 contributions 移到 future work。

文件路径(便于追溯):
- `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/logs/webshop_qwen3b_duet_v39b.log`
- `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v39b.yaml`
- `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/agentevolver/module/exp_manager/het_actor.py` (lines 1090, 1500–1666, 1755–1810)
- `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/agentevolver/module/exp_manager/het_core_algos.py` (lines 237–630, 1900–2050)
- `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/agentevolver/module/trainer/ae_ray_trainer.py` (lines 3260–3450, 3500–3610)
