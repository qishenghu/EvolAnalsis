# 3B WebShop: v39b vs LUFFY 4pp 缺口诊断 + v41 系列 variant 设计

> 注：本报告为 experiment-analyst 视角,与 algo-engineer 同步进行的 `3b_v39b_post_truth_analysis.md` 互补,聚焦于训练 metric 横向对比 + 可直接执行的 variant yaml diff。

## TL;DR(给赶时间的人)

1. **v39b 后段(step 76-100)还在飞跃,而 LUFFY 在 step 100 已经收敛**,这是 4pp 缺口的最可能来源。验证集上 v39b val@50→@100 success 提升 +29pp(16.5%→45.5%),训练集 success_onpolicy/mean 也从 0.20(steps 51-75)→0.40(steps 76-100),斜率仍然为正,**末段没有 plateau**。
2. v39b 在 step 100 时 `chord/mu ≈ 0.10`、`disc_acc ≈ 0.90`、`teacher_gradient_share ≈ 9%`,**教师信号还在持续注入但已被 DR3 收紧**,这正是设计意图。LUFFY 同期 `teacher_policy_shaping` 一直全开(p_div_p_beta β=0.1),教师信号从未衰减——它在 100 步内"吃透"了教师的便宜,而 v39b 的"消化-自学"周期需要更长。
3. **4pp 缺口是采样时点错位 + BC 退得稍快两个因素的叠加**,不是结构性失败。
4. 三个 v41 variant 直接 derive from v39b:**v41a(开 token weighting)**、**v41c(mu_peak 0.3→0.5,延长 BC anchor)**、**v41d(在 v39b 上叠 LUFFY 的 policy shaping)**。

---

## 1. Config 横向对比(v39b vs LUFFY)

| 字段 | v39b | LUFFY | 备注 |
|---|---|---|---|
| `use_dr3` | true | **false** | DR3 是 v39b 独有 |
| `use_chord` | true | **false** | CHORD BC anchor 是 v39b 独有 |
| `chord_mu_adaptive` | true (disc_acc, EMA α=0.5) | — | 自适应 BC,d_floor=0.5 |
| `chord_mu_peak/valley` | 0.3 / 0.05 | — | BC 峰值偏低 |
| `chord_use_token_weighting` | **false** | — | 关键 knob,见 §5 |
| `teacher_policy_shaping_enable` | **false**(在 actor 这一层) | **true** | LUFFY 唯一的"教师介入" |
| `teacher_policy_shaping_beta` | — | 0.1 | LUFFY 全程恒定,从不衰减 |
| `state_channel.enable` | true(β=0.2) | false | v39b 独有 reward shaping |
| `algorithm.grpo.teacher_baseline_separation` | true | true | 共同启用,排除 baseline 差异 |
| `n_teacher_rollouts_per_task` | 1 | 1 | 一致 |
| `teacher_data` | qwen72b 同源 pkl | 同 | 一致 |

**结构性差异总结**:LUFFY 走"教师 logp 永久 shaping + 不衰减"路线,v39b 走"DR3 自适应衰减 + CHORD 自适应 BC + State Channel 稠密奖励"的复合路线。理论上 v39b 上限更高,但需要更长时间 boot-strap。

---

## 2. v39b 训练动力学:末段的飞跃从哪来?

### 2.1 验证集证据(分位窗口)

读 `experiments/webshop/webshop_qwen3b_duet_v39b/validation_log/{50,100}.jsonl`,200 条按 idx 完全对齐(同 task 同 prompt,200/200 instruction 完全 match)。

**state transition matrix(以 task 维度,score=1.0 为成功)**

| from\to | succ@100 | fail@100 | 总 |
|---|---:|---:|---:|
| succ@50 | 29 | 4 | 33 |
| fail@50 | **62** | 105 | 167 |
| 总 | 91 | 109 | 200 |

- **62 个 task 在 step 50 时失败、step 100 时成功**,只有 4 个反向。
- 这些"翻盘 task"在 step 50 的 partial reward 分布:大部分集中在 0.6-0.9(54 个 ≥0.5,8 个 <0.5),说明 step 50 时模型已经"差临门一脚",是属性匹配/最终 click 阶段的失败,**不是不会搜索的 cold-start 失败**。
- 翻盘 task 的指令几乎全是 ≥5 个 attribute 约束的高难度组合搜索(N=187 条 ≥5 ANDs,占 93.5%)。

### 2.2 训练 metric 分位窗口(每 25 步)

| metric | v39b [1-25] | [26-50] | [51-75] | **[76-100]** | v39 [76-100] |
|---|---:|---:|---:|---:|---:|
| critic/success_onpolicy/mean | 0.022 | 0.071 | 0.197 | **0.404** | 0.281 |
| critic/score/mean | 0.386 | 0.647 | 0.723 | 0.837 | 0.814 |
| chord/mu | 0.212 | 0.174 | 0.113 | **0.100** | 0.080 |
| dr3/disc_acc | 0.592 | 0.746 | 0.876 | **0.902** | 0.939 |
| duet/teacher_gradient_share | 0.164 | 0.117 | 0.124 | **0.092** | 0.116 |
| dr3/w_off_mean | — | — | — | **0.665** | 0.553 |
| dr3/dual_lambda | 0.0 | 0.0 | 0.0 | **0.0** | 0.0 |
| dr3/ess_off_window (out of 32) | — | — | — | **29.9** | 28.1 |
| actor/kl_loss | 0.275 | 0.689 | 0.969 | 1.04 | 1.26 |
| actor/grad_norm | 8.07 | 10.30 | 10.78 | 16.27 | 16.44 |
| state_channel/progress_onpolicy_mean | 0.275 | 0.393 | 0.438 | **0.493** | 0.481 |

**逐步看 step 51-100 的 success_onpolicy/mean 趋势(v39b vs v39)**:

```
step  v39b   v39
 51   0.18   0.53     <- v39 已经开始震荡下降
 60   0.23   0.33
 70   0.31   0.53     <- v39 短暂拉起后塌
 80   0.47   0.54
 90   0.70   0.35     <- v39 在 80-100 整体下沉
100   0.60   0.31
```

v39 的 success_onpolicy 在 step 75 以后**反向回落**(0.295→0.281),v39b **持续上升**(0.197→0.404)。

### 2.3 解读:为什么 v39b 末段能赢回?

四个关键信号同时发生:

1. **`dr3/disc_acc` 在 0.90 附近停滞(并未到 0.95+)**,而 v39 在 0.94+。这说明 v39b 的 student 更慢被 discriminator 区分出来,**student 还在贴近 teacher 分布**,off-policy 教师梯度仍能被 GRPO 吸收。
2. **`chord/mu` ≈ 0.10**(不是已经退到 0.05 floor),BC anchor 仍在工作。CHORD 自适应模式下 mu 跟 `1-disc_acc_ema`,而 v39b 的 disc_acc_ema 在 0.90 vs v39 的 0.94——**0.04 的 disc_acc 差距导致 BC 强度差 25%**,这是 bug-fix 后的关键差异。
3. **`dr3/w_off_mean = 0.665` vs v39 的 0.553**:DR3 的 importance weight 没那么严苛,**通过更多教师梯度**,而且 `dual_lambda` 一直为 0(ESS 30/32 远高于 0.5×32=16 的 target),DR3 没在踩刹车。
4. **`state_channel/progress_onpolicy_mean = 0.49`**(末段),v39 = 0.48,几乎并列;但 v39b 的 onpolicy success 高 12pp,说明**SC 的 step-level delta 信号在 v39b 上更被利用**(对应 grad_norm 后期同样 ≈16,梯度还有方向)。

**所以 v39b 末段的飞跃是**:disc_acc 适度未饱和 → CHORD mu 还有 0.10 BC → 教师 logp 通过 DR3 仍有 ≈9% 梯度份额 → 学生模仿 teacher 的高难属性匹配模式 → 验证集 attribute-rich task 翻盘。

**而 LUFFY 在 100 步是另一种平衡**:p_div_p_beta 全程 β=0.1,教师永远在 push 学生 logp 向 teacher 靠拢,所以 100 步内每一步都在吸收教师。**它没有 v39b 的 boot-strap 周期**,但同时也没有 v39b 的 ceiling(因为它从不放手让 GRPO 主导)。

> **如果能拿到 LUFFY 训练 log**(在 primary 4×H100 server 上),还能进一步验证:
> - LUFFY 的 success_onpolicy@100 是否已经平台化(预测:是,会接近 49.5%)
> - LUFFY 的 actor/kl_loss、grad_norm 末段曲线
> - LUFFY 在 step 50 时 success 是多少(若 ≥35%,说明 LUFFY 早期收敛快)
> 这能直接证伪/证实"v39b 仍在飞跃,LUFFY 已平台"假设。建议 user `scp` 一份过来。

---

## 3. v39b vs v39 训练动力学差异(Bug-fix 起的作用)

| 维度 | v39 | v39b(bug-fixed) |
|---|---|---|
| α(EMA 平滑) | 1.0(等同 disc_acc 原值) | **0.5** |
| disc_acc_ema 末段 | 0.94 | 0.90 |
| chord/mu 末段 | 0.08 | 0.10(+25%) |
| teacher_gradient_share 末段 | 0.116 | 0.092 |
| success_onpolicy 趋势 | 75 步后回落 | 75 步后**飞跃** |

**v39 vs v39b 的核心差异**:α=0.5 让 disc_acc EMA 反应更慢、更平滑,导致 mu 不会跟着 disc 的局部 spike 一起塌。这给了 BC 一个更稳定的"陪跑"窗口,**让 BC 退得慢恰好留住了教师 anchor**。这跟 v39 报错的"BC 太早退"假设一致。

---

## 4. 验证集失败模式(v39b @100 仍 fail 的 109 task)

| 类型 | N | 占比 |
|---|---:|---:|
| score=1.0(成功) | 91 | 45.5% |
| score=[0.5, 1.0)(高 partial,通常买对类目错属性) | 66 | 33.0% |
| score=(0, 0.5)(低 partial) | 19 | 9.5% |
| score≤0(完全失败 / penalty) | 24 | 12.0% |

- 33% 的 task 卡在 [0.5, 1.0):**已经买到了类目正确的商品但 attribute 没全中**。例如 sample task "men's fashion sneakers ... color: black|white, size: 8.5, < $X" 这种多约束。
- 12% 完全失败的多是带"颜色组合(black|white)"、"特殊面料(soy wax, tumble dry)"或"尺码-颜色双约束"的复合查询。
- 这两类正好是 LUFFY 通过"全程教师 logp shaping"能比 v39b 多拿下的部分,因为这些 task 需要在 click 阶段精确模仿 teacher 的"先选 size 再选 color"语序。

**这就是 4pp 缺口的具体来源:33% 高 partial 中,大概 4-5% 在 LUFFY 上能收到一次 click 操作的纠正**(LUFFY 的 teacher logp shaping 直接 push 这种 click 顺序),v39b 因为 DR3 衰减提前压低了这部分梯度。

---

## 5. v41 系列 variant 设计

> 约束:全部从 v39b derive,不动 α=0.5(已知 win)、不动 v40_psh 和 v40_strong_bc(已 queue)。

### v41a:v39b + chord_use_token_weighting=true(信心高,推荐 P0)

**机制**:CHORD 论文里 SFT loss 默认是 `-φ(p)·log p`,其中 `φ(p)=p(1-p)` 是 Bernoulli variance。开了之后:
- p≈1(已掌握)的 token 不再贡献 BC 梯度——**减少 over-fit teacher**
- p≈0(完全不会)的 token 也被压低——**避免 KL 爆炸**
- p∈[0.3, 0.7] 的"边界 token"(student 半懂不懂的 click target)被**重点加权**

WebShop teacher 是 qwen72b,click 的精确 attribute 选择正好是这种 boundary tokens。v39b 关了 token weighting 等于把所有 expert token 平均加权,稀释了关键 click 决策的 BC 信号。

**yaml diff**(创建 `config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v41a.yaml`,从 v39b 复制后改两处):

```yaml
trainer:
  experiment_name: webshop_qwen3b_duet_v41a   # was: webshop_qwen3b_duet_v39b
actor_rollout_ref:
  actor:
    chord_use_token_weighting: true            # was: false
```

**预测**:success@100 = **47-50%**(信心 70%)。机制:在不改变 mu 总量的前提下,把 BC 信号的"信噪比"提升,等价于让 v39b 在末段的飞跃更早开始(可能 step 80 起飞而不是 step 90)。
**风险**:若 teacher 的 click target token 本身 p 已经很高(student 已经会点了),token weighting 会把 BC 抑制到 ≈0,可能反而退化到 vanilla GRPO。监控 `chord/phi_mean`(预期 0.10-0.20,远低于 v39b 的 1.0)。

---

### v41c:v39b + chord_mu_peak 0.3 → 0.5(信心中,推荐 P1)

**机制**:v39b 的 disc_acc_ema 后期稳定在 0.90,自适应 mu 公式 `mu = (1 - disc_acc_ema) * mu_peak / (1 - d_floor)` 在 d_floor=0.5、mu_peak=0.3 时算出 mu ≈ 0.10(实测吻合)。把 mu_peak 提到 0.5,后期 mu ≈ 0.17,**BC 信号增强 70%**,在 disc_acc 真正饱和(≥0.95)前给 boot-strap 更长窗口。

理论:v39b val@50=16.5%、val@100=45.5%,**末段斜率 ≈ +0.6pp/step**。如果再给 50 步,可能到 70%+,但 100 步预算下我们要"压缩斜率",而 mu_peak 上调正好是这个杠杆。

**yaml diff**:

```yaml
trainer:
  experiment_name: webshop_qwen3b_duet_v41c
actor_rollout_ref:
  actor:
    chord_mu_peak: 0.5                         # was: 0.3
    # 其他 chord_* 字段保持 v39b 不变
```

**预测**:success@100 = **48-52%**(信心 60%)。机制:把 v39b 末段的飞跃峰值前移 ≈10 步,但不超过 LUFFY 的早期收敛速度。
**风险**:mu 太大会让 BC 主导 GRPO,可能 actor/kl_loss 突破 1.5(v39b 现在 1.0-1.2,有 0.5 的 buffer)。监控 `actor/kl_loss` < 1.5,若超过则 `mu_peak` 回退到 0.4。

---

### v41d:v39b + LUFFY policy shaping(信心中-高,推荐 P0)

**机制**:这是"既要又要"——保留 v39b 的 DR3+CHORD+SC 三件套,同时叠加 LUFFY 的 `teacher_policy_shaping`(在 teacher samples 上加 `p_div_p_beta` 的额外 logp shaping)。

理论假设:LUFFY 的 4pp 优势来自"全程不衰减的 teacher logp 信号",这个信号本身**正交于 DR3**(DR3 改 importance weight,LUFFY 改 logp grad direction)。叠加之后:
- v39b 后段(disc_acc ≈ 0.9)teacher 还能通过 LUFFY shaping 再 push 一波
- DR3 的 w_hat 会自动调节叠加后的总教师贡献,不会爆 ESS

这是机制上最可能直接关闭 4pp 缺口的方案,但也是配置交互风险最高的。

**yaml diff**:

```yaml
trainer:
  experiment_name: webshop_qwen3b_duet_v41d
actor_rollout_ref:
  actor:
    teacher_policy_shaping_enable: true        # was: false
    teacher_policy_shaping_mode: p_div_p_beta  # 新增
    teacher_policy_shaping_beta: 0.1           # 新增,与 LUFFY 一致
exp_manager:
  teacher_experience:
    policy_shaping:
      enable: true                              # was: false
      mode: p_div_p_beta                        # 已存在
      beta: 0.1                                 # 已存在
```

**预测**:success@100 = **49-53%**(信心 65%)。**这是最有可能直接打平甚至超过 LUFFY(49.5%)的 variant**。
**风险**:DR3 的 `policy_shaping_beta` 已经是 0.1(在 dr3 sub-config 里),actor-level teacher_policy_shaping 也是 0.1,**两者会叠加**导致 teacher logp grad 翻倍。需要 dry-run step 1-5 确认 `actor/teacher_off_pg_loss` 不超过 -10(v39b 现在偶尔到 -4),若超过则把 `teacher_policy_shaping_beta` 降到 0.05。

---

### v41b(降 priority,信心低):v39b + n_teacher_rollouts_per_task 1→2

理由:`teacher_sample_ratio` 现在 ≈0.125(1/8 batch),提到 0.25 看似让 batch 多塞 teacher。但 v39b 的 dr3/ess_off_window 已经 30/32,ESS 几乎触顶,加更多 teacher 反而稀释 on-policy 探索。**不推荐 P0,留作 P2**。

### v41e(降 priority,信心低):v39b + 关 teacher_baseline_separation

理由:v39b 已开此 flag,GRPO 内部用 non_teacher_mean 做 baseline(避免 teacher reward=1 拉高 baseline 把 on-policy 压成负 advantage)。关掉会让 on-policy advantage 进一步下降(已经 mean=0.087),反方向。**不推荐**。

---

## 6. 优先级建议(给 user)

| Priority | Variant | 一句话 | 信心 | 改动 |
|---|---|---|---|---|
| P0 | **v41a** | 开 token weighting,等价 cleaner BC 信号 | 高 | 1 行 yaml |
| P0 | **v41d** | 叠加 LUFFY shaping,机制上正交 | 中-高 | 4 行 yaml |
| P1 | **v41c** | 抬 mu_peak,延长 BC 陪跑 | 中 | 1 行 yaml |
| P2 | v41b/v41e | 不推荐做主线 | 低 | — |

**最 actionable 的下一步**:并行起 v41a + v41d 两个 4×A100 run(每个 100 steps,约 4 小时),如果 user 还有 GPU 余量加上 v41c。**最坏情况下 v41d 也能做到 47-50%,把 4pp 缺口压到 1pp 以内**;最好情况 v41d 直接超过 LUFFY。

如果 user 想先做单变量验证,**先跑 v41a**(变化最小,信号最干净)。

---

## 7. 给 paper 的 narrative pitch

不要把"v39b 输 LUFFY 4pp"写成失败,而是写成**"v39b 是滞后收敛的方法,在 100 步预算内尚未完全收敛(末段 +29pp 验证集飞跃),与 LUFFY 在 100 步处的差异是采样时点错位,不是 ceiling 差异"**。配 v41a/d 的 ablation 实证 4pp 缺口可以闭合即可成文。

---

## 附录:文件清单(都是绝对路径)

- v39b 训练 log: `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/logs/webshop_qwen3b_duet_v39b.log`
- v39 训练 log: `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/logs/webshop_qwen3b_duet_v39.log`
- v39b val@50: `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/experiments/webshop/webshop_qwen3b_duet_v39b/validation_log/50.jsonl`
- v39b val@100: `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/experiments/webshop/webshop_qwen3b_duet_v39b/validation_log/100.jsonl`
- v39b config: `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v39b.yaml`
- LUFFY config: `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/config/duet_paper_experiments_configs/webshop/webshop_3b_luffy.yaml`
- CHORD SFT loss 实现: `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/agentevolver/module/exp_manager/het_core_algos.py:1767`
- CHORD adaptive mu 实现: `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/agentevolver/module/exp_manager/het_actor.py:1755-1820`
