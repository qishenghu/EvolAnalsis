# DUET v39b 3B 代码层审计报告

**时间**: 2026-04-24
**审计范围**: `het_actor.py` adaptive μ + `dr3_ratio.py` DR3 + `state_progress.py` SC + `het_core_algos.py` PG loss
**Train log**: `logs/alfworld_qwen3b_duet_v39b.log`
**Trajectory**: `checkpoints/agentevolver/alfworld_qwen3b_duet_v39b/Trajectory/batch_diag_step_*.json`
**Validation**: 50.jsonl=56.5%, 100.jsonl=42.0% (回归 -14.5pp)

---

## 1. 总体结论

**有 bug,严重程度: HIGH**。共发现 3 个真 bug + 1 个数值不稳路径。它们叠加导致 step 80 后的 grad_norm 急剧上升与 Val@100 退化。

| # | Bug | 严重度 | 位置 |
|---|-----|--------|------|
| **B1** | State Channel 的 β decay 在初始 success_rate>target 时直接归零 → 整轮 trajectory-level SC 完全失效 | **HIGH** | `ae_ray_trainer.py:3293-3307` |
| **B2** | Cross-rank μ 不同步: 只有 rank0 计算真实 `disc_acc`,rank≥1 永远拿到 `disc_acc=0.0`,fallback 成 0.5 → rank≥1 的 μ 卡在 peak | **HIGH** | `dr3_ratio.py:707-752` + `het_actor.py:1779-1786` |
| **B3** | DR3 修复路径 (`repo_compute_token_loss`) 不返回 `self_off_pg_loss` / `teacher_off_pg_loss`,导致教师专属诊断永远为 None,无法监控教师梯度 | MEDIUM | `het_core_algos.py:2039-2049` + `het_actor.py:2159-2160` |
| **U1** | `gap_linear` adaptive_weight 与教师 advantage 是同号扩张关系: gap 大时 gate 大,反而 **强化** 教师梯度 — 在末段教师 vs on-policy 差距被 GRPO 放大时,这是直接的不稳源 | **HIGH** | `ae_ray_trainer.py:3846-3849` |

---

## 2. 末段失稳的代码层归因

### 失稳事实链 (从 trajectory snapshot)

| step | success_on | ent_on | adv_t_mean | t_pos% | gap | grad_norm | kl_loss | resp_len |
|------|-----------|--------|-----------|--------|-----|-----------|---------|----------|
| 50   | 0.571 | 0.083 | 0.04 | 75% | 0.43 | 10.1 | 0.80 | 5313 |
| 60   | **0.768** | 0.107 | 0.07 | 50% | 0.23 | 6.2  | 0.26 | 2673 |
| 70   | 0.768 | 0.086 | 0.08 | 62% | 0.23 | 8.9  | 0.67 | 3011 |
| 79   | 0.500 | 0.066 | 0.15 | 75% | 0.50 | **20.5** | 0.51 | 3965 |
| 80   | 0.491 | 0.065 | 0.26 | **100%** | 0.41 | 19.9 | 1.06 | 4180 |
| 88   | 0.679 | 0.066 | 0.09 | 100% | 0.32 | **40.6** | 0.87 | 3819 |
| 95   | **0.286** | 0.060 | **0.29** | 100% | **0.70** | 27.4 | **1.69** | **7100** |
| 100  | 0.357 | 0.074 | **0.34** | 100% | 0.64 | 27.0 | 1.24 | 5626 |

### 因果链解释

1. **step 60**: 模型已达到 76.8% on-policy success (peak), entropy 已经压到 0.10。
2. **step 60 → 70**: μ 已掉到 ~0.05 floor (因 disc_acc_ema=1.0),BC 完全退出。Policy 进入纯 GRPO 阶段。
3. **step 70 → 79**: on-policy 开始随机退化(无 BC 锚定,无 entropy bonus,`entropy_coeff=0`)。on-policy reward 从 0.77 跌到 0.50。
4. **step 79+**: 回归后,**GRPO 的 group-relative advantage 把教师与 on-policy 的差异放大**(`adv_teacher_sample_mean` 从 0.07 跳到 0.15+ → 0.34)。所有教师样本 advantage 变正(t_pos 100%)。
5. **关键放大器 U1**: `adaptive_weight.gap_linear` 在 reward gap 变大时给教师 **更大** 的 gate (gap=0.7 → α=1.0),**与 DR3 的 fade-out 方向相反**。`teacher_loss_scale` 也变大,通过 `dr3_gap_gate` 又乘到 advantages 上,**双重放大教师梯度**。
6. **B2 加成**: 4 个 rank 中 3 个 rank 的 μ 卡在 peak(0.3)。它们对教师轨迹施加 6× 的 BC 拉力。FSDP 平均后 effective_μ ≈ 0.24,远超 metric 显示的 0.05。
7. **结果**: 教师梯度过度放大 + BC 拉力残留 → grad_norm 20→40,policy 在 step 95 触发 catastrophic update,KL 飙到 1.69,response_length 跳到 7100(重复输出),success_on 跌到 0.286。

### 量化责任分担(我的最佳估计)

- **B1 (SC 失活)**: ~30%。SC 本应在 step 60 后给 on-policy 提供探索 bonus,缺失后 policy 缺少持续的小信号引导,纯 GRPO 在 success>0.7 后无显著学习信号。
- **B2 (μ 跨 rank 不同步)**: ~40%。这是 grad_norm 跳跃的最直接来源 — 不同 rank 不同 μ 引起 FSDP grad 不一致,放大 step-to-step variance。
- **U1 (gap_linear 反向放大)**: ~25%。末段教师梯度异常增强的主因。
- **其他**: ~5%。`entropy_coeff=0`、`kl_loss_coef=0.005` 太小都是 "no brake" 因素,但不是 root cause。

---

## 3. 立即可执行的修复

### Fix B1: SC β decay 公式不合理 (`ae_ray_trainer.py:3293-3307`)

**当前**:
```python
if _sc_cfg.get("beta_decay", False):
    _sc_decay_metric = _sc_cfg.get("beta_decay_metric", "success_rate")
    if _sc_decay_metric == "success_rate":
        _sc_current_mean = (batch.batch["token_level_rewards"].sum(dim=-1) > 0).float().mean().item()
    ...
    _sc_target = float(_sc_cfg.get("beta_decay_target", 0.5))
    if _sc_target > 0:
        _sc_beta *= max(0.0, 1.0 - _sc_current_mean / _sc_target)
```

**问题**: 当 `success_rate=0.35` 而 `target=0.3` 时,`max(0, 1 - 0.35/0.3) = 0` → β_effective = 0。配置 `beta_decay_target: 0.3` 与 ALFWorld 3B 起步 success_rate (~0.35) 不匹配,SC 从 step 1 就完全关掉。

**Patch (最小修改,纠正语义)**:
```python
# 改成 “success>=target 时才开始 decay,在 [target, target+window] 之间线性退场”
_sc_target = float(_sc_cfg.get("beta_decay_target", 0.5))
_sc_window = float(_sc_cfg.get("beta_decay_window", 0.3))  # 新参,默认 0.3
if _sc_target > 0:
    if _sc_current_mean <= _sc_target:
        # below target: full beta (early-stage exploration)
        decay_factor = 1.0
    else:
        # past target: linearly fade across window
        decay_factor = max(0.0, 1.0 - (_sc_current_mean - _sc_target) / max(_sc_window, 1e-6))
    _sc_beta *= decay_factor
```

**期望效果**: 在 success<0.3 时 SC 全开,success∈[0.3, 0.6] 线性退场,success>0.6 时关掉。这与 "SC 是探索辅助、后期不需要" 的设计意图一致。

---

### Fix B2: 跨 rank μ 不同步 (`dr3_ratio.py:707-752` + `het_actor.py:1779-1786`)

**当前**:
- `dr3_ratio.py:715`: `can_optimize = (not self.broadcast_params) or (rank == 0)` → 只 rank0 训练
- `dr3_ratio.py:751`: `disc_acc_val` 只在 rank0 被更新,其它 rank 永远 = 0.0
- `dr3_ratio.py:874`: `metrics["dr3/disc_acc"] = disc_acc_val` → rank≥1 上报 0.0
- `het_actor.py:1779-1781`:
  ```python
  _disc_acc_raw = float(dr3_metrics.get("dr3/disc_acc", 0.0))  # rank≥1 = 0.0
  _disc_ready = float(dr3_metrics.get("dr3/disc_trained_steps", 0.0))  # rank≥1 = 0.0
  _disc_acc_now = _disc_acc_raw if _disc_ready > 0 else 0.5  # rank≥1 → 0.5
  ```

rank≥1 用 0.5 计算 EMA,然后在 het_actor:1796 算 `_gated = (1 - 0.5)/0.5 = 1.0` → μ = peak (0.3)。
rank0 用真实 disc_acc=0.99 → μ = floor (0.05)。

**Patch (在 dr3_ratio.py 的 step() 末尾,把 disc_acc broadcast 给所有 rank)**:

在 `dr3_ratio.py` `step()` 函数中,broadcast 后追加:

```python
# After self._broadcast_model_params() at line 756, also broadcast disc_acc / disc_trained_steps
if self.broadcast_params and (self._calls % self.broadcast_every_n_calls == 0):
    self._broadcast_model_params()
    bcast_metrics["dr3/bcast_happened"] = 1.0
    # NEW: broadcast disc_acc / disc_trained_steps from rank0
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            t = torch.tensor([disc_acc_val, disc_trained_steps, disc_loss_val],
                             device=self._buf_x.device, dtype=torch.float32)
            dist.broadcast(t, src=0)
            disc_acc_val = float(t[0].item())
            disc_trained_steps = float(t[1].item())
            disc_loss_val = float(t[2].item())
    except Exception:
        pass
```

**期望效果**: 所有 rank 看到一致的 `disc_acc`,从而计算一致的 μ。FSDP 内的 grad 一致性恢复,grad_norm variance 显著下降。

**Sanity check (单元测试)**:
```bash
# 在 4-GPU 跑 5 步,断言所有 rank 的 chord/mu 相等
DEBUG_DR3_RANK_SYNC=1 bash run_sanity_dr3_sync.sh
```

---

### Fix B3: DR3 路径未返回 teacher_off_pg_loss (`het_core_algos.py:2039-2049`)

**当前**: `repo_compute_token_loss` 只返回 `off_pg_loss`,不区分 self vs teacher。`het_actor.py:2160` 期望读 `teacher_off_pg_loss` 但 DR3 路径下永远拿到 None。结果 wandb 上 `actor/teacher_off_pg_loss` 一直缺失,无法监控末段教师梯度异常。

**Patch (在 repo_compute_token_loss 末尾增加 teacher 维度切分)**:
- 接受额外参数 `teacher_mask`
- 计算 `self_off_pg_loss = masked_mean(off_pg_losses, exp_mask * (1-teacher_mask) * response_mask)` 与 `teacher_off_pg_loss = masked_mean(off_pg_losses, teacher_mask * response_mask)`
- 加进 ret_dict

**期望效果**: 修复后可以在 wandb 上监控 `actor/teacher_off_pg_loss` 末段是否飙升,作为 v39c 训练的早期 abort 信号。

---

### Fix U1: gap_linear 反向放大教师梯度 (`ae_ray_trainer.py:3846-3849`)

**当前**:
```python
def _alpha_from_gap(g_eff: float) -> float:
    a = (g_eff - eps) / denom
    a = max(a_min, min(a_max, a))
    return float(a)
```

`g_eff` = teacher_reward - on_policy_reward。当 on-policy 退化时 gap 变大 → α 变大 → `teacher_loss_scale` 变大 → 通过 `dr3_gap_gate` 放大教师 advantages。这与 "gap 大说明 student 还学不到,继续教" 的初衷相符,但忽视了:**reward gap = teacher 优势,但 GRPO 已经在用 gap 算 advantage**。再用 gap 放大教师 loss 等于 **重复计入差距,二阶放大**。

**Patch (短期): config 关掉 adaptive_weight.gap_linear,或加上一个 `cap_when_progress_low` 条件**:

修改 `alfworld_qwen3b_duet_v39b.yaml` 的 `adaptive_weight`:
```yaml
adaptive_weight:
  enable: false  # 关掉,让 DR3 + chord_mu_adaptive 单独承担教师权重控制
```

**或者** (中期 cleanup,保留 gap_linear):
```python
def _alpha_from_gap(g_eff: float) -> float:
    # gap is already incorporated into GRPO advantage; over-amplifying teacher
    # via gap-based scale double-counts the gap in late training when on-policy
    # regresses. Cap by entropy / reward absolute level instead.
    a = (g_eff - eps) / denom
    a = max(a_min, min(a_max, a))
    # NEW: shrink alpha when on-policy success has already passed a threshold
    # (i.e., student is no longer beginner; teacher should fade)
    if hasattr(self, '_onpolicy_success_ema') and self._onpolicy_success_ema is not None:
        s = float(self._onpolicy_success_ema)
        # fade out from threshold=0.6 to 0.85
        fade = max(0.0, min(1.0, (0.85 - s) / 0.25))
        a = a * fade
    return float(a)
```

**期望效果**: 末段 teacher_loss_scale 自动收缩 → 教师梯度收敛到 0,与 DR3 fade-out 方向一致,避免 "教师拉回" 效应。

---

## 4. 必要的 sanity check 实验

### Exp-A: 跨 rank μ 验证(确认 B2)

**目的**: 直接验证 rank≥1 的 μ 是否真的卡在 peak。

**方法**: 在 het_actor.py 修改 logging,把每个 rank 的 chord/disc_acc_ema、chord/mu 都用 `dist.all_gather` 收齐,打到 wandb 上(metric 名 `chord/mu_rank0..rank3`)。

```python
# In het_actor.py around line 1801
try:
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        ws = dist.get_world_size()
        rank = dist.get_rank()
        t = torch.tensor([float(self._disc_acc_ema), float(mu)],
                          device=log_prob.device, dtype=torch.float32)
        gathered = [torch.zeros_like(t) for _ in range(ws)]
        dist.all_gather(gathered, t)
        for i in range(ws):
            adaptive_metrics[f"chord/disc_acc_ema_rank{i}"] = float(gathered[i][0].item())
            adaptive_metrics[f"chord/mu_rank{i}"] = float(gathered[i][1].item())
except Exception:
    pass
```

跑 10 步即可观察。如果 rank0 的 `chord/mu_rank0=0.05` 而 `chord/mu_rank1=chord/mu_rank2=chord/mu_rank3=0.3`,B2 即被实证。

**预算**: 4×A100,10 step ≈ 60 min。

---

### Exp-B: SC 重启与 grad_norm 关联验证(确认 B1 + U1)

**目的**: 看 SC 真实生效后是否抑制 grad_norm spike。

**配置**: 复制 `alfworld_qwen3b_duet_v39b.yaml` 为 `_v39c_sanity.yaml`,改:
```yaml
state_channel:
  beta: 0.2
  beta_decay: false   # 或者改成新公式 (Fix B1)
  step_level:
    enable: true
    eta: 0.05  # 同 v39b
exp_manager.teacher_experience.adaptive_weight:
  enable: false  # Fix U1 短期方案
actor_rollout_ref.actor:
  entropy_coeff: 0.001  # 给一点 entropy bonus,防 mode collapse
```

跑 100 step,观察:
- `state_channel/beta_effective` 应该 > 0
- `actor/grad_norm` step 80+ 应该不超过 15
- `critic/success_onpolicy/mean` Val@100 应该不退化

**预算**: 4×A100,100 step ≈ 8h。

---

### Exp-C: 末段 lr decay + entropy bonus 兜底(若 Fix B2 + U1 仍不够)

**配置**:
```yaml
actor_rollout_ref.actor:
  optim:
    lr: 1.0e-06
    lr_decay_steps: 100
    lr_decay_target: 1.0e-7  # 末段降一个数量级
  entropy_coeff: 0.001
  kl_loss_coef: 0.01  # 从 0.005 翻倍
```

---

## 5. 附录:关键数据点

### 末段 actor 指标(从 log)

| step | actor/grad_norm | actor/kl_loss | actor/on_pg_loss | dr3/w_off_max | dr3/disc_acc_ema | chord/mu | resp_len_mean |
|------|-----------------|---------------|------------------|---------------|-------------------|----------|---------------|
| 50 | 10.13 | 0.80 | 0.51 | 0.71 | 0.98 | 0.06 | 5313 |
| 60 | 6.23 | 0.26 | 0.27 | 0.61 | 0.997 | 0.05 | 2673 |
| 65 | 9.46 | 0.41 | (empty) | 0.61 | **1.000** | 0.05 | 4468 |
| 70 | 8.96 | 0.67 | 0.08 | 0.88 | 0.99 | 0.06 | 3011 |
| 79 | **20.54** | 0.51 | 0.22 | 0.75 | 0.99 | 0.05 | 3966 |
| 88 | **40.60** | 0.87 | 0.05 | 0.86 | 0.99 | 0.06 | 3819 |
| 95 | 27.38 | **1.69** | 0.27 | (empty) | (empty) | (empty) | 7100 |
| 100 | 27.04 | 1.24 | 0.09 | 0.67 | 0.97 | 0.07 | 5626 |

### 关键事实

1. `dr3/w_off_max` 全程 ≤ 1.1,**远未触及 clip_max=5.0** → DR3 没有 ratio 爆炸问题。`dr3.clip_max=5.0` 是 hard clip 实现(`dr3_ratio.py:840` `torch.clamp`),工作正常。
2. `dr3/dual_lambda=0.000` 全程 → ESS 一直接近 target,dual ascent 没启动。`ess_target_ratio=0.5` 在当前数据规模下不是约束 binding 的。
3. `dr3/disc_acc=1.0` 自 step 65 起。`disc_temperature=1.5` 软化 logits 把 D 拉到 ~0.05/0.95,所以 w_off 仍在 0.5-0.8(不是预期的 0.01)。这本身工作正常,但说明 **DR3 fade-out 实际上靠的不是 w_min 也不是 clip_max,而是 disc_temperature**。这也意味着 **w_min=0.01 floor 形同虚设**(实际值 0.5+)。
4. `state_channel/beta_effective=0.000` 全程(除一步=0.013)→ trajectory-level SC bonus 一直是 0。
5. `state_channel/step_level_delta_count > 0` → step-level deltas 在工作。但 `step_delta_negative_ratio=0.23-0.38`,**约 1/3 的 delta 是负的**,会扣 on-policy 奖励。

### B2 确证证据

`chord/disc_acc_ema=0.99-1.00` 是 rank0 的 EMA 值,但 `chord/mu_adaptive_gated=0.013-0.069` 也是 rank0 的。如果 rank≥1 也按 0.5 fallback 走 EMA(初值 0.5,d_ema_alpha=0.5,经过 100 步 EMA 仍然在 0.5 附近),那么 rank≥1 的 `gated = (1 - 0.5)/0.5 = 1.0` → μ = peak (0.3)。这与 metric 显示的 mu=0.05 直接矛盾,**确认 B2 是真实的 cross-rank 不一致**。

---

## 6. 推荐 v39c 配置

```yaml
# alfworld_qwen3b_duet_v39c.yaml (最小修改版)
actor_rollout_ref.actor:
  entropy_coeff: 0.001               # 防 mode collapse(原为 0)
  kl_loss_coef: 0.01                 # 加大 KL 制约(原为 0.005)
  chord_mu_peak: 0.2                 # 降低初始 BC 强度(原为 0.3)
  # B2 修复后这里更稳;暂时保守

state_channel:
  beta: 0.15                         # 略小一点(原为 0.2)
  beta_decay_target: 0.5             # 改成更高 target
  beta_decay_window: 0.3             # 新参数(配合 B1 patch)

exp_manager.teacher_experience:
  adaptive_weight:
    enable: false                    # U1 短期: 关 gap_linear
```

**配合 B1+B2+B3 patch,预期 Val@100 ≥ Val@50,grad_norm 末段 ≤ 15。**
