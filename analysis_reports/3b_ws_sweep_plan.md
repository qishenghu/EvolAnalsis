---
name: WebShop v39b Sweep Plan — DUET* targeting > 53% (single-seed, BC-only sweep)
created: 2026-04-28
updated: 2026-04-28 (v2: removed v1cfg variants — BC params only for fair baseline comparison)
purpose: push DUET* WebShop val@100 above 53% (beat all baselines) by tuning v39b's BC schedule
---

# WS v39b Sweep Plan (v2)

## Goal

```
Beat: DUET v1 = 53.0%   (target)
       LUFFY    = 49.5%
       CHORD    = 39.0%

Strategy: Sweep ONLY v39b-specific BC schedule parameters
          (peak / valley / d_floor / d_ema_alpha)
          
Infrastructure stays IDENTICAL to v39b's existing config:
  ppo_micro_batch_size_per_gpu = 2
  log_prob_micro_batch_size_per_gpu = 2
  param_offload = false
  optimizer_offload = false
  gpu_memory_utilization = 0.65
  max_env_worker = 64
  
Reason: changing infrastructure makes DUET* not directly comparable to baselines
        (which used some other server config). Sweep should reflect algorithmic gain only.
```

## Phase A — 12 single-seed runs (~42h, 1.75 days)

每 run 之前重启 webshop env (defense vs memory leak)。

| #   | Config name                  | peak | valley | d_floor | d_ema_alpha | 测试什么                              |
| --- | ---------------------------- | ---- | ------ | ------- | ----------- | ------------------------------------- |
| 01  | ws_swA_01_v39b_default       | 0.3  | 0.05   | 0.5     | 0.5         | v39b baseline 复核                    |
| 02  | ws_swA_02_peak02             | 0.2  | 0.05   | 0.5     | 0.5         | low BC                                |
| 03  | ws_swA_03_peak04             | 0.4  | 0.05   | 0.5     | 0.5         | medium BC                             |
| 04  | ws_swA_04_peak05             | 0.5  | 0.05   | 0.5     | 0.5         | strong BC ⭐                          |
| 05  | ws_swA_05_peak06             | 0.6  | 0.05   | 0.5     | 0.5         | very strong BC                        |
| 06  | ws_swA_06_peak07             | 0.7  | 0.05   | 0.5     | 0.5         | extreme BC (over-imitation risk?)    |
| 07  | ws_swA_07_ema02              | 0.3  | 0.05   | 0.5     | 0.2         | slow EMA → BC stays high longer       |
| 08  | ws_swA_08_ema08              | 0.3  | 0.05   | 0.5     | 0.8         | fast EMA → BC fades quick              |
| 09  | ws_swA_09_floor04            | 0.3  | 0.05   | 0.4     | 0.5         | lower d_floor → μ_raw 触底更快        |
| 10  | ws_swA_10_pk05_ema02         | 0.5  | 0.05   | 0.5     | 0.2         | strong BC + slow fade ⭐              |
| 11  | ws_swA_11_pk05_v10           | 0.5  | 0.10   | 0.5     | 0.5         | strong BC + high floor (BC 不彻底退出) ⭐ |
| 12  | ws_swA_12_pk05_ema02_v10     | 0.5  | 0.10   | 0.5     | 0.2         | full combo: 强 BC + 慢退 + 高 floor ⭐⭐ |

⭐ = 预测 candidate winners

---

## Hypothesis behind the sweep

```
WebShop reward 是 partial-credit (reach the right product = 1.0, near match = 0.6-0.9).
Student 没 BC 时容易找"近似商品" stuck 在 0.6-0.8，不努力做精确 attribute matching.
BC 强 (peak=0.5+) 把 student 拉向 teacher 的精确动作 → 拿到 1.0.

但 BC 太强 (peak=0.7+) 又会 over-imitate → student 失去 RL 探索能力, 
反而陷入 mode collapse.

Sweep 想找 sweet spot: BC strength enough to anchor但不窒息 GRPO.
```

---

## Phase A 决策树

```
读 12 个 single-seed val@100:

Top-1 ≥ 53%   →  ✓ 已超过 DUET v1, paper 头条 OK. 进 Phase C confirm.
Top-1 45-53%  →  ~ 接近 LUFFY/v1 但未超. 进 Phase B 在 best 周围 ±0.05 微调.
Top-1 < 45%   →  ✗ 整体偏弱. 考虑 Phase B 加更激进改动:
                  (a) 调 dr3.policy_shaping_beta 0.05/0.15
                  (b) 调 SC β/η
                  (c) 调 kl_loss_coef 0.0005/0.002
```

## Phase B (TBD, ~6-10 runs ~21-35h)

Phase A 完后基于数据立刻设计。围绕 Phase A 的 best 在邻域微调。

## Phase C (~3-5 runs)

Best Phase B config × 3-5 seeds (如时间允许) 拿 mean±std。

---

## 时间表

```
现在 (04-28 09:00) → AF 跑中 (v_gap_af_a step 20)
04-28 14:00       → AF v_gap_af_a val@100
04-29 01:00       → AF v_gap_af_b val@100, AF queue 完
                  ↓ 自动启 (run_after_af_then_ws_sweep.sh PID 2747671)
04-29 01:00 → 04-30 19:00     Phase A (42h, 12 runs)
04-30 19:00 → 05-02 06:00     Phase B (~35h, 6-10 runs)
05-02 06:00 → 05-03 00:00     Phase C (~18h, 5 runs)
                              ↓
05-04 → 05-07: paper writing & polish (~3 days)
```

---

## Critical clarification (vs v1 plan)

**Removed**: all `v1cfg` variants (which would have changed infrastructure).
**Reason**: Sweep must isolate the **algorithmic** contribution of v39b's BC schedule.
Infrastructure (batch_size, offload, env_worker, gpu_mem) is **frozen** at v39b's
defaults to remain comparable to other on-server experiments.
