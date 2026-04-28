---
name: WebShop v39b Sweep Plan — DUET* targeting > 53% (single-seed, time-bounded)
created: 2026-04-28
purpose: push DUET* WebShop val@100 above 53% (beat all baselines) with single-seed efficiency
---

# WS v39b Sweep Plan

## Goal

```
Beat: DUET v1 = 53.0%   (target)
       LUFFY    = 49.5%
       CHORD    = 39.0%

Strategy: single-seed efficiency — sweep WIDER instead of multi-seed deep
          每个 run 之前重启 webshop env (defense vs memory leak)
```

## Critical observation from data audit

```
所有现存 WS data:
  Run                              val@100   config (key knobs)
  ──────────────────────────────────────────────────────────────
  webshop_3b_duet_0409_ema (v1)    53.0% ✓   no BC, batch=1, offload=ON, env_worker=32
  v39b (04-25)                     45.5%     BC peak=.30 ema=.5, batch=2, offload=OFF, env_worker=64
  v39b sanity (04-28)              12.5%     SAME yaml as 04-25 v39b
  v39                              32.0%     BC peak=.30 ema=.2
  v_clean_ws                       36.0%     BC peak=.30 closed-form
  v_no_bc_ws                        1.0% ✗   no BC, batch=2, offload=OFF, env_worker=64
```

**Smoking gun**: v_no_bc_ws (1.0%) vs DUET v1 (53.0%) — **同算法 (no BC) 但不同 config knobs**。

→ **Hypothesis**: v39b 的 config knobs (batch=2, offload=OFF, env_worker=64) 在 WS 上有副作用。

---

## Phase A — 12 single-seed runs (~42h, 1.75 days)

每个 run 单 seed，每 run 之前重启 webshop env。

### Diagnostic 阶段（先确认 hypothesis）

| #   | Config name              | Base   | BC | peak | valley | ema  | 价值                                  |
| --- | ------------------------ | ------ | -- | ---- | ------ | ---- | ------------------------------------- |
| 01  | ws_swA_01_v1cfg_no_bc    | v1cfg  | ✗  | —    | —      | —    | **pipeline sanity** (expect ~53%)     |
| 02  | ws_swA_02_v39b_default   | v39b   | ✓  | 0.3  | 0.05   | 0.5  | v39b baseline confirmation             |

### BC peak sweep (with v1cfg base)

| #   | Config name              | peak | valley | ema  | 测试什么                              |
| --- | ------------------------ | ---- | ------ | ---- | ------------------------------------- |
| 03  | ws_swA_03_v1cfg_peak02   | 0.2  | 0.05   | 0.5  | low BC ceiling                        |
| 04  | ws_swA_04_v1cfg_peak03   | 0.3  | 0.05   | 0.5  | v39b peak baseline (with v1cfg)        |
| 05  | ws_swA_05_v1cfg_peak04   | 0.4  | 0.05   | 0.5  | medium BC ceiling                     |
| 06  | ws_swA_06_v1cfg_peak05   | 0.5  | 0.05   | 0.5  | strong BC                             |
| 07  | ws_swA_07_v1cfg_peak07   | 0.7  | 0.05   | 0.5  | very strong BC                        |

### EMA speed sweep (peak=0.3 default + v1cfg)

| #   | Config name             | peak | valley | ema  | 测试什么                              |
| --- | ----------------------- | ---- | ------ | ---- | ------------------------------------- |
| 08  | ws_swA_08_v1cfg_ema02   | 0.3  | 0.05   | 0.2  | slow EMA → BC stays high longer       |
| 09  | ws_swA_09_v1cfg_ema08   | 0.3  | 0.05   | 0.8  | fast EMA → BC fades quick              |

### Strong-BC + slow-fade combos (best-bet candidates)

| #   | Config name                   | peak | valley | ema  | 测试什么                              |
| --- | ----------------------------- | ---- | ------ | ---- | ------------------------------------- |
| 10  | ws_swA_10_v1cfg_pk05_ema02    | 0.5  | 0.05   | 0.2  | strong BC + slow fade                  |
| 11  | ws_swA_11_v1cfg_pk05_v10      | 0.5  | 0.10   | 0.5  | strong BC + high floor (BC 不彻底退出) |
| 12  | ws_swA_12_v1cfg_pk05_ema02_v10| 0.5  | 0.10   | 0.2  | full combo: 强 BC + 慢退 + 高 floor    |

---

## Phase A 决策树

```
读 val@100 (single seed):

A. 01_v1cfg_no_bc < 45%  →  pipeline 已坏，进 deep debug（不是 paper 主线问题）
B. 01_v1cfg_no_bc ≥ 45%  →  pipeline OK
   ├─ 02_v39b_default ≈ 04-05 swA_04 (peak=0.3 + v1cfg) → config 不影响
   └─ 02_v39b_default << 04 → config knobs 是问题（v1cfg 必备）

C. 03-12 中找 best:
   ├─ 若 best ≥ 53%   →  Phase B 跳过，直接 Phase C 多 seed 确认 1-2 run
   ├─ 若 best 45-53%  →  Phase B 在 best 周围 ±0.05 微调
   └─ 若 best < 45%   →  Phase B 加大胆改动（policy_shaping_beta、kl_loss_coef）
```

---

## Phase B — refinement around Phase A best (待定, ~6-10 runs ~21-35h)

Phase A 结果出来后立刻设计。围绕 best config 在邻域微调（peak±0.05, ema±0.1, valley±0.02）。

## Phase C — final confirmation (~3-5 runs)

Best Phase B config 多 seed 一次（如时间允许）拿 mean±std。如时间不够，单 seed 报。

---

## 时间表

```
现在 (04-28 08:35) → AF 跑中 (v_gap_af_a step 13)
04-28 13:50      → AF v_gap_af_a val@100
04-29 01:00      → AF v_gap_af_b val@100, AF queue 完
                  ↓ 自动启 (run_after_af_then_ws_sweep.sh PID 2747671)
04-29 01:00 → 04-30 19:00     Phase A (42h, 12 runs)
04-30 19:00 → 05-02 06:00     Phase B (~35h, 8-10 runs)
05-02 06:00 → 05-03 00:00     Phase C (~18h, 5 runs)
                              ↓
05-04 → 05-07: paper writing & polish (~3 days)
```

---

## Auto-launch chain

```
1. run_diagnose_and_af.sh (PID 2010041, 当前 AF queue)
2. run_after_af_then_ws_sweep.sh (PID 2747671, 监听 1 退出)
3. run_ws_sweep_phase_a.sh (auto-launch when 1 done; 12 single-seed runs)
4. (Phase B/C 我会在 Phase A 完后基于数据立刻起)

Phase A 出来后用:
   python scripts/analyze_ws_sweep.py --phase A
   会显示 12 个 run 的 val@100 表 + 跟 baseline 对比 + 推荐下一步
```
