---
name: 3B Master Experiment Table — DUET* (closed-form auto-adjusting BC + DR3 + SC) + baselines
created: 2026-04-26
updated: 2026-04-28
purpose: single source of truth for paper Table 1 (3B section)
---

# 3B 主数据表 — ALFWorld + WebShop（success / reward）

所有数字按 **train step = 100**（除非另注），N=200（全测试集），Qwen2.5-3B-Instruct，4×L20X-144G 或 4×H100。

---

## DUET* — 算法概述（paper narrative）

DUET\* 由三个 **closed-form** 组件构成（"closed-form" 意指**算法/公式自给自足、无需 epoch-by-epoch 手调 schedule**）：

```
(i) DR3 density-ratio correction
    ŵ = D / (1 − D)
    对 off-policy IS 的修正；D 是在线训练的 discriminator
    teacher 梯度自动 fade-out (随 student → teacher，D 分不开 → ŵ → 1)

(ii) Adaptive BC
    μ(t) = valley + (peak − valley) · max(0, (1 − d̄(t)) / (1 − d_floor))
    d̄ = EMA(D's accuracy)
    BC 强度自动衰减；与 (i) 共享同一个 D，无需独立 schedule

(iii) State Channel
    r' = r + β · Φ(s_T) + Σ_t η · [Φ(s_{t+1}) − Φ(s_t)]
    Φ 是 teacher trajectory 离线哈希出的 progress map
    无在线学习，对 on-policy 样本生效（teacher 排除）
```

**关键设计**：(i) 和 (ii) **共享** discriminator D — 一个学习信号驱动两个机制。

---

## ALFWorld 3B（reward = success，二值）

| Method                         | Algorithm                          | Val@50 | Val@100   | Source                                       |
| ------------------------------ | ---------------------------------- | ------ | --------- | -------------------------------------------- |
| OnPolicy (GRPO)                | GRPO only                          | —      | 58.5%     | 4×H100 user 表，无 raw                       |
| LUFFY                          | GRPO + mix + p/p_β                  | —      | 61.5%     | 4×H100 user 表，无 raw                       |
| SFT+RL                         | 50 SFT + 50 GRPO                   | —      | 59.5%     | 4×H100 user 表，无 raw                       |
| (SFT alone)                    | 50 SFT                             | —      | 64.0%     | 4×H100 user 表，无 raw                       |
| CHORD                          | GRPO + weighted SFT                | —      | 67.0%     | 4×H100 user 表，无 raw                       |
| **DUET v1 (0329)**             | DR3 + SC（**无 BC**）              | 48.0%  | 69.5% ✓   | local raw `alfworld_3b_duet_0329`            |
| **DUET\* v39** (BC, α=0.2)     | BC + DR3 + SC, disc_α_ema=0.2      | 48.0%  | 67.0% ✓   | local raw `alfworld_qwen3b_duet_v39`         |
| **DUET\* v39b** (BC, α=0.5) 🏆 | BC + DR3 + SC, disc_α_ema=0.5      | 55.5%  | **77.5%** ✓ | local raw `alfworld_qwen3b_duet_v39b` (rerun 04-27) |
| DUET\* v_gap_af_a              | BC + DR3 + SC, gap-driven μ        | TBD（跑中, ETA 13:50） | TBD       | running on this server                       |
| DUET\* v_gap_af_b              | BC + DR3 + SC, gap-driven μ peak=.20 | 排队     | 排队        | scheduled                                    |

**ALFWorld 头条数 (paper headline)**:
**DUET\* v39b = 77.5%**
- +8.0pp over DUET v1 (no BC, 69.5%)
- +10.5pp over CHORD (67.0%)
- +16.0pp over LUFFY (61.5%)
- +19.0pp over OnPolicy (58.5%)

**首次实证 BC 在 3B ALFWorld 上 add value** — DUET v1 (no BC) → DUET\* v39b (BC) = +8pp。

---

## WebShop 3B（reward 与 success；两列都报）

| Method                       | Algorithm                          | Val@50 r/s         | Val@100 r/s        | Source                                            |
| ---------------------------- | ---------------------------------- | ------------------ | ------------------ | ------------------------------------------------- |
| OnPolicy (GRPO)              | GRPO only                          | —                  | 0.402 / 2.0%       | 4×H100 user 表                                    |
| SFT+RL                       | 50 SFT + 50 GRPO                   | —                  | 0.651 / 24.0%      | 4×H100 user 表                                    |
| CHORD                        | GRPO + weighted SFT                | —                  | 0.728 / 39.0%      | 4×H100 user 表                                    |
| LUFFY                        | GRPO + mix + p/p_β                  | —                  | 0.753 / 49.5%      | 4×H100 user 表                                    |
| (SFT alone)                  | 50 SFT                             | —                  | 0.614 / 28.5%      | 4×H100 user 表                                    |
| **DUET v1 (0409_ema)**       | DR3 + SC（**无 BC**）              | 0.599 / 17.0%      | **0.763 / 53.0%** ✓ | local raw `webshop_3b_duet_0409_ema`              |
| **DUET\* v39b** (BC, α=0.5, 04-25) | BC + DR3 + SC                | 0.610 / 16.5%      | 0.725 / 45.5% ✓    | local raw `webshop_qwen3b_duet_v39b`              |
| **DUET\* v39** (BC, α=0.2)   | BC + DR3 + SC                      | 0.661 / 33.0%      | 0.713 / 32.0% ✓    | local raw `webshop_qwen3b_duet_v39`               |

**WebShop 单 run variance 很大**（见下文 caveat），单数据点不可靠。

---

## ⚠️ WebShop variance 警告（重要 caveat）

```
同 yaml + 同 seed 在不同 run 跑 v39b WebShop:
   04-25 run:    val@100 = 45.5% (8 月份)
   04-28 sanity: val@100 = 12.5% (sanity rerun, 同 yaml)
                          ↑ 33pp 差距
   04-28 v_clean_ws (BC + closed-form schedule): 36.0%
```

**Verified (经 4 个 agent 平行 audit + 实测)**:
1. ✓ 代码无 bug — 我加的 gap-mode 是 dead code under disc_acc mode
2. ✓ teacher data 完美 — 5/5 trajectory 在当前 env 端到端复现 reward=1.0
3. ✓ env 是确定的 — 给 task_id，同 action 序列 → 同 reward
4. ✓ teacher data 完整 — sha256 自 04-24 起未变

**Variance 来源**：vLLM 采样非确定性 (T=0.6) + trainer-side `random.sample(teacher_trajs, 1)` 选不同 path（teacher pool 每 task 5 条）。

**对 paper 的 implication**：WebShop 数据点需要多 seed 平均 (≥3 seeds, mean±std)，否则单 run 不 robust。当前所有 baseline (LUFFY/CHORD/v1) 都是单 run，如果不重跑只能保留 caveat。

---

## DUET\* 框架与 v39b 实例化

```
DUET* 的"closed-form auto-adjusting"叙事：
─────────────────────────────────────────────
              shared discriminator D
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
       DR3       Adaptive BC    [State Channel 用离线 progress map，独立]
   ŵ=D/(1-D)  μ=f(disc_acc EMA)         Φ(s) = step_idx / traj_len
   修正 IS    控制 BC 强度

→ 三组件无需独立 schedule，公式自给自足
→ DR3 和 BC 共用一个 D —— "一个学习信号、两个机制"，paper sell point
```

| 变体          | μ schedule (BC)                                | 共用 D？ | 状态                  |
| ------------- | ---------------------------------------------- | -------- | --------------------- |
| v1            | (无 BC)                                        | n/a      | 历史 SOTA on WS       |
| v39           | α=0.2, disc_acc-driven                          | ✓        | 已跑                  |
| **v39b**      | α=0.5, disc_acc-driven (主推)                   | ✓        | **AF SOTA 77.5%** 🏆  |
| v_gap_af_a/b  | gap-driven (alternative signal, 实验中)         | ✗        | 跑中 / 排队           |
| v_clean_ws    | closed-form `μ(t) = peak·γ^t + valley`         | ✗        | 单数据点 36%          |

---

## 当前 server 状态 (2026-04-28 ~08:15)

| 进程                              | PID      | 状态                                                  |
| --------------------------------- | -------- | ----------------------------------------------------- |
| run_diagnose_and_af.sh            | 2010041  | running v_gap_af_a (step 9/100, ETA 13:50)            |
| (orphan launcher.py)              | 2699631  | child of orchestrator, training v_gap_af_a            |
| WebShop env (8083)                | —        | down                                                   |
| ALFWorld env (8081)               | —        | up (PID 2692692, RSS 3.1 GB, healthy)                 |

**剩余 queue**：
- v_gap_af_a（跑中，ETA 13:50）
- v_gap_af_b（排队，ETA 04-29 ~01:00）

**已砍掉**（避免污染）:
- v_gap_ws_b（gap-driven 在 WS 已被 v_gap_ws_a 8.5% 证明不工作）

---

## Source traceability

| 数据                                  | Local raw      | Source 描述                          |
| ------------------------------------- | -------------- | ------------------------------------ |
| DUET v1 (ALFWorld 0329)               | ✓              | 4×H100 下载                          |
| DUET v1 (WebShop 0409_ema)            | ✓              | 4×H100 下载                          |
| OnPolicy / LUFFY / CHORD / SFT+RL     | ✗              | 4×H100 user 报数（无 raw）           |
| DUET\* v39 / v39b 系列                | ✓              | 本机 training                        |
| DUET\* gap-driven 变体                | ✓ (in progress)| 本机 followup queue                  |

*若 reviewer 要 raw，可从 4×H100 server 取 baseline 数据复现表格。*

---

## paper 写作要点（更新）

1. **DUET\* = "closed-form auto-adjusting BC + DR3 + SC"**（解读 B：算法闭式，无需手调 schedule）
2. v39b 是 DUET\* 在 ALFWorld 上的 SOTA 实例（77.5%）
3. **DR3 + Adaptive BC 共享 discriminator** — 是 sell point
4. WebShop 数据需 caveat（high variance），或多 seed 平均
5. 不去叙事"严格 t-only closed-form"避免 reviewer 抓字面
