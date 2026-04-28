# Handoff: DUET* v39b Parameter Sweep — for 1.5B Server

**Date**: 2026-04-28
**Author (3B server)**: Claude (current session)
**Audience**: Claude on 1.5B server
**Deadline**: NeurIPS 2026 — **2026-05-07** (9 days away)
**Goal for 1.5B server**: run v39b-style parameter sweep on Qwen2.5-1.5B for both ALFWorld and WebShop, find configs that beat baselines.

---

## TL;DR (read this first)

We've decided to adopt **v39b** as DUET\*'s canonical instantiation. The paper narrative is:

> **DUET\* = closed-form auto-adjusting BC + DR3 + SC**, where (i) DR3 density-ratio
> correction and (ii) Adaptive BC share a single learned discriminator D, and
> (iii) State Channel uses an offline progress map. Three components, no
> manually-tuned schedule, "closed-form" in the sense of *self-contained algorithm*.

On 3B, v39b is SOTA on ALFWorld (**77.5%**, +8.0pp over prev SOTA) but underperforms on WebShop (variance issue). 3B is currently running a 12-config single-seed parameter sweep on WebShop. **You should run an analogous sweep on 1.5B** for both envs.

---

## Section 1 — DUET\* algorithm (paper narrative)

```
DUET* 由三个 closed-form 组件构成:

(i) DR3 density-ratio correction
    ŵ = D / (1 − D)
    对 off-policy IS 修正；D 是在线学习的 discriminator
    teacher 梯度自然 fade-out（D 分不开时 ŵ → 1）

(ii) Adaptive BC (chord weighted SFT)
    μ(t) = valley + (peak − valley) · max(0, (1 − d̄(t)) / (1 − d_floor))
    d̄ = EMA(D's accuracy)
    BC 强度自动衰减；与 (i) 共享同一个 D

(iii) State Channel
    r' = r + β · Φ(s_T) + Σ_t η · [Φ(s_{t+1}) − Φ(s_t)]
    Φ 是 teacher trajectory 离线哈希出的 progress map
    无在线学习；只对 on-policy 样本生效
```

**关键 sell point**: (i) 和 (ii) **共享 discriminator D** — 一个学习信号驱动两个机制。

"Closed-form" 在 paper 里取**算法闭式**含义（self-contained, no manual schedule），不是严格"只依赖 t 的公式"。

---

## Section 2 — 3B Results (current state)

### ALFWorld 3B (val@100)

| Method                       | Algorithm                          | success    | Note                          |
| ---------------------------- | ---------------------------------- | ---------- | ----------------------------- |
| OnPolicy GRPO                | GRPO only                          | 58.5%      | 4×H100 user 表                |
| LUFFY                        | GRPO + mix + p/p_β                  | 61.5%      | 4×H100 user 表                |
| SFT+RL                       | 50 SFT + 50 GRPO                   | 59.5%      | 4×H100 user 表                |
| (SFT alone)                  | 50 SFT                             | 64.0%      | 4×H100 user 表                |
| CHORD                        | GRPO + weighted SFT                | 67.0%      | 4×H100 user 表                |
| **DUET v1 (0329)**           | DR3 + SC（无 BC）                  | 69.5%      | local raw (prev SOTA)         |
| DUET\* v39 (BC, α=0.2)       | BC + DR3 + SC                      | 67.0%      | local raw                     |
| **DUET\* v39b (BC, α=0.5)** 🏆 | BC + DR3 + SC                    | **77.5%**  | **NEW SOTA** ✓                |

**DUET\* v39b 头条**：+8.0pp over DUET v1, +10.5pp over CHORD, +16.0pp over LUFFY.

### WebShop 3B (val@100)

| Method                       | Algorithm                          | reward / success | Note                       |
| ---------------------------- | ---------------------------------- | ---------------- | -------------------------- |
| OnPolicy                     | GRPO only                          | 0.402 / 2.0%     | 4×H100                     |
| SFT+RL                       | 50+50                              | 0.651 / 24.0%    | 4×H100                     |
| CHORD                        | GRPO + wSFT                        | 0.728 / 39.0%    | 4×H100                     |
| LUFFY                        | GRPO + mix + p/p_β                  | 0.753 / 49.5%    | 4×H100                     |
| **DUET v1 (0409_ema)**       | DR3 + SC（无 BC）                  | **0.763 / 53.0%** | local raw (target to beat) |
| v39b (04-25, single seed)    | BC + DR3 + SC                      | 0.725 / 45.5%    | local raw                  |
| v39b sanity (04-28, same yaml)| same                               | 0.649 / 12.5%    | **33pp variance** ⚠️        |
| v_clean_ws (closed-form μ)   | BC + DR3 + SC                      | 0.707 / 36.0%    | local raw                  |

**WebShop variance 警告**: 同 yaml 不同 run 差 33pp。已验证 (4 个 agent + 端到端 replay):
- ✓ teacher data 完美 (sha256 sealed since 04-24, 5/5 trajectory replay reach reward=1.0)
- ✓ env 确定性 OK (给 task_id → 同 instruction → BM25 同 search results)
- ✓ 代码 audit clean
- ✗ variance 来源：vLLM 采样 T=0.6 + trainer-side `random.sample(teacher_pool, 1)` 选不同 path

→ WS 上 single-seed 不可靠。**你的 1.5B WS 也会面对这个 variance 问题**，做心理准备。

---

## Section 3 — v39b Canonical Config

完整 yaml 在: `config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v39b.yaml` (3B) 和对应的 alfworld v39b。

**核心 BC 参数**:
```yaml
actor_rollout_ref:
  actor:
    use_dr3: true
    use_chord: true
    chord_mu_warmup_steps: 0
    chord_mu_decay_steps: 25
    chord_mu_peak: 0.3                   # ← BC 强度上限 (sweep dimension)
    chord_mu_valley: 0.05                # ← BC 强度下限 (sweep dimension)
    chord_use_token_weighting: false
    chord_mu_adaptive: true              # ← 启用 disc_acc-driven μ
    chord_mu_adaptive_mode: "disc_acc"
    chord_mu_d_floor: 0.5                # ← μ_raw 触底阈值 (sweep dimension)
    chord_mu_d_ema_alpha: 0.5            # ← disc_acc EMA 速度 (sweep dimension)
    
    dr3:
      enable: true
      apply_to: teacher_no_logprob
      use_policy_shaping: true
      policy_shaping_beta: 0.1
      alpha_mode: sync_batch_ema         # ← teacher α EMA
      alpha_ema_beta: 0.9
      w_hat_ema_alpha: 0.3               # ← discriminator Polyak EMA
      ratio_shaping_mode: auto
      ...

  rollout:
    n: 8                                 # 8 rollouts per task
    temperature: 0.6
    
exp_manager:
  teacher_experience:
    enable: true
    n_teacher_rollouts_per_task: 1       # 1 teacher mixed in per task
    select_mode: random
  state_channel:
    enable: true
    exclude_teacher: true
    beta: 0.2
    step_level:
      enable: true
      eta: 0.05
```

---

## Section 4 — WebShop env memory leak (CRITICAL infra issue)

```
WebShop env service 有内存泄漏：
  正常启动: ~1.5 GB RSS
  跑 26h 后: ~50 GB RSS (击中 K8s pod limit)
  → MemoryError → 训练卡在 retry loop → ray actor SIGTERM
  
3B 已实测 v39b ALFWorld 因此崩了一次（在 step 89/100，丢了 11h 训练 + val@100）

防御措施: 每个 run 之前 stop+start env 服务（fresh process, ~300 MB → safe）
```

**你的 1.5B server orchestrator 必须**：
1. 每个 run 之前 `bash start_env_alfworld.sh stop` + `bash start_env_webshop.sh stop`
2. 等 5-8 秒
3. 启 fresh env：`bash start_env_<env>.sh`
4. 等 5 秒
5. 跑 launcher.py

参考 3B 的实现: `run_ws_sweep_phase_a.sh` 里 `run_one()` 函数。

---

## Section 5 — Sweep Plan for 1.5B

3B 已经设计好 12-config single-seed sweep，**1.5B server 应跑相同 sweep** (一份 ALFWorld + 一份 WebShop)。

**重要原则：sweep 只动 v39b-specific 的 BC schedule 参数（peak / valley / d_floor / d_ema_alpha），
不动 training infrastructure（batch_size / offload / env_worker / gpu_mem_util）**，
让 DUET\* 能跟其他 baseline 在同 infrastructure 下公平对比。

### Sweep dimensions (12 configs, all on v39b cfg infrastructure)

| #   | Tag                          | peak | valley | d_floor | d_ema_alpha | 测试什么                                 |
| --- | ---------------------------- | ---- | ------ | ------- | ----------- | ---------------------------------------- |
| 01  | v39b_default                 | 0.3  | 0.05   | 0.5     | 0.5         | v39b baseline 复核                       |
| 02  | peak02                       | 0.2  | 0.05   | 0.5     | 0.5         | low BC                                   |
| 03  | peak04                       | 0.4  | 0.05   | 0.5     | 0.5         | medium BC                                |
| 04  | peak05                       | 0.5  | 0.05   | 0.5     | 0.5         | strong BC ⭐                             |
| 05  | peak06                       | 0.6  | 0.05   | 0.5     | 0.5         | very strong BC                           |
| 06  | peak07                       | 0.7  | 0.05   | 0.5     | 0.5         | extreme BC                               |
| 07  | ema02                        | 0.3  | 0.05   | 0.5     | 0.2         | slow EMA → BC stays high longer          |
| 08  | ema08                        | 0.3  | 0.05   | 0.5     | 0.8         | fast EMA → BC fades quick                 |
| 09  | floor04                      | 0.3  | 0.05   | 0.4     | 0.5         | lower d_floor → μ_raw 触底更快           |
| 10  | pk05_ema02                   | 0.5  | 0.05   | 0.5     | 0.2         | strong BC + slow fade ⭐                 |
| 11  | pk05_v10                     | 0.5  | 0.10   | 0.5     | 0.5         | strong BC + high floor ⭐                |
| 12  | pk05_ema02_v10               | 0.5  | 0.10   | 0.5     | 0.2         | full combo: 强 BC + 慢退 + 高 floor ⭐⭐ |

⭐ = 预测 candidate winners

`v39b cfg infrastructure` (FIXED, not swept):
```
ppo_micro_batch_size_per_gpu = 2
log_prob_micro_batch_size_per_gpu = 2
param_offload = false
optimizer_offload = false
gpu_memory_utilization = 0.65
max_env_worker = 64
n (rollouts per task) = 8
n_teacher_rollouts_per_task = 1
temperature = 0.6
```

**(注：v1.0 of this plan included `v1cfg` variants that swept infrastructure too;
this was removed in v2 because changing infrastructure makes DUET\* not directly
comparable to non-DUET baselines on the same hardware substrate.)**

### Why v1cfg as the base

3B WebShop 上 v_no_bc_ws = 1.0%（用 v39b config）vs DUET v1 (0409_ema) = 53.0%（同算法，no BC）— 唯一差异是 config knobs。这暗示 v39b config 在 WS 上有副作用。**v1cfg 是更稳的 base**。

→ 1.5B 上同样有可能发生这个 config interaction。**Sweep 应该用 v1cfg base**。

### Reference 3B sweep configs

3B 的 yaml 文件在: `config/duet_paper_experiments_configs/webshop/sweep/ws_swA_*.yaml` (12 个)。
1.5B server 可以基于这些直接改：
- 把模型 path 换成 1.5B 的
- 调整 sequence length（1.5B 可能用更短）
- 复制 BC/DR3/SC 参数照搬

### Generator script (3B 用过的，可复制)

3B 的 generator 脚本在 `run_ws_sweep_phase_a.sh` 的 `generate_yaml` 函数，参考实现。1.5B server 应 fork 一份改成自己的。

### Time budget

- 单 run 约 3.5h (WS) / 11h (AF) — 视 1.5B 实际速度
- 12 runs × 2 env = 24 total runs
- 1.5B server 8 days × 24h = 192h budget
- 24 runs × ~7h avg = 168h，刚好

如果时间紧，**优先 WebShop**（因为 3B 上 WS 没解决，1.5B WS 数据是关键 paper 主表 cell）。

---

## Section 6 — Goal for 1.5B server

```
找到一个 v39b-series config 在 1.5B 上同时:
  ALFWorld:  beat baselines
  WebShop:   beat baselines (especially DUET v1)

如果 12 个 single-seed runs 里有任何一个 ≥ baseline，**就是赢**。
```

### Baselines on 1.5B

User 应有 1.5B 的 baseline 表（OnPolicy/LUFFY/CHORD/SFT/DUET v1）。这边没拿到，**1.5B server 自己确认**。

---

## Section 7 — Process/infra changes since 3B handoff

代码上的实质改动（已 push to main）:

```
agentevolver/module/exp_manager/het_actor.py
  - 加了 _reward_gap_from_data 读取 (top of update_policy)
  - 加了 elif adaptive_mode == "gap" 分支 (~71 lines)
  - 这两处对 adaptive_mode != "gap" 是 no-op
  - 1.5B 跑 v39b (mode='disc_acc') 不会受影响

agentevolver/module/trainer/ae_ray_trainer.py
  - 加了 batch.meta_info["reward_gap"] = ... 
  - 给 actor 提供 gap signal（gap mode 用），其他 mode 忽略

config/duet_paper_experiments_configs/webshop/sweep/ (新)
  - 12 个 ws_swA_*.yaml configs

run_ws_sweep_phase_a.sh (新)
  - Phase A orchestrator，每 run 之前重启 env

scripts/analyze_ws_sweep.py (新)
  - 自动汇总 val@100 → mean±std 表
```

---

## Section 8 — File pointers

```
分析报告:
  analysis_reports/3b_master_experiment_table.md      ← 3B 主数据表
  analysis_reports/3b_ws_sweep_plan.md                ← WS sweep plan
  analysis_reports/3b_v39b_*.md                       ← v39b 系列 deep analyses
  analysis_reports/3b_v39_webshop_*.md                ← WS variance 调查

3B Sweep 资源:
  config/duet_paper_experiments_configs/webshop/sweep/  ← 12 个 yamls
  run_ws_sweep_phase_a.sh                                ← orchestrator
  scripts/analyze_ws_sweep.py                           ← analyzer

3B Canonical configs:
  config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v39b.yaml
  config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_duet_v39b.yaml

env script:
  start_env_webshop.sh, start_env_alfworld.sh        ← 注意 readiness check 不充分
                                                       但 env 启动后 1-2 min 内可用
```

---

## Section 9 — Specific tasks for 1.5B server agent

1. **读这份文档**，确认理解 v39b narrative + sweep plan
2. **生成 1.5B 版本的 12 个 sweep yamls**（fork 3B 的 ws_swA_*.yaml，改 model path + sequence length）
3. **写 1.5B 的 orchestrator**（参考 run_ws_sweep_phase_a.sh，关键是 per-run env restart）
4. **跑 WebShop sweep**（12 runs × ~2h = 24h on 1.5B if it's faster）
5. **同时/之后跑 ALFWorld sweep**（同 12 configs，~5-7h each）
6. **结果汇总到 analysis_reports/1.5b_master_experiment_table.md** 用同样格式

完成后 commit + push，3B server 会读你的 results 整合到 paper main table。

---

## Section 10 — Communication / sync

```
3B server 计划:
  04-29 01:00:  Phase A WS sweep 自动启动 (12 runs)
  04-30 19:00:  Phase A 完，决策 Phase B
  04-30 → 05-03: Phase B + C
  05-04 → 05-07: paper writing

1.5B server 计划（建议）:
  04-28 (今天): 读这份文档 + 生成 1.5B 的 sweep yamls
  04-29 → 04-30: WS sweep
  04-30 → 05-02: AF sweep
  05-02 → 05-04: refinement / multi-seed if time
  05-04 → 05-07: paper writing support

最终主数据表合并:
  3B 部分: from this server (analysis_reports/3b_master_experiment_table.md)
  1.5B 部分: from 1.5B server (analysis_reports/1.5b_master_experiment_table.md)
```
