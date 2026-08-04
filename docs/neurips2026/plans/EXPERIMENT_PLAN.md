# DUET 7B 实验计划

> 目标：NeurIPS 2026 投稿（截止 ~2026-05-07）
> 服务器：8x A100-SXM4-80GB, CUDA 12.8
> 更新日期：2026-04-10

---

## 实验总览

### 已完成：3B 实验结果 (Qwen2.5-3B-Instruct, 4xGPU, bs=8, 800 tasks)

| 环境 | 指标 | DUET (0409_ema) | LUFFY | CHORD | On-policy GRPO |
|------|------|:-:|:-:|:-:|:-:|
| ALFWorld | Success Rate | **69.5%** | 61.5% | 54.5% | 58.5% |
| WebShop | Avg Reward | **0.763** | 0.753 | -0.100 | 0.402 |
| WebShop | Success Rate | **53.0%** | 49.5% | 0.0% | 2.0% |

### 进行中：7B 实验 (Qwen2.5-7B-Instruct, 8xGPU, bs=16, 1600 tasks)

---

## Phase 1: 7B Baseline 实验（当前阶段）

目标：先拿到所有 baseline 数据点，再跑 DUET。

### 实验配置变更（相对 3B）

| 参数 | 3B | 7B |
|------|----|----|
| 模型 | Qwen2.5-3B-Instruct | Qwen2.5-7B-Instruct |
| 模型路径 | `/data/shared_models/Qwen2.5-3B-Instruct` | `/data/shared_models/Qwen2.5-7B-Instruct` |
| GPU 数 | 4 | 8 |
| TP size | 1 | 2 |
| train_batch_size | 8 | 16 |
| ppo_mini_batch_size | 8 | 16 |
| max_train_tasks | 800 | 1600 |
| rollout.n | 8 | 8 (不变) |
| 学习率 | 1.0e-06 | 5.0e-07 |
| FSDP offload | param+optimizer | param+optimizer |
| 每 epoch 总步数 | 800/8=100 | 1600/16=100 |

### 实验清单

| # | 环境 | 算法 | 配置文件 | 状态 | 结果 |
|---|------|------|----------|:----:|------|
| 1 | WebShop | LUFFY | `webshop/webshop_7b_luffy.yaml` | TODO | |
| 2 | WebShop | On-policy GRPO | `webshop/webshop_7b_onpolicy.yaml` | TODO | |
| 3 | WebShop | CHORD | `webshop/webshop_7b_chord.yaml` | TODO | |
| 4 | ALFWorld | LUFFY | `alfworld/alfworld_7b_luffy.yaml` | TODO | |
| 5 | ALFWorld | On-policy GRPO | `alfworld/alfworld_7b_onpolicy.yaml` | TODO | |
| 6 | ALFWorld | CHORD | `alfworld/alfworld_7b_chord.yaml` | TODO | |

**环境启动：**
```bash
# 启动 ALFWorld 环境 (AgentGym :36001 + env_service :8081)
bash start_env_alfworld.sh

# 启动 WebShop 环境 (AgentGym :36003 + env_service :8083)
bash start_env_webshop.sh
```

**运行命令：**
```bash
# WebShop (确保 webshop 环境已启动)
conda activate duet
python launcher.py --conf config/duet_paper_experiments_configs/webshop/webshop_7b_luffy.yaml --kill
python launcher.py --conf config/duet_paper_experiments_configs/webshop/webshop_7b_onpolicy.yaml --kill
python launcher.py --conf config/duet_paper_experiments_configs/webshop/webshop_7b_chord.yaml --kill

# ALFWorld (确保 alfworld 环境已启动)
conda activate duet
python launcher.py --conf config/duet_paper_experiments_configs/alfworld/alfworld_7b_luffy.yaml --kill
python launcher.py --conf config/duet_paper_experiments_configs/alfworld/alfworld_7b_onpolicy.yaml --kill
python launcher.py --conf config/duet_paper_experiments_configs/alfworld/alfworld_7b_chord.yaml --kill
```

**停止环境：**
```bash
bash start_env_alfworld.sh stop
bash start_env_webshop.sh stop
```

### 建议运行顺序

同环境实验串行（共用 env_service），不同环境可在不同机器并行：

```
WebShop: LUFFY → On-policy → CHORD  （预计每个实验 ~10-15h）
ALFWorld: LUFFY → On-policy → CHORD
```

---

## Phase 2: 7B DUET 实验

Phase 1 baseline 完成后启动。需先将 7B DUET 配置对齐 3B 最佳参数 (0409_ema)。

### 待更新参数（webshop_7b_duet.yaml / alfworld_7b_duet.yaml）

| 参数 | 当前 7B 值 | 应改为 (对齐 0409_ema) | 说明 |
|------|-----------|----------------------|------|
| `dr3.w_hat_ema_alpha` | 未设置 | **0.3** | 核心突破 |
| `dr3.use_policy_shaping` | 未设置 | **true** | Hybrid 模式 |
| `dr3.policy_shaping_beta` | 未设置 | **0.1** | |
| `dr3.gap_gate_enable` | true | **false** | 避免双重抑制 |
| `dr3.disc_temperature` | 1.2 | **1.0** | 更锐利判别器 |
| `dr3.disc_label_smoothing` | 0.05 | **0.1** | |
| `dr3.clip_max` | 10.0 | **5.0** | |
| `state_channel.match_mode` | stage (WS) | **attribute_aware** (WS) | 覆盖率 0%→100% |
| `state_channel.beta` | 0.5 | **0.2** | |
| `state_channel.beta_decay` | true | **false** | |
| `state_channel.grpo_decouple` | 未设置 | **true** | 防优势扭曲 |
| `state_channel.exclude_teacher` | 未设置 | **true** | |
| `state_channel.step_level.enable` | false | **true** (eta=0.05) | |
| `adaptive_weight.enable` | true | **false** | |

| # | 环境 | 算法 | 配置文件 | 状态 | 结果 |
|---|------|------|----------|:----:|------|
| 7 | WebShop | DUET | `webshop/webshop_7b_duet.yaml` | TODO | |
| 8 | ALFWorld | DUET | `alfworld/alfworld_7b_duet.yaml` | TODO | |

---

## Phase 3: 多种子验证 + 消融实验

在 7B 主实验完成后，根据需要安排。

| # | 任务 | 说明 | 状态 |
|---|------|------|:----:|
| 9 | WebShop 3B 多种子 | DUET vs LUFFY 仅 +3.5pp，需 3+ 种子 | TODO |
| 10 | ALFWorld 3B 多种子 | 验证 +8pp 的统计显著性 | TODO |
| 11 | 7B 多种子 | 如果 7B 结果 margin 小 | TODO |
| 12 | DR3-only 消融 | 仅 Action Channel, 无 SC | TODO |
| 13 | SC-only 消融 | 仅 State Channel, 无 DR3 | TODO |

---

## Phase 4: 论文写作

| # | 任务 | 截止日期 | 状态 |
|---|------|----------|:----:|
| 14 | 主实验结果表 (3B+7B) | 04-25 | TODO |
| 15 | 消融实验表 | 04-28 | TODO |
| 16 | 学习曲线图 | 04-25 | TODO |
| 17 | Agent 行为分析 (case study) | 04-28 | TODO |
| 18 | 论文初稿 | 05-01 | TODO |
| 19 | 论文定稿 | 05-05 | TODO |

---

## 关键风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 7B 训练 OOM | 阻塞实验 | 已启用 FSDP offload; 可降 gpu_memory_utilization |
| CHORD 7B 格式崩溃 | 数据点缺失 | 3B 已观测到 WebShop 崩溃，7B 可能重现；ALFWorld 应正常 |
| 训练时间过长 | 拖延进度 | 100 steps/epoch 与 3B 一致；8GPU 应加速 |
| baseline 结果异常 | 对比失真 | 对比 3B 趋势，及时发现问题 |

---

## 监控指标 (wandb)

| 指标 | 说明 |
|------|------|
| `critic/success_onpolicy/mean` | 主要性能指标 |
| `critic/avg_reward/mean` | 平均奖励 |
| `actor/kl_loss` | 策略稳定性，应 < 0.5 |
| `diag/teacher_sample_ratio` | Teacher 混合比例 (LUFFY/CHORD) |
