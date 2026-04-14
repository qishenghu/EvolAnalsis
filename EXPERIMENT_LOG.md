# DUET Paper — Experiment Log

> 持续更新的实验记录，包含计划、进度、结论和复现命令。

**目标**: NeurIPS 2026 submission (~2026-05-07)  
**核心主张**: DUET (DR3 + State Channel) 提供 data-driven teacher curriculum，在 capability gap 大时显著提升，gap 小时安全退出（graceful degradation）  
**实验策略**: Qwen2.5 × 3 尺度 (1.5B, 3B, 7B) × 2 环境 (ALFWorld, WebShop)。1.5B/3B 为主要结果，7B 为 scaling analysis（展示 DR3 fade-out 的正确性）

---

## 一、实验矩阵总览

### 1.1 主实验：1.5B + 3B（Paper 主表）

| 环境 | 模型 | OnPolicy | LUFFY | CHORD | DUET | 状态 |
|------|------|----------|-------|-------|------|------|
| ALFWorld | Qwen2.5-1.5B | ⬚ 待跑 | ⬚ 待跑 | ⬚ 待跑 | ⬚ 待跑 | **优先** |
| ALFWorld | Qwen2.5-3B | ✅ 已完成 | ✅ 已完成 | ❌ 需重跑 | ✅ 已完成 | 缺 CHORD |
| WebShop | Qwen2.5-1.5B | ⬚ 待跑 | ⬚ 待跑 | ⬚ 待跑 | ⬚ 待跑 | **优先** |
| WebShop | Qwen2.5-3B | ✅ 已完成 | ✅ 已完成 | ✅ 已完成 | ✅ 已完成 | 全部完成 |

### 1.2 Scaling Analysis：7B（Paper 分析节，非主表）

| 环境 | 模型 | OnPolicy | LUFFY | CHORD | DUET | 用途 |
|------|------|----------|-------|-------|------|------|
| ALFWorld | Qwen2.5-7B | ✅ | ✅ | ✅ (86步) | ✅ | DR3 fade-out 验证 |
| WebShop | Qwen2.5-7B | ✅ | ✅ | ✅ (崩溃) | ✅ (93步) | Teacher gap collapse 分析 |

> **7B 的叙事定位**：不是"DUET 在 7B 上没用"，而是"DR3 正确检测到 teacher 已无用并自动降权至 1.3%，DUET 安全退出（+1.5%），而 LUFFY 因缺乏 fade-out 反而有害（-2.5% ~ -3.7%）"。

> **Llama-3.2-3B 实验已放弃** (2026-04-14)：Instruct 版本的 tool-calling 微调导致模型强制输出 JSON 格式，无法使用 XML action tags，前 50 步 0% 成功率。

---

## 二、已完成实验结果

### 2.1 Qwen2.5-3B — WebShop（基准结果）

| 方法 | Val@50 | Val@100 | vs OnPolicy | wandb run |
|------|--------|---------|-------------|-----------|
| OnPolicy | 0.276 | 0.402 | — | `27ysbdvi` |
| LUFFY | 0.509 | 0.753 | +87.3% | `o405qtk1` |
| CHORD | 0.572 | 0.728 | +81.1% | `lawzxf7d` |
| **DUET** | **0.599** | **0.763** | **+89.8%** | `v1df0dep` |

**结论**: DUET 在 WebShop 3B 上全面领先。OnPolicy 在 step 60-72 发生灾难性退化 (0.53→0.10)。

### 2.2 Qwen2.5-3B — ALFWorld（基准结果，缺 CHORD）

| 方法 | Val@50 | Val@100 | vs OnPolicy | wandb run |
|------|--------|---------|-------------|-----------|
| OnPolicy | 0.475 | 0.585 | — | `is8xwpd4` |
| LUFFY | 0.475 | 0.615 | +5.1% | `rdnmotb3` |
| **DUET** | 0.480 | **0.695** | **+18.8%** | `9ryexv2i` |
| CHORD | — | — | — | 需重跑 |

### 2.3 Qwen2.5-7B — Scaling Analysis 汇总

| 方法 | ALFWorld Val@100 | WebShop Val@100 | 备注 |
|------|----------------|-----------------|------|
| OnPolicy | 0.850 | 0.760 | 7B 自身已足够强 |
| LUFFY | 0.825 (**-2.9%**) | 0.755 (**-3.7%**) | **有害** |
| DUET | 0.865 (+1.8%) | N/A (93步崩溃) | 安全/中性 |

---

## 三、待执行实验计划

### 3.1 Phase 1: Qwen2.5-1.5B × ALFWorld（最高优先级）

**运行命令**:
```bash
nohup bash run_qwen1.5b_alfworld.sh > logs/qwen1.5b_alfworld_all.log 2>&1 &
```

**脚本**: `run_qwen1.5b_alfworld.sh`  
**执行方式**: 2 Rounds × 2 并行实验 (4 GPU each)  
- Round 1: OnPolicy (GPU 0-3) + LUFFY (GPU 4-7)
- Round 2: CHORD (GPU 0-3) + DUET (GPU 4-7)

**Configs** (已优化: offload=false, gpu_mem=0.75, micro_batch=2, env_worker=64):
| 方法 | Config 路径 |
|------|------------|
| OnPolicy | `config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_onpolicy.yaml` |
| LUFFY | `config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_luffy.yaml` |
| CHORD | `config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_chord.yaml` |
| DUET | `config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_duet.yaml` |

### 3.2 Phase 2: Qwen2.5-1.5B × WebShop

**运行命令**:
```bash
nohup bash run_qwen1.5b_webshop.sh > logs/qwen1.5b_webshop_all.log 2>&1 &
```

**脚本**: `run_qwen1.5b_webshop.sh`  
**Configs**:
| 方法 | Config 路径 |
|------|------------|
| OnPolicy | `config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_onpolicy.yaml` |
| LUFFY | `config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_luffy.yaml` |
| CHORD | `config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_chord.yaml` |
| DUET | `config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet.yaml` |

### 3.3 Phase 3: Qwen2.5-3B 重跑（补 CHORD + 可选多 seed）

**运行命令**:
```bash
# ALFWorld 全部 4 方法
nohup bash run_qwen3b_alfworld.sh > logs/qwen3b_alfworld_all.log 2>&1 &

# WebShop 全部 4 方法（可选，已有结果）
nohup bash run_qwen3b_webshop.sh > logs/qwen3b_webshop_all.log 2>&1 &
```

**脚本**: `run_qwen3b_alfworld.sh`, `run_qwen3b_webshop.sh`  
**Configs** (已优化: offload=false, gpu_mem=0.65, micro_batch=2, env_worker=64):
| 环境 | 方法 | Config 路径 |
|------|------|------------|
| ALFWorld | OnPolicy | `config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_onpolicy.yaml` |
| ALFWorld | LUFFY | `config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_luffy.yaml` |
| ALFWorld | CHORD | `config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_chord.yaml` |
| ALFWorld | DUET | `config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_duet.yaml` |
| WebShop | OnPolicy | `config/duet_paper_experiments_configs/webshop/webshop_qwen3b_onpolicy.yaml` |
| WebShop | LUFFY | `config/duet_paper_experiments_configs/webshop/webshop_qwen3b_luffy.yaml` |
| WebShop | CHORD | `config/duet_paper_experiments_configs/webshop/webshop_qwen3b_chord.yaml` |
| WebShop | DUET | `config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet.yaml` |

---

## 四、Config 设计说明

### 4.1 公平对比原则

所有尺度的实验使用**相同的超参**（lr、batch size、DUET 参数等），仅 `model.path` 和基础设施参数不同：

| 参数 | 1.5B | 3B | 7B |
|------|------|----|----|
| `lr` | 1.0e-6 | 1.0e-6 | 5.0e-7 |
| `train_batch_size` | 8 | 8 | 8 |
| `n` (rollout) | 8 | 8 | 8 |
| `max_train_tasks` | 800 | 800 | 800 |
| `n_gpus_per_node` | 4 | 4 | 4 |
| `tensor_model_parallel_size` | 1 | 1 | 2 |
| `param_offload` | false | false | true |
| `gpu_memory_utilization` | 0.75 | 0.65 | 0.65 |
| `ppo_micro_batch_size_per_gpu` | 2 | 2 | 1 |
| `max_env_worker` | 64 | 64 | 64 |

### 4.2 各方法核心差异

| 方法 | Teacher 数据 | 关键机制 |
|------|-------------|---------|
| **OnPolicy (GRPO)** | 无 | 纯 on-policy RL，`teacher_baseline_separation: false` |
| **LUFFY** | 有 | `policy_shaping: enable=true, beta=0.1`，固定 teacher 混入 |
| **CHORD** | 有 | `use_chord: true`，mu 从 0.9 衰减到 0.05 (25 步)，GRPO + weighted SFT |
| **DUET** | 有 | DR3 (density ratio) + State Channel (progress shaping) |

### 4.3 CHORD 统一配置 (chord_mu_0410)

所有 CHORD 实验统一使用 decay schedule:
```yaml
chord_mu_warmup_steps: 0
chord_mu_decay_steps: 25
chord_mu_peak: 0.9        # 初始 SFT 权重
chord_mu_valley: 0.05     # 最终 SFT 权重
chord_use_token_weighting: false
```

### 4.4 DUET 配置差异 (ALFWorld vs WebShop)

| 参数 | ALFWorld | WebShop |
|------|----------|---------|
| `kl_loss_coef` | 0.005 | 0.001 |
| `dr3.gap_gate_enable` | true | false |
| `dr3.use_policy_shaping` | — | true (beta=0.1) |
| `dr3.disc_temperature` | 1.5 | 1.0 |
| `dr3.w_hat_ema_alpha` | — | 0.3 |
| `state_channel.match_mode` | hash | attribute_aware |
| `state_channel.beta_decay` | true (target=0.3) | false |
| `state_channel.grpo_decouple` | — | true |
| `adaptive_weight.enable` | true | false |

---

## 五、数据存放路径

### 5.1 模型
| 模型 | 路径 | 大小 |
|------|------|------|
| Qwen2.5-1.5B-Instruct | `/data/shared_models/Qwen2.5-1.5B-Instruct` | 2.9G |
| Qwen2.5-3B-Instruct | `/data/shared_models/Qwen2.5-3B-Instruct` | 6.5G |
| Qwen2.5-7B-Instruct | `/data/shared_models/Qwen2.5-7B-Instruct` | 15G |

### 5.2 Teacher 数据
| 环境 | 路径 |
|------|------|
| ALFWorld | `data/teacher_trajectories/qwen72b/alfworld_qwen72b_filtered_react_tags.pkl` |
| WebShop | `data/teacher_trajectories/qwen72b/webshop_qwen72b_filtered.pkl` |

### 5.3 实验输出
- Trajectory: `checkpoints/agentevolver/{experiment_name}/Trajectory/`
- Validation: `experiments/{env}/{experiment_name}/validation_log/`
- Training logs: `logs/{experiment_name}.log`

### 5.4 已有 wandb run 索引

**Qwen2.5-3B**:
| 实验 | wandb run id |
|------|-------------|
| webshop_3b_onpolicy | `27ysbdvi` |
| webshop_3b_luffy | `o405qtk1` |
| webshop_3b_chord_mu_0410 | `lawzxf7d` |
| webshop_3b_duet_0409_ema | `v1df0dep` |
| alfworld_3b_grpo_react_tags | `is8xwpd4` |
| alfworld_3b_luffy | `rdnmotb3` |
| alfworld_3b_duet_0329 | `9ryexv2i` |

**Qwen2.5-7B**:
| 实验 | wandb run id |
|------|-------------|
| alfworld_7b_onpolicy | `vpaqwmtw` |
| alfworld_7b_luffy | `cj640iqp` |
| alfworld_7b_chord | `9t00wfz7` |
| alfworld_7b_duet | `y903pomp` |
| webshop_7b_onpolicy | `lpglyxik` |
| webshop_7b_luffy | `b62rlamz` |
| webshop_7b_chord | `f6jm7m4n` |
| webshop_7b_duet | `wn5t64wb` |

---

## 六、分析报告索引

| 报告 | 路径 | 内容 |
|------|------|------|
| WebShop 7B 综合分析 | `analysis_reports/FINAL_7b_analysis_report.md` | Teacher Gap Collapse 核心结论 |
| WebShop 7B vs 3B 数据分析 | `analysis_reports/exp_analysis_7b_vs_3b.md` | 训练曲线、验证性能对比 |
| WebShop 7B 轨迹案例分析 | `analysis_reports/case_analysis_7b.md` | 行为模式、失败模式 |
| WebShop 7B 理论分析 | `analysis_reports/theory_analysis_7b.md` | DR3/SC 理论框架 |
| ALFWorld 7B vs 3B 综合分析 | `analysis_reports/alfworld_7b_vs_3b_analysis.md` | 跨环境验证 |

---

## 七、阶段性结论

### 结论 1: DUET 在 3B 上跨环境一致有效 (2026-04-14)

DUET 在 3B 上的优势在两个环境均显著（ALFWorld +18.8%, WebShop +89.8%），证明 DR3 + State Channel 双通道对弱模型的学习效率有实质性提升。

### 结论 2: 7B 验证 DR3 fade-out 的正确性 (2026-04-14)

7B 结果不是"方法失效"，而是 DR3 正确检测到 teacher 不再有用（gradient share 从 16.6% 降至 1.3%）。DUET 在 7B 上安全退出（+1.5%），而 LUFFY 因缺乏 fade-out 反而有害（-2.5% ~ -3.7%）。

### 结论 3: 实验策略转向 1.5B+3B 主实验 + 7B scaling analysis (2026-04-14)

- Llama-3.2-3B 因 tool-calling 先验放弃
- 增加 Qwen2.5-1.5B 构建三点 scaling curve (1.5B → 3B → 7B)
- Paper 叙事："DUET 随 capability gap 自适应调节，gap 大时大幅提升，gap 小时安全退出"

### 结论 4: Llama-Instruct 模型的 tool-calling 先验不兼容 XML tag 格式 (2026-04-14)

Llama-3.2-3B-Instruct 即使在 system prompt 明确要求 XML 格式 + "Do NOT respond with JSON" 的情况下，仍然 100% 输出 JSON tool-calling 格式。前 50 步 0% 成功率。需要约 70 步 RL 训练才能纠正格式，代价太大。Chat template 本身不注入 tool 指令（`tools=none` 时无 JSON 提示），问题在模型权重。

---

## 八、Paper 预期结果结构

### 主表 (Table 1): 1.5B + 3B 主实验

| 环境 | 模型 | OnPolicy | LUFFY | CHORD | DUET |
|------|------|----------|-------|-------|------|
| ALFWorld | 1.5B | ? | ? | ? | **?** |
| ALFWorld | 3B | 0.585 | 0.615 | ? | **0.695** |
| WebShop | 1.5B | ? | ? | ? | **?** |
| WebShop | 3B | 0.402 | 0.753 | 0.728 | **0.763** |

### Scaling Analysis (Figure/Table): DUET 优势 vs 模型尺度

| 尺度 | Teacher Gap (step 50) | DUET vs OnP | LUFFY vs OnP | 结论 |
|------|---------------------|-------------|-------------|------|
| 1.5B | 很大（预期） | **很大（预期）** | 正面 | Teacher 不可或缺 |
| 3B | 中等 | +18~89% | +5~87% | DUET 领先 |
| 7B | 近零 | +1.5~6.5% | **-2.5~-3.7%** | DUET 安全，LUFFY 有害 |

---

*最后更新: 2026-04-14*
