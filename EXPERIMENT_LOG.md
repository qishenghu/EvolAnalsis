# DUET Paper — Experiment Log

> 持续更新的实验记录，包含计划、进度、结论和复现命令。

**目标**: NeurIPS 2026 submission (~2026-05-07)  
**核心主张**: DUET (DR3 + State Channel) 提供 data-driven teacher curriculum，在 capability gap 大时显著提升，gap 小时安全退出（graceful degradation）  
**实验策略**: Qwen2.5 × 3 尺度 (1.5B, 3B, 7B) × 2 环境 (ALFWorld, WebShop)。1.5B/3B 为主要结果，7B 为 scaling analysis（展示 DR3 fade-out 的正确性）

---

## 一、实验矩阵总览

### 1.1 主实验：1.5B + 3B（Paper 主表）

| 环境 | 模型 | OnPolicy | LUFFY | CHORD | DUET | SFT+RL | 状态 |
|------|------|----------|-------|-------|------|--------|------|
| ALFWorld | Qwen2.5-1.5B | ✅ 1.0% | ✅ 5.5% | ✅ 27.0% | ✅ **32.5%** | ✅ 30.0% | 全部完成 |
| ALFWorld | Qwen2.5-3B | ✅ 58.5% | ✅ 61.5% | 🔄 运行中 | ✅ **69.5%** | ⬚ 待跑 | CHORD 补跑中 |
| WebShop | Qwen2.5-1.5B | ✅ 0.152 | ✅ 0.573 | ✅ **0.603** | ✅ 0.549 / 🔄 v2 | ✅ 0.641 | DUET v2 运行中 |
| WebShop | Qwen2.5-3B | ✅ 0.402 | ✅ 0.753 | ✅ 0.728 | ✅ **0.763** | ⬚ 待跑 | 全部完成 |

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

---

## 2026-07-31: Rebuttal 补充实验收官(A100)

- **−SC s2027 复现**(`alfworld_qwen1.5b_duet_minus_sc_s2027`,task_seed=2026): val@50 = 43.0%,val@100 = **33.0%**(n=200,strict)。原 −SC 为 35.5/31.0 → −SC val@100 现有 n=2 {31.0, 33.0},均值 32.0。对比 DUET 全量 n=7 = 44.6±6.6,SC 移除代价 ~12.6pp,bDeY W1(item 2)引用的 31.0% 已有第二 seed 支撑。
- **Llama-3.2-3B DUET 8 卡复跑**(`alfworld_llama3b_duet_a100_both8`,micro=1×8 卡,clock 不变量 8 保持,SC β=0.05/η=0.0125):2026-07-31 01:24 启动,目标 100 步,val@100 出来后兑现 UyKJ Q1 讨论期承诺(val@50=15.0% 已有)。
- **2026-07-31 03:5x**: 按用户指示停止全部 rebuttal 实验:8 卡 Llama both8 复跑(step ~30 附近,val@100 未达,val@50=15.0% 仍为可引用数字)、runner/watcher、ALFWorld 主 env service(8081/36001)与辅 env service(18091, envsvc_aux Ray 会话)全部干净退出,8 卡 GPU 已释放。rebuttal 实验阶段就此收官,转入 ICLR 2027 准备(见 ICLR2027_PLAN.md)。
- **2026-07-31 磁盘清理**(用户授权,三 agent 对抗验证后执行):Phase 1 缓存类 ~128G(wandb 85G→12K 经官方 sync --clean 核对、logs 18G→557M gzip、.git 12G→161M gc、tmp_ckpts 14G);Phase 2 证据驱动剪枝 ~245G(71 个 dev-sweep experiments 扁平目录、112 个无权重死 checkpoint 目录、11 个 pre-July 论文 cell 的 optimizer 分片修剪——model 分片全保留,merge/再验证能力不受影响,唯一副作用是原配置名 auto-resume 会 fail-fast 报 FileNotFoundError,修复方法记录在 manifest)。合计回收 **~373G**。删除清单:`archive/PRUNE_MANIFEST_2026-07-31.tsv`(228 行);嵌套树 `experiments/{env}/<run>/validation_log/` 全部保留(论文数字 provenance 完整);4 个无验证日志 run 的 evaluation 记录抢救至 `archive/pruned_eval_provenance/`。

---

## 2026-07-31: ICLR 新栈首发(Qwen3.5-4B GRPO baselines)

**基础设施**:方案 (b) 落地——`duet2`(torch 2.9.1+cu128 / transformers 5.5.1 / vendored verl 0.4.0.dev0 + 7 项 t5x 补丁 / flash-attn 2.8.3 / fla+causal-conv1d)训练,`vllm2`(vllm 0.19.1)起外置 rollout server,每步权重同步(FSDP 导出 → collective_rpc reload,实测 36-43s/步)。

**新栈兼容问题清单(全部在胶水层,DUET 算法代码零改动)**:①duet2 缺 4 个非 PyPI 模块(best_logger/beast_logger/jieba/cachetools,从 duet 复制);②`env_client.py` URL 双斜杠(urllib3 2.7 不再折叠 → 404),改 `lstrip('/')`;③verl FSDP wrap 策略遇 `_no_split_modules` 中缺失类即抛错(Qwen3.5 声明含 VisionBlock 而我们纯文本加载)→ 补丁 P5 改为跳过、全空才报错;④权重同步命名:训练侧导出 `model.layers.*`,HF 布局为 `model.language_model.*`、**vLLM 内部布局为 `language_model.model.*`**,扩展里按服务端实际布局自适应重映射;⑤transformers 5.x `apply_chat_template(tokenize=True)` 返回 BatchEncoding 而非 list → 新增 `chat_template_ids()` helper,改 14 处调用点;⑥thinking 模板下 `step_parser` 的多轮 diff 失效(末轮 assistant 被注入空 think 块、历史轮被剥离,前缀关系不成立)→ 改字面 header 路径。

**Gate-S**(Qwen3.5-2B ALFWorld GRPO 冒烟,4×A100):step 2 成功率 20.3%、response 5477 tok、KL 0.001、277s/步 —— 通过。

**首发(双 lane,各 4 卡,100 步)**:
- lane A(GPU0-3):`alfworld_qwen35_4b_grpo` step1 成功率 **31.2%**、response 5970 tok、KL 0.001、428s/步
- lane B(GPU4-7):`webshop_qwen35_4b_grpo` step1 成功率 **12.5%**、response 9076 tok、KL 0.003、417s/步
- SciWorld 排队(agentenv-sciworld env + openjdk-8 已装,`start_env_sciworld.sh` 就绪),待任一 lane 空出接力。

**教师数据**:DeepSeek-v4-pro 经 OpenRouter 采集,ALFWorld 试点 50/50 成功(100%),全量 800 任务 × 10 条进行中(成功率 ~93%,`<think>` 与 action 格式与 react_tags 管线兼容)。
- **2026-07-31 下午:Qwen3.5-4B GRPO baseline 三次发车与 reasoning-context 决策**
  - 第一次(B=0,strip-all):ALFWorld step-15 崩溃(SR 31.2%→0.0%,response 5970→4828,带梯度 token 8.8%→2.7%,全部 episode 撞 30 轮上限)后 OOM;WebShop KL 冲到 2.59、SR 平于 12.5%。**根因:历史轮 action token 带梯度但其 think 被剥离 → 训练优化 π(a|ctx) 而采样自 π(a|ctx,think),有偏梯度。**
  - 第二次(B=−1,keep-all,`-thinkhist` 模板):ALFWorld step-1 SR **57.8%**(vs 31.2%)、KL 0.001、带梯度 token 31.3%,step-5 SR 35.9%/KL 0.004;但 **step 6 OOM**(248K 词表下 32768-token 微批 logits = 16.3GB)。
  - 第三次(B=1024,`-thinkraw` 模板,max_model_len 收到 24576):两 lane 运行中,ALFWorld step-5 SR 32.8%/KL 0.005/带梯度 19.1%,WebShop step-5 SR 12.5%/KL 0.040。
  - 诊断:**长度-only AUC = 0.558**(教师 n=400 实测),DR3 长度捷径不存在,特征硬化取消;教师 assistant token 中 **85.2% 是 think**(B 对 BC 通道的价值 ×6)。
  - 工程:权重同步移至 `/dev/shm`(88s→34-39s);teacher loss_mask 与 on-policy 对称(`force_training`);三处 think 解析条件修正。
- **2026-07-31 夜:根因订正 + v2 发车**。前述 ALFWorld/WebShop 多次崩溃的主因是 `/no_think` 注入与 thinking 生成提示词冲突(非熵坍缩);新增 `native_qwen35` 模式修复,文本级验收全绿(0/64 畸形、标签 64/64 完整、输出长度恢复 17×)。实验重命名为 `{env}_qwen35_4b_grpo_v2`。教师采集脚本收尾截断 bug 已修,ALFWorld 重采中。
