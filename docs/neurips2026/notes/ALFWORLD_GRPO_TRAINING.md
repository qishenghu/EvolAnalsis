# AlfWorld GRPO 训练指南

本指南说明如何使用 AgentEvolver 在 AlfWorld 环境中训练 Qwen/Qwen2.5-3B-Instruct 模型，使用 GRPO 算法和 Experience Pool (ReMe)。

## 📋 前置条件

### 1. 确保 AgentGym AlfWorld 环境已安装

```bash
cd AgentGym/agentenv-alfworld
bash setup.sh
```

### 2. 确保 AlfWorld 数据已下载

```bash
export ALFWORLD_DATA=~/.cache/alfworld
# 如果数据未下载，运行：
alfworld-download
```

### 3. 确保 ReMe (Experience Pool) 服务可用

ReMe 服务需要单独启动，用于存储和检索经验。确保：
- ReMe 服务已安装
- 可以访问 `http://127.0.0.1:8001`

## 🚀 启动训练

### 方式 1: 使用 launcher（推荐）

```bash
python launcher.py \
  --conf config/alfworld_grpo.yaml \
  --with-alfworld \
  --with-reme
```

这会自动：
1. 启动 AlfWorld 环境服务（`http://127.0.0.1:8080`）
2. 启动 ReMe Experience Pool 服务（`http://127.0.0.1:8001`）
3. 开始 GRPO 训练

### 方式 2: 手动启动服务

如果选择手动启动，需要按顺序执行：

#### 步骤 1: 启动 AlfWorld 环境服务

```bash
cd env_service/launch_script
bash alfworld.sh
```

或者启动 AgentGym 的 AlfWorld HTTP 服务器：

```bash
cd AgentGym/agentenv-alfworld
python -m agentenv_alfworld.server
```

#### 步骤 2: 启动 ReMe Experience Pool 服务

```bash
# 在 ReMe 服务目录中
reme \
  config=default \
  backend=http \
  thread_pool_max_workers=256 \
  http.host="127.0.0.1" \
  http.port=8001 \
  http.limit_concurrency=256 \
  llm.default.model_name=qwen-max-2025-01-25 \
  embedding_model.default.model_name=text-embedding-v4 \
  vector_store.default.backend=local \
  op.rerank_memory_op.params.enable_llm_rerank=false
```

#### 步骤 3: 启动训练

```bash
python launcher.py --conf config/alfworld_grpo.yaml
```

## 📝 配置说明

### 关键配置项

1. **环境配置**：
   - `env_service.env_type: "alfworld"` - 使用 AlfWorld 环境
   - `env_service.env_url: "http://127.0.0.1:8080"` - 环境服务地址

2. **模型配置**：
   - `actor_rollout_ref.model.path: "Qwen/Qwen2.5-3B-Instruct"` - 训练模型

3. **任务配置**：
   - `task_manager.n: 0` - 不生成合成任务，只使用原始 2420 个训练任务
   - `task_manager.mixture.synthetic_data_ratio: 0.0` - 不使用合成任务
   - `data.train_files: null` - 从环境服务加载任务

4. **Experience Pool 配置**：
   - `exp_manager.reme.enable_summarizer: true` - 启用经验总结
   - `exp_manager.reme.enable_context_generator: true` - 启用经验检索
   - `exp_manager.train_rollout_mode: "mixed"` - 训练时使用经验池
   - `exp_manager.rollout_ratio: 0.5` - 50% 的 rollout 使用经验

5. **算法配置**：
   - `algorithm.adv_estimator: grpo` - 使用 GRPO 算法

## 📊 训练数据

- **训练任务**: 2420 个任务（从 `mappings_train.json` 加载）
- **验证任务**: 200 个任务（从 `mappings_test.json` 加载）
- **任务 ID 范围**: 
  - 训练: [0, 2420)
  - 测试: [2420, 2620)

## 🔍 训练流程

1. **任务加载**: `TaskManager.load_tasks_from_environment()` 从 AlfWorld 环境加载 2420 个训练任务
2. **Rollout**: 在每个训练步骤中，对任务进行 rollout，生成轨迹
3. **Experience Pool**: 
   - 轨迹通过 `exp_manager.submit_summary_task()` 提交到 ReMe 服务
   - ReMe 服务总结并存储经验
   - 后续 rollout 通过 `call_context_generator()` 检索历史经验
4. **GRPO 训练**: 
   - 计算 rewards 和 advantages（GRPO）
   - 更新 actor/critic 模型

## 📁 输出目录

训练输出将保存在：
- 实验目录: `experiments/alfworld/alfworld_qwen25-3b_grpo_expool/`
- 验证日志: `experiments/alfworld/alfworld_qwen25-3b_grpo_expool/validation_log/`
- Rollout 日志: `experiments/alfworld/alfworld_qwen25-3b_grpo_expool/rollout_log/`

## ⚙️ 自定义配置

如果需要修改配置，可以：

1. **直接修改配置文件** `config/alfworld_grpo.yaml`
2. **通过命令行参数覆盖**：
   ```bash
   python launcher.py \
     --conf config/alfworld_grpo.yaml \
     --with-alfworld \
     --with-reme \
     actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
     data.train_batch_size=64
   ```

## 🐛 常见问题

### 1. AlfWorld 环境服务无法启动

- 检查 `ALFWORLD_DATA` 环境变量是否设置
- 确认 AlfWorld 数据已下载
- 检查端口 8080 是否被占用

### 2. ReMe 服务连接失败

- 确认 ReMe 服务已启动并运行在 `http://127.0.0.1:8001`
- 检查 `exp_manager.reme.base_url` 配置是否正确

### 3. 任务加载失败

- 确认 `mappings_train.json` 和 `mappings_test.json` 文件存在
- 检查环境服务是否正常运行

### 4. 内存不足

- 减小 `data.train_batch_size`
- 减小 `actor_rollout_ref.rollout.n`
- 减小 `actor_rollout_ref.rollout.max_env_worker`

## 📚 相关文档

- [AlfWorld 环境整合指南](ALFWORLD_INTEGRATION.md)
- [Experience Pool 使用指南](docs/guidelines/exp_manager.md)
- [GRPO 训练文档](docs/guidelines/trainer.md)

