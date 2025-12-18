# AlfWorld 环境整合指南

本指南说明如何将 AgentGym 的 AlfWorld 环境整合到 AgentEvolver 框架中，以支持 Experience Pool 和 GRPO 训练。

## 📋 整合完成清单

✅ **已完成的工作**：
1. ✅ 创建 `alfworld_env.py` - 实现 `BaseEnv` 接口
2. ✅ 创建 `alfworld.sh` - 启动脚本
3. ✅ 更新 `launcher.py` - 添加 `--with-alfworld` 支持
4. ✅ 创建环境模块 `__init__.py`
5. ✅ 编写整合文档

## 🚀 快速开始

### 步骤 1: 确保 AgentGym AlfWorld 已安装

```bash
cd AgentGym/agentenv-alfworld
bash setup.sh
```

确保：
- AlfWorld 数据已下载（`~/.cache/alfworld`）
- 配置文件存在（`configs/base_config.yaml`）

### 步骤 2: 配置环境变量（可选）

编辑 `.env` 文件或设置环境变量：

```bash
export ALFWORLD_DATA=~/.cache/alfworld
export AGENTGYM_ROOT=/path/to/AgentEvolver/AgentGym/agentenv-alfworld
```

### 步骤 3: 启动训练

```bash
# 方式 1: 使用 launcher（推荐）
python launcher.py \
  --conf config/your_config.yaml \
  --with-alfworld \
  --with-reme  # 如果使用 Experience Pool

# 方式 2: 手动启动环境服务
cd env_service/launch_script
bash alfworld.sh
# 然后在另一个终端启动训练
python -m agentevolver.main_ppo --config-path ... --config-name ...
```

## 📝 配置文件示例

### 训练配置 (YAML)

```yaml
env_service:
  env_type: "alfworld"
  env_url: "http://127.0.0.1:8080"

data:
  train_files: null  # 从环境服务加载任务
  val_files: null

exp_manager:
  reme:
    enable_summarizer: true
    enable_context_generator: true
    # ... 其他配置
```

## 🔍 工作原理

### 1. Seed Task 加载

- `TaskManager.load_tasks_from_environment()` 调用 `env_service.get_env_profile(env_type="alfworld", split="train")`
- `AlfworldEnv.get_query_list()` 从 `mappings_train.json` 读取任务列表
- 返回任务 ID 列表（游戏索引：0, 1, 2, ...）

### 2. 环境实例创建

- `EnvWorker.execute()` 调用 `env.create_instance(env_type="alfworld", task_id=...)`
- `AlfworldEnv.__init__()` 创建环境实例
- `get_init_state()` 使用 `task_id` 作为游戏索引加载特定游戏

### 3. 交互循环

- Agent 生成动作 → `step()` 执行 → 返回观察和奖励
- 重复直到任务完成或达到最大步数

### 4. Experience Pool 集成

- 训练过程中，trajectories 通过 `exp_manager.submit_summary_task()` 提交
- ReMe 服务总结并存储经验
- 后续 rollout 可以通过 `call_context_generator()` 检索历史经验

### 5. GRPO 训练

- Trajectories 转换为训练 batch
- 计算 rewards 和 advantages（GRPO）
- 可选：应用 ADCA-GRPO 重写 advantages
- 更新 actor/critic 模型

## 🎯 关键文件位置

```
AgentEvolver/
├── env_service/
│   ├── environments/
│   │   └── alfworld/
│   │       ├── __init__.py
│   │       ├── alfworld_env.py      # ⭐ 核心实现
│   │       └── README.md
│   └── launch_script/
│       └── alfworld.sh              # ⭐ 启动脚本
├── launcher.py                      # ⭐ 已更新支持 --with-alfworld
└── AgentGym/
    └── agentenv-alfworld/           # AgentGym 环境
        ├── agentenv_alfworld/
        │   └── env_wrapper.py
        └── configs/
            ├── mappings_train.json
            └── mappings_test.json
```

## ⚠️ 注意事项

1. **任务 ID 格式**：
   - Seed tasks 使用游戏索引（整数转字符串："0", "1", "2", ...）
   - 不是语义化的任务描述

2. **环境初始化**：
   - 每个 `AlfworldEnv` 实例对应一个游戏会话
   - 使用懒加载避免初始化问题

3. **数据路径**：
   - 默认：`~/.cache/alfworld`
   - 可通过 `params["data_path"]` 或环境变量 `ALFWORLD_DATA` 设置

4. **世界类型**：
   - 支持 "Text", "Embody", "Hybrid"
   - 默认：`"Text"`

## 🐛 常见问题

### Q: 启动时提示 "Environment 'alfworld' not found"

**A**: 检查：
1. `alfworld_env.py` 是否在正确位置
2. `@Registry.register("alfworld")` 装饰器是否存在
3. `env_service/env_service.py` 能否正确导入模块

### Q: "Failed to create AlfWorld environment"

**A**: 
1. 检查 `ALFWORLD_DATA` 是否设置
2. 确认 `configs/base_config.yaml` 存在
3. 验证 AlfWorld 依赖已安装

### Q: 如何自定义任务列表？

**A**: 修改 `get_query_list()` 方法，或直接编辑 `mappings_train.json` / `mappings_test.json`

## 📚 下一步

1. **测试整合**：运行简单训练循环验证功能
2. **配置 Experience Pool**：设置 ReMe 服务
3. **调优超参数**：针对 AlfWorld 调整 GRPO/PPO 参数
4. **监控训练**：使用 logview 跟踪进度

## 🔗 相关文档

- [AlfWorld 环境 README](env_service/environments/alfworld/README.md)
- [Experience Pool 文档](docs/guidelines/exp_manager.md)
- [GRPO 训练文档](docs/tutorial/quick_start.md)

