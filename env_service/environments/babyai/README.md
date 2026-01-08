# BabyAI 环境集成

## 概述

BabyAI 是一个基于 MiniGrid 的指令跟随环境。Agent 需要在网格世界中完成各种导航和操作任务，如 "go to the red ball"、"pick up the blue key" 等。

## 环境特点

- **40 个关卡**：从简单的导航到复杂的多步骤任务
- **程序化生成**：每个关卡可以用不同种子生成不同实例
- **丰富的动作空间**：包括移动、拾取、放下、开门等

## 快速开始

### 1. 安装 AgentGym BabyAI 服务器

```bash
cd AgentGym/agentenv-babyai
conda create --name agentenv-babyai python=3.8
conda activate agentenv-babyai
pip install -e .
```

### 2. 启动 BabyAI 服务器

```bash
babyai --host 0.0.0.0 --port 36002
```

### 3. 启动环境服务

```bash
cd env_service/launch_script
bash babyai.sh
```

## 配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `BABYAI_SERVER_URL` | `http://127.0.0.1:36002` | AgentGym BabyAI 服务器地址 |

## 任务 ID 格式

- `task_id = level_idx + seed * 40`
- 总共 40 个关卡 × 50 个种子 = 2000 个任务
- 训练集：前 1600 个任务
- 测试集：后 400 个任务

## 关卡列表

1. GoToRedBallGrey - 导航到红球（灰色世界）
2. GoToRedBall - 导航到红球
3. GoToRedBallNoDists - 导航到红球（无干扰物）
4. GoToObjS6 - 导航到物体
5. GoToLocalS8N7 - 局部导航
... （共 40 个关卡）

## API 参考

### `get_init_state(params)`

初始化环境，返回初始观察和目标。

### `step(action, params)`

执行动作，支持的动作格式：
- `turn left` / `turn right`
- `move forward`
- `pickup [object]`
- `drop`
- `toggle`
- `go to [object]`
- `go through [door]`

### `evaluate()`

返回任务完成度（0.0 - 1.0）。

