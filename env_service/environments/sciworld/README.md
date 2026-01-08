# ScienceWorld 环境集成

## 概述

ScienceWorld 是一个基于文本的科学实验模拟环境。Agent 需要在模拟的实验室环境中，通过一系列操作完成各种科学实验任务。

## 环境特点

- **30+ 科学任务**：涵盖物理、化学、生物等领域
- **真实的实验流程**：需要正确的步骤顺序
- **丰富的交互对象**：仪器、材料、容器等

## 快速开始

### 1. 安装 AgentGym ScienceWorld 服务器

**注意：需要 Java 1.8+ 环境**

```bash
cd AgentGym/agentenv-sciworld
conda create --name agentenv-sciworld python=3.8
conda activate agentenv-sciworld
pip install -e .
```

### 2. 启动 ScienceWorld 服务器

```bash
sciworld --host 0.0.0.0 --port 36004
```

### 3. 启动环境服务

```bash
cd env_service/launch_script
bash sciworld.sh
```

## 配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `SCIWORLD_SERVER_URL` | `http://127.0.0.1:36004` | AgentGym ScienceWorld 服务器地址 |

## 任务 ID 格式

- `task_id = data_idx`（组合了 taskName 和 variationIdx）
- 总共约 3000+ 任务变体
- 训练集：80%
- 测试集：20%

## 常用动作

ScienceWorld 支持多种实验操作：

### 移动与观察
- `look around` - 观察周围环境
- `look at [object]` - 仔细观察物体
- `move to [location]` - 移动到某位置

### 物体操作
- `pick up [object]` - 拾取物体
- `put [object] in [container]` - 放置物体
- `open [object]` - 打开容器/门
- `close [object]` - 关闭容器/门

### 实验操作
- `activate [device]` - 开启设备
- `deactivate [device]` - 关闭设备
- `use [object] on [target]` - 使用物体
- `pour [liquid] into [container]` - 倒入液体
- `mix [container]` - 混合容器内物质
- `focus on [object]` - 聚焦观察
- `wait` - 等待反应发生

## 任务类型示例

1. **测量质量** - 使用天平测量物体质量
2. **加热物体** - 使用热源加热物体
3. **电路实验** - 连接电路点亮灯泡
4. **化学反应** - 混合物质观察反应
5. **生命周期** - 观察植物/动物生命周期
... （共 30+ 种任务类型）

## API 参考

### `get_init_state(params)`

初始化环境，返回任务描述和初始观察。

### `step(action, params)`

执行动作，返回观察结果和进度奖励。

### `evaluate()`

返回实验完成度（0.0 - 1.0），基于完成的子目标数量。

## 注意事项

1. 需要 Java 1.8+ 环境
2. ScienceWorld 启动时会加载任务模板，首次启动较慢
3. 某些任务可能需要较长的步数才能完成
4. 注意观察环境反馈，某些操作可能失败

