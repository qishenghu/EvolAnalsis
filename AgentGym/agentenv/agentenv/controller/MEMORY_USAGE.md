# Memory模块使用指南

本文档介绍如何在AgentGym中使用Memory模块来存储和检索经验。

## 概述

Memory模块允许Agent在解决任务时：
1. **检索**相似的历史经验
2. **注入**检索到的经验到conversation中，帮助Agent做出更好的决策
3. **存储**新完成的经验，供未来使用

## 基本使用

### 1. 创建Memory实例

```python
from agentenv.controller import RawMemory, NullMemory

# 使用RawMemory（向量存储）
memory = RawMemory(
    local_exp_pool_path="./exp_pool",  # 可选：已有经验池路径
    collection_name="exp_pool",
    success_only=True,  # 只存储成功的经验
    k_retrieval=1,  # 检索k个最相似的经验
)

# 或者使用NullMemory（不存储也不检索，用于baseline）
memory = NullMemory()
```

### 2. 在Task中使用Memory

```python
from agentenv.controller import APIAgent, AlfWorldTask
from agentenv.controller.memory import RawMemory

# 创建Memory
memory = RawMemory(
    local_exp_pool_path="",  # 从空的经验池开始
    success_only=True,
    k_retrieval=1,
)

# 创建Task时传入Memory
task = AlfWorldTask(
    client_args={
        "env_server_base": "http://localhost:3000",
        "data_len": 100,
    },
    memory=memory,  # 传入Memory实例
)

# 创建Agent
agent = APIAgent(
    api_key="your-api-key",
    base_url="http://localhost:8000/v1",
    model="qwen-plus",
)

# 使用Memory支持的方法生成experience
experiences = task.generate_experience_with_memory(
    agent=agent,
    idxs=[0, 1, 2],  # 任务索引列表
)
```

### 3. 禁用/启用存储

在评估阶段，你可能想禁用存储但保留检索功能：

```python
# 禁用存储（只检索，不存储新经验）
memory.disable_storage()

# 启用存储
memory.enable_storage()
```

### 4. 持久化Memory

对于RawMemory，可以将经验池持久化到文件：

```python
# 持久化到jsonl文件
memory.persist("./exp_pool_snapshot.jsonl")
```

## 工作流程

### 训练阶段

```python
# 1. 创建Memory（从空或已有经验池）
memory = RawMemory(
    local_exp_pool_path="",  # 从空开始
    success_only=True,
    k_retrieval=1,
)

# 2. 创建Task和Agent
task = AlfWorldTask(client_args={...}, memory=memory)
agent = APIAgent(...)

# 3. 在训练循环中使用
for idx in train_indices:
    experience = task.generate_experience_with_memory(
        agent=agent,
        idxs=idx,
    )
    # Memory会自动存储成功的经验
    
# 4. 定期保存经验池
memory.persist(f"./exp_pool_step_{step}.jsonl")
```

### 评估阶段

```python
# 1. 加载已有经验池
memory = RawMemory(
    local_exp_pool_path="./exp_pool_step_100.jsonl",  # 加载已有经验
    success_only=True,
    k_retrieval=1,
)

# 2. 禁用存储（只检索，不存储）
memory.disable_storage()

# 3. 评估
task = AlfWorldTask(client_args={...}, memory=memory)
agent = APIAgent(...)

results = task.generate_experience_with_memory(
    agent=agent,
    idxs=test_indices,
)
```

## Memory类型

### NullMemory
- **用途**: Baseline，不存储也不检索
- **适用场景**: 对比实验

### RawMemory
- **用途**: 存储完整的原始轨迹
- **存储方式**: Chroma向量数据库
- **检索方式**: 向量相似度搜索
- **适用场景**: 需要完整轨迹信息的场景

## 注意事项

1. **向后兼容**: 如果不传入`memory`参数，Task会使用`NullMemory`，行为与原来完全一致
2. **经验注入位置**: 记忆会被注入在第一个实际任务消息（state）之前
3. **存储条件**: RawMemory默认只存储成功的经验（reward > 0），可通过`success_only=False`修改
4. **依赖**: RawMemory需要安装`langchain`和`chromadb`

## 使用Evaluator

`Evaluator`类也支持Memory功能，通过`use_memory`参数控制：

```python
from agentenv.controller import Evaluator, APIAgent, AlfWorldTask
from agentenv.controller.memory import RawMemory

# 创建Memory
memory = RawMemory(
    local_exp_pool_path="./exp_pool.jsonl",
    success_only=True,
    k_retrieval=1,
)

# 创建Task（传入Memory）
task = AlfWorldTask(
    client_args={"env_server_base": "http://localhost:3000", "data_len": 100},
    memory=memory,
)

# 创建Agent和Evaluator
agent = APIAgent(api_key="your-api-key", base_url="http://localhost:8000/v1", model="qwen-plus")
evaluator = Evaluator(agent=agent, tasks=[task])

# 不使用Memory评估
result_without_memory = evaluator.eval(
    idxs=[0, 1, 2],
    use_memory=False,  # 默认值，使用原来的generate_experience
)

# 使用Memory评估
result_with_memory = evaluator.eval(
    idxs=[0, 1, 2],
    use_memory=True,  # 使用generate_experience_with_memory
)

print(f"Without memory: score={result_without_memory.score}, success={result_without_memory.success}")
print(f"With memory: score={result_with_memory.score}, success={result_with_memory.success}")
```

## 与原有代码的对比

### 原有方式（不使用Memory）
```python
task = AlfWorldTask(client_args={...})
experiences = task.generate_experience(agent, idxs=[0, 1, 2])

# 或使用Evaluator
evaluator = Evaluator(agent=agent, tasks=[task])
result = evaluator.eval(idxs=[0, 1, 2])  # use_memory默认为False
```

### 新方式（使用Memory）
```python
memory = RawMemory(...)
task = AlfWorldTask(client_args={...}, memory=memory)

# 方式1：直接使用Task方法
experiences = task.generate_experience_with_memory(agent, idxs=[0, 1, 2])

# 方式2：使用Evaluator（推荐）
evaluator = Evaluator(agent=agent, tasks=[task])
result = evaluator.eval(idxs=[0, 1, 2], use_memory=True)
```

**注意**: 
- `generate_experience`方法仍然存在且未被修改，保持完全向后兼容
- `Evaluator.eval()`的`use_memory`参数默认为`False`，保持向后兼容
- 如果Task没有传入Memory，即使`use_memory=True`，也会使用`NullMemory`（不会报错）

