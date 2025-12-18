# AgentEnv AIME

AIME (American Invitational Mathematics Examination) 数学推理环境，用于评估AI模型在数学问题求解上的能力。

## 特点

- **Single-Turn任务**：每个问题只需要一次交互即可完成，step函数会立即返回done=True
- **答案验证**：自动提取和验证答案（0-999之间的整数）
- **数据支持**：支持AIME2024和AIME2025数据集

## 安装

```bash
cd AgentGym/agentenv-aime
pip install -e .
```

## 数据准备

### 方式1：使用HuggingFace数据集（推荐）

环境会自动从HuggingFace加载数据：
- AIME 2024: `Maxwell-Jia/AIME_2024`
- AIME 2025: `selectdataset/AIME_2025`

### 方式2：使用本地数据

将数据文件放在 `agentenv-aime/data/` 目录下：
- `train.jsonl`: 训练数据
- `test.jsonl`: 测试数据（64个问题）

数据格式（JSONL，每行一个JSON对象）：
```json
{"problem": "问题描述", "answer": "答案（字符串格式）"}
```

## 启动服务器

```bash
aime --port 8001 --host 0.0.0.0
```

或者使用Python：
```python
from agentenv_aime import launch
launch()
```

## 使用示例

### 在代码中使用

```python
from agentenv.envs import AIMETask

# 创建任务
task = AIMETask(
    client_args={
        "env_server_base": "http://127.0.0.1:8001",
        "data_len": 64,  # test数据大小
        "is_eval": True,
        "test_data_start_index": 0,  # test数据起始索引
        "timeout": 300,
    },
    n_clients=1,
)

# 生成experience
from agentenv.controller import APIAgent
agent = APIAgent(...)  # 初始化你的agent
experience = task.generate_experience(agent, idxs=0)
```

## 数据划分

- **Test数据**：前64个问题（索引0-63）
- **Train数据**：剩余所有问题（索引64+）

## 答案格式

Agent的响应应该包含答案，可以通过以下方式提供：
1. 使用`<answer>数字</answer>`标签
2. 或者在响应末尾直接提供数字

答案会被自动提取和验证。

## 环境变量

- `AIME_DATA_PATH`: 本地数据路径（默认：`agentenv-aime/data/`）
- `AGENTENV_DEBUG`: 启用调试模式

## Single-Turn特性

与multi-turn任务（如SearchQA、Alfworld）不同，AIME是single-turn任务：
- 每次调用`step()`函数都会立即返回`done=True`
- 任务只进行一次交互：Agent提供答案，环境验证并返回结果
- 这符合数学竞赛的特点：给出最终答案即可判断对错
