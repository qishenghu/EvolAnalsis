# AgentEnv GPQA-Diamond (Single-Choice)

基于 HuggingFace `fingertap/GPQA-Diamond` 的单轮选择题环境（split 默认 `test`）。

## 特点
- 单轮任务：一次回答即结束（无效格式可重试）
- 答案解析：支持 `<answer>A</answer>` 或文本中的 A/B/C/D
- 数据加载：`datasets.load_dataset('fingertap/GPQA-Diamond', split='test')`，可用环境变量 `GPQA_SPLIT` 覆盖

## 安装
```bash
cd AgentGym/agentenv-gpqa
pip install -e .
```

## 启动服务
```bash
gpqa --port 8003 --host 0.0.0.0
```

## 使用示例
```python
from agentenv.envs import GPQATask

task = GPQATask(
    client_args={
        "env_server_base": "http://127.0.0.1:8003",
        "data_len": 198,  # GPQA-Diamond test集大小
        "is_eval": True,
        "test_data_start_index": 0,
        "timeout": 300,
    },
    n_clients=1,
)
```

## 答案格式
- 推荐：`<answer>A</answer>`
- 也接受：直接输出 `A/B/C/D`
- 模型必须先在 `<think>...</think>` 内推理，再给出 `<answer>...</answer>`
