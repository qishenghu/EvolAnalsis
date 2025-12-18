# AgentEnv MathBench (Single-Choice)

基于 MathBench `high_mix/single_choice_en_shuffled.jsonl` 的单轮选择题环境。

## 特点
- 单轮任务：`step` 一次即结束，返回 `done=True`
- 自动解析答案：支持 `<answer>A</answer>` 或直接输出选项字母/文本
- 默认数据源：`/home/qisheng/agent/EvolCL/mathbench_v1/high_mix/single_choice_en_shuffled.jsonl`
- 可通过 `MATHBENCH_DATA_PATH` 环境变量切换数据路径

## 安装
```bash
cd AgentGym/agentenv-mathbench
pip install -e .
```

## 启动服务
```bash
mathbench --port 8002 --host 0.0.0.0
```

## 使用示例
```python
from agentenv.envs import MathBenchTask

task = MathBenchTask(
    client_args={
        "env_server_base": "http://127.0.0.1:8002",
        "data_len": 400,  # 根据数据量设置
        "is_eval": True,
        "test_data_start_index": 0,
        "timeout": 300,
    },
    n_clients=1,
)
```

## 数据格式
每行 JSON：
```json
{
  "question": "...",
  "options": ["opt1", "opt2", ...],
  "answer": "A",
  "topic": "..."
}
```

## 答案输出要求
- 推荐：`<answer>A</answer>`
- 也接受：直接输出 `A` 或包含选项文本，系统会匹配字母。
