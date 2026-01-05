# LUFFY Teacher Experience Replay 集成分析

> **状态**: ✅ 实现完成  
> **更新日期**: 2026-01-04

本文档分析如何在 AgentEvolver 中集成 LUFFY 风格的 Teacher Experience Replay 算法，在现有 ExGRPO（self-generated experience replay）基础上增加外部 Teacher 轨迹支持。

## ⭐ 快速开始

### 1. 准备 Teacher 轨迹

```bash
# 1. 启动环境服务（确保已运行）
# 例如：python -m agentevolver.env_service --port 8000

# 2. 从环境服务生成 task_id 列表（与 TaskManager.load_tasks_from_environment 一致）
python scripts/generate_task_ids.py \
    --env_url http://localhost:8000 \
    --env_type alfworld \
    --split train \
    --output data/alfworld/task_ids.txt

# 3. 采集 Teacher 轨迹（使用 vLLM）
python scripts/collect_teacher_trajectories.py \
    --env alfworld \
    --backend vllm \
    --model_path /data/models/Qwen2.5-72B-Instruct \
    --task_file data/alfworld/task_ids.txt \
    --output data/teacher_trajectories/alfworld_qwen72b.jsonl

# 4. 验证轨迹
python scripts/validate_teacher_trajectories.py \
    --input data/teacher_trajectories/alfworld_qwen72b.jsonl
```

### 2. 配置 Teacher Experience

在配置文件中添加：

```yaml
exp_manager:
  teacher_experience:
    enable: true
    data_path: "data/teacher_trajectories/alfworld_qwen72b.jsonl"
    exp_ratio: 0.2
    max_trajectories_per_task: 3
    use_log_prob: true  # vLLM 模型有 log_prob
```

### 3. 启动训练

```bash
python train.py --config config/alfworld_grpo_3b_teacher.yaml
```

---

## 目录

1. [背景与动机](#背景与动机)
2. [LUFFY vs ExGRPO 对比](#luffy-vs-exgrpo-对比)
3. [需要添加的组件](#需要添加的组件)
4. [详细设计](#详细设计)
5. [数据格式定义](#数据格式定义)
6. [配置项设计](#配置项设计)
7. [实现计划](#实现计划)
8. [采集脚本详细设计](#采集脚本详细设计)
9. [测试计划](#测试计划)
10. [附录](#附录)

---

## 背景与动机

### 现有实现：ExGRPO（Self-Generated Experience Replay）

AgentEvolver 已实现类似 ExGRPO 的 self-generated experience replay：
- **数据来源**：模型自身在训练过程中生成的成功轨迹
- **存储位置**：`ExperienceManager.task2trajectories`
- **使用方式**：复用历史成功轨迹，提高样本效率
- **重要性采样**：使用 `recorded_old_log_probs` 校正 off-policy 权重

### LUFFY 的优势

LUFFY 引入外部 Teacher 模型的轨迹：
- **突破能力边界**：弱模型可以学习超出自身能力的推理模式
- **加速学习**：直接从高质量轨迹学习，而不是等待自身探索
- **稳定训练**：Teacher 轨迹提供高质量的 positive examples

### 集成目标

在 AgentEvolver 中同时支持：
1. **Self-Generated Experience**（ExGRPO 风格）：复用自身历史成功轨迹
2. **Teacher Experience**（LUFFY 风格）：使用外部 Teacher 模型的轨迹

---

## LUFFY vs ExGRPO 对比

| 维度 | ExGRPO (Self-Generated) | LUFFY (Teacher) |
|------|-------------------------|-----------------|
| **数据来源** | 模型自身历史轨迹 | 外部 Teacher 模型 |
| **old_log_prob** | 有（收集时保存） | 无（需要简化计算） |
| **数据质量** | 部分正确（intermediate difficulty） | 高质量（通常全部正确） |
| **采集时机** | 训练过程中动态采集 | 预先采集或训练时在线采集 |
| **Importance Sampling** | `ratio = π_current / π_old` | `ratio = π_current / 1 = π_current`（简化） |
| **Policy Shaping** | 可选 | 推荐使用 `f(x) = x/(x+β)` |

### 关键公式差异

**ExGRPO（有 old_log_prob）**:
```
ratio = exp(log_prob_current - old_log_prob_historical)
     = π_current / π_old
```

**LUFFY（无 old_log_prob，简化计算）**:
```
# 假设 π_old = 1（均匀分布或忽略重要性采样）
ratio = exp(log_prob_current - 0) = exp(log_prob_current) = π_current

# 配合 Policy Shaping
shaped_ratio = ratio / (ratio + β)  # 放大低概率信号
```

### ⭐ Teacher Trajectory 的灵活处理

Teacher trajectory 的 log_prob 可用性取决于采集方式：

| 采集方式 | log_prob 可用性 | 推荐处理方式 |
|----------|----------------|-------------|
| **开源模型**（如 Qwen、Llama、DeepSeek） | ✅ 有 | 使用 ExGRPO 形式：`ratio = π_current / π_old` |
| **闭源 API**（如 GPT-4、Claude） | ❌ 无 | 使用 LUFFY 形式：`ratio = π_current`（分母=1） |
| **OpenAI API with logprobs** | ⚠️ 部分 | 可选使用，需配置 `logprobs=True` |

**设计原则**：通过配置项 `teacher_experience.use_log_prob` 让用户决定：
- `use_log_prob: true`：使用标准重要性采样（需要 log_prob）
- `use_log_prob: false`：使用 LUFFY 简化形式（无需 log_prob）

---

## 需要添加的组件

### 1. Teacher Trajectory Collector（核心新增）

**功能**：在各种环境（如 ALFWorld）下采集 Teacher agent 的轨迹

**设计决策**：复用 VERL/AgentEvolver 的 Rollout 框架

**选择复用现有框架的原因**：
1. **数据格式一致**：生成的 Trajectory 与 on-policy rollout 格式完全相同，无需额外对齐
2. **Multi-turn 天然支持**：复用 `EnvWorker`、`Linear_CMT` 等，自动处理 multi-turn 对话
3. **Tokenization 一致**：使用相同的 tokenizer 和 `tokenize_steps()` 逻辑
4. **Mask 生成一致**：`loss_mask`、`response_mask`、`exp_mask` 的生成逻辑完全复用
5. **减少维护成本**：不需要为 Teacher trajectory 单独维护一套数据处理逻辑

#### Teacher LLM 后端支持

Teacher LLM 需要兼容两种调用方式：

| 后端类型 | 适用场景 | Log Prob 支持 | 部署方式 |
|---------|---------|--------------|---------|
| **OpenAI-compatible API** | GPT-4, Claude, DashScope, DeepSeek API, vLLM Server | ⚠️ 部分支持（需 API 支持 logprobs） | 远程 API 调用 |
| **vLLM 本地推理** | Qwen, Llama, DeepSeek, Mistral 等开源模型 | ✅ 完全支持 | 本地 GPU 推理 |

**设计原则**：
- 定义 `BaseTeacherLLM` 抽象基类，统一接口
- 实现 `OpenAITeacherLLM` 支持所有 OpenAI-compatible API
- 实现 `VLLMTeacherLLM` 支持本地 vLLM 推理
- 使用工厂函数 `create_teacher_llm()` 根据配置创建实例
- VERL 框架默认使用 vLLM 作为推理引擎，我们复用这一设计

**实现方式**：
- 创建一个 `TeacherAgentFlow`，替换原有的 `agent_flow`，使用 Teacher LLM（GPT-4/Claude）
- 复用 `EnvManager.rollout()` 和 `EnvWorker.execute()` 的整体流程
- Teacher LLM 通过 OpenAI-compatible API 调用

**输入**：
- Teacher agent 配置（如 GPT-4、Claude、DeepSeek-R1 等）
- 环境配置（复用现有配置）
- 任务列表

**输出**：
- `Trajectory` 对象列表（与 on-policy rollout 格式完全相同）
- 可选：log_prob（如果 Teacher API 支持，如 OpenAI 的 `logprobs` 参数）

**位置**：
- `agentevolver/module/exp_manager/teacher_collector.py`（新增，主要逻辑）
- `agentevolver/agent/teacher_agent_flow.py`（新增，Teacher LLM 的 agent_flow）

### 2. Teacher Experience Storage（扩展现有 ExperienceManager）

**功能**：存储和管理 Teacher 轨迹

**设计决策**：
- ✅ **方案 A（采用）**：扩展现有 `ExperienceManager`，增加 `teacher_task2trajectories`
- ❌ **方案 B（不采用）**：新建 `TeacherExperienceManager` 类

**选择方案 A 的原因**：
1. **混合使用场景**：未来会同时使用 LUFFY（teacher）和 ExGRPO（self-generated），统一管理更方便
2. **共享基础设施**：复用 `difficulty2task_dict`、`skip_uid_set` 等现有逻辑
3. **统一的 API**：对外暴露一致的接口，简化 Trainer 的调用逻辑
4. **Multi-turn 一致性**：Teacher 轨迹和 Self-generated 轨迹使用相同的 `exp_mask`、`response_mask` 处理逻辑

**修改位置**：`agentevolver/module/exp_manager/exp_manager.py`（扩展）

### 3. Teacher Trajectory Loader（集成到 ExperienceManager）

**功能**：从磁盘加载预采集的 Teacher 轨迹

**支持格式**：
- JSONL（每行一个轨迹，推荐）
- Pickle（快速加载）

**位置**：集成到 `ExperienceManager.load_teacher_trajectories()` 方法

### 4. Loss 计算适配

**功能**：支持没有 `old_log_prob` 的 Teacher 轨迹

**修改位置**：`agentevolver/module/exp_manager/het_core_algos.py`

**新增模式**：
- `teacher_no_old_logprob`：使用 `ratio = exp(log_prob)` 
- `teacher_with_shaping`：使用 `ratio = π / (π + β)`

### 5. ExperienceMixCollateFn 扩展

**功能**：混合 on-policy、self-generated off-policy 和 teacher off-policy

**修改位置**：`agentevolver/module/exp_manager/experience_collate.py`

### 6. 配置项扩展

**位置**：`config/` 下的 yaml 文件

---

## 详细设计

### 4.1 TeacherTrajectoryCollector（复用 EnvManager 框架）

**核心设计**：复用现有的 `EnvManager.rollout()` 流程，只替换 LLM 调用部分为 Teacher Agent。

```python
# agentevolver/module/exp_manager/teacher_collector.py

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger

from agentevolver.schema.trajectory import Trajectory
from agentevolver.schema.task import Task
from agentevolver.module.env_manager.env_manager import ParallelEnvManager
from agentevolver.agent.teacher_agent_flow import TeacherAgentFlow


@dataclass
class TeacherConfig:
    """Teacher agent 配置"""
    teacher_type: str = "api"  # "api" | "local" | "vllm"
    model_name: str = "gpt-4"
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 4096
    
    # ⭐ Log Prob 采集配置
    collect_log_prob: bool = False  # 是否采集 log_prob
                                    # - API (OpenAI): 设置 logprobs=True
                                    # - vLLM/Local: 模型生成时返回 log_prob
    
    # Multi-turn 相关
    max_turns: int = 50  # 最大对话轮数
    

class TeacherTrajectoryCollector:
    """
    Teacher 轨迹采集器
    
    ⭐ 核心设计：复用 EnvManager/EnvWorker 框架
    - 生成的 Trajectory 与 on-policy rollout 格式完全相同
    - Multi-turn 场景自动支持（复用 Linear_CMT）
    - Tokenization、mask 生成完全一致
    """
    
    def __init__(
        self, 
        config: TeacherConfig, 
        env_manager: ParallelEnvManager,
        tokenizer,
    ):
        self.config = config
        self.env_manager = env_manager
        self.tokenizer = tokenizer
        
        # 创建 Teacher Agent Flow（替代原有的 agent_flow）
        self.teacher_agent_flow = TeacherAgentFlow(
            model_name=config.model_name,
            api_base=config.api_base,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            collect_log_prob=config.collect_log_prob,
        )
    
    def collect_trajectories(
        self, 
        tasks: List[Task],
        n_trajectories_per_task: int = 1,
        filter_success_only: bool = True,
        mode: str = "sample",  # "sample" | "greedy"
    ) -> List[Trajectory]:
        """
        为每个 task 采集 Teacher 轨迹
        
        ⭐ 复用 EnvManager.rollout()，只替换 agent_flow
        
        Args:
            tasks: 任务列表
            n_trajectories_per_task: 每个任务采集的轨迹数
            filter_success_only: 是否只保留成功的轨迹
            mode: 采样模式
            
        Returns:
            Trajectory 列表（与 on-policy rollout 格式完全相同）
        """
        # 构建 task_exp_configs（复用现有逻辑）
        task_exp_configs = self._build_task_exp_configs(tasks, n_trajectories_per_task)
        
        # ⭐ 复用 EnvManager.rollout()，传入 teacher_agent_flow
        trajectories = self.env_manager.rollout(
            tasks=tasks,
            task_exp_configs=task_exp_configs,
            mode=mode,
            epoch="teacher_collect",
            agent_flow=self.teacher_agent_flow,  # ⭐ 关键：使用 Teacher agent
        )
        
        # 标记为 Teacher 轨迹
        for traj in trajectories:
            traj.metadata["is_teacher"] = True
            traj.metadata["teacher_model"] = self.config.model_name
            traj.metadata["has_log_prob"] = self.config.collect_log_prob
        
        # 过滤成功的轨迹
        if filter_success_only:
            trajectories = [
                traj for traj in trajectories 
                if traj.reward and traj.reward.outcome == 1.0
            ]
        
        logger.info(f"[TeacherCollector] Collected {len(trajectories)} trajectories "
                   f"({len(tasks)} tasks, {n_trajectories_per_task} per task)")
        
        return trajectories
    
    def _build_task_exp_configs(self, tasks: List[Task], n_per_task: int):
        """构建 task_exp_configs（复用现有逻辑）"""
        from agentevolver.module.exp_manager.exp_manager import TaskExpConfig
        
        task_exp_configs = {}
        for task in tasks:
            task_exp_configs[task.task_id] = TaskExpConfig(
                add_exp=[False] * n_per_task,
                train_mode="discard",
            )
        return task_exp_configs
    
    def save_trajectories(self, trajectories: List[Trajectory], save_path: str):
        """
        保存采集的轨迹到磁盘
        
        格式：JSONL，每行一个轨迹
        """
        import json
        import os
        
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        
        with open(save_path, 'w') as f:
            for traj in trajectories:
                traj_dict = {
                    "task_id": traj.task_id,
                    "data_id": traj.data_id,
                    "rollout_id": traj.rollout_id,
                    "messages": traj.steps,  # Multi-turn 对话历史
                    "reward": traj.reward.outcome if traj.reward else 0.0,
                    "success": traj.reward.outcome == 1.0 if traj.reward else False,
                    "metadata": traj.metadata,
                }
                f.write(json.dumps(traj_dict, ensure_ascii=False) + '\n')
        
        logger.info(f"[TeacherCollector] Saved {len(trajectories)} trajectories to {save_path}")
```

### 4.1.1 Teacher LLM 后端设计（支持 OpenAI-compatible API 和 vLLM）

**设计原则**：Teacher LLM 既可能是 OpenAI-compatible API（GPT-4/Claude/DashScope），也可能是开源模型在本地通过 vLLM 加载调用。我们设计一个统一的抽象接口。

#### 4.1.1.1 抽象基类

```python
# agentevolver/module/teacher/base_teacher_llm.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple


class BaseTeacherLLM(ABC):
    """
    Teacher LLM 抽象基类
    
    ⭐ 核心：统一 OpenAI-compatible API 和 vLLM 开源模型的接口
    
    接口要求（与 EnvWorker 兼容）：
    - __call__(messages, **kwargs) -> Tuple[str, Optional[Dict]]
    - 返回：(response_text, metadata)
    """
    
    @abstractmethod
    def __call__(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Tuple[str, Optional[Dict]]:
        """
        调用 Teacher LLM 生成响应
        
        Args:
            messages: 对话历史 [{"role": "user/assistant", "content": "..."}]
            **kwargs: 其他参数
            
        Returns:
            - response_text: LLM 生成的文本
            - metadata: 可选的元数据（如 log_prob, tokens）
        """
        pass
    
    @abstractmethod
    def reset(self):
        """重置状态（如果有的话）"""
        pass
    
    @property
    @abstractmethod
    def supports_log_prob(self) -> bool:
        """是否支持返回 log_prob"""
        pass
```

#### 4.1.1.2 OpenAI-Compatible API 后端

```python
# agentevolver/module/teacher/openai_teacher_llm.py

import os
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from loguru import logger

from .base_teacher_llm import BaseTeacherLLM


class OpenAITeacherLLM(BaseTeacherLLM):
    """
    OpenAI-Compatible API Teacher LLM 后端
    
    ⭐ 支持：OpenAI GPT-4/4o、Claude via OpenAI API、DashScope、DeepSeek API 等
    
    关键特性：
    - 通过 api_base 参数兼容各种 OpenAI-compatible 服务
    - 支持采集 log_prob（如果 API 支持）
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        collect_log_prob: bool = False,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.collect_log_prob = collect_log_prob
        self._supports_log_prob = collect_log_prob  # 由用户配置决定
        
        # 初始化 OpenAI client（兼容 OpenAI-compatible API）
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=api_base or "https://api.openai.com/v1",
        )
        
        logger.info(f"[OpenAITeacherLLM] Initialized with model={model_name}, "
                   f"api_base={api_base or 'default'}, collect_log_prob={collect_log_prob}")
    
    def __call__(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Tuple[str, Optional[Dict]]:
        """调用 OpenAI-compatible API 生成响应"""
        try:
            # 构建请求参数
            request_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            
            # 如果需要采集 log_prob（OpenAI API 支持）
            if self.collect_log_prob:
                request_params["logprobs"] = True
                request_params["top_logprobs"] = 1
            
            # 调用 API
            response = self.client.chat.completions.create(**request_params)
            
            # 提取响应
            response_text = response.choices[0].message.content
            
            # 提取 log_prob（如果有）
            metadata = {}
            if self.collect_log_prob and response.choices[0].logprobs:
                logprobs = response.choices[0].logprobs.content
                if logprobs:
                    # 提取每个 token 的 log_prob
                    token_logprobs = [t.logprob for t in logprobs]
                    metadata["log_probs"] = token_logprobs
                    metadata["tokens"] = [t.token for t in logprobs]
            
            return response_text, metadata
            
        except Exception as e:
            logger.error(f"[OpenAITeacherLLM] API call failed: {e}")
            raise
    
    def reset(self):
        """OpenAI API 无状态，无需重置"""
        pass
    
    @property
    def supports_log_prob(self) -> bool:
        return self._supports_log_prob
```

#### 4.1.1.3 vLLM 开源模型后端

```python
# agentevolver/module/teacher/vllm_teacher_llm.py

from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import torch

from .base_teacher_llm import BaseTeacherLLM


class VLLMTeacherLLM(BaseTeacherLLM):
    """
    vLLM 开源模型 Teacher LLM 后端
    
    ⭐ 支持：Qwen、Llama、DeepSeek、Mistral 等本地开源模型
    
    关键特性：
    - 使用 vLLM 进行高效推理
    - 天然支持 log_prob 采集（开源模型可以获取完整概率分布）
    - 支持 tensor_parallel 多卡推理
    
    参考：VERL 框架的 vLLMRollout 实现
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer: Any = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        collect_log_prob: bool = True,  # 开源模型默认采集 log_prob
        trust_remote_code: bool = True,
    ):
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.collect_log_prob = collect_log_prob
        
        # 延迟导入 vLLM
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("vLLM is required for VLLMTeacherLLM. "
                            "Install with: pip install vllm")
        
        # 初始化 tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=trust_remote_code
            )
        else:
            self.tokenizer = tokenizer
        
        # 初始化 vLLM 引擎
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
        )
        
        # 默认采样参数
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=1 if collect_log_prob else 0,  # 采集 top-1 log_prob
        )
        
        logger.info(f"[VLLMTeacherLLM] Initialized with model={model_path}, "
                   f"tp={tensor_parallel_size}, collect_log_prob={collect_log_prob}")
    
    def __call__(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Tuple[str, Optional[Dict]]:
        """调用 vLLM 开源模型生成响应"""
        from vllm import SamplingParams
        
        try:
            # 使用 chat template 格式化消息
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 更新采样参数（如果有覆盖）
            sampling_params = SamplingParams(
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                logprobs=1 if self.collect_log_prob else 0,
            )
            
            # 调用 vLLM 生成
            outputs = self.llm.generate([prompt], sampling_params)
            output = outputs[0]
            
            # 提取响应文本
            response_text = output.outputs[0].text
            
            # 提取 log_prob（如果采集）
            metadata = {}
            if self.collect_log_prob and output.outputs[0].logprobs:
                logprobs_list = output.outputs[0].logprobs
                # vLLM 返回的 logprobs 是 List[Dict[token_id, Logprob]]
                token_logprobs = []
                tokens = []
                for lp_dict in logprobs_list:
                    if lp_dict:
                        # 获取生成的 token 的 logprob
                        for token_id, logprob_obj in lp_dict.items():
                            token_logprobs.append(logprob_obj.logprob)
                            tokens.append(logprob_obj.decoded_token)
                            break  # 只取第一个（实际生成的 token）
                
                metadata["log_probs"] = token_logprobs
                metadata["tokens"] = tokens
                # 同时保存 token_ids，方便后续对齐
                metadata["token_ids"] = output.outputs[0].token_ids
            
            return response_text, metadata
            
        except Exception as e:
            logger.error(f"[VLLMTeacherLLM] Generation failed: {e}")
            raise
    
    def reset(self):
        """vLLM 无状态，无需重置"""
        pass
    
    @property
    def supports_log_prob(self) -> bool:
        """vLLM 开源模型始终支持 log_prob"""
        return True


# ============================================================
# 工厂函数：根据配置创建 Teacher LLM
# ============================================================

def create_teacher_llm(config: Dict[str, Any]) -> BaseTeacherLLM:
    """
    工厂函数：根据配置创建 Teacher LLM 实例
    
    Args:
        config: Teacher LLM 配置
            - type: "openai" 或 "vllm"
            - 其他参数根据 type 决定
            
    Returns:
        BaseTeacherLLM 实例
        
    Example config for OpenAI:
        {
            "type": "openai",
            "model_name": "gpt-4",
            "api_base": "https://api.openai.com/v1",
            "api_key": "sk-xxx",
            "temperature": 0.0,
            "max_tokens": 4096,
            "collect_log_prob": false
        }
        
    Example config for vLLM:
        {
            "type": "vllm",
            "model_path": "/path/to/qwen-72b",
            "temperature": 0.0,
            "max_tokens": 4096,
            "tensor_parallel_size": 4,
            "gpu_memory_utilization": 0.85,
            "collect_log_prob": true
        }
    """
    llm_type = config.get("type", "openai").lower()
    
    if llm_type == "openai":
        return OpenAITeacherLLM(
            model_name=config.get("model_name", "gpt-4"),
            api_base=config.get("api_base"),
            api_key=config.get("api_key"),
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens", 4096),
            collect_log_prob=config.get("collect_log_prob", False),
        )
    elif llm_type == "vllm":
        return VLLMTeacherLLM(
            model_path=config["model_path"],
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens", 4096),
            tensor_parallel_size=config.get("tensor_parallel_size", 1),
            gpu_memory_utilization=config.get("gpu_memory_utilization", 0.85),
            collect_log_prob=config.get("collect_log_prob", True),
            trust_remote_code=config.get("trust_remote_code", True),
        )
    else:
        raise ValueError(f"Unsupported teacher LLM type: {llm_type}. "
                        f"Supported types: 'openai', 'vllm'")
```

#### 4.1.1.4 与 AgentFlow 接口的兼容

```python
# agentevolver/module/teacher/teacher_agent_flow.py

from typing import List, Dict, Any, Optional, Callable
from omegaconf import DictConfig

from agentevolver.module.agent_flow.base_agent_flow import BaseAgentFlow
from .base_teacher_llm import BaseTeacherLLM
from .vllm_teacher_llm import create_teacher_llm


class TeacherAgentFlow(BaseAgentFlow):
    """
    Teacher Agent Flow - 包装 Teacher LLM 以兼容 BaseAgentFlow 接口
    
    ⭐ 核心：作为 BaseTeacherLLM 和 EnvWorker 之间的适配层
    
    使用方式：
    1. 创建 TeacherAgentFlow 替代标准 AgentFlow
    2. TeacherAgentFlow 内部使用 Teacher LLM（OpenAI 或 vLLM）
    3. EnvWorker 无需修改，调用接口完全一致
    """
    
    def __init__(
        self,
        teacher_llm: BaseTeacherLLM,
        tokenizer: Any,
        config: DictConfig = None,
        **kwargs
    ):
        """
        初始化 Teacher Agent Flow
        
        Args:
            teacher_llm: Teacher LLM 实例（OpenAI 或 vLLM）
            tokenizer: Tokenizer（用于与 BaseAgentFlow 兼容）
            config: 配置
        """
        # 创建一个包装 llm_chat_fn
        def teacher_llm_chat_fn(messages, **kw):
            response_text, metadata = teacher_llm(messages, **kw)
            # 返回格式与标准 llm_chat_fn 一致
            return {"role": "assistant", "content": response_text, "_metadata": metadata}
        
        super().__init__(
            llm_chat_fn=teacher_llm_chat_fn,
            tokenizer=tokenizer,
            config=config,
            **kwargs
        )
        
        self.teacher_llm = teacher_llm
    
    @property
    def supports_log_prob(self) -> bool:
        """是否支持 log_prob"""
        return self.teacher_llm.supports_log_prob
```

### 4.2 ExperienceManager 扩展（支持 Teacher 轨迹）

**设计原则**：在现有 `ExperienceManager` 中增加 Teacher 轨迹支持，而不是创建新类。

```python
# agentevolver/module/exp_manager/exp_manager.py
# 
# ⭐ 扩展现有 ExperienceManager，添加 Teacher 轨迹支持

class ExperienceManager(object):
    
    def __init__(self, config: DictConfig):
        # ... 现有初始化代码 ...
        
        # ⭐ 新增：Teacher Experience 相关属性
        teacher_config = self.exp_manager_config.get("teacher_experience", {})
        self.teacher_enabled = teacher_config.get("enable", False)
        self.teacher_data_path = teacher_config.get("data_path", None)
        self.teacher_exp_ratio = teacher_config.get("exp_ratio", 0.2)
        self.teacher_max_per_task = teacher_config.get("max_trajectories_per_task", 3)
        self.teacher_select_mode = teacher_config.get("select_mode", "random")
        
        # ⭐ 新增：Teacher 轨迹存储（与 self-generated 分开存储）
        self.teacher_task2trajectories: Dict[str, List[Trajectory]] = defaultdict(list)
        
        # 加载预采集的 Teacher 轨迹
        if self.teacher_enabled and self.teacher_data_path:
            self.load_teacher_trajectories(self.teacher_data_path)
    
    # ==================== Teacher Experience 相关方法 ====================
    
    def load_teacher_trajectories(self, data_path: str) -> int:
        """
        从磁盘加载 Teacher 轨迹
        
        ⭐ Multi-turn 支持：加载的轨迹格式与 on-policy rollout 一致
        
        支持格式：
        - .jsonl: 每行一个 JSON 对象（推荐）
        - .pkl: Pickle 序列化
        
        Returns:
            加载的轨迹数量
        """
        import json
        import pickle
        import os
        
        if not os.path.exists(data_path):
            logger.warning(f"[ExperienceManager] Teacher data path not found: {data_path}")
            return 0
        
        count = 0
        
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    traj_dict = json.loads(line.strip())
                    traj = self._dict_to_teacher_trajectory(traj_dict)
                    self.teacher_task2trajectories[traj.task_id].append(traj)
                    count += 1
                    
        elif data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                trajectories = pickle.load(f)
                for traj in trajectories:
                    # 标记为 Teacher 轨迹
                    traj.metadata["is_teacher"] = True
                    self.teacher_task2trajectories[traj.task_id].append(traj)
                    count += 1
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        logger.info(f"[ExperienceManager] Loaded {count} teacher trajectories from {data_path}")
        logger.info(f"[ExperienceManager] Teacher tasks: {len(self.teacher_task2trajectories)}")
        
        return count
    
    def _dict_to_teacher_trajectory(self, traj_dict: Dict) -> Trajectory:
        """
        将字典转换为 Trajectory 对象
        
        ⭐ Multi-turn 关键：保持与 on-policy 轨迹相同的结构
        ⭐ Log Prob 处理：根据数据中是否有 log_prob 设置 has_log_prob 标记
        """
        from agentevolver.schema.trajectory import Trajectory
        from agentevolver.schema.reward import Reward
        
        # 创建 Trajectory 对象
        traj = Trajectory(
            task_id=traj_dict.get("task_id", ""),
            data_id=traj_dict.get("data_id", ""),
            rollout_id=traj_dict.get("rollout_id", ""),
            steps=traj_dict.get("messages", []),  # ⭐ Multi-turn 对话历史
        )
        
        # 设置 reward
        reward_val = traj_dict.get("reward", 0.0)
        success = traj_dict.get("success", reward_val == 1.0)
        traj.reward = Reward(outcome=reward_val if success else 0.0)
        
        # 设置 metadata
        traj.metadata = traj_dict.get("metadata", {})
        traj.metadata["is_teacher"] = True
        traj.metadata["is_experience_replay"] = True  # 标记为 off-policy
        traj.metadata["teacher_model"] = traj_dict.get("teacher_model", 
                                                        traj.metadata.get("teacher_model", "unknown"))
        
        # ⭐ 处理 log_prob（关键）
        # 检查轨迹数据中是否包含 log_prob
        # 可能在 traj_dict 顶层或 metadata 中
        log_probs = (traj_dict.get("log_probs") or 
                     traj_dict.get("metadata", {}).get("old_log_probs"))
        
        if log_probs and len(log_probs) > 0:
            traj.metadata["old_log_probs"] = log_probs
            traj.metadata["has_log_prob"] = True
            logger.debug(f"[ExperienceManager] Teacher trajectory {traj.task_id} has log_prob")
        else:
            traj.metadata["has_log_prob"] = False
            logger.debug(f"[ExperienceManager] Teacher trajectory {traj.task_id} without log_prob")
        
        return traj
    
    def get_teacher_trajectories(
        self, 
        task_ids: List[str],
        num_per_task: int = 1,
    ) -> List[Trajectory]:
        """
        获取指定 task 的 Teacher 轨迹
        
        ⭐ 支持多种选择模式：
        - "all": 返回所有轨迹
        - "first": 返回前 N 个
        - "random": 随机选择 N 个
        - "entropy": 按 entropy 从低到高排序
                     * 对于有 log_prob 的轨迹：直接计算 entropy 并排序
                     * 对于没有 log_prob 的轨迹：在 trainer 中使用当前 policy 计算 entropy
                     * 两部分合并后按 entropy 从低到高排序
        
        Args:
            task_ids: 任务 ID 列表
            num_per_task: 每个任务获取的轨迹数
            
        Returns:
            Teacher 轨迹列表（格式与 self-generated 一致）
        """
        import copy
        import numpy as np
        
        trajectories = []
        
        for task_id in task_ids:
            if task_id not in self.teacher_task2trajectories:
                continue
            
            task_trajs = self.teacher_task2trajectories[task_id]
            
            if self.teacher_select_mode == "all":
                selected = task_trajs
            elif self.teacher_select_mode == "first":
                selected = task_trajs[:num_per_task]
            elif self.teacher_select_mode == "random":
                selected = random.sample(task_trajs, min(num_per_task, len(task_trajs)))
            elif self.teacher_select_mode == "entropy":
                # ⭐ Entropy 模式：按 entropy 从低到高排序
                trajs_with_logprob = []
                trajs_without_logprob = []
                
                for traj in task_trajs:
                    if traj.metadata.get("has_log_prob", False) and "old_log_probs" in traj.metadata:
                        trajs_with_logprob.append(traj)
                    else:
                        trajs_without_logprob.append(traj)
                
                # 对于有 log_prob 的轨迹，计算 entropy
                if trajs_with_logprob:
                    traj_entropys = []
                    for traj in trajs_with_logprob:
                        # 从 old_log_probs 计算 entropy
                        # entropy = -sum(p * log(p))，其中 p = exp(log_prob)
                        old_log_probs = np.array(traj.metadata["old_log_probs"])
                        probs = np.exp(old_log_probs)
                        # 避免 log(0)
                        probs = np.clip(probs, 1e-10, 1.0)
                        entropy = -np.sum(probs * np.log(probs))
                        traj_entropys.append((entropy, traj))
                    
                    # 按 entropy 从低到高排序
                    trajs_with_logprob = [traj for _, traj in sorted(traj_entropys, key=lambda x: x[0])]
                
                # 合并：先是有 log_prob 的（已排序），然后是没有 log_prob 的（需要在 trainer 中处理）
                selected = trajs_with_logprob + trajs_without_logprob
                selected = selected[:num_per_task]
                
                if trajs_without_logprob:
                    logger.debug(
                        f"Task {task_id}: {len(trajs_with_logprob)} trajectories with log_prob "
                        f"(sorted by entropy), {len(trajs_without_logprob)} without log_prob "
                        f"(will use current policy to compute entropy in trainer)"
                    )
            else:
                selected = task_trajs[:num_per_task]
            
            # 深拷贝以避免修改原始轨迹
            for traj in selected:
                traj_copy = copy.deepcopy(traj)
                traj_copy.metadata["is_experience_replay"] = True
                traj_copy.metadata["is_teacher"] = True
                trajectories.append(traj_copy)
        
        return trajectories
    
    def get_valid_teacher_task_ids(self) -> List[str]:
        """
        获取所有有 Teacher 轨迹的 task_id
        
        排除已在 skip_uid_set 中的 task（已完全解决）
        """
        valid_ids = [
            task_id for task_id in self.teacher_task2trajectories.keys()
            if task_id not in self.skip_uid_set
        ]
        return valid_ids
    
    def has_teacher_trajectory(self, task_id: str) -> bool:
        """检查某个 task 是否有 Teacher 轨迹"""
        return (task_id in self.teacher_task2trajectories 
                and len(self.teacher_task2trajectories[task_id]) > 0)
    
    def get_teacher_stats(self) -> Dict:
        """获取 Teacher 轨迹统计信息"""
        total_tasks = len(self.teacher_task2trajectories)
        total_trajectories = sum(len(trajs) for trajs in self.teacher_task2trajectories.values())
        
        return {
            "total_tasks": total_tasks,
            "total_trajectories": total_trajectories,
            "avg_trajectories_per_task": total_trajectories / total_tasks if total_tasks > 0 else 0,
        }
    
    # ==================== 统一的 Off-policy 获取接口 ====================
    
    def get_mixed_offpolicy_trajectories(
        self,
        self_exp_task_ids: List[str],
        teacher_task_ids: List[str],
        num_per_task: int = 1,
    ) -> Tuple[List[Trajectory], List[Trajectory]]:
        """
        获取混合的 off-policy 轨迹
        
        ⭐ 统一接口，同时支持 self-generated 和 teacher 轨迹
        
        Returns:
            - self_trajectories: Self-generated 轨迹列表
            - teacher_trajectories: Teacher 轨迹列表
        """
        # 获取 self-generated 轨迹（使用现有逻辑）
        self_trajectories = self.get_offpolicy_batch(
            tasks=[Task(task_id=tid) for tid in self_exp_task_ids],
            num_trajectories_per_task=num_per_task,
        ) if self_exp_task_ids else []
        
        # 获取 teacher 轨迹
        teacher_trajectories = self.get_teacher_trajectories(
            task_ids=teacher_task_ids,
            num_per_task=num_per_task,
        ) if teacher_task_ids else []
        
        return self_trajectories, teacher_trajectories
    
    # ==================== 保存/加载扩展 ====================
    
    def save_experience_pool_to_disk(self, save_dir: str):
        """
        保存 experience pool 到磁盘
        
        ⭐ 扩展：同时保存 self-generated 和 teacher 轨迹信息
        """
        import json
        import pickle
        import os
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存现有内容（self-generated）
        # ... 现有保存逻辑 ...
        
        # ⭐ 新增：保存 Teacher 轨迹统计（轨迹本身从文件加载，不需要重复保存）
        teacher_info = {
            "enabled": self.teacher_enabled,
            "data_path": self.teacher_data_path,
            "stats": self.get_teacher_stats(),
        }
        with open(os.path.join(save_dir, "teacher_info.json"), 'w') as f:
            json.dump(teacher_info, f, indent=2)
        
        logger.info(f"[ExperienceManager] Saved experience pool to {save_dir}")
```

### 4.3 Loss 计算适配

在 `het_core_algos.py` 中添加对 Teacher 轨迹的支持，根据 `use_log_prob` 配置决定使用哪种 ratio 计算方式：

```python
# agentevolver/module/exp_manager/het_core_algos.py

def het_compute_token_on_off_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    exp_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    off_cliprange_high=1.0,
    # ⭐ 新增参数
    teacher_mask=None,              # 标记哪些是 Teacher 轨迹
    teacher_use_log_prob=False,     # ⭐ 是否使用 log_prob（关键配置）
    teacher_policy_shaping_enable=True,  # 是否启用 policy shaping
    teacher_policy_shaping_mode="p_div_p_beta",  # Policy shaping 模式
    teacher_policy_shaping_beta=0.1,  # Policy shaping 参数 β
    teacher_use_clip=False,         # Teacher 轨迹是否使用 clipping
    loss_agg_mode: str = "token-mean",
    **kwargs,
):
    """
    计算混合 on-policy、self-generated off-policy 和 teacher off-policy 的 loss
    
    ⭐ Teacher 轨迹的两种处理模式：
    1. use_log_prob=True: 使用 ExGRPO 形式，ratio = π_current / π_old（需要 log_prob）
    2. use_log_prob=False: 使用 LUFFY 形式，ratio = π_current（分母=1）
    
    Args:
        teacher_mask: [batch, seq_len] - 1 表示 Teacher 轨迹，0 表示非 Teacher
        teacher_use_log_prob: 是否使用 log_prob 进行重要性采样
        teacher_policy_shaping_enable: 是否启用 policy shaping（use_log_prob=False 时推荐）
        teacher_policy_shaping_mode: Policy shaping 模式
        teacher_policy_shaping_beta: β 参数
        teacher_use_clip: Teacher 轨迹是否使用 clipping
    """
    
    # 标准的 ratio 计算（用于 on-policy 和 self-generated off-policy）
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    
    # ⭐ Teacher 轨迹的处理
    if teacher_mask is not None:
        if teacher_use_log_prob:
            # ========== 模式 1: 使用 log_prob（ExGRPO 形式） ==========
            # Teacher 轨迹有 log_prob，使用标准重要性采样
            # ratio = π_current / π_old = exp(log_prob - old_log_prob)
            # 此时 old_log_prob 已经包含 Teacher 的 log_prob
            teacher_ratio = ratio  # 直接使用标准 ratio
            
            # Policy shaping 可选（有 log_prob 时通常不需要）
            if teacher_policy_shaping_enable:
                teacher_ratio = _apply_policy_shaping(
                    teacher_ratio, 
                    mode=teacher_policy_shaping_mode,
                    beta=teacher_policy_shaping_beta,
                )
        else:
            # ========== 模式 2: 无 log_prob（LUFFY 形式） ==========
            # Teacher 轨迹无 log_prob，使用简化计算
            # 假设 π_old = 1，ratio = π_current / 1 = π_current
            teacher_ratio = torch.exp(log_prob)
            
            # Policy shaping（LUFFY 推荐使用）
            if teacher_policy_shaping_enable:
                teacher_ratio = _apply_policy_shaping(
                    teacher_ratio,
                    mode=teacher_policy_shaping_mode,
                    beta=teacher_policy_shaping_beta,
                )
        
        # 混合 ratio：根据 teacher_mask 选择使用哪个 ratio
        # - teacher_mask=1: Teacher 轨迹，使用 teacher_ratio
        # - teacher_mask=0: 非 Teacher（on-policy 或 self-generated），使用标准 ratio
        ratio = torch.where(
            teacher_mask.bool(),
            teacher_ratio,
            ratio
        )
    
    # =============== Loss 计算 ===============
    
    # On-policy loss（与现有逻辑相同）
    on_pg_losses = -advantages * ratio
    on_pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    on_pg_losses = torch.maximum(on_pg_losses, on_pg_losses2)
    on_pg_loss = verl_F.masked_mean(on_pg_losses, (1.0 - exp_mask) * response_mask)
    
    # Off-policy loss（包括 self-generated 和 teacher）
    off_pg_losses = -advantages * ratio
    
    if teacher_mask is not None:
        # Self-generated off-policy：使用 clipping
        self_off_pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + off_cliprange_high)
        self_off_pg_losses = torch.maximum(off_pg_losses, self_off_pg_losses2)
        
        # Teacher off-policy：根据配置决定是否使用 clipping
        if teacher_use_clip:
            teacher_off_pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + off_cliprange_high)
            teacher_off_pg_losses = torch.maximum(off_pg_losses, teacher_off_pg_losses2)
        else:
            teacher_off_pg_losses = off_pg_losses  # 不 clip
        
        # 根据 teacher_mask 选择
        off_pg_losses = torch.where(
            teacher_mask.bool(),
            teacher_off_pg_losses,  # Teacher
            self_off_pg_losses       # Self-generated
        )
    else:
        off_pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + off_cliprange_high)
        off_pg_losses = torch.maximum(off_pg_losses, off_pg_losses2)
    
    off_pg_loss = verl_F.masked_mean(off_pg_losses, exp_mask * response_mask)
    
    # 合并 loss
    pg_losses = off_pg_losses * exp_mask + on_pg_losses * (1.0 - exp_mask)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    
    return {
        "pg_loss": pg_loss,
        "on_pg_loss": on_pg_loss,
        "off_pg_loss": off_pg_loss,
        # ... 其他 metrics
    }


def _apply_policy_shaping(ratio, mode, beta=0.1):
    """
    应用 Policy Shaping
    
    Args:
        ratio: 原始 ratio
        mode: shaping 模式
        beta: β 参数
    
    Returns:
        shaped_ratio: 处理后的 ratio
    """
    if mode == "p_div_p_beta":
        # LUFFY 的 policy shaping: f(x) = x / (x + β)
        # 放大低概率信号，抑制高概率信号
        return ratio / (ratio + beta)
    elif mode == "sqrt":
        return torch.sqrt(ratio)
    elif mode == "log":
        # 直接使用 log_prob 作为权重
        return torch.log(ratio + 1e-8)
    elif mode == "no_shaping":
        return ratio
    else:
        raise ValueError(f"Unknown policy_shaping_mode: {mode}")
```

### 4.3.1 Teacher log_prob 的处理流程

```python
# 在 ae_ray_trainer.py 中处理 Teacher 轨迹的 old_log_prob

def _prepare_teacher_old_log_probs(self, batch, teacher_trajectories, teacher_config):
    """
    准备 Teacher 轨迹的 old_log_prob
    
    ⭐ 根据 use_log_prob 配置决定处理方式
    """
    use_log_prob = teacher_config.get("use_log_prob", False)
    
    if use_log_prob:
        # ========== 模式 1: 使用 log_prob ==========
        # 从 Teacher 轨迹中提取 log_prob
        # 替换 batch 中对应位置的 old_log_prob
        for traj in teacher_trajectories:
            if traj.metadata.get("has_log_prob", False):
                recorded_log_probs = traj.metadata.get("old_log_probs")
                # 替换到 batch 中...
            else:
                logger.warning(f"Teacher trajectory {traj.task_id} missing log_prob, "
                             "but use_log_prob=True. Using current policy's log_prob.")
    else:
        # ========== 模式 2: 不使用 log_prob ==========
        # 设置 old_log_prob = 0（即 π_old = 1）
        # 这样 ratio = exp(log_prob - 0) = exp(log_prob) = π_current
        # 在 loss 计算中会被 LUFFY 的简化公式处理
        pass  # 不需要替换，loss 函数会根据 teacher_mask 特殊处理
    
    return batch
```

### 4.3.2 Teacher 轨迹的 Entropy 模式选择

当 `teacher_select_mode == "entropy"` 时，对于没有 log_prob 的轨迹，需要在 trainer 中使用当前 policy 计算 entropy：

```python
# 在 ae_ray_trainer.py 中处理 Teacher 轨迹的 entropy 选择

def _select_best_teacher_by_current_entropy(
    self,
    teacher_trajectories: List,
    tasks: List,
    num_trajectories_per_task: int = 1,
) -> List:
    """
    使用当前 policy 计算 entropy，选择每个 task 的最优 teacher 轨迹。
    
    ⭐ 对于有 log_prob 的轨迹：已经由 get_teacher_trajectories 按 entropy 排序
    ⭐ 对于没有 log_prob 的轨迹：使用当前 policy 计算 entropy 并排序
    ⭐ Multi-turn 关键：只对 LLM 响应部分（loss_mask=1）计算 entropy。
    
    工作流程：
    1. 分离有/无 log_prob 的轨迹
    2. 对于没有 log_prob 的轨迹：
       - 转换为 CMT 格式
       - 使用当前 policy 计算 log_prob
       - 计算 entropy（只考虑 LLM 响应部分）
       - 按 entropy 从低到高排序
    3. 合并两部分轨迹（都按 entropy 排序）
    4. 选择前 num_trajectories_per_task 个
    """
    # 分离有/无 log_prob 的轨迹
    trajs_with_logprob = []
    trajs_without_logprob = []
    
    for traj in teacher_trajectories:
        if traj.metadata.get("has_log_prob", False) and "old_log_probs" in traj.metadata:
            trajs_with_logprob.append(traj)  # 已在 get_teacher_trajectories 中排序
        else:
            trajs_without_logprob.append(traj)  # 需要在这里计算 entropy
    
    # 对于没有 log_prob 的轨迹，使用当前 policy 计算 entropy
    if trajs_without_logprob:
        # 转换为 CMT 并计算 entropy（类似 _select_best_offpolicy_by_current_entropy）
        candidate_cmts = self.env_manager.convert_offpolicy_to_cmt(...)
        log_prob_result = self.actor_rollout_wg.compute_log_prob(candidate_batch)
        entropys = log_prob_result.batch["entropys"]
        
        # 使用 loss_mask 计算 LLM 响应部分的平均 entropy
        response_masks = candidate_batch.batch["loss_mask"][:, -response_length:]
        # 计算每个轨迹的平均 entropy 并排序
        trajs_without_logprob = sorted_by_entropy(trajs_without_logprob, entropys, response_masks)
    
    # 合并：先是有 log_prob 的（已排序），然后是没有 log_prob 的（已排序）
    all_sorted = trajs_with_logprob + trajs_without_logprob
    return all_sorted[:num_trajectories_per_task]
```

**在 trainer 中的调用**：

```python
# 在 ae_ray_trainer.py 的 fit() 方法中

if teacher_exp_tasks:
    teacher_offpolicy_trajectories = self.exp_manager.get_teacher_offpolicy_batch(...)
    
    # ⭐ 如果 teacher_select_mode == 'entropy'，对没有 log_prob 的轨迹使用当前 policy 计算 entropy
    teacher_select_mode = teacher_exp_config.get("select_mode", "random")
    if teacher_select_mode == "entropy":
        teacher_offpolicy_trajectories = self._select_best_teacher_by_current_entropy(
            teacher_trajectories=teacher_offpolicy_trajectories,
            tasks=teacher_exp_tasks,
            num_trajectories_per_task=teacher_num_per_task,
        )
```

### 4.4 ExperienceMixCollateFn 扩展

```python
# agentevolver/module/exp_manager/experience_collate.py

class ExperienceMixCollateFn:
    """
    扩展的 Experience 混合函数，支持三种数据类型：
    1. On-policy: 当前策略生成的新轨迹
    2. Self-generated off-policy: 自身历史成功轨迹（ExGRPO）
    3. Teacher off-policy: 外部 Teacher 模型的轨迹（LUFFY）
    
    ⭐ 设计变更：使用统一的 ExperienceManager（不再分离 TeacherExperienceManager）
    """
    
    def __init__(
        self,
        exp_manager,  # ⭐ 统一的 ExperienceManager（同时管理 self-generated 和 teacher）
        train_task_manager,
        # Self-generated experience 配置
        self_exp_ratio: float = 0.3,
        # Teacher experience 配置
        teacher_exp_ratio: float = 0.2,
        teacher_exp_enabled: bool = True,
        # 共同配置
        n_rollout: int = 8,
        offpolicy_trajectories_per_task: int = 1,
        replay_start_ratio: float = 0.35,
        **kwargs,
    ):
        self.exp_manager = exp_manager  # ⭐ 统一的 manager
        self.train_task_manager = train_task_manager
        self.self_exp_ratio = self_exp_ratio
        self.teacher_exp_ratio = teacher_exp_ratio
        self.teacher_exp_enabled = teacher_exp_enabled and exp_manager.teacher_enabled
        self.n_rollout = n_rollout
        self.offpolicy_trajectories_per_task = offpolicy_trajectories_per_task
        self.replay_start_ratio = replay_start_ratio
    
    def __call__(
        self,
        training_tasks: List[Task],
        training_progress: float,
        enable_replay: bool = True,
    ) -> Tuple[List[Task], List[Task], List[Task]]:
        """
        混合三种类型的 tasks
        
        ⭐ Multi-turn 支持：所有轨迹格式一致，无需额外处理
        
        Returns:
            - self_exp_tasks: 使用 self-generated experience 的 tasks
            - teacher_exp_tasks: 使用 teacher experience 的 tasks
            - on_policy_tasks: 纯 on-policy 的 tasks
        """
        batch_size = len(training_tasks)
        
        # 检查是否达到 replay 开始条件
        if not enable_replay or training_progress < self.replay_start_ratio:
            return [], [], training_tasks
        
        # 计算各类型的 task 数量
        target_self_exp_count = int(batch_size * self.self_exp_ratio)
        target_teacher_exp_count = int(batch_size * self.teacher_exp_ratio) if self.teacher_exp_enabled else 0
        
        # 获取可用的 self-generated experience task_ids（从统一的 exp_manager）
        valid_self_exp_task_ids = self.exp_manager.get_valid_replay_task_ids()
        
        # 获取可用的 teacher experience task_ids（从统一的 exp_manager）
        valid_teacher_task_ids = []
        if self.teacher_exp_enabled:
            valid_teacher_task_ids = self.exp_manager.get_valid_teacher_task_ids()
        
        # 采样 self-generated experience tasks
        n_self_exp = min(len(valid_self_exp_task_ids), target_self_exp_count)
        sampled_self_exp_task_ids = random.sample(valid_self_exp_task_ids, n_self_exp) if n_self_exp > 0 else []
        
        # 采样 teacher experience tasks
        # 优先选择没有在 self_exp 中的 task，避免同一 task 同时用两种 off-policy
        available_teacher_task_ids = [
            tid for tid in valid_teacher_task_ids 
            if tid not in sampled_self_exp_task_ids
        ]
        n_teacher_exp = min(len(available_teacher_task_ids), target_teacher_exp_count)
        sampled_teacher_task_ids = random.sample(available_teacher_task_ids, n_teacher_exp) if n_teacher_exp > 0 else []
        
        # 转换为 Task 对象
        self_exp_tasks = self._task_ids_to_tasks(sampled_self_exp_task_ids)
        teacher_exp_tasks = self._task_ids_to_tasks(sampled_teacher_task_ids)
        
        # 补充 on-policy tasks
        used_task_ids = set(sampled_self_exp_task_ids + sampled_teacher_task_ids)
        on_policy_tasks = [t for t in training_tasks if t.task_id not in used_task_ids]
        n_on_policy = batch_size - len(self_exp_tasks) - len(teacher_exp_tasks)
        on_policy_tasks = on_policy_tasks[:n_on_policy]
        
        logger.debug(f"[ExperienceMixCollateFn] Batch split: "
                    f"self_exp={len(self_exp_tasks)}, teacher_exp={len(teacher_exp_tasks)}, "
                    f"on_policy={len(on_policy_tasks)}")
        
        return self_exp_tasks, teacher_exp_tasks, on_policy_tasks
    
    def _task_ids_to_tasks(self, task_ids: List[str]) -> List[Task]:
        """将 task_id 转换为 Task 对象"""
        tasks = []
        for task_id in task_ids:
            task = self.train_task_manager.get_task_by_id(task_id)
            if task is not None:
                tasks.append(task)
        return tasks
```

### 4.5 Multi-turn 场景的关键考虑

**现有 ExGRPO 实现的 Multi-turn 支持**（Teacher Experience 需要保持一致）：

| 组件 | Multi-turn 处理 | Teacher Experience 适配 |
|------|----------------|------------------------|
| `exp_mask` | 只对 LLM 响应位置设为 1 | ✅ 自动复用（通过 `convert_offpolicy_to_cmt`） |
| `response_mask` | 基于 `loss_mask`，只包含 LLM tokens | ✅ 自动复用（通过 `tokenize_steps`） |
| `old_log_probs` | 保存完整 response 部分 | ⚠️ Teacher 通常无 log_prob，需要特殊处理 |
| `teacher_mask` | N/A | ⭐ 新增，标记 Teacher 轨迹位置 |

**关键实现点**：

```python
# 在 env_manager.convert_offpolicy_to_cmt() 中处理 Teacher 轨迹

def convert_offpolicy_to_cmt(self, offpolicy_trajectories, config, tokenizer, task_id_to_data_id=None):
    """
    将 off-policy 轨迹转换为 CMT 对象
    
    ⭐ Multi-turn + Teacher 支持：
    - 复用现有的转换逻辑
    - 通过 metadata["is_teacher"] 标记 Teacher 轨迹
    """
    for traj in offpolicy_trajectories:
        cmt = Linear_CMT(config, tokenizer)
        
        # ... 现有转换逻辑 ...
        
        # ⭐ 标记 Teacher 轨迹
        cmt.metadata["is_teacher"] = traj.metadata.get("is_teacher", False)
        cmt.metadata["has_log_prob"] = traj.metadata.get("has_log_prob", False)
        
        # ⭐ Multi-turn 关键：LLM 消息保持 author="llm"
        # 这样 loss_mask=1，参与 off-policy loss 计算
        for step in cmt.steps:
            if step["role"] == "assistant":
                step["author"] = "llm"  # 不是 "llm(do_not_train)"
```

---

## 数据格式定义

### 5.1 Teacher Trajectory 文件格式（JSONL）

#### 无 log_prob（GPT-4/Claude 采集）

```json
{"task_id": "alfworld_task_001", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "reward": 1.0, "success": true, "teacher_model": "gpt-4", "metadata": {"env": "alfworld", "collected_at": "2024-01-01"}}
{"task_id": "alfworld_task_002", "messages": [...], "reward": 1.0, "success": true, "teacher_model": "gpt-4", "metadata": {...}}
```

#### 有 log_prob（开源模型采集）

```json
{"task_id": "alfworld_task_001", "messages": [...], "reward": 1.0, "success": true, "teacher_model": "qwen-72b", "log_probs": [-0.12, -0.05, -0.23, ...], "metadata": {"env": "alfworld"}}
{"task_id": "alfworld_task_002", "messages": [...], "reward": 1.0, "success": true, "teacher_model": "qwen-72b", "log_probs": [...], "metadata": {...}}
```

### 5.2 Teacher Trajectory Schema

```python
@dataclass
class TeacherTrajectorySchema:
    # ========== 必需字段 ==========
    task_id: str                    # 任务 ID
    messages: List[Dict[str, Any]]  # 完整对话历史（Multi-turn）
    reward: float                   # 最终 reward
    success: bool                   # 是否成功
    teacher_model: str              # Teacher 模型名称
    
    # ========== 可选字段 ==========
    log_probs: Optional[List[float]] = None  # ⭐ Token 级别的 log_prob
                                              # 开源模型采集时提供
                                              # 闭源 API 采集时为 None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # ========== 运行时标记（自动生成） ==========
    is_teacher: bool = True         # 标记为 Teacher 轨迹
    has_log_prob: bool = False      # ⭐ 是否有 log_prob（根据 log_probs 字段自动判断）
```

### 5.3 不同采集来源的处理

| 采集来源 | log_probs 字段 | has_log_prob | use_log_prob 推荐值 | Ratio 计算 |
|----------|---------------|--------------|-------------------|-----------|
| GPT-4 / Claude | 无 | False | False | `ratio = π_current` (LUFFY) |
| OpenAI API + logprobs | 有 | True | True | `ratio = π_current / π_old` (ExGRPO) |
| Qwen / Llama / DeepSeek | 有 | True | True | `ratio = π_current / π_old` (ExGRPO) |
| vLLM serving | 有 | True | True | `ratio = π_current / π_old` (ExGRPO) |

---

## 配置项设计

### 6.1 YAML 配置示例

**⭐ 方案 C 比例分配说明**：

```
experience_replay.exp_ratio 定义总 off-policy 比例
teacher_experience.exp_ratio 从总 off-policy 中分走一部分给 Teacher
剩余的 off-policy 比例给 self-generated

示例：exp_ratio=0.5, teacher.exp_ratio=0.2
  → self_exp = 0.3 (30%), teacher_exp = 0.2 (20%), on_policy = 0.5 (50%)

好处：
1. 向后兼容：添加 teacher_experience 不需要修改原 exp_ratio
2. on-policy 比例保持稳定（始终为 1 - exp_ratio）
```

```yaml
exp_manager:
  # ===== Self-Generated Experience Replay (ExGRPO) =====
  experience_replay:
    enable: true
    exp_ratio: 0.5                    # ⭐ 总 off-policy 比例（包含 self + teacher）
    replay_start_ratio: 0.35          # 训练进度达到 35% 时开始 replay
    offpolicy_trajectories_per_task: 1
    experience_lbound: 0
    experience_rbound: 8
    exp_select_mode: "argmin"
    exp_is_correct: true
    max_trajectories_per_task: 5
    use_current_policy_entropy: true
  
  # ===== Teacher Experience Replay (LUFFY) =====
  teacher_experience:
    enable: true                      # 是否启用 Teacher Experience
    data_path: "data/teacher_trajectories/alfworld_gpt4.jsonl"  # Teacher 轨迹文件路径
    exp_ratio: 0.2                    # ⭐ Teacher 从总 off-policy 中占比
                                      # 实际分配：self=30%, teacher=20%, on_policy=50%
    max_trajectories_per_task: 3      # 每个 task 最多使用的 Teacher 轨迹数
    select_mode: "entropy"            # ⭐ 轨迹选择模式
                                      # - "random": 随机选择（默认）
                                      # - "first": 选择前 N 个
                                      # - "all": 返回所有轨迹
                                      # - "entropy": 按 entropy 从低到高排序选择
                                      #   * 对于有 log_prob 的轨迹：直接计算 entropy
                                      #   * 对于没有 log_prob 的轨迹：使用当前 policy 计算 entropy
                                      #   * 两部分合并后按 entropy 从低到高排序
                                      #   推荐：与 self-generated experience 的 exp_select_mode="argmin" 一致
    
    # ⭐ Log Prob 配置（关键选项）
    use_log_prob: false               # 是否使用 log_prob 进行重要性采样
                                      # - true: 使用 ExGRPO 形式 ratio = π_current / π_old
                                      #         适用于：开源模型采集的轨迹（有 log_prob）
                                      # - false: 使用 LUFFY 形式 ratio = π_current（分母=1）
                                      #          适用于：闭源 API 采集的轨迹（无 log_prob）
    
    # Policy Shaping 配置（当 use_log_prob=false 时使用）
    policy_shaping:
      enable: true                    # 是否启用 policy shaping
      mode: "p_div_p_beta"            # Policy shaping 模式
                                      # - "p_div_p_beta": f(x) = x / (x + β)，LUFFY 推荐
                                      # - "sqrt": f(x) = sqrt(x)
                                      # - "no_shaping": 不使用 shaping
      beta: 0.1                       # β 参数（mode="p_div_p_beta" 时使用）
    
    # Loss 计算配置
    loss:
      use_clip: false                 # Teacher 轨迹是否使用 clipping
      clip_upper_bound: 1.0           # 如果使用 clipping，上界
  
  # ===== Teacher Trajectory Collector（在线采集） =====
  
  # ===== Teacher 轨迹选择模式详解 =====
  # 
  # select_mode: "entropy" 模式说明：
  # 
  # 1. 对于有 log_prob 的轨迹（如 vLLM 采集的轨迹）：
  #    - 直接使用轨迹中的 old_log_probs 计算 entropy
  #    - entropy = -sum(p * log(p))，其中 p = exp(log_prob)
  #    - 按 entropy 从低到高排序（低 entropy = 高确定性 = 高质量）
  # 
  # 2. 对于没有 log_prob 的轨迹（如 GPT-4 API 采集的轨迹）：
  #    - 在 trainer 中使用当前 policy 计算 log_prob
  #    - 然后计算 entropy 并排序
  #    - 与 self-generated experience 的 exp_select_mode="argmin" 行为一致
  # 
  # 3. 最终选择：
  #    - 两部分轨迹合并后按 entropy 从低到高排序
  #    - 选择前 num_trajectories_per_task 个
  # 
  # 推荐配置：
  #   - 如果 Teacher 轨迹有 log_prob：select_mode="entropy"（推荐）
  #   - 如果 Teacher 轨迹无 log_prob：select_mode="entropy" 或 "random"
  #   - 与 self-generated 的 exp_select_mode="argmin" 保持一致
  # =====
  teacher_collector:
    enable: false                     # 是否启用在线采集（通常预采集更高效）
    n_trajectories_per_task: 1        # 每个 task 采集的轨迹数
    filter_success_only: true         # 只保留成功的轨迹
    output_path: "data/teacher_trajectories/output.jsonl"  # 保存路径
    
    # ===== Teacher LLM 后端配置 =====
    # 支持两种后端：OpenAI-compatible API 或 vLLM 本地模型
    
    teacher_llm:
      type: "openai"                  # ⭐ 后端类型："openai" 或 "vllm"
      
      # --- OpenAI-compatible API 配置（type="openai" 时使用）---
      model_name: "gpt-4"
      api_base: "https://api.openai.com/v1"
      api_key: "${OPENAI_API_KEY}"
      temperature: 0.0
      max_tokens: 4096
      collect_log_prob: false         # 是否采集 log_prob（OpenAI API 可选支持）
      
      # --- vLLM 本地模型配置（type="vllm" 时使用）---
      # model_path: "/path/to/qwen-72b"
      # tensor_parallel_size: 4
      # gpu_memory_utilization: 0.85
      # trust_remote_code: true
      # collect_log_prob: true        # vLLM 默认支持 log_prob

actor_rollout_ref:
  actor:
    # Loss 计算配置
    off_policy_shaping_mode: "exgrpo_policy_shaping"  # Self-generated: ExGRPO 方式
    off_policy_shaping_beta: 0.1
    
    # Teacher 轨迹特殊配置（当 use_log_prob=false 时生效）
    teacher_policy_shaping_mode: "p_div_p_beta"       # Teacher: LUFFY 方式
    teacher_policy_shaping_beta: 0.1
```

### 6.2 配置场景示例

#### 场景 1：使用 GPT-4 采集的轨迹（无 log_prob）

```yaml
exp_manager:
  teacher_experience:
    enable: true
    data_path: "data/teacher_trajectories/alfworld_gpt4.jsonl"
    use_log_prob: false               # ⭐ 无 log_prob，使用 LUFFY 形式
    policy_shaping:
      enable: true
      mode: "p_div_p_beta"
      beta: 0.1
```

#### 场景 2：使用 Qwen-72B 采集的轨迹（有 log_prob）

```yaml
exp_manager:
  teacher_experience:
    enable: true
    data_path: "data/teacher_trajectories/alfworld_qwen72b.jsonl"
    use_log_prob: true                # ⭐ 有 log_prob，使用 ExGRPO 形式
    policy_shaping:
      enable: false                   # 有 log_prob 时通常不需要 shaping
```

#### 场景 3：同时使用 Self-generated + Teacher（混合模式）

```yaml
exp_manager:
  experience_replay:
    enable: true
    exp_ratio: 0.3                    # 30% self-generated
  
  teacher_experience:
    enable: true
    exp_ratio: 0.2                    # 20% teacher
    use_log_prob: false               # Teacher 来自 GPT-4，无 log_prob
    
# 最终 batch 构成：
# - 30% self-generated off-policy（使用 ExGRPO ratio）
# - 20% teacher off-policy（使用 LUFFY ratio）
# - 50% on-policy（使用标准 ratio ≈ 1.0）
```

#### 场景 4：使用 vLLM 本地模型采集 Teacher 轨迹

```yaml
exp_manager:
  teacher_collector:
    enable: true
    n_trajectories_per_task: 2
    filter_success_only: true
    output_path: "data/teacher_trajectories/alfworld_qwen72b.jsonl"
    
    teacher_llm:
      type: "vllm"                    # ⭐ 使用 vLLM 后端
      model_path: "/data/models/Qwen2.5-72B-Instruct"
      tensor_parallel_size: 4         # 4卡张量并行
      gpu_memory_utilization: 0.85
      trust_remote_code: true
      temperature: 0.0
      max_tokens: 4096
      collect_log_prob: true          # ⭐ vLLM 采集 log_prob

  teacher_experience:
    enable: true
    data_path: "data/teacher_trajectories/alfworld_qwen72b.jsonl"
    use_log_prob: true                # ⭐ 使用采集的 log_prob
```

#### 场景 5：使用 DashScope/DeepSeek API 作为 Teacher

```yaml
exp_manager:
  teacher_collector:
    enable: true
    n_trajectories_per_task: 1
    filter_success_only: true
    output_path: "data/teacher_trajectories/alfworld_deepseek.jsonl"
    
    teacher_llm:
      type: "openai"                  # ⭐ OpenAI-compatible API
      model_name: "deepseek-chat"
      api_base: "https://api.deepseek.com/v1"  # DeepSeek API
      api_key: "${DEEPSEEK_API_KEY}"
      temperature: 0.0
      max_tokens: 4096
      collect_log_prob: false         # DeepSeek API 不支持 logprobs
```

#### 场景 6：使用 vLLM OpenAI-Compatible Server

```yaml
# 假设已经启动了 vLLM server: 
# python -m vllm.entrypoints.openai.api_server \
#   --model /path/to/model --port 8000

exp_manager:
  teacher_collector:
    enable: true
    
    teacher_llm:
      type: "openai"                  # ⭐ 使用 OpenAI client 访问 vLLM server
      model_name: "Qwen/Qwen2.5-72B-Instruct"
      api_base: "http://localhost:8000/v1"   # vLLM server 地址
      api_key: "EMPTY"                # vLLM server 通常不需要 key
      collect_log_prob: true          # vLLM server 支持 logprobs
```

---

## 实现计划

### Phase 1: 基础设施（1-2 天）

1. **扩展 ExperienceManager**
   - 添加 `teacher_task2trajectories` 存储
   - 实现 `load_teacher_trajectories()` 方法
   - 实现 `get_teacher_trajectories()`、`get_valid_teacher_task_ids()` 等接口
   - 保持与现有 self-generated experience 的兼容性

2. **扩展配置项**
   - 添加 `teacher_experience` 配置节
   - 更新配置解析逻辑（在 `ExperienceManager.__init__` 中）

### Phase 2: 数据采集框架（2-3 天）

3. **实现 Teacher LLM 抽象层**
   - 创建 `agentevolver/module/teacher/` 目录
   - 创建 `base_teacher_llm.py`：定义 `BaseTeacherLLM` 抽象基类
   - 创建 `openai_teacher_llm.py`：实现 `OpenAITeacherLLM`（OpenAI-compatible API）
   - 创建 `vllm_teacher_llm.py`：实现 `VLLMTeacherLLM`（本地 vLLM 模型）
   - 实现 `create_teacher_llm()` 工厂函数

4. **实现 TeacherAgentFlow**
   - 创建 `agentevolver/module/teacher/teacher_agent_flow.py`
   - 包装 `BaseTeacherLLM` 以兼容 `BaseAgentFlow` 接口
   - 确保与 `EnvWorker` 无缝对接
   - 统一处理两种后端的输出格式

5. **实现 TeacherTrajectoryCollector**
   - 创建 `agentevolver/module/exp_manager/teacher_collector.py`
   - 复用 `EnvManager.rollout()` 流程
   - 传入 `TeacherAgentFlow` 替代原有 agent_flow
   - 根据后端类型决定是否保存 log_prob
   - 保存轨迹到 JSONL 格式

6. **创建采集脚本**
   - 创建 `scripts/collect_teacher_trajectories.py`（主采集脚本）
   - 创建 `scripts/generate_task_ids.py`（生成 task_ids 文件）
   - 创建 `scripts/validate_teacher_trajectories.py`（验证输出格式）
   - 支持 `--task_file` 指定采集哪些 tasks
   - 支持 `--backend openai/vllm` 选择后端
   - 支持 `--collect_log_prob auto/true/false`（vLLM 默认 true）
   - 支持 `--resume` 断点续采
   - 支持分布式采集（`--task_start/--task_end`）

### Phase 3: 训练集成（2-3 天）

7. **扩展 ExperienceMixCollateFn**
   - 修改 `agentevolver/module/exp_manager/experience_collate.py`
   - 支持三种数据类型混合
   - 使用统一的 `exp_manager` 获取轨迹

8. **修改 Loss 计算**
   - 修改 `agentevolver/module/exp_manager/het_core_algos.py`
   - 添加 `teacher_mask` 参数
   - 实现 LUFFY 风格的 policy shaping（`ratio = π / (π + β)`）
   - Teacher 轨迹不使用 clipping

9. **修改 Trainer**
   - 修改 `agentevolver/module/trainer/ae_ray_trainer.py`
   - 在数据准备阶段获取 Teacher 轨迹
   - 生成 `teacher_mask` 并传递给 loss 计算
   - 处理 Teacher 轨迹的 dataproto 转换

10. **扩展 env_manager**
    - 修改 `convert_offpolicy_to_cmt()` 识别 Teacher 轨迹
    - 在 `samples_to_dataproto()` 中生成 `teacher_mask`

### Phase 4: 测试与验证（1-2 天）

11. **单元测试**
    - `test_exp_manager_teacher.py`：ExperienceManager Teacher 功能
    - `test_teacher_llm_backends.py`：测试 OpenAI 和 vLLM 两种后端
    - `test_teacher_collector.py`：TeacherTrajectoryCollector
    - `test_loss_computation_teacher.py`：LUFFY loss 计算
    - `test_experience_mix_three_types.py`：三种数据类型混合

12. **集成测试**
    - 预采集 Teacher 轨迹（使用 GPT-4 或本地 vLLM 模型）
    - 运行混合训练（Self-generated + Teacher + On-policy）
    - 验证 Multi-turn 场景的 mask 正确性
    - 对比 ExGRPO-only vs LUFFY+ExGRPO 训练效果
    - 对比有/无 log_prob 的 Teacher 轨迹训练效果

---

## 采集脚本详细设计

本节详细规划 Teacher Trajectory 采集脚本的设计和使用方法，以 ALFWorld 环境为例。

### 8.1 采集脚本设计原则

1. **Task 索引文件**：允许用户提供 `task_ids.txt` 指定从哪些 task 采集
2. **双后端支持**：同时支持 OpenAI-compatible API 和 vLLM 开源模型
3. **默认收集 log_prob**：vLLM 后端默认开启，OpenAI 后端可选
4. **断点续采**：支持从上次中断的位置继续采集
5. **并行采集**：支持多进程/异步并行加速采集

### 8.2 采集脚本实现

```python
#!/usr/bin/env python3
# scripts/collect_teacher_trajectories.py
"""
Teacher Trajectory 采集脚本

使用示例：
  # 使用 vLLM 本地模型采集
  python scripts/collect_teacher_trajectories.py \
      --env alfworld \
      --backend vllm \
      --model_path /data/models/Qwen2.5-72B-Instruct \
      --task_file data/alfworld/task_ids.txt \
      --output data/teacher_trajectories/alfworld_qwen72b.jsonl \
      --n_per_task 2

  # 使用 OpenAI API 采集
  python scripts/collect_teacher_trajectories.py \
      --env alfworld \
      --backend openai \
      --model_name gpt-4 \
      --api_base https://api.openai.com/v1 \
      --task_file data/alfworld/task_ids.txt \
      --output data/teacher_trajectories/alfworld_gpt4.jsonl \
      --collect_log_prob false
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from tqdm import tqdm

from loguru import logger
from omegaconf import OmegaConf, DictConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect teacher trajectories for LUFFY training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # ===== 环境配置 =====
    parser.add_argument(
        "--env", type=str, required=True,
        choices=["alfworld", "webshop", "sciworld"],
        help="Environment name"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to base config file (optional, for environment setup)"
    )
    
    # ===== Task 索引配置 =====
    parser.add_argument(
        "--task_file", type=str, required=True,
        help="Path to task IDs file (one task_id per line)"
    )
    parser.add_argument(
        "--task_start", type=int, default=0,
        help="Start index in task file (for distributed collection)"
    )
    parser.add_argument(
        "--task_end", type=int, default=None,
        help="End index in task file (exclusive, None for all)"
    )
    
    # ===== Teacher LLM 后端配置 =====
    parser.add_argument(
        "--backend", type=str, required=True,
        choices=["openai", "vllm"],
        help="Teacher LLM backend: 'openai' for API, 'vllm' for local model"
    )
    
    # OpenAI 后端参数
    parser.add_argument(
        "--model_name", type=str, default="gpt-4",
        help="[OpenAI] Model name (e.g., gpt-4, gpt-4-turbo, claude-3-opus)"
    )
    parser.add_argument(
        "--api_base", type=str, default="https://api.openai.com/v1",
        help="[OpenAI] API base URL"
    )
    parser.add_argument(
        "--api_key", type=str, default=None,
        help="[OpenAI] API key (default: use OPENAI_API_KEY env var)"
    )
    
    # vLLM 后端参数
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="[vLLM] Path to local model"
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1,
        help="[vLLM] Tensor parallel size for multi-GPU inference"
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.85,
        help="[vLLM] GPU memory utilization ratio"
    )
    
    # ===== 通用生成参数 =====
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=4096,
        help="Maximum tokens per response"
    )
    parser.add_argument(
        "--collect_log_prob", type=str, default="auto",
        choices=["true", "false", "auto"],
        help="Collect log probabilities. 'auto': true for vllm, false for openai"
    )
    
    # ===== 采集配置 =====
    parser.add_argument(
        "--n_per_task", type=int, default=1,
        help="Number of trajectories to collect per task"
    )
    parser.add_argument(
        "--filter_success", action="store_true", default=True,
        help="Only keep successful trajectories"
    )
    parser.add_argument(
        "--no_filter_success", action="store_false", dest="filter_success",
        help="Keep all trajectories including failed ones"
    )
    parser.add_argument(
        "--max_retries", type=int, default=3,
        help="Maximum retries per task on failure"
    )
    
    # ===== 输出配置 =====
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing output file (skip already collected tasks)"
    )
    parser.add_argument(
        "--save_every", type=int, default=10,
        help="Save checkpoint every N tasks"
    )
    
    # ===== 并行配置 =====
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help="Number of parallel workers (for OpenAI API)"
    )
    
    return parser.parse_args()


def load_task_ids(task_file: str, start: int = 0, end: Optional[int] = None) -> List[str]:
    """
    从文件加载 task IDs
    
    文件格式：每行一个 task_id
    示例：
        pick_cool_apple_0
        pick_cool_apple_1
        put_book_on_table_0
        ...
    """
    with open(task_file, 'r') as f:
        task_ids = [line.strip() for line in f if line.strip()]
    
    # 支持切片（用于分布式采集）
    if end is None:
        end = len(task_ids)
    task_ids = task_ids[start:end]
    
    logger.info(f"Loaded {len(task_ids)} task IDs from {task_file} "
               f"(range: [{start}, {end}))")
    return task_ids


def load_completed_task_ids(output_file: str) -> set:
    """
    从已有输出文件中加载已完成的 task IDs（用于断点续采）
    """
    completed = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        traj = json.loads(line)
                        completed.add(traj.get("task_id", ""))
                    except json.JSONDecodeError:
                        continue
        logger.info(f"Found {len(completed)} already completed tasks in {output_file}")
    return completed


def create_teacher_llm(args):
    """
    根据命令行参数创建 Teacher LLM 实例
    """
    from agentevolver.module.teacher.vllm_teacher_llm import create_teacher_llm as factory
    
    # 确定是否收集 log_prob
    if args.collect_log_prob == "auto":
        collect_log_prob = (args.backend == "vllm")  # vLLM 默认收集
    else:
        collect_log_prob = (args.collect_log_prob == "true")
    
    if args.backend == "openai":
        config = {
            "type": "openai",
            "model_name": args.model_name,
            "api_base": args.api_base,
            "api_key": args.api_key or os.environ.get("OPENAI_API_KEY"),
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "collect_log_prob": collect_log_prob,
        }
    elif args.backend == "vllm":
        if not args.model_path:
            raise ValueError("--model_path is required for vllm backend")
        config = {
            "type": "vllm",
            "model_path": args.model_path,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "collect_log_prob": collect_log_prob,
        }
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    
    logger.info(f"Creating Teacher LLM with config: {config}")
    return factory(config)


def setup_environment(env_name: str, config_path: Optional[str] = None) -> DictConfig:
    """
    设置环境配置
    """
    if config_path:
        config = OmegaConf.load(config_path)
    else:
        # 使用默认配置
        default_configs = {
            "alfworld": "config/alfworld_grpo_3b_exp_replay.yaml",
            "webshop": "config/webshop_grpo_3b.yaml",
            "sciworld": "config/sciworld_grpo_3b.yaml",
        }
        config_path = default_configs.get(env_name)
        if config_path and os.path.exists(config_path):
            config = OmegaConf.load(config_path)
        else:
            # 创建最小配置
            config = OmegaConf.create({
                "env": {"name": env_name},
                "actor_rollout_ref": {
                    "rollout": {
                        "max_model_len": 8192,
                        "max_env_len": 4096,
                        "multi_turn": {"max_steps": 30},
                    }
                }
            })
    
    return config


def create_env_manager(config: DictConfig, tokenizer):
    """
    创建环境管理器
    """
    from agentevolver.module.env_manager.env_manager import ParallelEnvManager
    # 这里需要根据实际实现进行调整
    # 简化版本：直接创建环境
    return None  # Placeholder - 实际实现需要正确初始化


def collect_single_task(
    task_id: str,
    teacher_llm,
    env_config: DictConfig,
    n_trajectories: int = 1,
    max_retries: int = 3,
    filter_success: bool = True,
) -> List[Dict[str, Any]]:
    """
    采集单个 task 的 Teacher 轨迹
    
    Returns:
        轨迹列表（字典格式，便于 JSON 序列化）
    """
    from agentevolver.module.teacher.teacher_agent_flow import TeacherAgentFlow
    from agentevolver.module.env_manager.env_worker import EnvWorker
    
    trajectories = []
    
    for rollout_idx in range(n_trajectories):
        for retry in range(max_retries):
            try:
                # 创建 env worker
                worker = EnvWorker(
                    task_id=task_id,
                    config=env_config,
                )
                
                # 创建 teacher agent flow
                agent_flow = TeacherAgentFlow(
                    teacher_llm=teacher_llm,
                    tokenizer=worker.tokenizer,
                    config=env_config,
                )
                
                # 执行 rollout
                trajectory = worker.execute(
                    data_id=task_id,
                    rollout_id=f"{task_id}_teacher_{rollout_idx}",
                    agent_flow=agent_flow,
                )
                
                # 检查是否成功
                success = (trajectory.reward and 
                          trajectory.reward.outcome == 1.0)
                
                if filter_success and not success:
                    logger.debug(f"Task {task_id} rollout {rollout_idx} failed, "
                               f"retry {retry+1}/{max_retries}")
                    continue
                
                # 转换为字典格式
                traj_dict = {
                    "task_id": task_id,
                    "data_id": trajectory.data_id,
                    "rollout_id": trajectory.rollout_id,
                    "messages": trajectory.steps,  # Multi-turn 对话历史
                    "reward": trajectory.reward.outcome if trajectory.reward else 0.0,
                    "success": success,
                    "metadata": {
                        "is_teacher": True,
                        "teacher_model": getattr(teacher_llm, 'model_name', 
                                                getattr(teacher_llm, 'model_path', 'unknown')),
                        "has_log_prob": teacher_llm.supports_log_prob,
                        "collected_at": datetime.now().isoformat(),
                    }
                }
                
                # 如果有 log_prob，添加到 metadata
                if hasattr(trajectory, 'log_probs') and trajectory.log_probs:
                    traj_dict["log_probs"] = trajectory.log_probs
                    traj_dict["metadata"]["old_log_probs"] = trajectory.log_probs
                
                trajectories.append(traj_dict)
                break  # 成功，跳出重试循环
                
            except Exception as e:
                logger.warning(f"Task {task_id} rollout {rollout_idx} error: {e}, "
                             f"retry {retry+1}/{max_retries}")
                if retry == max_retries - 1:
                    logger.error(f"Task {task_id} rollout {rollout_idx} failed after "
                               f"{max_retries} retries")
    
    return trajectories


def save_trajectories_jsonl(trajectories: List[Dict], output_file: str, mode: str = 'a'):
    """
    保存轨迹到 JSONL 文件
    """
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, mode) as f:
        for traj in trajectories:
            f.write(json.dumps(traj, ensure_ascii=False) + '\n')


def main():
    args = parse_args()
    
    # 设置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    log_file = args.output.replace('.jsonl', '.log')
    logger.add(log_file, level="DEBUG", rotation="100 MB")
    
    logger.info(f"=" * 60)
    logger.info(f"Teacher Trajectory Collection")
    logger.info(f"=" * 60)
    logger.info(f"Backend: {args.backend}")
    logger.info(f"Task file: {args.task_file}")
    logger.info(f"Output: {args.output}")
    
    # 1. 加载 task IDs
    task_ids = load_task_ids(args.task_file, args.task_start, args.task_end)
    
    # 2. 断点续采：加载已完成的 tasks
    if args.resume:
        completed = load_completed_task_ids(args.output)
        task_ids = [tid for tid in task_ids if tid not in completed]
        logger.info(f"Resuming: {len(task_ids)} tasks remaining")
    
    if not task_ids:
        logger.info("No tasks to collect. Exiting.")
        return
    
    # 3. 创建 Teacher LLM
    teacher_llm = create_teacher_llm(args)
    logger.info(f"Teacher LLM supports log_prob: {teacher_llm.supports_log_prob}")
    
    # 4. 设置环境配置
    env_config = setup_environment(args.env, args.config)
    
    # 5. 采集轨迹
    all_trajectories = []
    
    with tqdm(total=len(task_ids), desc="Collecting") as pbar:
        for i, task_id in enumerate(task_ids):
            try:
                trajectories = collect_single_task(
                    task_id=task_id,
                    teacher_llm=teacher_llm,
                    env_config=env_config,
                    n_trajectories=args.n_per_task,
                    max_retries=args.max_retries,
                    filter_success=args.filter_success,
                )
                
                all_trajectories.extend(trajectories)
                
                # 定期保存
                if (i + 1) % args.save_every == 0:
                    save_trajectories_jsonl(
                        all_trajectories, 
                        args.output, 
                        mode='a' if args.resume else 'w' if i < args.save_every else 'a'
                    )
                    all_trajectories = []
                    logger.info(f"Checkpoint saved at task {i+1}/{len(task_ids)}")
                
            except Exception as e:
                logger.error(f"Error collecting task {task_id}: {e}")
            
            pbar.update(1)
            pbar.set_postfix({"success": len(all_trajectories)})
    
    # 保存剩余轨迹
    if all_trajectories:
        save_trajectories_jsonl(all_trajectories, args.output, mode='a')
    
    # 6. 统计结果
    total_collected = sum(1 for _ in open(args.output))
    logger.info(f"=" * 60)
    logger.info(f"Collection completed!")
    logger.info(f"Total trajectories: {total_collected}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"=" * 60)


if __name__ == "__main__":
    main()
```

### 8.3 Task 索引文件格式

Task 索引文件是一个简单的文本文件，每行一个 `task_id`：

```text
# data/alfworld/task_ids.txt
# ALFWorld 任务 ID 列表
# 格式：每行一个 task_id

pick_cool_apple_from_microwave_0
pick_cool_apple_from_microwave_1
pick_cool_apple_from_microwave_2
put_book_on_table_in_bedroom_0
put_book_on_table_in_bedroom_1
heat_egg_in_microwave_0
heat_egg_in_microwave_1
clean_mug_and_put_in_cabinet_0
examine_book_with_lamp_0
...
```

**从环境服务生成 task_ids 文件**：

脚本 `scripts/generate_task_ids.py` 与 `TaskManager.load_tasks_from_environment()` 行为一致，
从环境服务获取 task_ids：

```bash
# 基本用法：从环境服务获取 train split 的所有 task_ids
python scripts/generate_task_ids.py \
    --env_url http://localhost:8000 \
    --env_type alfworld \
    --split train \
    --output data/alfworld/train_task_ids.txt

# 高级用法：限制数量并采样
python scripts/generate_task_ids.py \
    --env_url http://localhost:8000 \
    --env_type alfworld \
    --split train \
    --max_tasks 500 \
    --sample_ratio 0.5 \
    --seed 42 \
    --output data/alfworld/train_task_ids_sampled.txt

# 查看可用的 task_ids（不保存）
python scripts/generate_task_ids.py \
    --env_url http://localhost:8000 \
    --env_type alfworld \
    --split val \
    --dry_run
```

**核心参数说明**：
- `--env_url`: 环境服务 URL（默认 `http://localhost:8000`）
- `--env_type`: 环境类型（`alfworld`, `webshop`, `sciworld`, `appworld`）
- `--split`: 数据集划分（`train`, `val`, `dev`, `test`）
- `--shuffle`: 是否打乱顺序（默认 True，与 TaskManager 一致）
- `--max_tasks`: 最大 task 数量限制
- `--sample_ratio`: 采样比例（0.0-1.0）
- `--seed`: 随机种子（默认 42）

### 8.4 使用示例

#### 8.4.1 使用 vLLM 本地 Qwen-72B 采集

```bash
# 1. 准备 task_ids 文件（从环境服务获取）
python scripts/generate_task_ids.py \
    --env_url http://localhost:8000 \
    --env_type alfworld \
    --split train \
    --output data/alfworld/train_task_ids.txt

# 2. 使用 vLLM 采集（4 卡并行，默认收集 log_prob）
python scripts/collect_teacher_trajectories.py \
    --env alfworld \
    --backend vllm \
    --model_path /data/models/Qwen2.5-72B-Instruct \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.85 \
    --task_file data/alfworld/train_task_ids.txt \
    --output data/teacher_trajectories/alfworld_qwen72b.jsonl \
    --n_per_task 2 \
    --temperature 0.0 \
    --filter_success

# 3. 查看采集结果
head -1 data/teacher_trajectories/alfworld_qwen72b.jsonl | python -m json.tool
```

#### 8.4.2 使用 GPT-4 API 采集

```bash
# 设置 API Key
export OPENAI_API_KEY="sk-xxx"

# 采集（不收集 log_prob）
python scripts/collect_teacher_trajectories.py \
    --env alfworld \
    --backend openai \
    --model_name gpt-4-turbo \
    --api_base https://api.openai.com/v1 \
    --task_file data/alfworld/train_task_ids.txt \
    --output data/teacher_trajectories/alfworld_gpt4.jsonl \
    --n_per_task 1 \
    --collect_log_prob false \
    --temperature 0.0 \
    --max_retries 5
```

#### 8.4.3 使用 DashScope API 采集

```bash
export DASHSCOPE_API_KEY="sk-xxx"

python scripts/collect_teacher_trajectories.py \
    --env alfworld \
    --backend openai \
    --model_name qwen-max \
    --api_base https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --api_key $DASHSCOPE_API_KEY \
    --task_file data/alfworld/train_task_ids.txt \
    --output data/teacher_trajectories/alfworld_qwen_max.jsonl \
    --n_per_task 1 \
    --collect_log_prob false
```

#### 8.4.4 分布式采集（多节点并行）

```bash
# Node 1: 采集前 500 个 tasks
python scripts/collect_teacher_trajectories.py \
    --env alfworld \
    --backend vllm \
    --model_path /data/models/Qwen2.5-72B-Instruct \
    --task_file data/alfworld/train_task_ids.txt \
    --task_start 0 \
    --task_end 500 \
    --output data/teacher_trajectories/alfworld_qwen72b_part1.jsonl

# Node 2: 采集后 500 个 tasks
python scripts/collect_teacher_trajectories.py \
    --env alfworld \
    --backend vllm \
    --model_path /data/models/Qwen2.5-72B-Instruct \
    --task_file data/alfworld/train_task_ids.txt \
    --task_start 500 \
    --task_end 1000 \
    --output data/teacher_trajectories/alfworld_qwen72b_part2.jsonl

# 合并结果
cat data/teacher_trajectories/alfworld_qwen72b_part*.jsonl > \
    data/teacher_trajectories/alfworld_qwen72b.jsonl
```

#### 8.4.5 断点续采

```bash
# 如果采集中断，使用 --resume 继续
python scripts/collect_teacher_trajectories.py \
    --env alfworld \
    --backend vllm \
    --model_path /data/models/Qwen2.5-72B-Instruct \
    --task_file data/alfworld/train_task_ids.txt \
    --output data/teacher_trajectories/alfworld_qwen72b.jsonl \
    --resume  # 跳过已采集的 tasks
```

### 8.5 采集配置文件（可选）

除了命令行参数，也支持使用 YAML 配置文件：

```yaml
# config/teacher_collector/alfworld_qwen72b.yaml

env: alfworld
backend: vllm

# Task 配置
task_file: data/alfworld/train_task_ids.txt
n_per_task: 2
filter_success: true
max_retries: 3

# vLLM 配置
model_path: /data/models/Qwen2.5-72B-Instruct
tensor_parallel_size: 4
gpu_memory_utilization: 0.85

# 生成配置
temperature: 0.0
max_tokens: 4096
collect_log_prob: true  # vLLM 默认收集

# 输出配置
output: data/teacher_trajectories/alfworld_qwen72b.jsonl
save_every: 10
resume: true
```

```yaml
# config/teacher_collector/alfworld_gpt4.yaml

env: alfworld
backend: openai

# Task 配置
task_file: data/alfworld/train_task_ids.txt
n_per_task: 1
filter_success: true
max_retries: 5

# OpenAI 配置
model_name: gpt-4-turbo
api_base: https://api.openai.com/v1
# api_key: 从环境变量 OPENAI_API_KEY 读取

# 生成配置
temperature: 0.0
max_tokens: 4096
collect_log_prob: false  # GPT-4 通常不收集

# 输出配置
output: data/teacher_trajectories/alfworld_gpt4.jsonl
save_every: 10
```

使用配置文件：

```bash
python scripts/collect_teacher_trajectories.py \
    --config config/teacher_collector/alfworld_qwen72b.yaml
```

### 8.6 输出格式验证

采集完成后，验证输出格式：

```python
# scripts/validate_teacher_trajectories.py
"""
验证采集的 Teacher 轨迹格式

Usage:
    python scripts/validate_teacher_trajectories.py \
        --input data/teacher_trajectories/alfworld_qwen72b.jsonl
"""

import json
import argparse
from collections import Counter


def validate(input_file: str):
    stats = Counter()
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                traj = json.loads(line)
                
                # 必需字段检查
                assert "task_id" in traj, f"Line {line_num}: missing task_id"
                assert "messages" in traj, f"Line {line_num}: missing messages"
                assert "reward" in traj, f"Line {line_num}: missing reward"
                assert "success" in traj, f"Line {line_num}: missing success"
                assert "metadata" in traj, f"Line {line_num}: missing metadata"
                assert traj["metadata"].get("is_teacher") == True, \
                    f"Line {line_num}: is_teacher should be True"
                
                # 统计
                stats["total"] += 1
                if traj["success"]:
                    stats["success"] += 1
                if traj["metadata"].get("has_log_prob"):
                    stats["has_log_prob"] += 1
                
            except json.JSONDecodeError as e:
                print(f"Line {line_num}: Invalid JSON - {e}")
                stats["invalid_json"] += 1
            except AssertionError as e:
                print(f"Validation error: {e}")
                stats["validation_error"] += 1
    
    print(f"\n{'='*40}")
    print(f"Validation Results for {input_file}")
    print(f"{'='*40}")
    print(f"Total trajectories: {stats['total']}")
    print(f"Successful: {stats['success']} ({100*stats['success']/max(stats['total'],1):.1f}%)")
    print(f"Has log_prob: {stats['has_log_prob']}")
    print(f"Invalid JSON: {stats['invalid_json']}")
    print(f"Validation errors: {stats['validation_error']}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()
    validate(args.input)
```

---

## 测试计划

### 9.1 单元测试

```python
# tests/test_teacher_experience_replay.py

def test_teacher_exp_manager_load():
    """测试 TeacherExperienceManager 加载功能"""
    pass

def test_teacher_trajectory_collector():
    """测试 TeacherTrajectoryCollector 采集功能"""
    pass

def test_loss_computation_teacher():
    """测试 Teacher 轨迹的 loss 计算"""
    pass

def test_experience_mix_collate_three_types():
    """测试三种数据类型的混合"""
    pass
```

### 9.2 集成测试

```bash
# 1. 预采集 Teacher 轨迹（使用 vLLM）
python scripts/collect_teacher_trajectories.py \
    --env alfworld \
    --backend vllm \
    --model_path /data/models/Qwen2.5-72B-Instruct \
    --task_file data/alfworld/test_task_ids.txt \
    --output data/teacher_trajectories/alfworld_qwen72b_test.jsonl \
    --n_per_task 1

# 2. 验证轨迹格式
python scripts/validate_teacher_trajectories.py \
    --input data/teacher_trajectories/alfworld_qwen72b_test.jsonl

# 3. 运行混合训练
python -m agentevolver.main \
    --config config/alfworld_grpo_3b_luffy.yaml
```

---

## 附录

### A. 设计决策总结

| 决策点 | 选择 | 原因 |
|--------|------|------|
| Teacher Experience 存储 | 扩展现有 `ExperienceManager` | 便于混合使用 LUFFY + ExGRPO，共享基础设施 |
| Teacher Trajectory 采集 | 复用 `EnvManager.rollout()` | 数据格式一致，Multi-turn 自动支持 |
| LLM 调用 | 通过 `TeacherAgentFlow` 替换 `agent_flow` | 保持接口一致，最小化修改 |
| Teacher LLM 后端 | 支持 OpenAI API + vLLM 双后端 | 兼容闭源 API 和开源本地模型，VERL 默认使用 vLLM |
| Log Prob 处理 | 配置化 `use_log_prob` 选项 | 灵活支持有/无 log_prob 的不同场景 |

### A.1 Teacher LLM 后端选择指南

| 使用场景 | 推荐后端 | 配置 `type` | Log Prob | 说明 |
|---------|---------|------------|---------|------|
| 调用 GPT-4/Claude 等闭源模型 | OpenAI API | `"openai"` | ❌ 通常不可用 | 使用 LUFFY 简化计算 |
| 调用 DashScope/DeepSeek API | OpenAI API | `"openai"` | ⚠️ 取决于 API | 配置 `collect_log_prob` |
| 本地运行 Qwen/Llama/DeepSeek | vLLM | `"vllm"` | ✅ 始终可用 | 推荐采集 log_prob |
| 使用 vLLM Server (OpenAI 兼容) | OpenAI API | `"openai"` | ✅ 可用 | 通过 API 访问 vLLM |
| 需要高吞吐量的批量采集 | vLLM | `"vllm"` | ✅ 始终可用 | vLLM 批处理效率高 |
| GPU 资源有限 | OpenAI API | `"openai"` | 取决于 API | 远程调用无需本地 GPU |

### B. 与现有实现的兼容性

| 组件 | 是否需要修改 | 修改程度 | 修改内容 |
|------|-------------|---------|----------|
| `ExperienceManager` | ⭐ 中等 | 扩展 | 添加 `teacher_task2trajectories`、`load_teacher_trajectories()` 等 |
| `het_core_algos.py` | 中等 | 扩展 | 添加 `teacher_mask` 和 LUFFY policy shaping |
| `ExperienceMixCollateFn` | 中等 | 扩展 | 支持三种数据类型，使用统一的 `exp_manager` |
| `ae_ray_trainer.py` | 中等 | 扩展 | 处理 `teacher_mask`，获取 Teacher 轨迹 |
| `env_manager.py` | 轻微 | 扩展 | `convert_offpolicy_to_cmt` 识别 Teacher 轨迹 |
| ⭐ 新增 `teacher_collector.py` | 新文件 | - | Teacher 轨迹采集器 |
| ⭐ 新增 `teacher_agent_flow.py` | 新文件 | - | Teacher LLM 调用封装 |

### C. Multi-turn 一致性检查清单

复用 Teacher Experience 时，确保以下 Multi-turn 特性保持一致：

- [ ] **exp_mask 生成**：只对 LLM 响应位置（`loss_mask=1`）设置 `exp_mask=1`
- [ ] **response_mask 计算**：基于 `loss_mask`，只包含 LLM tokens
- [ ] **teacher_mask 新增**：标记 Teacher 轨迹位置，用于 loss 计算时区分
- [ ] **data_id 分配**：同一 task 的 on-policy 和 off-policy 共享 data_id（GRPO 分组）
- [ ] **author 设置**：LLM 消息保持 `author="llm"`，确保 `loss_mask=1`

#### ⭐ Log Prob 处理检查

- [ ] **has_log_prob 标记**：加载 Teacher 轨迹时正确设置 `metadata["has_log_prob"]`
- [ ] **use_log_prob 配置**：根据采集来源设置正确的 `use_log_prob` 值
- [ ] **old_log_probs 替换**：
  - `use_log_prob=True`：使用 Teacher 轨迹中的 `old_log_probs` 替换
  - `use_log_prob=False`：不替换，loss 函数使用 LUFFY 简化公式
- [ ] **policy_shaping 配置**：`use_log_prob=False` 时推荐启用 `policy_shaping`

### D. 参考资料

1. **LUFFY 论文**: Learning to Reason under Off-Policy Guidance
   - Project: https://github.com/ElliottYan/LUFFY
   - 关键点：`off_ratio = exp(log_prob)` 简化计算，`f(x) = x/(x+β)` policy shaping

2. **ExGRPO 论文**: Learning to Reason from Experience
   - Code: https://github.com/ElliottYan/LUFFY/tree/main/ExGRPO
   - 关键点：基于 entropy 选择轨迹，experience replay buffer 管理

3. **现有实现文档**: 
   - `docs/guidelines/experience_replay_manual.md`（ExGRPO 实现详解）
   - `docs/guidelines/experience_replay_design.md`（设计文档）

### E. 文件新增/修改一览

> **✅ 已实现文件** - 以下所有文件已经创建并完成实现

```
agentevolver/
├── module/
│   ├── exp_manager/
│   │   ├── exp_manager.py           # ✅ 修改：添加 teacher_task2trajectories、load_teacher_trajectories() 等
│   │   ├── experience_collate.py    # ✅ 修改：新增 TeacherExperienceMixCollateFn 支持三种数据类型
│   │   ├── het_core_algos.py        # ✅ 修改：新增 het_compute_teacher_aware_loss() 支持 teacher_mask
│   │   └── teacher_collector.py     # ✅ 新增：TeacherTrajectoryCollector 轨迹采集器
│   │
│   ├── teacher/                     # ✅ 新增：Teacher LLM 后端模块
│   │   ├── __init__.py              # ✅ 模块导出
│   │   ├── base_teacher_llm.py      # ✅ 抽象基类 BaseTeacherLLM
│   │   ├── openai_teacher_llm.py    # ✅ OpenAI-compatible API 后端
│   │   ├── vllm_teacher_llm.py      # ✅ vLLM 本地推理后端 + create_teacher_llm() 工厂函数
│   │   └── teacher_agent_flow.py    # ✅ TeacherAgentFlow 适配层
│   │
│   └── trainer/
│       └── ae_ray_trainer.py        # ✅ 修改：集成 TeacherExperienceMixCollateFn
│
└── ...

config/
├── examples/
│   └── teacher_experience_config.yaml  # ✅ 新增：Teacher Experience 配置示例
└── ...

scripts/
├── collect_teacher_trajectories.py  # ✅ 新增：Teacher 轨迹采集主脚本
├── generate_task_ids.py             # ✅ 新增：生成 task_ids 文件
└── validate_teacher_trajectories.py # ✅ 新增：验证轨迹格式

data/
├── alfworld/
│   └── train_task_ids.txt           # 用户提供：Task ID 索引文件
└── teacher_trajectories/            # 用户创建：采集的轨迹输出目录
    ├── alfworld_qwen72b.jsonl       # 示例：Qwen-72B 采集的轨迹
    └── alfworld_gpt4.jsonl          # 示例：GPT-4 采集的轨迹
```

### F. vLLM 与 OpenAI API 代码示例

#### F.1 使用 vLLM 本地模型采集

```python
from agentevolver.module.teacher.vllm_teacher_llm import VLLMTeacherLLM

# 初始化 vLLM Teacher LLM
teacher_llm = VLLMTeacherLLM(
    model_path="/data/models/Qwen2.5-72B-Instruct",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.85,
    temperature=0.0,
    max_tokens=4096,
    collect_log_prob=True,  # 采集 log_prob
)

# 调用生成
messages = [{"role": "user", "content": "What should I do next?"}]
response_text, metadata = teacher_llm(messages)

print(f"Response: {response_text}")
print(f"Log probs: {metadata.get('log_probs', 'N/A')}")
```

#### F.2 使用 OpenAI-compatible API 采集

```python
from agentevolver.module.teacher.openai_teacher_llm import OpenAITeacherLLM

# 初始化 OpenAI Teacher LLM（支持 GPT-4、Claude、DashScope 等）
teacher_llm = OpenAITeacherLLM(
    model_name="gpt-4",
    api_base="https://api.openai.com/v1",
    api_key="sk-xxx",
    temperature=0.0,
    max_tokens=4096,
    collect_log_prob=False,  # GPT-4 不建议采集（费用高）
)

# 调用生成
messages = [{"role": "user", "content": "What should I do next?"}]
response_text, metadata = teacher_llm(messages)

print(f"Response: {response_text}")
```

#### F.3 使用工厂函数创建

```python
from agentevolver.module.teacher.vllm_teacher_llm import create_teacher_llm

# 从配置创建（自动选择后端）
config = {
    "type": "vllm",
    "model_path": "/data/models/Qwen2.5-72B-Instruct",
    "tensor_parallel_size": 4,
    "collect_log_prob": True,
}
teacher_llm = create_teacher_llm(config)

# 或者
config = {
    "type": "openai",
    "model_name": "gpt-4",
    "api_base": "https://api.openai.com/v1",
}
teacher_llm = create_teacher_llm(config)
```

