import json
import os
import random
import re
from typing import Optional, Sequence
import multiprocessing
from functools import partial
import time

import numpy as np
from transformers import GenerationConfig

from . import Agent, APIAgent, BaseTask
from .types import (
    ActionFormat,
    ActionWithTought,
    ConversationMessage,
    EvaluationOutput,
    ExperienceOutput,
    APIExperienceOutput,
)

# 延迟导入 MEMORY_CLS_MAP 以避免循环导入
# 在需要使用的地方再导入

INVOKING_FUNCTION_PROMPT = """

If you want to invoke a provided function or tool, please reply in the following *JSON* format:
```json
{
    "thought": "I think ...",
    "function_name": "function_name",
    "arguments": <valid json object of args>
}
```
Only reply the *JSON* object, no other text should be present.
"""

WRITE_CODE_PROMPT = """

If you want to call these functions, please reply the python code block:
```python
# Write you thought in the code comment before you call any function.
<write valid python code here.>
```
Only reply the code block with "```python" and "```",  no other text should be present.
"""

def format_function_call_prompt(function_description: Sequence) -> str:
    prompt = "You have the following functions available:\n\n"

    tool_descs = [{"type": "function", "function": f} for f in function_description]
    prompt += "\n".join(
        [json.dumps(f, ensure_ascii=False, indent=2) for f in tool_descs]
    )
    prompt += INVOKING_FUNCTION_PROMPT

    return prompt


def generate_function_signatures(function_descriptions: Sequence):
    function_strings = []
    for func in function_descriptions:
        name = func["name"]
        description = func["description"]
        params = func["parameters"]["properties"]
        required_params = func["parameters"].get("required", [])

        # Generate function signature
        signature_params = ", ".join(
            [
                f"{param}='{param}'" if param not in required_params else param
                for param in params
            ]
        )
        function_signature = f"def {name}({signature_params}):"

        # Generate docstring
        docstring = f'    """\n    {description}\n\n'
        for param, details in params.items():
            docstring += (
                f"    :param {param} ({details['type']}): {details['description']}\n"
            )
        docstring += '    """'

        # Combine signature and docstring
        function_strings.append(f"{function_signature}\n{docstring}\n")

    return "\n".join(function_strings)


def format_code_as_action_prompt(function_description: Sequence) -> str:
    prompt = "Here are the signatures and docstrings of these functions:\n\n```python\n"
    prompt += generate_function_signatures(function_description)
    prompt += "\n```"
    prompt += WRITE_CODE_PROMPT

    return prompt


_python_comment_pattern = re.compile(r"#.*")


def parse_python_code_comments(code: str) -> str:
    comments = _python_comment_pattern.findall(code)
    comments = [c.strip() for c in comments]
    comments = [c if c else "\n" for c in comments]
    return " ".join(comments)


def extract_python_code_blocks(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("\n", 1)[0]
    return text


class BaseAdapter:
    conversation_start_dict: dict[
        ActionFormat, tuple[ConversationMessage, ConversationMessage]
    ]

    @staticmethod
    def parse_react(text: str) -> ActionWithTought:
        """
        ReAct format:
        ```
        Thought:
        I think ...

        Action:
        action
        ```
        """
        invalid_format_flg = False
        _split = text.rsplit("Action:", 1)
        if len(_split) == 0:
            _thought, _action = text
            invalid_format_flg = True
        elif len(_split) == 1:
            if "search[" in text or "click[" in text:
                _thought, _action = "", _split[0]
            else:
                _thought, _action = _split[0], ""
            invalid_format_flg = True
        else:
            assert len(_split) == 2
            _thought, _action = _split

        thought = _thought.split("Thought:")
        if len(thought) == 1:
            thought = thought[0]
            invalid_format_flg = True
        else:
            thought = thought[1].strip()
        action = _action.strip()
        if invalid_format_flg:
            print(
                "The text is not in the correct format. Parsing result may not be accurate."
            )
            print("###RAW TEXT:\n", text)
            print("\n###PARSED THOUGHT:\n", thought)
            print("\n###PARSED ACTION:\n", action)
        return ActionWithTought(thought, action)

    @staticmethod
    def to_react(action_with_thought: ActionWithTought) -> str:
        return f"Thought:\n{action_with_thought.thought}\n\nAction:\n{action_with_thought.action}"

    @staticmethod
    def parse_function_calling(text: str) -> ActionWithTought:
        """
        Function Calling format:
        ```json
        {
            "function_name": "function_name",
            "args": {"kwarg1": "value1", "kwarg2": "value2"}
        }
        ```
        """
        raise NotImplementedError

    @staticmethod
    def to_function_calling(action_with_thought: ActionWithTought) -> str:
        raise NotImplementedError

    @staticmethod
    def parse_code_as_action(text: str) -> ActionWithTought:
        """
        Code as Action format:
        ```
        code
        ```
        """
        raise NotImplementedError

    @staticmethod
    def to_code_as_action(action_with_thought: ActionWithTought) -> str:
        raise NotImplementedError

    @classmethod
    def action_parser(cls, action: str, action_format: ActionFormat) -> str:
        if action_format == ActionFormat.REACT:
            return cls.parse_react(action).action
        elif action_format == ActionFormat.FUNCTION_CALLING:
            return cls.parse_function_calling(action).action
        elif action_format == ActionFormat.CODE_AS_ACTION:
            return cls.parse_code_as_action(action).action
        else:
            raise NotImplementedError


class BaseAgentEnvController:
    def __init__(self, agent: Agent | APIAgent, tasks: Sequence[BaseTask]) -> None:
        self.agent = agent
        self.tasks = tasks

    def generate_experience(
        self,
        idxs: Sequence[int] | Sequence[Sequence[int]] | None = None,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
        use_memory: bool = False,
        enable_memory_storage: bool = True,
    ) -> list[ExperienceOutput | APIExperienceOutput]:
        """
        生成experience
        
        Args:
            idxs: 任务索引或索引列表
            generation_config: 生成配置
            max_rounds: 最大轮数
            use_memory: 是否使用Memory（如果为True，会调用generate_experience_with_memory）
        """
        experience = []
        if idxs is None or len(idxs) == 0:
            raise ValueError("idxs cannot be None or empty")
        
        # 检查idxs的第一个元素类型
        first_elem = idxs[0]
        if isinstance(first_elem, int):
            if use_memory:
                experience += self.tasks[0].generate_experience_with_memory(
                    self.agent,
                    idxs,
                    generation_config,
                    max_rounds,
                    enable_memory_storage=enable_memory_storage,
                )
            else:
                experience += self.tasks[0].generate_experience(
                    self.agent,
                    idxs,
                    generation_config,
                    max_rounds,
                )
        elif isinstance(first_elem, Sequence):
            for idx, task in enumerate(self.tasks):
                if use_memory:
                    experience += task.generate_experience_with_memory(
                        self.agent,
                        idxs[idx],
                        generation_config,
                        max_rounds,
                        enable_memory_storage=enable_memory_storage,
                    )
                else:
                    experience += task.generate_experience(
                        self.agent,
                        idxs[idx],
                        generation_config,
                        max_rounds,
                    )
        else:
            raise ValueError("Incorrect Format for idxs")

        return experience


class Evaluator(BaseAgentEnvController):
    def eval(
        self,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
        idxs: Sequence[int] | Sequence[Sequence[int]] | None = None,
        use_memory: bool = False,
    ) -> EvaluationOutput:
        """
        评估任务
        
        Args:
            generation_config: 生成配置
            max_rounds: 最大轮数
            idxs: 任务索引或索引列表
            use_memory: 是否使用Memory（如果为True，会调用generate_experience_with_memory）
        """
        exps = self.generate_experience(
            idxs=(
                idxs
                if idxs is not None
                else [list(range(len(task.clients[0]))) for task in self.tasks]
            ),
            generation_config=generation_config,
            max_rounds=max_rounds,
            use_memory=use_memory,
            enable_memory_storage=False, # Disable memory storage during evaluation
        )
        rewards = np.array([exp.reward for exp in exps])
        print(f"Rewards: {rewards}")
        return EvaluationOutput(
            experiences=exps, score=rewards.mean(), success=((rewards == 1) | (rewards == 100)).mean()
        )
    
    def _process_single_eval(
        self,
        idx: int,
        generation_config: Optional[GenerationConfig],
        max_rounds: Optional[int],
        use_memory: bool,
        enable_memory_storage: bool,
    ) -> tuple[float, float, ExperienceOutput | APIExperienceOutput]:
        """
        处理单个任务索引的评估（用于多进程）
        
        Returns:
            (score, success, experience) 元组
        """
        try:
            exps = self.generate_experience(
                idxs=[idx],
                generation_config=generation_config,
                max_rounds=max_rounds,
                use_memory=use_memory,
                enable_memory_storage=enable_memory_storage,
            )
            if len(exps) == 0:
                return 0.0, 0.0, None
            exp = exps[0]
            score = exp.reward
            success = 1.0 if (exp.reward == 1 or exp.reward == 100) else 0.0
            return score, success, exp
        except Exception as e:
            print(f"Error processing idx {idx}: {e}")
            return 0.0, 0.0, None
    
    def eval_multiprocess(
        self,
        idxs: Sequence[int] | None = None,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
        use_memory: bool = False,
        enable_memory_storage: bool = False,
        num_processes: int = 4,
    ) -> EvaluationOutput:
        """
        多进程评估任务
        
        Args:
            idxs: 任务索引列表
            generation_config: 生成配置
            max_rounds: 最大轮数
            use_memory: 是否使用Memory
            enable_memory_storage: 是否启用Memory存储
            num_processes: 进程数量
            
        Returns:
            EvaluationOutput包含所有experiences和统计信息
            
        Note:
            - 每个进程会独立初始化Evaluator，因此Agent和Task会在每个进程中重新创建
            - 如果使用Memory，每个进程会有独立的Memory实例，但它们共享同一个经验池路径（如果提供）
            - 多进程环境下，Memory的写入可能会有竞争，建议在评估时设置enable_memory_storage=False
        """
        normalized_idxs, raw_results = self._run_multiprocess_evaluations(
            idxs=idxs,
            generation_config=generation_config,
            max_rounds=max_rounds,
            use_memory=use_memory,
            enable_memory_storage=enable_memory_storage,
            num_processes=num_processes,
        )
        
        if len(normalized_idxs) == 0 or len(raw_results) == 0:
            return EvaluationOutput(experiences=[], score=0.0, success=0.0)
        
        experiences = []
        scores = []
        successes = []
        for _, score, success, exp in raw_results:
            if exp is not None:
                experiences.append(exp)
                scores.append(score)
                successes.append(success)
        
        if len(scores) == 0:
            return EvaluationOutput(experiences=[], score=0.0, success=0.0)
        
        return EvaluationOutput(
            experiences=experiences,
            score=np.array(scores).mean(),
            success=np.array(successes).mean(),
        )
    
    def generate_experience_multiprocess(
        self,
        idxs: Sequence[int],
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
        use_memory: bool = False,
        enable_memory_storage: bool = True,
        num_processes: int = 4,
    ) -> list[ExperienceOutput | APIExperienceOutput]:
        """
        多进程生成experience
        
        Args:
            idxs: 任务索引列表
            generation_config: 生成配置
            max_rounds: 最大轮数
            use_memory: 是否使用Memory
            enable_memory_storage: 是否启用Memory存储
            num_processes: 进程数量
            
        Returns:
            ExperienceOutput或APIExperienceOutput列表
        """
        eval_output = self.eval_multiprocess(
            idxs=idxs,
            generation_config=generation_config,
            max_rounds=max_rounds,
            use_memory=use_memory,
            enable_memory_storage=enable_memory_storage,
            num_processes=num_processes,
        )
        return eval_output.experiences
    
    def stability_analysis(
        self,
        history_path: str,
        idxs: Sequence[int] | None = None,
        generation_config: Optional[GenerationConfig] = None,
        baseline_generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
        num_processes: int = 4,
        seed: Optional[int] = None,
        use_memory: bool = True,
        enable_memory_storage: bool = False,
    ) -> dict:
        """
        运行稳定性(stability)分析
        
        Args:
            history_path: 存储/读取 results_history 的本地路径
            idxs: 可选的自定义索引列表；若为空则自动采样
            generation_config: 使用Memory评估时的生成配置
            baseline_generation_config: 生成baseline (无memory) 的配置
            max_rounds: 最大轮数
            num_processes: 并行进程数
            seed: 采样随机种子
            use_memory: 是否在稳定性评估阶段启用Memory
            enable_memory_storage: 稳定性评估阶段是否允许写Memory, 默认不允许
        
        Returns:
            {
                "rr": float,  # 平均retention ratio
                "per_task": dict,
                "history_path": str,
                "history": dict,
                "evaluated_indices": list[int],
                "experiences": list[ExperienceOutput | APIExperienceOutput],
            }
        """
        # rng = random.Random(seed)
        candidate_idxs = sorted(self._normalize_eval_indices(idxs))
        print(f"[Candidate indices]: {candidate_idxs}")
        if len(candidate_idxs) == 0:
            raise ValueError("No indices available for stability analysis.")
        
        results_history = self._load_results_history(history_path)
        missing_baseline_idxs = [idx for idx in candidate_idxs if str(idx) not in results_history]
        
        if missing_baseline_idxs:
            print(f"[Missing baseline performance]: {len(missing_baseline_idxs)} tasks")
            _, baseline_results = self._run_multiprocess_evaluations(
                idxs=missing_baseline_idxs,
                generation_config=baseline_generation_config or generation_config,
                max_rounds=max_rounds,
                use_memory=False,
                enable_memory_storage=False,
                num_processes=num_processes,
            )
            for idx, _, success, _ in baseline_results:
                results_history[str(idx)] = {"peak_performance": float(success)}
            self._save_results_history(history_path, results_history)
            print(f"[Saved baseline performance]: {len(results_history)} tasks")
            time.sleep(3)
        
        _, stability_results = self._run_multiprocess_evaluations(
            idxs=candidate_idxs,
            generation_config=generation_config,
            max_rounds=max_rounds,
            use_memory=use_memory,
            enable_memory_storage=enable_memory_storage,
            num_processes=num_processes,
        )
        
        per_task = {}
        total_ratio = 0.0
        valid_tasks = 0
        experiences = []
        
        for idx, _, success, exp in stability_results:
            key = str(idx)
            history_entry = results_history.get(key, {"peak_performance": 0.0})
            peak = history_entry.get("peak_performance", 0.0)
            updated_peak = max(float(peak), float(success))
            history_entry["peak_performance"] = updated_peak
            results_history[key] = history_entry
            
            ratio = (float(success) / updated_peak) if updated_peak > 0 else 0.0
            per_task[key] = {
                "current_performance": float(success),
                "peak_performance": updated_peak,
                "retention_ratio": ratio,
            }
            total_ratio += ratio
            valid_tasks += 1
            
            if exp is not None:
                experiences.append(exp)
        
        mean_rr = (total_ratio / valid_tasks) if valid_tasks > 0 else 0.0
        
        return {
            "rr": mean_rr,
            "per_task": per_task,
            "history_path": history_path,
            "history": results_history,
            "evaluated_indices": candidate_idxs,
            "experiences": experiences,
        }
    
    def plasticity_analysis(
        self,
        history_path: str,
        idxs: Sequence[int] | None = None,
        generation_config: Optional[GenerationConfig] = None,
        baseline_generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
        num_processes: int = 4,
        use_memory: bool = True,
        enable_memory_storage: bool = False,
    ) -> dict:
        """
        运行plasticity分析，用于衡量启用Memory后的forward transfer增益
        
        Args:
            history_path: baseline结果存储路径
            idxs: 待评估索引（None 时遍历任务的全部索引）
            generation_config: 启用Memory时的生成配置
            baseline_generation_config: baseline配置；为空时复用generation_config
            max_rounds: 最大交互轮数
            num_processes: 并行进程数量
            use_memory: plasticity评估时是否启用Memory
            enable_memory_storage: plasticity评估阶段是否允许写Memory（默认False）
        """
        candidate_idxs = sorted(self._normalize_eval_indices(idxs))
        print(f"[Plasticity] Candidate indices: {candidate_idxs}")
        if len(candidate_idxs) == 0:
            raise ValueError("No indices available for plasticity analysis.")
        
        results_history = self._load_results_history(history_path)
        missing_baseline_idxs = [idx for idx in candidate_idxs if str(idx) not in results_history]
        
        if missing_baseline_idxs:
            print(f"[Plasticity] Missing baseline for {len(missing_baseline_idxs)} tasks, generating...")
            _, baseline_results = self._run_multiprocess_evaluations(
                idxs=missing_baseline_idxs,
                generation_config=baseline_generation_config or generation_config,
                max_rounds=max_rounds,
                use_memory=False,
                enable_memory_storage=False,
                num_processes=num_processes,
            )
            for idx, _, success, _ in baseline_results:
                results_history[str(idx)] = {"peak_performance": float(success)}
            self._save_results_history(history_path, results_history)
            print(f"[Plasticity] Baseline saved for {len(results_history)} tasks")
            time.sleep(2)
        
        eval_candidate_idxs = [
            idx
            for idx in candidate_idxs
            if float(results_history.get(str(idx), {}).get("peak_performance", 0.0)) == 0.0
        ] # 对于plasticity可塑性测试，只评估baseline失败的任务
        if len(eval_candidate_idxs) == 0:
            print("[Plasticity] All candidates solved by baseline, nothing to evaluate.")
            plasticity_results = []
        else:
            print(f"[Plasticity] Evaluating plasticity on {len(eval_candidate_idxs)} tasks (baseline failed).")
            _, plasticity_results = self._run_multiprocess_evaluations(
                idxs=eval_candidate_idxs,
                generation_config=generation_config,
                max_rounds=max_rounds,
                use_memory=use_memory,
                enable_memory_storage=enable_memory_storage,
                num_processes=num_processes,
            )
        
        per_task = {}
        total_ft = 0.0
        valid_tasks = 0
        experiences = []
        
        for idx, _, success, exp in plasticity_results:
            key = str(idx)
            baseline_entry = results_history.get(key, {"peak_performance": 0.0})
            baseline_perf = float(baseline_entry.get("peak_performance", 0.0))
            ft_gain = float(success) - baseline_perf
            per_task[key] = {
                "memory_performance": float(success),
                "baseline_performance": baseline_perf,
                "ft_gain": ft_gain,
            }
            total_ft += ft_gain
            valid_tasks += 1
            if exp is not None:
                experiences.append(exp)
        
        mean_ft = (total_ft / valid_tasks) if valid_tasks > 0 else 0.0
        return {
            "ft": mean_ft,
            "per_task": per_task,
            "history_path": history_path,
            "history": results_history,
            "evaluated_indices": eval_candidate_idxs,
            "experiences": experiences,
        }
    
    def _get_agent_kwargs(self) -> dict:
        """获取Agent的初始化参数（用于多进程序列化）"""
        if isinstance(self.agent, APIAgent):
            # 从OpenAI client获取api_key和base_url
            api_key = ""
            base_url = ""
            if hasattr(self.agent, 'client') and self.agent.client is not None:
                # OpenAI client的api_key和base_url可能在不同位置
                if hasattr(self.agent.client, 'api_key'):
                    api_key = self.agent.client.api_key
                elif hasattr(self.agent.client, '_client') and hasattr(self.agent.client._client, 'api_key'):
                    api_key = self.agent.client._client.api_key
                
                if hasattr(self.agent.client, 'base_url'):
                    base_url = str(self.agent.client.base_url)
                elif hasattr(self.agent.client, '_client') and hasattr(self.agent.client._client, 'base_url'):
                    base_url = str(self.agent.client._client.base_url)
            
            return {
                "api_key": api_key,
                "base_url": base_url,
                "model": self.agent.model,
                "max_tokens": self.agent.max_tokens,
                "temperature": self.agent.temperature,
                "top_p": self.agent.top_p,
            }
        else:
            # 对于本地Agent，可能需要特殊处理
            # 这里返回基本信息，实际使用时可能需要调整
            return {
                "model_path": "local_model",  # 需要根据实际情况调整
            }
    
    def _get_task_kwargs(self, task: BaseTask) -> dict:
        """获取Task的初始化参数（用于多进程序列化）"""
        # 尝试从task的clients中获取client_args
        if hasattr(task, 'clients') and len(task.clients) > 0:
            client = task.clients[0]
            kwargs = {}
            if hasattr(client, 'env_server_base'):
                kwargs['env_server_base'] = client.env_server_base
            if hasattr(client, 'data_len'):
                kwargs['data_len'] = client.data_len
            if hasattr(client, 'timeout'):
                kwargs['timeout'] = client.timeout
            if hasattr(client, 'is_eval'):
                kwargs['is_eval'] = client.is_eval
            if hasattr(client, 'test_data_start_index'):
                kwargs['test_data_start_index'] = client.test_data_start_index
            return kwargs
        return {}
    
    def _get_memory_kwargs(self) -> dict | None:
        """获取Memory的初始化参数（用于多进程序列化）"""
        if hasattr(self.tasks[0], 'memory') and self.tasks[0].memory is not None:
            memory = self.tasks[0].memory
                        # NullMemory
            if getattr(memory, "type", None) == "null":
                return {"memory_type": "null"}
            
            return {
                "memory_type": memory.type,
                "reme_base_url": getattr(memory, "base_url", "http://localhost:8123/"),
                "reme_workspace_id": getattr(memory, "workspace_id", "task_workspace"),
                "success_only": getattr(memory, "success_only", True),
                "k_retrieval": getattr(memory, "k_retrieval", 1),
            }

            # # ReMe RawMemory
            # if getattr(memory, "type", None) == "raw":
            #     return {
            #         "memory_type": memory.type,
            #         "reme_base_url": getattr(memory, "base_url", "http://localhost:8123/"),
            #         "reme_workspace_id": getattr(memory, "workspace_id", "task_workspace"),
            #         "success_only": getattr(memory, "success_only", True),
            #         "k_retrieval": getattr(memory, "k_retrieval", 1),
            #     }
            # # ReMe StrategyMemory
            # if getattr(memory, "type", None) == "strategy":
            #     return {
            #         "memory_type": memory.type,
            #         "reme_base_url": getattr(memory, "base_url", "http://localhost:8123/"),
            #         "reme_workspace_id": getattr(memory, "workspace_id", "task_workspace"),
            #         "success_only": getattr(memory, "success_only", True),
            #         "k_retrieval": getattr(memory, "k_retrieval", 1),
            #     }
            # # ReMe StrategyRewriteMemory
            # if getattr(memory, "type", None) == "strategy_rewrite":
            #     return {
            #         "memory_type": memory.type,
            #         "reme_base_url": getattr(memory, "base_url", "http://localhost:8123/"),
            #         "reme_workspace_id": getattr(memory, "workspace_id", "task_workspace"),
            #         "success_only": getattr(memory, "success_only", True),
            #         "k_retrieval": getattr(memory, "k_retrieval", 1),
            #     }

        return None
    
    def _normalize_eval_indices(
        self,
        idxs: Sequence[int] | Sequence[Sequence[int]] | None,
    ) -> list[int]:
        """
        将idxs转换为扁平的整数列表
        """
        if idxs is None:
            normalized: list[int] = []
            for task in self.tasks:
                task_len = len(task.clients[0])
                normalized.extend(range(task_len))
            return normalized
        
        if isinstance(idxs, Sequence):
            idx_list = list(idxs)
        else:
            idx_list = [idxs]  # type: ignore[arg-type]
        
        if len(idx_list) == 0:
            return []
        
        first_elem = idx_list[0]
        if isinstance(first_elem, (list, tuple, Sequence)) and not isinstance(first_elem, (str, bytes)):
            flattened: list[int] = []
            for item in idx_list:
                if isinstance(item, (list, tuple, Sequence)) and not isinstance(item, (str, bytes)):
                    flattened.extend(int(i) for i in item)
                else:
                    flattened.append(int(item))  # type: ignore[arg-type]
            return flattened
        
        return [int(item) for item in idx_list]
    
    def _run_multiprocess_evaluations(
        self,
        idxs: Sequence[int] | None,
        generation_config: Optional[GenerationConfig],
        max_rounds: Optional[int],
        use_memory: bool,
        enable_memory_storage: bool,
        num_processes: int,
    ) -> tuple[list[int], list[tuple[int, float, float, ExperienceOutput | APIExperienceOutput | None]]]:
        normalized_idxs = self._normalize_eval_indices(idxs)
        if len(normalized_idxs) == 0:
            return [], []
        
        init_args = {
            "agent_type": type(self.agent).__name__,
            "agent_kwargs": self._get_agent_kwargs(),
            "task_classes": [type(task).__name__ for task in self.tasks],
            "task_kwargs_list": [self._get_task_kwargs(task) for task in self.tasks],
            "use_memory": use_memory,
            "memory_kwargs": self._get_memory_kwargs() if use_memory else None,
        }
        
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(
            processes=num_processes,
            initializer=_init_worker_evaluator,
            initargs=(init_args,)
        ) as pool:
            process_func = partial(
                _process_worker_eval,
                generation_config=generation_config,
                max_rounds=max_rounds,
                use_memory=use_memory,
                enable_memory_storage=enable_memory_storage,
            )
            results = pool.map(process_func, normalized_idxs)
        
        raw_results: list[tuple[int, float, float, ExperienceOutput | APIExperienceOutput | None]] = []
        for idx, (score, success, exp) in zip(normalized_idxs, results):
            raw_results.append((idx, score, success, exp))
        return normalized_idxs, raw_results
    
    def _load_results_history(self, history_path: Optional[str]) -> dict[str, dict]:
        if not history_path or not os.path.exists(history_path):
            return {}
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[Evaluator] Failed to load results history from {history_path}: {exc}")
        return {}
    
    def _save_results_history(self, history_path: Optional[str], history: dict) -> None:
        if not history_path:
            return
        dir_name = os.path.dirname(history_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)


# 全局变量用于多进程（每个进程独立）
_worker_evaluator = None



def _init_worker_evaluator(init_args: dict):
    """初始化工作进程的Evaluator（每个进程调用一次）"""
    global _worker_evaluator
    
    from .agent import APIAgent
    from .reme_memory import RawMemory, HybridTrajInsightMemory, StrategyMemory, StrategyRewriteMemory, MEMORY_CLS_MAP
    from .memory import NullMemory
    
    # 重建Agent
    agent_type = init_args["agent_type"]
    agent_kwargs = init_args["agent_kwargs"]
    
    if agent_type == "APIAgent":
        agent = APIAgent(**agent_kwargs)
    else:
        # 对于其他类型的Agent，可能需要特殊处理
        raise ValueError(f"Unsupported agent type for multiprocessing: {agent_type}")
    
    # 重建Tasks
    # 注意：这里需要从agentenv.envs导入，而不是从controller导入
    try:
        from agentenv.envs import (
            AlfWorldTask, WebshopTask, SciworldTask, TextCraftTask,
            WebarenaTask, SqlGymTask, BabyAITask, TodoTask, MovieTask,
            SheetTask, WeatherTask, AcademiaTask, SearchQATask,
            MazeTask, WordleTask, MathBenchTask, GPQATask,
        )
    except ImportError:
        # 如果直接导入失败，尝试从相对路径导入
        import sys
        import os
        # 获取agentenv包的路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        agentenv_dir = os.path.join(current_dir, '..', '..')
        if agentenv_dir not in sys.path:
            sys.path.insert(0, agentenv_dir)
        from agentenv.envs import (
            AlfWorldTask, WebshopTask, SciworldTask, TextCraftTask,
            WebarenaTask, SqlGymTask, BabyAITask, TodoTask, MovieTask,
            SheetTask, WeatherTask, AcademiaTask, SearchQATask,
            MazeTask, WordleTask, MathBenchTask, GPQATask,
        )
    
    task_classes_map = {
        "AlfWorldTask": AlfWorldTask,
        "WebshopTask": WebshopTask,
        "SciworldTask": SciworldTask,
        "TextCraftTask": TextCraftTask,
        "WebarenaTask": WebarenaTask,
        "SqlGymTask": SqlGymTask,
        "BabyAITask": BabyAITask,
        "TodoTask": TodoTask,
        "MovieTask": MovieTask,
        "SheetTask": SheetTask,
        "WeatherTask": WeatherTask,
        "AcademiaTask": AcademiaTask,
        "SearchQATask": SearchQATask,
        "MazeTask": MazeTask,
        "WordleTask": WordleTask,
        "MathBenchTask": MathBenchTask,
        "GPQATask": GPQATask,
    }



    tasks = []
    for task_class_name, task_kwargs in zip(init_args["task_classes"], init_args["task_kwargs_list"]):
        task_class = task_classes_map.get(task_class_name)
        if task_class is None:
            raise ValueError(f"Unknown task class: {task_class_name}")
        
        # 创建Memory（如果需要）
        memory = None
        if init_args["use_memory"] and init_args["memory_kwargs"]:
            memory_kwargs = init_args["memory_kwargs"]
            memory_cls = MEMORY_CLS_MAP[memory_kwargs["memory_type"]]
            if memory_kwargs["memory_type"] == "null":
                memory = NullMemory()
            else:
                memory = memory_cls(
                        reme_base_url=memory_kwargs["reme_base_url"],
                        reme_workspace_id=memory_kwargs["reme_workspace_id"],
                        success_only=memory_kwargs["success_only"],
                        k_retrieval=memory_kwargs["k_retrieval"],
                    )
            
        task = task_class(client_args=task_kwargs, n_clients=1, memory=memory)
        tasks.append(task)
    
    # 创建Evaluator
    from .utils import Evaluator
    _worker_evaluator = Evaluator(agent=agent, tasks=tasks)


def _process_worker_eval(
    idx: int,
    generation_config: Optional[GenerationConfig],
    max_rounds: Optional[int],
    use_memory: bool,
    enable_memory_storage: bool,
) -> tuple[float, float, ExperienceOutput | APIExperienceOutput]:
    """工作进程处理单个任务索引"""
    global _worker_evaluator
    if _worker_evaluator is None:
        raise RuntimeError("Worker evaluator not initialized")
    
    return _worker_evaluator._process_single_eval(
        idx=idx,
        generation_config=generation_config,
        max_rounds=max_rounds,
        use_memory=use_memory,
        enable_memory_storage=enable_memory_storage,
    )


class BaseTrainer(BaseAgentEnvController):
    # def __init__(self, agent: Agent, tasks: Sequence[BaseTask]) -> None:
    #     super().__init__(agent, tasks)

    def train(self):
        pass

    def eval(
        self,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
        idxs: Sequence[int] | Sequence[Sequence[int]] = None,
        use_memory: bool = False,
    ) -> EvaluationOutput:
        """
        评估任务
        
        Args:
            generation_config: 生成配置
            max_rounds: 最大轮数
            idxs: 任务索引或索引列表
            use_memory: 是否使用Memory（如果为True，会调用generate_experience_with_memory）
        """
        exps = self.generate_experience(
            idxs=idxs,
            generation_config=generation_config,
            max_rounds=max_rounds,
            use_memory=use_memory,
            enable_memory_storage=False, # Disable memory storage during evaluation
        )
        rewards = np.array([exp.reward for exp in exps])
        return EvaluationOutput(
            experiences=exps, score=rewards.mean(), success=((rewards == 1) | (rewards == 100)).mean()
        )

    def save_model(self):
        pass
