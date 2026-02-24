"""
Teacher Trajectory Collector.

This module provides the TeacherTrajectoryCollector class for collecting
trajectories from Teacher LLM (GPT-4, Claude, or local vLLM models) in
various environments (e.g., ALFWorld).

⭐ Core Design: Reuse existing EnvManager.rollout() framework, only replace
the LLM call with Teacher LLM to ensure consistent data format.
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from loguru import logger
from omegaconf import DictConfig

from agentevolver.schema.trajectory import Trajectory
from agentevolver.schema.task import Task


@dataclass
class TeacherCollectorConfig:
    """Teacher Trajectory Collector 配置"""
    # Teacher LLM 配置
    llm_type: str = "openai"          # "openai" 或 "vllm"
    model_name: str = "gpt-4"         # OpenAI model name
    model_path: Optional[str] = None  # vLLM model path
    api_base: Optional[str] = None    # OpenAI API base URL
    api_key: Optional[str] = None     # OpenAI API key
    
    # 生成参数
    temperature: float = 0.0
    max_tokens: int = 4096
    collect_log_prob: bool = True     # 是否采集 log_prob
    
    # vLLM 特定参数
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    
    # 采集参数
    n_trajectories_per_task: int = 1
    filter_success_only: bool = True
    max_retries: int = 3


class TeacherTrajectoryCollector:
    """
    Teacher Trajectory Collector
    
    ⭐ 核心设计：复用 EnvManager 的 rollout 流程，只替换 LLM 调用部分
    
    使用方式：
    1. 创建 TeacherTrajectoryCollector
    2. 调用 collect_trajectories() 采集轨迹
    3. 调用 save_trajectories() 保存到磁盘
    """
    
    def __init__(
        self, 
        config: TeacherCollectorConfig,
        env_config: DictConfig,
        tokenizer,
    ):
        """
        初始化 Teacher Trajectory Collector。
        
        Args:
            config: Teacher Collector 配置
            env_config: 环境配置（用于创建 EnvWorker）
            tokenizer: Tokenizer
        """
        self.config = config
        self.env_config = env_config
        self.tokenizer = tokenizer
        
        # 创建 Teacher LLM
        self.teacher_llm = self._create_teacher_llm()
        
        # 创建 Teacher Agent Flow
        from agentevolver.module.teacher.teacher_agent_flow import TeacherAgentFlow
        self.teacher_agent_flow = TeacherAgentFlow(
            teacher_llm=self.teacher_llm,
            tokenizer=tokenizer,
            config=env_config,
        )
        
        logger.info(f"[TeacherCollector] Initialized with {config.llm_type} backend, "
                   f"model={config.model_name or config.model_path}")
    
    def _create_teacher_llm(self):
        """创建 Teacher LLM 实例"""
        from agentevolver.module.teacher import create_teacher_llm
        
        if self.config.llm_type == "openai":
            llm_config = {
                "type": "openai",
                "model_name": self.config.model_name,
                "api_base": self.config.api_base,
                "api_key": self.config.api_key,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "collect_log_prob": self.config.collect_log_prob,
            }
        elif self.config.llm_type == "vllm":
            llm_config = {
                "type": "vllm",
                "model_path": self.config.model_path,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "collect_log_prob": self.config.collect_log_prob,
            }
        else:
            raise ValueError(f"Unsupported llm_type: {self.config.llm_type}")
        
        return create_teacher_llm(llm_config)
    
    def collect_trajectories(
        self, 
        tasks: List[Task],
        n_per_task: int = None,
        filter_success: bool = None,
        show_progress: bool = True,
    ) -> List[Trajectory]:
        """
        为给定任务列表采集 Teacher 轨迹。
        
        Args:
            tasks: 任务列表
            n_per_task: 每个任务采集的轨迹数（默认使用配置值）
            filter_success: 是否只保留成功的轨迹（默认使用配置值）
            show_progress: 是否显示进度条
            
        Returns:
            Trajectory 列表
        """
        from agentevolver.module.env_manager.env_worker import EnvWorker
        from agentevolver.module.exp_manager.exp_manager import TrajExpConfig
        
        n_per_task = n_per_task or self.config.n_trajectories_per_task
        filter_success = filter_success if filter_success is not None else self.config.filter_success_only
        
        all_trajectories = []
        
        # 进度条
        task_iter = tasks
        if show_progress:
            try:
                from tqdm import tqdm
                task_iter = tqdm(tasks, desc="Collecting teacher trajectories")
            except ImportError:
                pass
        
        for task in task_iter:
            task_trajectories = []
            
            for rollout_idx in range(n_per_task):
                for retry in range(self.config.max_retries):
                    try:
                        # 创建 EnvWorker
                        worker = EnvWorker(
                            task=task.task,
                            config=self.env_config,
                            thread_index=0,
                            tokenizer=self.tokenizer,
                        )
                        
                        # 创建 traj_exp_config
                        traj_exp_config = TrajExpConfig(
                            add_exp=False,
                            train_mode="discard",
                            task_id=task.task_id,
                            data_id=task.task_id,
                            rollout_id=f"{task.task_id}_teacher_{rollout_idx}",
                            query="",
                        )
                        
                        # 执行 rollout（使用 Teacher Agent Flow 的 llm_chat_fn）
                        trajectory = worker.execute(
                            data_id=task.task_id,
                            rollout_id=f"{task.task_id}_teacher_{rollout_idx}",
                            traj_exp_config=traj_exp_config,
                            agent_flow=self.teacher_agent_flow,
                            tmux={'step': [0], 'token': [0]},
                            stop=[False],
                        )
                        
                        # 检查是否成功
                        # - Prefer success_rate (binary flag) if present
                        # - This allows env-specific success definitions (e.g., SciWorld: done && score>0)
                        success = bool(
                            trajectory.reward
                            and float(getattr(trajectory.reward, "success_rate", 0.0)) > 0.0
                        )
                        
                        # 标记为 Teacher 轨迹
                        trajectory.metadata = trajectory.metadata or {}
                        trajectory.metadata["is_teacher"] = True
                        trajectory.metadata["teacher_model"] = self.teacher_llm.model_name
                        trajectory.metadata["has_log_prob"] = self.teacher_llm.supports_log_prob
                        trajectory.metadata["collected_at"] = datetime.now().isoformat()
                        
                        # 如果有 log_prob，从 last_metadata 获取
                        if self.teacher_agent_flow.last_metadata:
                            log_probs = self.teacher_agent_flow.last_metadata.get("log_probs")
                            if log_probs:
                                trajectory.metadata["old_log_probs"] = log_probs
                        
                        # 过滤成功/失败
                        if filter_success and not success:
                            logger.debug(f"Task {task.task_id} rollout {rollout_idx} failed, "
                                       f"retry {retry+1}/{self.config.max_retries}")
                            continue
                        
                        task_trajectories.append(trajectory)
                        break  # 成功，跳出重试循环
                        
                    except Exception as e:
                        logger.warning(f"Task {task.task_id} rollout {rollout_idx} error: {e}, "
                                     f"retry {retry+1}/{self.config.max_retries}")
                        if retry == self.config.max_retries - 1:
                            logger.error(f"Task {task.task_id} rollout {rollout_idx} failed after "
                                       f"{self.config.max_retries} retries")
            
            all_trajectories.extend(task_trajectories)
        
        logger.info(f"[TeacherCollector] Collected {len(all_trajectories)} trajectories "
                   f"from {len(tasks)} tasks")
        
        return all_trajectories
    
    def save_trajectories(
        self, 
        trajectories: List[Trajectory], 
        save_path: str,
        mode: str = 'w',
    ) -> int:
        """
        保存采集的轨迹到磁盘。
        
        Args:
            trajectories: 轨迹列表
            save_path: 保存路径（.jsonl 格式）
            mode: 写入模式 ('w' 覆盖, 'a' 追加)
            
        Returns:
            保存的轨迹数量
        """
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        
        count = 0
        with open(save_path, mode) as f:
            for traj in trajectories:
                traj_dict = self._trajectory_to_dict(traj)
                f.write(json.dumps(traj_dict, ensure_ascii=False) + '\n')
                count += 1
        
        logger.info(f"[TeacherCollector] Saved {count} trajectories to {save_path}")
        return count
    
    def _trajectory_to_dict(self, traj: Trajectory) -> Dict[str, Any]:
        """
        将 Trajectory 对象转换为可 JSON 序列化的字典。
        """
        # 获取 log_probs（如果有）
        log_probs = traj.metadata.get("old_log_probs") if traj.metadata else None
        
        traj_dict = {
            "task_id": traj.task_id,
            "data_id": traj.data_id,
            "rollout_id": traj.rollout_id,
            "messages": traj.steps,  # Multi-turn 对话历史
            "reward": traj.reward.outcome if traj.reward else 0.0,
            "success": (float(getattr(traj.reward, "success_rate", 0.0)) > 0.0) if traj.reward else False,
            "teacher_model": traj.metadata.get("teacher_model", "unknown") if traj.metadata else "unknown",
            "metadata": {
                "is_teacher": True,
                "has_log_prob": traj.metadata.get("has_log_prob", False) if traj.metadata else False,
                "collected_at": traj.metadata.get("collected_at", "") if traj.metadata else "",
            },
        }
        
        # 如果有 log_probs，添加到顶层
        if log_probs:
            traj_dict["log_probs"] = log_probs
        
        return traj_dict


def load_tasks_from_file(task_file: str, start: int = 0, end: int = None) -> List[str]:
    """
    从文件加载 task IDs。
    
    Args:
        task_file: task ID 文件路径（每行一个 task_id）
        start: 起始索引
        end: 结束索引（不包含）
        
    Returns:
        task_id 列表
    """
    with open(task_file, 'r') as f:
        task_ids = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    if end is None:
        end = len(task_ids)
    
    return task_ids[start:end]


def load_completed_task_ids(output_file: str) -> set:
    """
    从已有输出文件中加载已完成的 task IDs（用于断点续采）。
    
    Args:
        output_file: 输出文件路径
        
    Returns:
        已完成的 task_id 集合
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
    return completed

