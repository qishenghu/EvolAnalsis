"""
Experience Mix Collate Function for Experience Replay.

This module provides the ExperienceMixCollateFn class that handles
mixing on-policy and off-policy tasks based on the exp_ratio configuration.
"""

import random
from typing import List, Tuple, Optional, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from agentevolver.schema.task import Task
    from agentevolver.module.exp_manager.exp_manager import ExperienceManager


class ExperienceMixCollateFn:
    """
    混合 on-policy 和 off-policy tasks 的 collate 函数。
    参考 ExGRPO 的 ExperienceMixCollateFn 设计。
    """
    def __init__(
        self,
        exp_manager: "ExperienceManager",
        train_task_manager,  # TaskManager
        exp_ratio: float = 0.5,
        replay_start_ratio: float = 0.35,
        offpolicy_trajectories_per_task: int = 1,
        n_rollout: int = 8,
    ):
        """
        初始化 ExperienceMixCollateFn。
        
        Args:
            exp_manager: ExperienceManager 实例
            train_task_manager: TaskManager 实例，用于从 task_id 获取 Task 对象
            exp_ratio: Experience tasks 的比例（0.0-1.0），默认 0.5
            replay_start_ratio: 训练进度达到此比例时开始使用 replay
            offpolicy_trajectories_per_task: 每个任务获取的 off-policy 轨迹数量
            n_rollout: 每个 task 的 rollout 数量
        """
        self.exp_manager = exp_manager
        self.train_task_manager = train_task_manager
        self.exp_ratio = exp_ratio
        self.replay_start_ratio = replay_start_ratio
        self.offpolicy_trajectories_per_task = offpolicy_trajectories_per_task
        self.n_rollout = n_rollout
    
    def __call__(
        self,
        training_tasks: List["Task"],
        training_progress: float,
        enable_replay: bool = True,
    ) -> Tuple[List["Task"], List["Task"]]:
        """
        混合 on-policy 和 off-policy tasks。
        
        Args:
            training_tasks: 原始 training tasks 列表（batch_size 个）
            training_progress: 当前训练进度（global_steps / total_training_steps）
            enable_replay: 是否启用 replay（需要同时满足 training_progress >= replay_start_ratio）
            
        Returns:
            Tuple[List[Task], List[Task]]: (experience_tasks, on_policy_tasks)
            - experience_tasks: 从 replaytaskpool 选择的 tasks（需要获取 off-policy trajectories）
            - on_policy_tasks: 纯 on-policy tasks（不需要 off-policy trajectories）
            - 总数量 = len(experience_tasks) + len(on_policy_tasks) = len(training_tasks)
        """
        batch_size = len(training_tasks)
        
        # 检查是否启用 replay
        if not enable_replay or training_progress < self.replay_start_ratio:
            # 阶段 1：只使用 training tasks，不进行 replay
            return [], training_tasks
        
        # 阶段 2：混合 experience tasks 和 on-policy tasks
        # 计算目标 experience task 数量
        target_exp_count = int(batch_size * self.exp_ratio)  # 例如：64 * 0.5 = 32
        
        # 从 replaytaskpool 采样 experience task_ids
        valid_exp_task_ids = self.exp_manager.get_valid_replay_task_ids()
        
        # 采样 experience task_ids（最多 target_exp_count 个）
        n_exp = min(len(valid_exp_task_ids), target_exp_count)
        if n_exp > 0:
            # 随机采样（可以后续支持按难度采样）
            sampled_exp_task_ids = random.sample(valid_exp_task_ids, n_exp)
        else:
            sampled_exp_task_ids = []
        
        # 将 experience task_ids 转换为 Task 对象
        # ⭐ 注意：n_offpolicy_trajectories 会在 get_offpolicy_batch 中根据实际获取的数量设置
        # 这里只做初始化（期望值），实际值可能更小
        experience_tasks = []
        for task_id in sampled_exp_task_ids:
            # 从 train_task_manager 获取 Task 对象
            task = self._get_task_by_id(task_id)
            if task is not None:
                # 初始化 metadata（期望值，实际值在 get_offpolicy_batch 中更新）
                task.metadata = task.metadata if hasattr(task, 'metadata') and task.metadata else {}
                task.metadata["n_offpolicy_trajectories"] = self.offpolicy_trajectories_per_task  # 期望值
                experience_tasks.append(task)
            else:
                logger.warning(f"Failed to get Task object for task_id={task_id}, skipping")
        
        # 计算需要补充的 on-policy tasks 数量
        n_exp_actual = len(experience_tasks)
        n_on_policy = batch_size - n_exp_actual  # 确保总数为 batch_size
        
        # 从 training_tasks 中选择 on-policy tasks
        on_policy_tasks = training_tasks[:n_on_policy]
        
        # 验证总数
        assert len(experience_tasks) + len(on_policy_tasks) == batch_size, \
            f"Total tasks mismatch: {len(experience_tasks)} + {len(on_policy_tasks)} != {batch_size}"
        
        if n_exp_actual > 0:
            logger.info(
                f"Mixed batch: {len(experience_tasks)} experience tasks + "
                f"{len(on_policy_tasks)} on-policy tasks = {batch_size} total"
            )
        
        return experience_tasks, on_policy_tasks
    
    def _get_task_by_id(self, task_id: str) -> Optional["Task"]:
        """
        从 train_task_manager 获取 Task 对象。
        
        Args:
            task_id: 任务 ID
            
        Returns:
            Task 对象，如果找不到则返回 None
        """
        # 尝试从 seed_tasks 中查找
        if hasattr(self.train_task_manager, 'seed_tasks'):
            for task in self.train_task_manager.seed_tasks:
                if task.task_id == task_id:
                    return task
        
        # 尝试从 _tasks 中查找
        if hasattr(self.train_task_manager, '_tasks'):
            for task in self.train_task_manager._tasks:
                if task.task_id == task_id:
                    return task
        
        return None

