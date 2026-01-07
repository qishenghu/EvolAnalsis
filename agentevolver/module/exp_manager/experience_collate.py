"""
Experience Mix Collate Function for Experience Replay.

This module provides:
- ExperienceMixCollateFn: Original collate function for ExGRPO self-generated experience replay (Task-level mixing)
- TeacherExperienceMixCollateFn: Extended collate function supporting both self-generated and teacher experience (Task-level)
- LUFFYTeacherRolloutMixer: LUFFY-style rollout-level mixing (each task mixes on-policy and teacher rollouts)

⭐ Key Design Difference:
- Task-level mixing (ExGRPO): Some tasks in batch are entirely off-policy, others entirely on-policy
- Rollout-level mixing (LUFFY): Each task's n rollouts mix on-policy and teacher rollouts

Backward Compatibility:
- ExperienceMixCollateFn returns Tuple[List[Task], List[Task]] (experience_tasks, on_policy_tasks)
- TeacherExperienceMixCollateFn returns Tuple[List[Task], List[Task], List[Task]] 
  (self_exp_tasks, teacher_exp_tasks, on_policy_tasks)
- LUFFYTeacherRolloutMixer.mix_trajectories returns Tuple[List[Trajectory], Dict] (mixed_trajectories, stats)
"""

import random
from typing import List, Tuple, Optional, Dict, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from agentevolver.schema.task import Task
    from agentevolver.schema.trajectory import Trajectory
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
        print(f"valid_exp_task_ids: {valid_exp_task_ids}")
        
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


class TeacherExperienceMixCollateFn(ExperienceMixCollateFn):
    """
    扩展的 Experience 混合函数，支持三种数据类型：
    1. On-policy: 当前策略生成的新轨迹
    2. Self-generated off-policy: 自身历史成功轨迹（ExGRPO）
    3. Teacher off-policy: 外部 Teacher 模型的轨迹（LUFFY）
    
    ⭐ 向后兼容设计：
    - 继承 ExperienceMixCollateFn
    - 返回三元组 (self_exp_tasks, teacher_exp_tasks, on_policy_tasks)
    - 原有 ExperienceMixCollateFn 不受影响
    """
    
    def __init__(
        self,
        exp_manager: "ExperienceManager",
        train_task_manager,
        # Self-generated experience 配置
        self_exp_ratio: float = 0.3,
        # Teacher experience 配置
        teacher_exp_ratio: float = 0.2,
        teacher_exp_enabled: bool = True,
        # 共同配置
        replay_start_ratio: float = 0.35,
        offpolicy_trajectories_per_task: int = 1,
        n_rollout: int = 8,
    ):
        """
        初始化 TeacherExperienceMixCollateFn。
        
        Args:
            exp_manager: ExperienceManager 实例（统一管理 self-generated 和 teacher）
            train_task_manager: TaskManager 实例
            self_exp_ratio: Self-generated experience 比例
            teacher_exp_ratio: Teacher experience 比例
            teacher_exp_enabled: 是否启用 teacher experience
            replay_start_ratio: 开始 replay 的训练进度
            offpolicy_trajectories_per_task: 每个任务的 off-policy 轨迹数
            n_rollout: 每个 task 的 rollout 数量
        """
        # 调用父类构造函数（用于 self-generated experience）
        super().__init__(
            exp_manager=exp_manager,
            train_task_manager=train_task_manager,
            exp_ratio=self_exp_ratio,  # 父类的 exp_ratio 用于 self-generated
            replay_start_ratio=replay_start_ratio,
            offpolicy_trajectories_per_task=offpolicy_trajectories_per_task,
            n_rollout=n_rollout,
        )
        
        self.self_exp_ratio = self_exp_ratio
        self.teacher_exp_ratio = teacher_exp_ratio
        # 检查 exp_manager 是否启用了 teacher
        self.teacher_exp_enabled = (teacher_exp_enabled and 
                                    getattr(exp_manager, 'teacher_enabled', False))
    
    def __call__(
        self,
        training_tasks: List["Task"],
        training_progress: float,
        enable_replay: bool = True,
    ) -> Tuple[List["Task"], List["Task"], List["Task"]]:
        """
        混合三种类型的 tasks。
        
        ⭐ Multi-turn 支持：所有轨迹格式一致，无需额外处理
        
        Args:
            training_tasks: 原始 training tasks 列表（batch_size 个）
            training_progress: 当前训练进度
            enable_replay: 是否启用 replay
            
        Returns:
            Tuple[List[Task], List[Task], List[Task]]:
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
        
        # 获取可用的 self-generated experience task_ids
        valid_self_exp_task_ids = self.exp_manager.get_valid_replay_task_ids()
        
        # 获取可用的 teacher experience task_ids
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
        self_exp_tasks = self._task_ids_to_tasks(sampled_self_exp_task_ids, is_teacher=False)
        teacher_exp_tasks = self._task_ids_to_tasks(sampled_teacher_task_ids, is_teacher=True)
        
        # 补充 on-policy tasks
        used_task_ids = set(sampled_self_exp_task_ids + sampled_teacher_task_ids)
        remaining_tasks = [t for t in training_tasks if t.task_id not in used_task_ids]
        n_on_policy = batch_size - len(self_exp_tasks) - len(teacher_exp_tasks)
        on_policy_tasks = remaining_tasks[:n_on_policy]
        
        # 如果 remaining_tasks 不够，从 training_tasks 补充
        if len(on_policy_tasks) < n_on_policy:
            needed = n_on_policy - len(on_policy_tasks)
            additional = [t for t in training_tasks if t not in on_policy_tasks][:needed]
            on_policy_tasks.extend(additional)
        
        if self_exp_tasks or teacher_exp_tasks:
            logger.info(f"[TeacherExperienceMixCollateFn] Batch split: "
                       f"self_exp={len(self_exp_tasks)}, teacher_exp={len(teacher_exp_tasks)}, "
                       f"on_policy={len(on_policy_tasks)}")
        
        return self_exp_tasks, teacher_exp_tasks, on_policy_tasks
    
    def _task_ids_to_tasks(
        self, 
        task_ids: List[str], 
        is_teacher: bool = False
    ) -> List["Task"]:
        """
        将 task_id 转换为 Task 对象。
        
        Args:
            task_ids: task_id 列表
            is_teacher: 是否是 teacher experience
            
        Returns:
            Task 对象列表
        """
        tasks = []
        for task_id in task_ids:
            task = self._get_task_by_id(task_id)
            if task is not None:
                # 初始化 metadata
                task.metadata = task.metadata if hasattr(task, 'metadata') and task.metadata else {}
                task.metadata["n_offpolicy_trajectories"] = self.offpolicy_trajectories_per_task
                task.metadata["is_teacher_task"] = is_teacher
                tasks.append(task)
            else:
                logger.warning(f"Failed to get Task object for task_id={task_id}, skipping")
        return tasks


class LUFFYTeacherRolloutMixer:
    """
    ⭐ LUFFY 风格的 Teacher Rollout 混合器。
    
    核心设计差异：
    - ExGRPO (Task 级别): batch 中某些 task 完全是 off-policy，某些完全是 on-policy
    - LUFFY (Rollout 级别): 每个 task 的 n 个 rollouts 中混合 on-policy 和 teacher rollouts
    
    例如：batch_size=8, n_rollout=8, n_teacher_rollouts_per_task=1
        Task 1: 7 on-policy + 1 teacher
        Task 2: 7 on-policy + 1 teacher
        ...
        Task 8: 7 on-policy + 1 teacher
        
    这样 GRPO 在同一 task 内计算 advantage 时，teacher 的高 reward 会影响
    on-policy rollouts 的 baseline，从而引导策略学习。
    """
    
    def __init__(
        self,
        exp_manager: "ExperienceManager",
        n_teacher_rollouts_per_task: int = 1,
        n_rollout: int = 8,
    ):
        """
        初始化 LUFFY Teacher Rollout Mixer。
        
        Args:
            exp_manager: ExperienceManager 实例
            n_teacher_rollouts_per_task: 每个 task 混入的 teacher rollout 数量
            n_rollout: 每个 task 的总 rollout 数量
        """
        self.exp_manager = exp_manager
        self.n_teacher_rollouts_per_task = n_teacher_rollouts_per_task
        self.n_rollout = n_rollout
        
        # 验证配置
        if n_teacher_rollouts_per_task >= n_rollout:
            raise ValueError(
                f"n_teacher_rollouts_per_task ({n_teacher_rollouts_per_task}) "
                f"must be less than n_rollout ({n_rollout})"
            )
    
    def get_n_onpolicy_rollouts_per_task(self) -> int:
        """返回每个 task 需要生成的 on-policy rollout 数量。"""
        return self.n_rollout - self.n_teacher_rollouts_per_task
    
    def mix_trajectories(
        self,
        on_policy_trajectories: List["Trajectory"],
        tasks: List["Task"],
    ) -> Tuple[List["Trajectory"], Dict[str, int]]:
        """
        将 teacher rollouts 混入 on-policy trajectories。
        
        ⭐ 核心逻辑：
        1. 对于每个 task，获取其 on-policy trajectories
        2. 获取该 task 的 teacher trajectories
        3. 将 teacher trajectories 添加到该 task 的 rollout 列表中
        4. 确保 teacher trajectories 的 data_id 与 task 一致（用于 GRPO 分组）
        
        Args:
            on_policy_trajectories: 从 env_manager.rollout 生成的 on-policy 轨迹
            tasks: 当前 batch 的 tasks
            
        Returns:
            Tuple[List[Trajectory], Dict[str, int]]:
            - 混合后的所有轨迹（on-policy + teacher）
            - 统计信息字典
        """
        from collections import defaultdict
        
        # 按 task_id 分组 on-policy trajectories
        task_id_to_onpolicy: Dict[str, List] = defaultdict(list)
        for traj in on_policy_trajectories:
            task_id = traj.data_id  # Trajectory 使用 data_id
            task_id_to_onpolicy[task_id].append(traj)
        
        # 获取每个 task 的 teacher rollouts
        teacher_rollouts_map = self.exp_manager.get_teacher_rollouts_for_luffy_mixing(
            tasks=tasks,
            n_teacher_rollouts_per_task=self.n_teacher_rollouts_per_task,
        )
        
        # 混合轨迹
        mixed_trajectories = []
        stats = {
            "total_tasks": len(tasks),
            "tasks_with_teacher": 0,
            "total_onpolicy": len(on_policy_trajectories),
            "total_teacher": 0,
        }
        
        for task in tasks:
            task_id = task.task_id
            
            # 获取该 task 的 on-policy trajectories
            onpolicy_trajs = task_id_to_onpolicy.get(task_id, [])
            
            # 获取该 task 的 teacher trajectories
            teacher_trajs = teacher_rollouts_map.get(task_id, [])
            
            if teacher_trajs:
                stats["tasks_with_teacher"] += 1
                stats["total_teacher"] += len(teacher_trajs)
                
                # ⭐ 确保 teacher trajectories 的 data_id 与 task 一致
                # 这是 GRPO 分组的关键：同一 data_id 的轨迹会被分到同一组
                for traj in teacher_trajs:
                    traj.data_id = task_id
                    traj.metadata["is_teacher"] = True
                    traj.metadata["source"] = "teacher"
            
            # 合并：on-policy + teacher
            task_trajs = onpolicy_trajs + teacher_trajs
            mixed_trajectories.extend(task_trajs)
        
        logger.info(
            f"[LUFFYMixer] Mixed {stats['total_onpolicy']} on-policy + "
            f"{stats['total_teacher']} teacher = {len(mixed_trajectories)} total trajectories. "
            f"Tasks with teacher: {stats['tasks_with_teacher']}/{stats['total_tasks']}"
        )
        
        return mixed_trajectories, stats
    
    def should_use_reduced_rollout(self) -> bool:
        """
        返回是否应该减少 on-policy rollout 数量。
        
        ⭐ LUFFY 设计：如果 n_teacher_rollouts_per_task > 0，
        需要减少 on-policy rollout 数量以保持总数不变。
        
        Returns:
            bool: 是否需要减少 on-policy rollout 数量
        """
        return self.n_teacher_rollouts_per_task > 0
    
    def get_rollout_config_override(self) -> Dict[str, int]:
        """
        返回需要覆盖的 rollout 配置。
        
        ⭐ 用于在 rollout 时减少生成的 on-policy 数量。
        
        Returns:
            Dict: {"n": reduced_n_rollout}
        """
        return {"n": self.get_n_onpolicy_rollouts_per_task()}

