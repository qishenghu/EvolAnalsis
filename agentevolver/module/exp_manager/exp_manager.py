import random
import re
import numpy as np
import torch
from loguru import logger
from dataclasses import dataclass, field
from omegaconf import DictConfig
from typing import List, Dict, Any, Optional, Literal, Tuple, Set
from collections import defaultdict
from itertools import groupby
from concurrent.futures import as_completed, Future
from concurrent.futures.thread import ThreadPoolExecutor
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory
from agentevolver.client.em_client import EMClient


@dataclass
class TaskExpConfig:
    add_exp: List[bool]
    train_mode: str = "discard"     # "keep" | "discard"

@dataclass
class TrajExpConfig:
    add_exp: bool = True
    train_mode: str = "discard"
    task_id: str = ""
    data_id: str = ""
    rollout_id: str = ""
    query: str = ""
    mode: str = "sample"            # "sample" | "validate"
    experience_list: List[str] = field(default_factory=list)



class ExperienceManager(object):

    def __init__(self, config: DictConfig):
        """
        Initializes the ExperienceManager with the provided configuration.

        Args:
            config (DictConfig): The configuration dictionary containing settings for the experience manager, rollout, and other components.
        """
        self.config: DictConfig = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.exp_manager_config = config.exp_manager
        self.reme_config = config.exp_manager.reme

        self.val_rollout_mode = self.exp_manager_config.val_rollout_mode
        self.train_rollout_mode = self.exp_manager_config.train_rollout_mode
        self.rollout_ratio = self.exp_manager_config.rollout_ratio
        self.train_sample_mode = self.exp_manager_config.train_sample_mode
        self.train_sample_keepratio = self.exp_manager_config.train_sample_keepratio

        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.thread_pool.max_workers)
        self.em_client = EMClient(base_url=self.reme_config.base_url)
        
        # ⭐ Experience Replay 相关属性 (Self-Generated / ExGRPO)
        exp_replay_config = self.exp_manager_config.get("experience_replay", {})
        self.difficulty2task_dict: Dict[int, List[str]] = defaultdict(list)  # 按难度分桶存储 task_id
        self.task2trajectories: Dict[str, List[Trajectory]] = defaultdict(list)  # 按 task_id 存储 Trajectory 列表
        self.skip_uid_set: Set[str] = set()  # 存储已经全部做对的 task_id，不再参与 replay
        self.replay_start_ratio = exp_replay_config.get("replay_start_ratio", 0.35)
        self.max_trajectories_per_task = exp_replay_config.get("max_trajectories_per_task", 10)
        self.experience_lbound = exp_replay_config.get("experience_lbound", 0)
        self.experience_rbound = exp_replay_config.get("experience_rbound", 8)
        self.exp_select_mode = exp_replay_config.get("exp_select_mode", "argmin")
        self.exp_ratio = exp_replay_config.get("exp_ratio", 0.5)
        
        # ⭐ Teacher Experience Replay 相关属性 (LUFFY)
        teacher_config = self.exp_manager_config.get("teacher_experience", {})
        self.teacher_enabled = teacher_config.get("enable", False)
        self.teacher_data_path = teacher_config.get("data_path", None)
        
        # ⭐ LUFFY 风格配置：混合模式
        # - "rollout_level": LUFFY 风格，每个 task 内混合 on-policy 和 teacher rollouts
        # - "task_level": ExGRPO 风格，batch 中某些 task 完全是 teacher experience
        self.teacher_mix_mode = teacher_config.get("mix_mode", "rollout_level")
        
        # ⭐ LUFFY 风格：每个 task 混入的 teacher rollout 数量
        # 例如：n_rollout=8, n_teacher_rollouts_per_task=1 → 7 on-policy + 1 teacher
        self.n_teacher_rollouts_per_task = teacher_config.get("n_teacher_rollouts_per_task", 1)
        
        # Task 级别配置（兼容旧版 ExGRPO 风格）
        self.teacher_exp_ratio = teacher_config.get("exp_ratio", 0.2)
        
        self.teacher_max_per_task = teacher_config.get("max_trajectories_per_task", 3)
        self.teacher_select_mode = teacher_config.get("select_mode", "random")
        self.teacher_use_log_prob = teacher_config.get("use_log_prob", False)
        
        # Teacher 轨迹存储（与 self-generated 分开存储）
        self.teacher_task2trajectories: Dict[str, List[Trajectory]] = defaultdict(list)
        
        # 如果配置了 data_path，尝试加载 Teacher 轨迹
        if self.teacher_enabled and self.teacher_data_path:
            try:
                self.load_teacher_trajectories(self.teacher_data_path)
            except Exception as e:
                logger.warning(f"Failed to load teacher trajectories from {self.teacher_data_path}: {e}")
        
        # ⭐ RePO (Replay-Enhanced Policy Optimization) 相关属性
        repo_config = self.exp_manager_config.get("repo", {})
        self.repo_enabled = repo_config.get("enable", False)
        self.repo_replay_start_epoch = repo_config.get("replay_start_epoch", 1)
        self.repo_replay_strategy = repo_config.get("replay_strategy", "recency")
        self.repo_g_on = repo_config.get("g_on", 8)
        self.repo_g_off = repo_config.get("g_off", 8)
        self.repo_max_trajectories_per_task = repo_config.get("max_trajectories_per_task", 32)
        self.repo_buffer_update_mode = repo_config.get("buffer_update_mode", "fifo")
        self.repo_store_all_trajectories = repo_config.get("store_all_trajectories", True)
        self.repo_advantage_mode = repo_config.get("advantage_mode", "separate")
        self.repo_clip_eps = repo_config.get("clip_eps", 0.2)
        self.repo_use_importance_clipping = repo_config.get("use_importance_clipping", True)
        
        # RePO Replay Manager (initialized lazily)
        self._repo_manager = None
        self._repo_mixer = None
        
        if self.repo_enabled:
            logger.info(
                f"[ExperienceManager] RePO enabled: "
                f"replay_start_epoch={self.repo_replay_start_epoch}, "
                f"strategy={self.repo_replay_strategy}, "
                f"g_on={self.repo_g_on}, g_off={self.repo_g_off}"
            )
    
    def summarize_in_batch(self, trajectories: List[Trajectory]) -> None:
        trajectories_sorted = sorted(trajectories, key=lambda traj: traj.task_id)
        grouped_trajectories = [list(group) for key, group in groupby(trajectories_sorted, key=lambda traj: traj.task_id)]
        batch_size = self.exp_manager_config.summary_batch_size
        all_batches = []
        for group in grouped_trajectories:
            for i in range(0, len(group), batch_size):
                all_batches.append(group[i:i + batch_size])
        
        futures = []
        for batch in all_batches:
            future = self.thread_pool.submit(
                self.em_client.call_summarizer,
                trajectories=batch,
                workspace_id=self.reme_config.workspace_id
            )
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error in summary task: {e}")
        
        return

    def submit_summary_task(self, trajectories: List[Trajectory], global_steps: int) -> Optional[Future]:
        """
        Submits a summary task to the thread pool for asynchronous processing.

        Args:
            trajectories (List[Trajectory]): A list of trajectory objects to be summarized.
            global_steps (int): The current global step count used to determine task submission timing.

        Returns:
            Optional[Future]: A Future object representing the submitted task, or None if the task
                            should not be submitted or submission fails.
        """
        if not self._should_submit_summary(global_steps):
            return None
        
        try:
            summary_task = self.thread_pool.submit(
                self.em_client.call_summarizer,
                trajectories=trajectories,
                workspace_id=self.reme_config.workspace_id
            )
            print(f"[Summary] Async task submitted at step {global_steps}")
            return summary_task
        except Exception as e:
            print(f"[Summary] Failed to submit task: {e}")
            return None

    def _should_submit_summary(self, global_steps: int) -> bool:
        """
        Determines whether a summary task should be submitted based on configuration settings.

        Args:
            global_steps (int): The current global step count.

        Returns:
            bool: True if the summary task should be submitted, False otherwise.
        """
        return (
            self.reme_config.enable_summarizer
            and self.reme_config.updated_freq
            and global_steps % self.reme_config.updated_freq == 0
        )
    

    def collect_summary_result(self, summary_task: Optional[Future]) -> Optional[float]:
        """
        Collects the result from a submitted summary task.

        Args:
            summary_task (Optional[Future]): The Future object representing the summary task to collect.
            timeout (Optional[float]): Maximum time in seconds to wait for the task completion.
                                    Defaults to None (wait indefinitely).

        Returns:
            Optional[float]: The time cost of the summary task in seconds, or None if the task
                            is None, times out, or encounters an error.
        """
        if summary_task is None:
            return None
        try:
            print("[Summary] Waiting for task completion...")
            summarizer_response, time_cost = summary_task.result()
            print(f"[Summary] Task completed in {time_cost:.2f}s")
            return time_cost
        except Exception as e:
            print(f"[Summary] Task failed: {e}")
            return None

    def get_complete_exp_configs(self, tasks: List[Task], mode: Literal["sample", "validate"]) -> List[TaskExpConfig]:
        """
        Generates complete experience configurations for the given tasks.

        Args:
            tasks (List[Task]): A list of Task objects for which to generate configurations.
            mode (Literal["sample", "validate"]): The mode of operation, either "sample" or "validate".

        Returns:
            List[TaskExpConfig]: A list of TaskExpConfig objects with allocated training modes and experience addition settings.
        """
        exp_manager_configs = self.allocate_train_mode(tasks)
        exp_manager_configs = self.allocate_add_exp(exp_manager_configs, mode)
        return exp_manager_configs

    def allocate_train_mode(self, tasks: List[Task]) -> List[TaskExpConfig]:
        """
        Allocates training modes for the given tasks based on the configured training sample experience mode.

        Args:
            tasks (List[Task]): A list of Task objects for which to allocate training modes.

        Returns:
            List[TaskExpConfig]: A list of TaskExpConfig objects with allocated training modes.
        """
        mode_to_ratio = {
            "allkeep": 1.0,
            "alldiscard": 0.0,
            "hybrid": self.train_sample_keepratio
        }
        keep_ratio = mode_to_ratio.get(
            self.train_sample_mode, self.train_sample_keepratio
        )
        keep_count = int(len(tasks) * keep_ratio)
        exp_modes = ['keep'] * keep_count + ['discard'] * (len(tasks) - keep_count)
        random.shuffle(exp_modes)
        return [TaskExpConfig(add_exp=[], train_mode=exp_mode) for exp_mode in exp_modes]
    
    def allocate_add_exp(self, exp_configs: List[TaskExpConfig], mode: Literal["sample", "validate"]) -> List[TaskExpConfig]:
        """
        Allocates experience addition settings for the given tasks based on the mode and configured experience modes.

        Args:
            exp_configs (List[TaskExpConfig]): A list of TaskExpConfig objects to be updated.
            mode (Literal["sample", "validate"]): The mode of operation, either "sample" or "validate".

        Returns:
            List[TaskExpConfig]: An updated list of TaskExpConfig objects with allocated experience addition settings.
        """
        is_validate = mode == "validate"
        rollout_n = self.rollout_config.val_kwargs.n if is_validate else self.rollout_config.n
        exp_mode = self.val_rollout_mode if is_validate else self.train_rollout_mode
        for task_exp_config in exp_configs:
            add_exp_choices = {
                "woexp": [False] * rollout_n,
                "mixed": sorted([i < round(rollout_n*self.rollout_ratio) for i in range(rollout_n)], key=lambda _: random.random()),
                "all": [True] * rollout_n
            }[exp_mode]
            task_exp_config.add_exp = add_exp_choices
        
        return exp_configs

    # ==================== Experience Replay Methods ====================
    
    def update_difficulty2task_dict(self, trajectories: List[Trajectory]) -> None:
        """
        根据当前 step 的 rollout 结果更新 difficulty2task_dict。
        
        Difficulty 定义：在一个 training step 中，某个 task 的 n 次 rollout 中，reward=1 的次数。
        
        Args:
            trajectories: 当前 step 的所有 on-policy trajectory 列表
        """
        # 按 task_id 分组统计
        task_id_to_trajectories: Dict[str, List[Trajectory]] = defaultdict(list)
        for traj in trajectories:
            task_id = traj.task_id
            task_id_to_trajectories[task_id].append(traj)
        
        # 计算每个 task 的 difficulty（成功次数）
        for task_id, trajs in task_id_to_trajectories.items():
            success_count = sum(
                1
                for traj in trajs
                if traj.reward and float(getattr(traj.reward, "success_rate", 0.0)) > 0.0
            )
            new_difficulty = success_count
            
            # 检查 task_id 是否已经在某个难度的桶中
            old_difficulty = None
            for diff, task_list in self.difficulty2task_dict.items():
                if task_id in task_list:
                    old_difficulty = diff
                    break
            
            # 如果 task_id 已经在旧桶中，且新 difficulty 不同，需要重新归类
            if old_difficulty is not None and old_difficulty != new_difficulty:
                # 从旧桶中移除
                self.difficulty2task_dict[old_difficulty].remove(task_id)
                logger.info(f"Moved task {task_id} from difficulty {old_difficulty} to difficulty {new_difficulty}")
            
            # 如果 task_id 不在新桶中，添加到新桶
            if task_id not in self.difficulty2task_dict[new_difficulty]:
                self.difficulty2task_dict[new_difficulty].append(task_id)
                if old_difficulty is None:
                    logger.debug(f"Added task {task_id} to difficulty {new_difficulty} bucket")

    def save_trajectories_to_memory(self, trajectories: List[Trajectory]) -> None:
        """
        将轨迹及其 old_log_probs 保存到内存中的 task2trajectories。
        如果某个 task 的轨迹数量超过 max_trajectories_per_task，则根据 exp_select_mode 替换。
        
        Args:
            trajectories: 包含 old_log_probs 和 entropy 的轨迹列表
        """
        for traj in trajectories:
            task_id = traj.task_id
            if task_id not in self.task2trajectories:
                self.task2trajectories[task_id] = []
            
            # 如果当前轨迹列表已满，根据 exp_select_mode 决定是否替换
            if len(self.task2trajectories[task_id]) >= self.max_trajectories_per_task:
                # 找到当前列表中 entropy 最高（或最低）的轨迹
                current_entropies = [t.metadata.get("entropy", float('inf')) for t in self.task2trajectories[task_id]]
                
                if self.exp_select_mode == "argmin":  # 保留 entropy 最低的
                    traj_entropy = traj.metadata.get("entropy", float('inf'))
                    # 如果新轨迹的 entropy 更低，则替换掉当前最高的
                    if traj_entropy < max(current_entropies):
                        max_entropy_idx = current_entropies.index(max(current_entropies))
                        self.task2trajectories[task_id][max_entropy_idx] = traj
                        logger.debug(f"Replaced trajectory for task {task_id} with lower entropy.")
                elif self.exp_select_mode == "argmax":  # 保留 entropy 最高的
                    traj_entropy = traj.metadata.get("entropy", float('-inf'))
                    # 如果新轨迹的 entropy 更高，则替换掉当前最低的
                    if traj_entropy > min(current_entropies):
                        min_entropy_idx = current_entropies.index(min(current_entropies))
                        self.task2trajectories[task_id][min_entropy_idx] = traj
                        logger.debug(f"Replaced trajectory for task {task_id} with higher entropy.")
                else:  # 默认 FIFO
                    self.task2trajectories[task_id].pop(0)  # 移除最旧的轨迹
                    self.task2trajectories[task_id].append(traj)
                    logger.debug(f"Replaced trajectory for task {task_id} using FIFO.")
            else:
                self.task2trajectories[task_id].append(traj)

    def get_offpolicy_trajectories_from_memory(
        self, 
        task_id: str, 
        num_trajectories: int = 1,
        use_saved_entropy: bool = True
    ) -> List[Trajectory]:
        """
        从内存中的 task2trajectories 获取指定任务的 off-policy trajectory。
        根据 exp_select_mode 选择轨迹。
        
        Args:
            task_id: 任务 ID
            num_trajectories: 获取的轨迹数量
            use_saved_entropy: 是否使用保存时的 entropy 进行选择
                - True: 使用保存时的 entropy（默认，快速但可能不是当前 policy 最优）
                - False: 返回所有候选轨迹，由调用方使用当前 policy 计算 entropy 后选择
            
        Returns:
            List[Trajectory]: Off-policy trajectory 列表
        """
        import copy
        available_trajectories = self.task2trajectories.get(task_id, [])
        if not available_trajectories:
            return []
        
        # 深拷贝以避免修改原始轨迹
        available_trajectories = [copy.deepcopy(t) for t in available_trajectories]
        
        if use_saved_entropy:
            # 使用保存时的 entropy 进行选择
            if self.exp_select_mode == "argmin":  # 选择 entropy 最低的
                available_trajectories.sort(key=lambda t: t.metadata.get("entropy", float('inf')))
            elif self.exp_select_mode == "argmax":  # 选择 entropy 最高的
                available_trajectories.sort(key=lambda t: t.metadata.get("entropy", float('-inf')), reverse=True)
            # 默认或 random 模式下，不排序，直接随机选择
            
            # 采样 num_trajectories 个轨迹
            sampled_trajectories = available_trajectories[:min(num_trajectories, len(available_trajectories))]
        else:
            # 返回所有候选轨迹，由调用方使用当前 policy 计算 entropy 后选择
            sampled_trajectories = available_trajectories
        
        # 标记为 experience replay
        # ⭐ 注意：不再设置 author="llm(do_not_train)"
        # Off-policy LLM 消息保持 author="llm"，参与 off-policy loss 计算
        # 在 convert_offpolicy_to_cmt 中会强制设置 author="llm"
        # 使用 exp_mask=1 来区分 on-policy 和 off-policy 数据
        for traj in sampled_trajectories:
            traj.metadata["is_experience_replay"] = True
        
        return sampled_trajectories
    
    def get_all_candidate_trajectories(
        self,
        task_id: str,
    ) -> List[Trajectory]:
        """
        获取指定任务的所有候选 off-policy trajectories，不进行排序。
        用于后续使用当前 policy 计算 entropy 后选择最优轨迹。
        
        Args:
            task_id: 任务 ID
            
        Returns:
            List[Trajectory]: 所有候选轨迹列表
        """
        return self.get_offpolicy_trajectories_from_memory(
            task_id=task_id,
            num_trajectories=999999,  # 获取所有
            use_saved_entropy=False
        )

    def update_skip_uid_set_and_filter_trajectories(
        self,
        trajectories: List[Trajectory],
        n_rollout: int,
        entropys: Optional[torch.Tensor] = None,
        response_mask: Optional[torch.Tensor] = None,
    ) -> List[Trajectory]:
        """
        根据 rollout 结果更新 skip_uid_set，并筛选符合条件的轨迹（非全对非全错，且选择 entropy 最低的成功轨迹）。
        
        只统计 on-policy 成功次数。判断"全部成功"时，使用实际的 on-policy rollout 数量（即该 task 的轨迹数）。
        
        Args:
            trajectories: 当前 step 的所有 on-policy trajectory 列表
            n_rollout: 基准 rollout 数量（用于 experience_rbound 判断）
            entropys: 当前 step 所有 on-policy 轨迹的 token 级 entropy (bs, response_len)
            response_mask: 当前 step 所有 on-policy 轨迹的 response mask (bs, response_len)
            
        Returns:
            List[Trajectory]: 筛选后符合条件的轨迹列表，用于保存到 task2trajectories
        """
        filtered_trajectories_to_save = []
        
        # 按 task_id 分组统计
        task_id_to_trajectories: Dict[str, List[Trajectory]] = defaultdict(list)
        task_id_to_entropy_list: Dict[str, List[Tuple[float, Trajectory]]] = defaultdict(list)
        
        for i, traj in enumerate(trajectories):
            task_id = traj.task_id
            task_id_to_trajectories[task_id].append(traj)
            
            # 计算轨迹的平均 entropy
            if entropys is not None and response_mask is not None and i < entropys.shape[0]:
                traj_entropys = entropys[i].cpu().numpy()
                traj_response_mask = response_mask[i].cpu().numpy()
                valid_entropys = traj_entropys[traj_response_mask.astype(bool)]
                avg_entropy = float(np.mean(valid_entropys)) if len(valid_entropys) > 0 else 0.0
            else:
                avg_entropy = 0.0
            
            traj.metadata["entropy"] = avg_entropy  # 保存平均 entropy 到 metadata
            task_id_to_entropy_list[task_id].append((avg_entropy, traj))
        
        for task_id, trajs in task_id_to_trajectories.items():
            success_count = sum(
                1
                for traj in trajs
                if traj.reward and float(getattr(traj.reward, "success_rate", 0.0)) > 0.0
            )
            # ⭐ 使用实际的 on-policy rollout 数量，而不是基准 n_rollout
            # 对于 experience task，实际 rollout 数量 = n_rollout - n_offpolicy
            actual_rollout_count = len(trajs)
            
            # 1. 更新 skip_uid_set
            if success_count == actual_rollout_count:  # 全部做对（基于实际 rollout 数量）
                if task_id not in self.skip_uid_set:
                    self.skip_uid_set.add(task_id)
                    # 从 difficulty2task_dict 中移除
                    for diff, task_list in list(self.difficulty2task_dict.items()):
                        if task_id in task_list:
                            self.difficulty2task_dict[diff].remove(task_id)
                            if not self.difficulty2task_dict[diff]:
                                del self.difficulty2task_dict[diff]  # 如果桶为空则删除
                    # 从 task2trajectories 中删除
                    if task_id in self.task2trajectories:
                        del self.task2trajectories[task_id]
                    logger.info(f"Task {task_id} fully solved, added to skip_uid_set and removed from replay pool.")
                continue  # 不再考虑加入经验池
            elif task_id in self.skip_uid_set:  # 如果之前在 skip_uid_set 但现在没全对，则移除
                self.skip_uid_set.remove(task_id)
                logger.info(f"Task {task_id} no longer fully solved, removed from skip_uid_set.")
            
            # 2. 筛选加入 experience replay pool 的 tasks (非全对非全错)
            if self.experience_lbound < success_count < self.experience_rbound:
                # ⭐ ExGRPO 设计：存储所有 reward 为正的 trajectories，取用时再选最优的
                # 不再在存储时只选择一个最优的
                successful_trajs = [
                    t
                    for e, t in task_id_to_entropy_list[task_id]
                    if t.reward and float(getattr(t.reward, "success_rate", 0.0)) > 0.0
                ]
                if successful_trajs:
                    # 将所有成功的轨迹都加入待保存列表
                    filtered_trajectories_to_save.extend(successful_trajs)
                    logger.debug(f"Task {task_id}: adding {len(successful_trajs)} successful trajectories to experience pool")
            else:
                logger.debug(f"Task {task_id} (success_count={success_count}) not within bounds [{self.experience_lbound}, {self.experience_rbound}], skipping for experience pool.")
        
        return filtered_trajectories_to_save

    def sample_tasks_from_replaypool(
        self,
        difficulty: Optional[int] = None,
        num_tasks: int = 2,
        strategy: str = "random"
    ) -> List[str]:
        """
        从 replaytaskpool 中采样指定数量的 task_id。
        
        Args:
            difficulty: 指定的难度值。如果为 None，则随机选择一个难度
            num_tasks: 需要采样的 task 数量
            strategy: 采样策略，"random" 或 "uniform"（按难度均匀采样）
            
        Returns:
            List[str]: 采样得到的 task_id 列表
        """
        if not self.difficulty2task_dict:
            return []
        
        if difficulty is None:
            # 随机选择一个难度
            available_difficulties = [d for d, tasks in self.difficulty2task_dict.items() if len(tasks) > 0]
            if not available_difficulties:
                return []
            difficulty = random.choice(available_difficulties)
        
        # 从指定难度的 task 列表中采样
        available_tasks = self.difficulty2task_dict.get(difficulty, [])
        if len(available_tasks) == 0:
            return []
        
        # 采样（允许重复）
        sampled_tasks = random.choices(available_tasks, k=min(num_tasks, len(available_tasks)))
        return sampled_tasks

    def get_offpolicy_batch(
        self, 
        tasks: List[Task], 
        num_trajectories_per_task: int = 1,
        use_saved_entropy: bool = True
    ) -> List[Trajectory]:
        """
        为给定的任务列表从内存中获取 off-policy trajectory。
        同时更新每个 task 的 metadata["n_offpolicy_trajectories"] 为实际获取的数量。
        
        Args:
            tasks: 任务列表
            num_trajectories_per_task: 每个任务期望获取的轨迹数量
            use_saved_entropy: 是否使用保存时的 entropy 进行选择
                - True: 使用保存时的 entropy（默认，快速但可能不是当前 policy 最优）
                - False: 返回所有候选轨迹，由调用方使用当前 policy 计算 entropy 后选择
            
        Returns:
            List[Trajectory]: Off-policy trajectory 列表
        """
        offpolicy_trajectories = []
        
        for task in tasks:
            try:
                trajs = self.get_offpolicy_trajectories_from_memory(
                    task_id=task.task_id,
                    num_trajectories=num_trajectories_per_task,
                    use_saved_entropy=use_saved_entropy
                )
                offpolicy_trajectories.extend(trajs)
                
                # ⭐ 更新 task 的 n_offpolicy_trajectories 为实际获取的数量
                # 这样 rollout 时可以根据实际数量调整 on-policy rollout 数量
                actual_count = len(trajs)
                if hasattr(task, 'metadata') and task.metadata:
                    task.metadata["n_offpolicy_trajectories"] = actual_count
                else:
                    task.metadata = {"n_offpolicy_trajectories": actual_count}
                    
                if actual_count < num_trajectories_per_task:
                    logger.debug(
                        f"Task {task.task_id}: requested {num_trajectories_per_task} off-policy trajectories, "
                        f"but only got {actual_count}"
                    )
            except Exception as e:
                logger.warning(f"Failed to get off-policy trajectory for task {task.task_id}: {e}")
                # 获取失败时，设置为 0
                if hasattr(task, 'metadata') and task.metadata:
                    task.metadata["n_offpolicy_trajectories"] = 0
                else:
                    task.metadata = {"n_offpolicy_trajectories": 0}
                continue
        
        return offpolicy_trajectories
    
    def get_all_candidates_batch(
        self, 
        tasks: List[Task]
    ) -> Dict[str, List[Trajectory]]:
        """
        为给定的任务列表获取所有候选 off-policy trajectories。
        用于后续使用当前 policy 计算 entropy 后选择最优轨迹。
        
        Args:
            tasks: 任务列表
            
        Returns:
            Dict[str, List[Trajectory]]: task_id -> 候选轨迹列表的映射
        """
        task_to_candidates: Dict[str, List[Trajectory]] = {}
        
        for task in tasks:
            try:
                candidates = self.get_all_candidate_trajectories(task_id=task.task_id)
                if candidates:
                    task_to_candidates[task.task_id] = candidates
            except Exception as e:
                logger.warning(f"Failed to get candidate trajectories for task {task.task_id}: {e}")
                continue
        
        return task_to_candidates
    
    def get_valid_replay_task_ids(self) -> List[str]:
        """
        获取所有有可用轨迹的 replay task_ids（排除 skip_uid_set 中的）。
        
        Returns:
            List[str]: 有效的 task_id 列表
        """
        valid_task_ids = []
        for difficulty, task_ids in self.difficulty2task_dict.items():
            for task_id in task_ids:
                if task_id not in self.skip_uid_set:
                    if task_id in self.task2trajectories and len(self.task2trajectories[task_id]) > 0:
                        valid_task_ids.append(task_id)
        return valid_task_ids

    # ==================== Teacher Experience Replay Methods (LUFFY) ====================
    
    def load_teacher_trajectories(self, data_path: str) -> int:
        """
        从磁盘加载预采集的 Teacher 轨迹。
        
        支持格式：
        - JSONL：每行一个轨迹（推荐）
        - Pickle：序列化的轨迹列表
        
        Args:
            data_path: Teacher 轨迹文件路径
            
        Returns:
            加载的轨迹数量
        """
        import json
        import pickle
        import os
        
        if not os.path.exists(data_path):
            logger.warning(f"[ExperienceManager] Teacher trajectory file not found: {data_path}")
            return 0
        
        count = 0
        
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        traj_dict = json.loads(line.strip())
                        traj = self._dict_to_teacher_trajectory(traj_dict)
                        self.teacher_task2trajectories[traj.task_id].append(traj)
                        count += 1
                    except Exception as e:
                        logger.warning(f"[ExperienceManager] Failed to parse teacher trajectory: {e}")
                        continue
                        
        elif data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                trajectories = pickle.load(f)
                for item in trajectories:
                    # ⭐ 支持两种格式：
                    # 1. Trajectory 对象（直接使用）
                    # 2. 字典（由 filter_teacher_trajectories.py 生成，需要转换）
                    if isinstance(item, dict):
                        # 字典格式：转换为 Trajectory 对象
                        traj = self._dict_to_teacher_trajectory(item)
                    else:
                        # Trajectory 对象：直接使用
                        traj = item
                        if not hasattr(traj, 'metadata') or traj.metadata is None:
                            traj.metadata = {}
                        traj.metadata["is_teacher"] = True
                    
                    self.teacher_task2trajectories[traj.task_id].append(traj)
                    count += 1
        else:
            raise ValueError(f"Unsupported file format: {data_path}. Use .jsonl or .pkl")
        
        logger.info(f"[ExperienceManager] Loaded {count} teacher trajectories from {data_path}")
        logger.info(f"[ExperienceManager] Teacher tasks: {len(self.teacher_task2trajectories)}")
        
        return count
    
    def _dict_to_teacher_trajectory(self, traj_dict: Dict) -> Trajectory:
        """
        将字典转换为 Trajectory 对象。
        
        ⭐ Multi-turn 关键：保持与 on-policy 轨迹相同的结构
        ⭐ Log Prob 处理：根据数据中是否有 log_prob 设置 has_log_prob 标记
        """
        from agentevolver.schema.trajectory import Reward
        
        # 获取 task_id（用于后续存储）
        task_id = traj_dict.get("task_id", "")
        
        # 创建 Trajectory 对象
        # 注意：Trajectory 类没有 task_id 属性，使用 data_id 存储
        traj = Trajectory(
            data_id=traj_dict.get("data_id", task_id),  # 使用 task_id 作为 data_id
            rollout_id=traj_dict.get("rollout_id", ""),
            steps=traj_dict.get("messages", []),  # ⭐ Multi-turn 对话历史
        )
        
        # ⭐ 将 task_id 存储到 metadata 中
        traj.task_id = task_id  # 动态添加属性
        
        # 设置 reward
        reward_val = traj_dict.get("reward", 0.0)
        # Default success heuristic: positive reward indicates success.
        # (If traj_dict already has an explicit success flag, respect it.)
        success = traj_dict.get("success", reward_val > 0)
        traj.reward = Reward(
            outcome=reward_val if success else 0.0,
            success_rate=1.0 if success else 0.0,
        )
        
        # 设置 metadata
        traj.metadata = traj_dict.get("metadata", {})
        traj.metadata["is_teacher"] = True
        traj.metadata["is_experience_replay"] = True  # 标记为 off-policy
        traj.metadata["teacher_model"] = traj_dict.get("teacher_model", 
                                                        traj.metadata.get("teacher_model", "unknown"))
        
        # ⭐ 处理 log_prob（关键）
        # 检查轨迹数据中是否包含 log_prob
        log_probs = (traj_dict.get("log_probs") or 
                     traj_dict.get("metadata", {}).get("old_log_probs"))
        
        if log_probs and len(log_probs) > 0:
            # ⭐ 保存到 log_probs 字段（供 _align_teacher_log_probs 使用）
            # 注意：这是累积的 log_probs，只包含 LLM 生成的 token
            traj.metadata["log_probs"] = log_probs
            traj.metadata["has_log_prob"] = True
            
            # 也保存分轮的 log_probs（如果有）
            log_probs_per_turn = traj_dict.get("log_probs_per_turn")
            if log_probs_per_turn:
                traj.metadata["log_probs_per_turn"] = log_probs_per_turn
        else:
            traj.metadata["has_log_prob"] = False
        
        return traj
    
    def get_teacher_trajectories(
        self, 
        task_ids: List[str],
        num_per_task: int = 1,
    ) -> List[Trajectory]:
        """
        获取指定 task 的 Teacher 轨迹。
        
        ⭐ 支持多种选择模式：
        - "all": 返回所有轨迹
        - "first": 返回前 N 个
        - "random": 随机选择 N 个
        - "entropy": 按 entropy 从低到高排序
                     * 对于有 log_prob 的轨迹：直接计算 entropy 并排序
                     * 对于没有 log_prob 的轨迹：在 trainer 中使用当前 policy 计算 entropy
                     * 两部分合并后按 entropy 从低到高排序
        - "confidence": 按 Teacher 的平均 log_prob 从高到低排序（越高 = 越确信）
                        ⭐ 适用于 OpenAI 等 API 返回的 log_prob
                        ⭐ 只能对有 log_prob 的轨迹排序
        
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
                # 只处理有 log_prob 的轨迹，没有 log_prob 的轨迹需要在 trainer 中处理
                trajs_with_logprob = []
                trajs_without_logprob = []
                
                for traj in task_trajs:
                    # 检查 log_probs 或 old_log_probs（兼容两种命名）
                    log_probs = (traj.metadata.get("log_probs") or 
                                traj.metadata.get("old_log_probs"))
                    if traj.metadata.get("has_log_prob", False) and log_probs:
                        trajs_with_logprob.append(traj)
                    else:
                        trajs_without_logprob.append(traj)
                
                # 对于有 log_prob 的轨迹，计算 "entropy"（实际是负平均 log_prob）
                # ⭐ 注意：这里的 "entropy" 实际上是 -mean(log_probs)，不是真正的信息论 entropy
                # 真正的 entropy H = -Σ P(x) * log P(x) 需要完整的概率分布
                # 但 API 只返回被选中 token 的 log_prob，无法计算真正的 entropy
                # 
                # 这里使用 -mean(log_probs) 作为 "不确定性" 的代理指标：
                # - 值越低 → Teacher 对这个响应越确信（log_prob 越接近 0）
                # - 值越高 → Teacher 对这个响应不确定（log_prob 越负）
                # 
                # 与 confidence 的关系：entropy ≈ -confidence
                if trajs_with_logprob:
                    traj_entropys = []
                    for traj in trajs_with_logprob:
                        log_probs = (traj.metadata.get("log_probs") or 
                                    traj.metadata.get("old_log_probs"))
                        old_log_probs = np.array(log_probs)
                        # ⭐ 使用 -mean(log_probs) 作为 "entropy" 代理
                        # 这与 confidence = mean(log_probs) 相反
                        entropy_proxy = -np.mean(old_log_probs)
                        traj_entropys.append((entropy_proxy, traj))
                    
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
            elif self.teacher_select_mode == "confidence":
                # ⭐ Confidence 模式：按 Teacher 的平均 log_prob 从高到低排序
                # 适用于 OpenAI 等 API 返回的 log_prob
                # 越高的平均 log_prob 意味着 Teacher 对这个响应越确信
                trajs_with_logprob = []
                trajs_without_logprob = []
                
                for traj in task_trajs:
                    # 检查是否有 log_probs（注意：可能是 log_probs 或 old_log_probs）
                    log_probs = (traj.metadata.get("log_probs") or 
                                traj.metadata.get("old_log_probs"))
                    if log_probs and len(log_probs) > 0:
                        trajs_with_logprob.append(traj)
                    else:
                        trajs_without_logprob.append(traj)
                
                # 对于有 log_prob 的轨迹，计算平均 log_prob
                if trajs_with_logprob:
                    traj_confidences = []
                    for traj in trajs_with_logprob:
                        log_probs = (traj.metadata.get("log_probs") or 
                                    traj.metadata.get("old_log_probs"))
                        log_probs = np.array(log_probs)
                        # 平均 log_prob（越高越好）
                        avg_log_prob = np.mean(log_probs)
                        traj_confidences.append((avg_log_prob, traj))
                    
                    # 按平均 log_prob 从高到低排序（reverse=True）
                    trajs_with_logprob = [traj for _, traj in sorted(traj_confidences, key=lambda x: x[0], reverse=True)]
                    
                    if len(traj_confidences) > 0:
                        best_conf = traj_confidences[0][0] if traj_confidences else 0
                        worst_conf = traj_confidences[-1][0] if traj_confidences else 0
                        logger.debug(
                            f"Task {task_id}: {len(trajs_with_logprob)} trajectories with log_prob "
                            f"(sorted by confidence: best={best_conf:.4f}, worst={worst_conf:.4f})"
                        )
                
                # 合并：先是有 log_prob 的（按 confidence 排序），然后是没有的
                selected = trajs_with_logprob + trajs_without_logprob
                selected = selected[:num_per_task]
                
                if trajs_without_logprob:
                    logger.debug(
                        f"Task {task_id}: {len(trajs_without_logprob)} trajectories without log_prob "
                        f"(cannot compute confidence, placed at end)"
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
        获取所有有 Teacher 轨迹的 task_id。
        
        排除已在 skip_uid_set 中的 task（已完全解决）。
        
        Returns:
            List[str]: 有效的 Teacher task_id 列表
        """
        valid_ids = [
            task_id for task_id in self.teacher_task2trajectories.keys()
            if task_id not in self.skip_uid_set
        ]
        return valid_ids
    
    def has_teacher_trajectory(self, task_id: str) -> bool:
        """
        检查某个 task 是否有 Teacher 轨迹。
        
        Args:
            task_id: 任务 ID
            
        Returns:
            bool: 是否有 Teacher 轨迹
        """
        return (task_id in self.teacher_task2trajectories 
                and len(self.teacher_task2trajectories[task_id]) > 0)
    
    def get_teacher_stats(self) -> Dict:
        """
        获取 Teacher 轨迹统计信息。
        
        Returns:
            Dict: 统计信息
        """
        total_tasks = len(self.teacher_task2trajectories)
        total_trajectories = sum(len(trajs) for trajs in self.teacher_task2trajectories.values())
        
        # 统计有/无 log_prob 的轨迹数量
        with_log_prob = 0
        without_log_prob = 0
        for trajs in self.teacher_task2trajectories.values():
            for traj in trajs:
                if traj.metadata.get("has_log_prob", False):
                    with_log_prob += 1
                else:
                    without_log_prob += 1
        
        return {
            "enabled": self.teacher_enabled,
            "total_tasks": total_tasks,
            "total_trajectories": total_trajectories,
            "avg_trajectories_per_task": total_trajectories / total_tasks if total_tasks > 0 else 0,
            "with_log_prob": with_log_prob,
            "without_log_prob": without_log_prob,
        }
    
    def get_teacher_offpolicy_batch(
        self, 
        tasks: List[Task], 
        num_trajectories_per_task: int = 1,
    ) -> List[Trajectory]:
        """
        为给定的任务列表获取 Teacher off-policy 轨迹。
        
        ⭐ 与 get_offpolicy_batch 类似，但从 teacher_task2trajectories 获取
        
        Args:
            tasks: 任务列表
            num_trajectories_per_task: 每个任务期望获取的轨迹数量
            
        Returns:
            List[Trajectory]: Teacher off-policy trajectory 列表
        """
        task_ids = [task.task_id for task in tasks]
        return self.get_teacher_trajectories(
            task_ids=task_ids,
            num_per_task=num_trajectories_per_task,
        )

    def get_teacher_rollouts_for_luffy_mixing(
        self,
        tasks: List[Task],
        n_teacher_rollouts_per_task: int = 1,
    ) -> Dict[str, List[Trajectory]]:
        """
        ⭐ LUFFY 风格：为每个 task 获取 teacher rollouts 用于 rollout 级别混合。
        
        与 get_teacher_offpolicy_batch 的区别：
        - get_teacher_offpolicy_batch: 返回扁平化的轨迹列表（Task 级别混合）
        - get_teacher_rollouts_for_luffy_mixing: 返回 task_id -> trajectories 的字典（Rollout 级别混合）
        
        Args:
            tasks: 任务列表（batch 中的所有 tasks）
            n_teacher_rollouts_per_task: 每个 task 需要混入的 teacher rollout 数量
            
        Returns:
            Dict[str, List[Trajectory]]: task_id -> teacher trajectories 的映射
            - 只返回有 teacher 轨迹的 task
            - 每个 task 最多返回 n_teacher_rollouts_per_task 条轨迹
        """
        result: Dict[str, List[Trajectory]] = {}
        
        for task in tasks:
            task_id = task.task_id
            
            # 检查该 task 是否有 teacher 轨迹
            if task_id not in self.teacher_task2trajectories:
                continue
            
            teacher_trajs = self.teacher_task2trajectories[task_id]
            if not teacher_trajs:
                continue
            
            # 选择 teacher 轨迹
            if len(teacher_trajs) <= n_teacher_rollouts_per_task:
                selected = teacher_trajs.copy()
            elif self.teacher_select_mode == "random":
                import random
                selected = random.sample(teacher_trajs, n_teacher_rollouts_per_task)
            elif self.teacher_select_mode == "first":
                selected = teacher_trajs[:n_teacher_rollouts_per_task]
            else:
                # 默认随机选择
                import random
                selected = random.sample(teacher_trajs, n_teacher_rollouts_per_task)
            
            # 为每条轨迹添加元数据
            for traj in selected:
                traj.metadata["is_teacher"] = True
                traj.metadata["source"] = "teacher"
                # 确保 data_id 与 task_id 一致（用于 GRPO 分组）
                if not traj.data_id:
                    traj.data_id = task_id
            
            result[task_id] = selected
            
        logger.info(
            f"[LUFFY] Got teacher rollouts for {len(result)}/{len(tasks)} tasks, "
            f"{n_teacher_rollouts_per_task} rollouts per task"
        )
        
        return result

    def save_experience_pool_to_disk(self, save_dir: str) -> None:
        """
        将 experience pool 保存到磁盘，用于断点续训。
        
        Args:
            save_dir: 保存目录路径
        """
        import os
        import json
        import pickle
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存 difficulty2task_dict
        difficulty2task_dict_path = os.path.join(save_dir, "difficulty2task_dict.json")
        with open(difficulty2task_dict_path, "w") as f:
            json.dump({str(k): v for k, v in self.difficulty2task_dict.items()}, f, indent=2)
        
        # 保存 skip_uid_set
        skip_uid_set_path = os.path.join(save_dir, "skip_uid_set.json")
        with open(skip_uid_set_path, "w") as f:
            json.dump(list(self.skip_uid_set), f, indent=2)
        
        # 保存 task2trajectories (使用 pickle 以保存复杂对象)
        task2trajectories_path = os.path.join(save_dir, "task2trajectories.pkl")
        with open(task2trajectories_path, "wb") as f:
            pickle.dump(self.task2trajectories, f)
        
        logger.info(f"Experience pool saved to {save_dir}")

    def load_experience_pool_from_disk(self, load_dir: str) -> None:
        """
        从磁盘加载 experience pool，用于断点续训。
        
        Args:
            load_dir: 加载目录路径
        """
        import os
        import json
        import pickle
        
        # 加载 difficulty2task_dict
        difficulty2task_dict_path = os.path.join(load_dir, "difficulty2task_dict.json")
        if os.path.exists(difficulty2task_dict_path):
            with open(difficulty2task_dict_path, "r") as f:
                loaded_dict = json.load(f)
                self.difficulty2task_dict = defaultdict(list)
                for k, v in loaded_dict.items():
                    self.difficulty2task_dict[int(k)] = v
            logger.info(f"Loaded difficulty2task_dict with {len(self.difficulty2task_dict)} difficulty buckets")
        
        # 加载 skip_uid_set
        skip_uid_set_path = os.path.join(load_dir, "skip_uid_set.json")
        if os.path.exists(skip_uid_set_path):
            with open(skip_uid_set_path, "r") as f:
                self.skip_uid_set = set(json.load(f))
            logger.info(f"Loaded skip_uid_set with {len(self.skip_uid_set)} task_ids")
        
        # 加载 task2trajectories
        task2trajectories_path = os.path.join(load_dir, "task2trajectories.pkl")
        if os.path.exists(task2trajectories_path):
            with open(task2trajectories_path, "rb") as f:
                self.task2trajectories = pickle.load(f)
            total_trajs = sum(len(trajs) for trajs in self.task2trajectories.values())
            logger.info(f"Loaded task2trajectories with {len(self.task2trajectories)} tasks and {total_trajs} trajectories")

    def update_difficulty2task_dict(self, trajectories: List[Trajectory]) -> None:
        """
        根据当前 rollout 的结果更新 difficulty2task_dict。
        统计每个 task 的成功次数，并将 task 放入对应的 difficulty bucket。
        
        Args:
            trajectories: 当前 step 的所有 on-policy trajectory 列表
        """
        # 按 task_id 分组统计成功次数
        task_id_to_success_count: Dict[str, int] = defaultdict(int)
        task_ids_seen: Set[str] = set()
        
        for traj in trajectories:
            task_id = traj.task_id
            task_ids_seen.add(task_id)
            if traj.reward and float(getattr(traj.reward, "success_rate", 0.0)) > 0.0:
                task_id_to_success_count[task_id] += 1
        
        # 更新 difficulty2task_dict
        for task_id in task_ids_seen:
            success_count = task_id_to_success_count.get(task_id, 0)
            
            # 首先从旧的 difficulty bucket 中移除
            for diff, task_list in list(self.difficulty2task_dict.items()):
                if task_id in task_list:
                    self.difficulty2task_dict[diff].remove(task_id)
                    if not self.difficulty2task_dict[diff]:
                        del self.difficulty2task_dict[diff]  # 如果桶为空则删除
                    break
            
            # 如果不在 skip_uid_set 中，则加入新的 difficulty bucket
            if task_id not in self.skip_uid_set:
                self.difficulty2task_dict[success_count].append(task_id)
                logger.debug(f"Task {task_id} moved to difficulty bucket {success_count}")

    # ==================== RePO (Replay-Enhanced Policy Optimization) Methods ====================
    
    def get_repo_manager(self):
        """
        Get or create the RePO Replay Manager.
        
        Returns:
            RePOReplayManager instance
        """
        if self._repo_manager is None:
            from agentevolver.module.exp_manager.experience_collate import RePOReplayManager
            self._repo_manager = RePOReplayManager(
                max_trajectories_per_task=self.repo_max_trajectories_per_task,
                buffer_update_mode=self.repo_buffer_update_mode,
                replay_strategy=self.repo_replay_strategy,
                g_off=self.repo_g_off,
            )
        return self._repo_manager
    
    def get_repo_mixer(self):
        """
        Get or create the RePO Rollout Mixer.
        
        Returns:
            RePORolloutMixer instance
        """
        if self._repo_mixer is None:
            from agentevolver.module.exp_manager.experience_collate import RePORolloutMixer
            self._repo_mixer = RePORolloutMixer(
                repo_manager=self.get_repo_manager(),
                replay_start_epoch=self.repo_replay_start_epoch,
                g_on=self.repo_g_on,
                g_off=self.repo_g_off,
            )
        return self._repo_mixer
    
    def repo_save_all_trajectories(
        self,
        trajectories: List[Trajectory],
        global_step: int,
    ) -> int:
        """
        Save ALL on-policy trajectories to RePO buffer.
        
        ⭐ Key difference from ExGRPO:
        - ExGRPO: Only saves successful trajectories (reward > 0)
        - RePO: Saves ALL trajectories, uses strategy to select later
        
        Args:
            trajectories: List of on-policy trajectories
            global_step: Current global training step
            
        Returns:
            Number of trajectories saved
        """
        if not self.repo_enabled:
            return 0
        
        return self.get_repo_manager().save_all_trajectories(
            trajectories=trajectories,
            global_step=global_step,
        )
    
    def repo_get_offpolicy_trajectories(
        self,
        task_id: str,
        k: Optional[int] = None,
    ) -> List[Trajectory]:
        """
        Get off-policy trajectories for a task from RePO buffer.
        
        Args:
            task_id: Task ID
            k: Number of trajectories to get (default: g_off)
            
        Returns:
            List of selected trajectories
        """
        if not self.repo_enabled:
            return []
        
        return self.get_repo_manager().get_offpolicy_trajectories(
            task_id=task_id,
            k=k,
        )
    
    def repo_should_enable_replay(self, current_epoch: int) -> bool:
        """
        Check if RePO replay should be enabled for current epoch.
        
        Args:
            current_epoch: Current training epoch (0-indexed)
            
        Returns:
            bool: True if replay should be enabled
        """
        if not self.repo_enabled:
            return False
        
        return self.get_repo_mixer().should_enable_replay(current_epoch)
    
    def repo_mix_trajectories(
        self,
        on_policy_cmt_array: List,
        tasks: List,
        env_manager,
        config,
        tokenizer,
        current_epoch: int,
    ) -> Tuple[List, Dict]:
        """
        Mix on-policy and off-policy trajectories for RePO training.
        
        Args:
            on_policy_cmt_array: On-policy CMT objects
            tasks: Current batch tasks
            env_manager: EnvManager instance
            config: Config object
            tokenizer: Tokenizer
            current_epoch: Current training epoch
            
        Returns:
            Tuple of (mixed_cmt_array, stats_dict)
        """
        if not self.repo_enabled:
            return on_policy_cmt_array, {"repo_enabled": False}
        
        return self.get_repo_mixer().mix_trajectories(
            on_policy_cmt_array=on_policy_cmt_array,
            tasks=tasks,
            env_manager=env_manager,
            config=config,
            tokenizer=tokenizer,
            current_epoch=current_epoch,
        )
    
    def repo_get_stats(self) -> Dict:
        """
        Get RePO buffer statistics.
        
        Returns:
            Dict with buffer statistics
        """
        if not self.repo_enabled:
            return {"repo_enabled": False}
        
        stats = self.get_repo_manager().get_stats()
        stats["repo_enabled"] = True
        stats["replay_start_epoch"] = self.repo_replay_start_epoch
        stats["advantage_mode"] = self.repo_advantage_mode
        return stats


class ExperienceWorker(object):
    def __init__(self, config: DictConfig):
        """
        Initializes the ExperienceWorker with the provided configuration.

        Args:
            config (DictConfig): Configuration settings for the experience worker.
        """
        self.config: DictConfig = config
        self.experience_template = self.config.exp_manager.experience_template
    
    def manage_rollout_context(self, init_messages: List[dict], traj_exp_config: TrajExpConfig) -> Tuple[List[dict], TrajExpConfig]:
        """
        Manages the context for the rollout phase, potentially adding historical experience.

        Args:
            init_messages (List[dict]): Initial messages for the rollout.
            traj_exp_config (TrajExpConfig): Configuration for the trajectory experience.

        Returns:
            Tuple[List[dict], TrajExpConfig]: Updated messages and modified trajectory experience config.
        """
        # check experience conditions
        if not self._should_process_experience(traj_exp_config):
            return init_messages, traj_exp_config
        
        # initialize em client
        self._ensure_em_client()
        
        # construct trajectory
        trajectory = Trajectory(
            data_id=traj_exp_config.data_id,
            rollout_id=traj_exp_config.rollout_id,
            steps=init_messages,
            query=traj_exp_config.query
        )

        # retrieve experience
        reme_config = self.config.exp_manager.reme
        history_experience = self.em_client.call_context_generator(
            trajectory=trajectory,
            retrieve_top_k=reme_config.retrieve_top_k,
            workspace_id=reme_config.workspace_id
        )

        # check empty condition
        if not history_experience:
            logger.info("Experience is empty!")
            return init_messages, traj_exp_config

        # apply experience to trajectory
        logger.info(f"Retrieved history experience: {history_experience}")
        formatted_experience = self.experience_template.format(history_experience)
        new_content = formatted_experience + trajectory.steps[-1]["content"]
        trajectory.steps[-1]["content"] = new_content
        traj_exp_config.experience_list = traj_exp_config.experience_list + [formatted_experience]

        return trajectory.steps, traj_exp_config
    
    def _should_process_experience(self, traj_exp_config: TrajExpConfig) -> bool:
        """
        Checks if experience processing should be performed.

        Args:
            traj_exp_config (TrajExpConfig): Configuration for the trajectory experience.

        Returns:
            bool: True if experience should be processed, False otherwise.
        """
        return (traj_exp_config.add_exp and
                self.config.exp_manager.reme.enable_context_generator)
    
    def _ensure_em_client(self) -> None:
        """
        Initializes the EM client if it doesn't exist.
        """
        if not hasattr(self, 'em_client'):
            self.em_client = EMClient(
                base_url=self.config.exp_manager.reme.base_url
            )



    def manage_training_context(self, message: str, metadata_config: Dict) -> Tuple[str, str]:
        """
        Extracts and removes experience information from the given message.

        Args:
            message (str): Input message potentially containing experience information.
            metadata_config (Dict): Configuration for the trajectory experience.

        Returns:
            Tuple[str, str]: Extracted experience and the message with experience information removed.
        """
        experience = ""
        cleaned_message = message

        if metadata_config.get("task_train_mode", "discard") == "discard": 
            pattern = re.escape(self.experience_template).replace(r'\{\}', '(.*?)')
            match = re.search(pattern, message, re.DOTALL)
            if match:
                experience = match.group(1)
                cleaned_message = re.sub(pattern, '', message, flags=re.DOTALL)

        
        return experience, cleaned_message

