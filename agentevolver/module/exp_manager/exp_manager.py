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
        
        # ⭐ Experience Replay 相关属性
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
        
        # 计算每个 task 的 difficulty（reward=1 的数量）
        for task_id, trajs in task_id_to_trajectories.items():
            success_count = sum(1 for traj in trajs if traj.reward and traj.reward.outcome == 1.0)
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
            success_count = sum(1 for traj in trajs if traj.reward and traj.reward.outcome == 1.0)
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
                    t for e, t in task_id_to_entropy_list[task_id] 
                    if t.reward and t.reward.outcome == 1.0
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
            if traj.reward and traj.reward.outcome == 1.0:
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

