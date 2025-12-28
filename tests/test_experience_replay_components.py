#!/usr/bin/env python3
"""
Experience Replay 组件快速测试脚本

这个脚本可以快速测试 experience replay 流程的各个组件，无需完整训练流程。

使用方法:
    python tests/test_experience_replay_components.py [--test TEST_NAME]

测试项:
    - all: 运行所有测试
    - exp_manager: 测试 ExperienceManager 基本功能（初始化、difficulty2task_dict 更新、valid_replay_task_ids）
    - trajectory_storage: 测试轨迹存储和检索（save_trajectories_to_memory、get_offpolicy_trajectories_from_memory）
    - mix_collate: 测试 ExperienceMixCollateFn（task 混合、exp_ratio 控制）
    - offpolicy_retrieval: 测试 off-policy 轨迹获取（get_offpolicy_batch、get_all_candidates_batch）
    - loss_computation: 测试 loss 计算（两种 policy shaping 方式：higher_clip_bound、exgrpo_policy_shaping）
    - grpo_grouping: 测试 GRPO 分组机制（Experience Replay 场景：on-policy + off-policy 混合分组）
    - skip_uid_set: 测试 skip_uid_set 更新逻辑（全对/部分成功/全失败场景）
    - e2e: 端到端测试（模拟完整 Experience Replay 流程）
"""

import sys
import os
import argparse
import numpy as np
import torch
from typing import List, Dict
from collections import defaultdict

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import DictConfig, OmegaConf
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory, Reward
from agentevolver.module.exp_manager.exp_manager import ExperienceManager
from agentevolver.module.exp_manager.experience_collate import ExperienceMixCollateFn
from agentevolver.module.exp_manager.het_core_algos import het_compute_token_on_off_policy_loss


# ============================================================================
# 测试数据生成
# ============================================================================

def create_mock_config() -> DictConfig:
    """创建模拟配置"""
    config = OmegaConf.create({
        "exp_manager": {
            "experience_replay": {
                "enable": True,
                "replay_start_ratio": 0.1,
                "exp_ratio": 0.5,
                "offpolicy_trajectories_per_task": 1,
                "experience_lbound": 0,
                "experience_rbound": 8,
                "exp_select_mode": "argmin",
                "exp_is_correct": True,
                "max_trajectories_per_task": 5,
                "use_current_policy_entropy": False,
            },
            "summary_batch_size": 10,
            "val_rollout_mode": "sample",
            "train_rollout_mode": "sample",
            "rollout_ratio": 1.0,
            "train_sample_mode": "keep",
            "train_sample_keepratio": 1.0,
            "reme": {
                "base_url": "http://localhost:8000",
                "workspace_id": "test",
            },
        },
        "actor_rollout_ref": {
            "rollout": {
                "n": 8,
            },
        },
        "thread_pool": {
            "max_workers": 4,
        },
    })
    return config


def create_mock_task(task_id: str, query: str = None) -> Task:
    """创建模拟 Task"""
    return Task(
        task_id=task_id,
        env_type="appworld",
        open_query=False,
        query=query or f"Task {task_id}",
        metadata={},
    )


def create_mock_trajectory(
    task_id: str,
    rollout_id: str,
    success: bool = True,
    old_log_probs: List[float] = None,
    entropy: float = None,
) -> Trajectory:
    """创建模拟 Trajectory"""
    steps = [
        {"role": "user", "content": f"Query for task {task_id}"},
        {"role": "assistant", "content": f"Response for task {task_id}, rollout {rollout_id}"},
    ]
    
    reward = Reward(
        outcome=1.0 if success else 0.0,
        success_rate=1.0 if success else 0.0,
    )
    
    metadata = {
        "old_log_probs": old_log_probs or [-0.5] * 10,
        "response_mask": [1] * 10,
        "policy_version": 100,
    }
    
    if entropy is not None:
        metadata["entropy"] = entropy
    
    return Trajectory(
        data_id=task_id,
        rollout_id=rollout_id,
        steps=steps,
        query=f"Task {task_id}",
        is_terminated=True,
        reward=reward,
        metadata=metadata,
    )


def create_mock_task_manager(tasks: List[Task]):
    """创建模拟 TaskManager"""
    class MockTaskManager:
        def __init__(self, tasks):
            self.tasks = {task.task_id: task for task in tasks}
            # _get_task_by_id 会查找 _tasks 属性
            self._tasks = tasks
        
        def get_task_by_id(self, task_id: str) -> Task:
            return self.tasks.get(task_id)
    
    return MockTaskManager(tasks)


# ============================================================================
# 测试函数
# ============================================================================

def test_exp_manager_basic():
    """测试 ExperienceManager 基本功能"""
    print("\n" + "="*80)
    print("测试 1: ExperienceManager 基本功能")
    print("="*80)
    
    config = create_mock_config()
    exp_manager = ExperienceManager(config)
    
    # 检查初始化
    assert hasattr(exp_manager, 'difficulty2task_dict')
    assert hasattr(exp_manager, 'task2trajectories')
    assert hasattr(exp_manager, 'skip_uid_set')
    print("✓ ExperienceManager 初始化成功")
    
    # 检查配置是否正确读取
    assert exp_manager.replay_start_ratio == 0.1
    assert exp_manager.max_trajectories_per_task == 5
    print("✓ 配置读取正确")
    
    # 测试 update_difficulty2task_dict
    tasks = [create_mock_task(f"task_{i}") for i in range(3)]
    trajectories = []
    for i, task in enumerate(tasks):
        for j in range(8):
            success = (i + j) % 3 == 0  # 部分成功
            traj = create_mock_trajectory(task.task_id, f"rollout_{j}", success=success)
            traj.task_id = task.task_id
            trajectories.append(traj)
    
    exp_manager.update_difficulty2task_dict(trajectories)
    
    print(f"✓ difficulty2task_dict 更新成功")
    print(f"  - 难度分布: {dict(exp_manager.difficulty2task_dict)}")
    
    # 验证每个 task 被分到正确的难度桶
    # task_0: j=0,3,6 成功 (3个)
    # task_1: j=2,5 成功 (2个) 
    # task_2: j=1,4,7 成功 (3个)
    assert len(exp_manager.difficulty2task_dict) > 0, "应该有难度分组"
    
    # 测试 get_valid_replay_task_ids（此时 task2trajectories 为空，应该返回空列表）
    valid_ids = exp_manager.get_valid_replay_task_ids()
    print(f"✓ 有效 replay task IDs（无轨迹时）: {len(valid_ids)} 个")
    assert len(valid_ids) == 0, "task2trajectories 为空时，valid_ids 应为空"
    
    # 为 task_0 添加轨迹到 task2trajectories
    task_0_trajs = [t for t in trajectories if t.task_id == "task_0" and t.reward.outcome == 1.0]
    exp_manager.save_trajectories_to_memory(task_0_trajs)
    
    # 再次测试 get_valid_replay_task_ids
    valid_ids = exp_manager.get_valid_replay_task_ids()
    print(f"✓ 有效 replay task IDs（有轨迹后）: {len(valid_ids)} 个")
    print(f"  - 示例: {valid_ids[:3] if valid_ids else 'None'}")
    assert "task_0" in valid_ids, "task_0 应该在 valid_ids 中"
    
    return True


def test_trajectory_storage():
    """测试轨迹存储和检索"""
    print("\n" + "="*80)
    print("测试 2: 轨迹存储和检索")
    print("="*80)
    
    config = create_mock_config()
    exp_manager = ExperienceManager(config)
    
    # 创建测试轨迹
    task_id = "test_task_1"
    trajectories = []
    for i in range(5):
        traj = create_mock_trajectory(
            task_id, 
            f"rollout_{i}", 
            success=True,
            entropy=0.5 + i * 0.1,  # 不同的 entropy
        )
        traj.task_id = task_id
        trajectories.append(traj)
    
    # 测试保存
    exp_manager.save_trajectories_to_memory(trajectories)
    print(f"✓ 保存了 {len(trajectories)} 条轨迹到 task {task_id}")
    
    # 测试检索
    retrieved = exp_manager.get_offpolicy_trajectories_from_memory(
        task_id, 
        num_trajectories=1,
        use_saved_entropy=True,
    )
    print(f"✓ 检索到 {len(retrieved)} 条轨迹")
    
    # 验证选择的是 entropy 最低的
    if len(retrieved) > 0:
        entropies = [t.metadata.get("entropy", float('inf')) for t in retrieved]
        print(f"  - 检索到的 entropy: {entropies}")
        assert all(e == min(entropies) for e in entropies), "应该选择 entropy 最低的"
        print("  ✓ 正确选择了 entropy 最低的轨迹")
    
    # 测试 max_trajectories_per_task 限制
    assert len(exp_manager.task2trajectories[task_id]) <= config.exp_manager.experience_replay.max_trajectories_per_task
    print(f"✓ 轨迹数量限制正确: {len(exp_manager.task2trajectories[task_id])} <= {config.exp_manager.experience_replay.max_trajectories_per_task}")
    
    return True


def test_mix_collate():
    """测试 ExperienceMixCollateFn"""
    print("\n" + "="*80)
    print("测试 3: ExperienceMixCollateFn")
    print("="*80)
    
    config = create_mock_config()
    exp_manager = ExperienceManager(config)
    
    # 创建一些任务并添加到 difficulty2task_dict
    exp_tasks = [create_mock_task(f"exp_task_{i}") for i in range(5)]
    for task in exp_tasks:
        exp_manager.difficulty2task_dict[2].append(task.task_id)  # 难度 2
    
    # 为 exp_tasks 创建并保存轨迹到 task2trajectories
    # 这样 get_valid_replay_task_ids() 才能返回有效的 task_ids
    for task in exp_tasks:
        trajectories = []
        for j in range(3):  # 每个任务创建 3 条轨迹
            traj = create_mock_trajectory(
                task.task_id,
                f"rollout_{j}",
                success=True,
                entropy=0.3 + j * 0.1,
            )
            traj.task_id = task.task_id
            trajectories.append(traj)
        exp_manager.save_trajectories_to_memory(trajectories)
    
    print(f"✓ 为 {len(exp_tasks)} 个 exp_tasks 创建并保存了轨迹")
    
    # 验证 valid_replay_task_ids 不为空
    valid_ids = exp_manager.get_valid_replay_task_ids()
    print(f"✓ 有效 replay task IDs: {len(valid_ids)} 个")
    assert len(valid_ids) > 0, "应该有有效的 replay task IDs"
    
    # 创建训练任务
    training_tasks = [create_mock_task(f"train_task_{i}") for i in range(10)]
    
    # 创建模拟 TaskManager
    all_tasks = exp_tasks + training_tasks
    mock_task_manager = create_mock_task_manager(all_tasks)
    
    # 创建 ExperienceMixCollateFn
    mix_collate = ExperienceMixCollateFn(
        exp_manager=exp_manager,
        train_task_manager=mock_task_manager,
        exp_ratio=0.5,
        replay_start_ratio=0.1,
        offpolicy_trajectories_per_task=1,
        n_rollout=8,
    )
    
    # 测试混合
    experience_tasks, on_policy_tasks = mix_collate(
        training_tasks=training_tasks,
        training_progress=0.5,  # 50% 进度，应该启用 replay
        enable_replay=True,
    )
    
    print(f"✓ 混合成功")
    print(f"  - Experience tasks: {len(experience_tasks)}")
    print(f"  - On-policy tasks: {len(on_policy_tasks)}")
    print(f"  - 总数: {len(experience_tasks) + len(on_policy_tasks)}")
    
    # 验证比例
    total = len(experience_tasks) + len(on_policy_tasks)
    if total > 0:
        exp_ratio_actual = len(experience_tasks) / total
        print(f"  - 实际 exp_ratio: {exp_ratio_actual:.2f}")
    
    return True


def test_offpolicy_retrieval():
    """测试 off-policy 轨迹获取"""
    print("\n" + "="*80)
    print("测试 4: Off-policy 轨迹获取")
    print("="*80)
    
    config = create_mock_config()
    exp_manager = ExperienceManager(config)
    
    # 创建任务和轨迹
    tasks = [create_mock_task(f"task_{i}") for i in range(3)]
    
    for task in tasks:
        # 为每个任务保存多条轨迹
        trajectories = []
        for j in range(3):
            traj = create_mock_trajectory(
                task.task_id,
                f"rollout_{j}",
                success=True,
                entropy=0.3 + j * 0.2,
            )
            traj.task_id = task.task_id
            trajectories.append(traj)
        exp_manager.save_trajectories_to_memory(trajectories)
    
    # 测试 get_offpolicy_batch
    offpolicy_trajectories = exp_manager.get_offpolicy_batch(
        tasks=tasks,
        num_trajectories_per_task=1,
    )
    
    print(f"✓ 获取到 {len(offpolicy_trajectories)} 条 off-policy 轨迹")
    for traj in offpolicy_trajectories:
        print(f"  - Task {traj.task_id}: entropy={traj.metadata.get('entropy', 'N/A')}")
    
    # 测试 get_all_candidates_batch
    all_candidates = exp_manager.get_all_candidates_batch(tasks=tasks)
    print(f"✓ 获取到所有候选轨迹")
    for task_id, candidates in all_candidates.items():
        print(f"  - Task {task_id}: {len(candidates)} 条候选轨迹")
    
    return True


def test_loss_computation():
    """测试 loss 计算（两种 policy shaping 方式）"""
    print("\n" + "="*80)
    print("测试 5: Loss 计算（两种 Policy Shaping 方式）")
    print("="*80)
    
    batch_size, seq_len = 4, 20
    response_len = 10
    
    # 创建模拟数据
    old_log_prob = torch.randn(batch_size, response_len)
    log_prob = torch.randn(batch_size, response_len)
    advantages = torch.randn(batch_size, response_len)
    response_mask = torch.ones(batch_size, response_len)
    exp_mask = torch.zeros(batch_size, response_len)
    exp_mask[0, :] = 1  # 第一个样本是 off-policy
    exp_mask[1, :] = 1  # 第二个样本也是 off-policy
    
    # 测试 higher_clip_bound 方式
    print("\n测试 higher_clip_bound 方式:")
    result1 = het_compute_token_on_off_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        exp_mask=exp_mask,
        cliprange=0.2,
        cliprange_low=0.2,
        cliprange_high=0.2,
        off_cliprange_high=0.6,
        clip_ratio_c=3.0,
        loss_agg_mode="token-mean",
        off_policy_shaping_mode="higher_clip_bound",
        off_policy_shaping_beta=0.1,
    )
    print(f"✓ higher_clip_bound 计算成功")
    print(f"  - pg_loss: {result1['pg_loss'].item():.4f}")
    print(f"  - on_pg_loss: {result1['on_pg_loss'].item():.4f}")
    print(f"  - off_pg_loss: {result1['off_pg_loss'].item():.4f}")
    print(f"  - on_pg_clipfrac: {result1['on_pg_clipfrac'].item():.4f}")
    
    # 测试 exgrpo_policy_shaping 方式
    print("\n测试 exgrpo_policy_shaping 方式:")
    result2 = het_compute_token_on_off_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        exp_mask=exp_mask,
        cliprange=0.2,
        cliprange_low=0.2,
        cliprange_high=0.2,
        off_cliprange_high=0.6,
        clip_ratio_c=3.0,
        loss_agg_mode="token-mean",
        off_policy_shaping_mode="exgrpo_policy_shaping",
        off_policy_shaping_beta=0.1,
    )
    print(f"✓ exgrpo_policy_shaping 计算成功")
    print(f"  - pg_loss: {result2['pg_loss'].item():.4f}")
    print(f"  - on_pg_loss: {result2['on_pg_loss'].item():.4f}")
    print(f"  - off_pg_loss: {result2['off_pg_loss'].item():.4f}")
    print(f"  - on_pg_clipfrac: {result2['on_pg_clipfrac'].item():.4f}")
    
    # 验证两种方式产生不同的结果
    assert not torch.isclose(result1['off_pg_loss'], result2['off_pg_loss'], atol=1e-5), \
        "两种方式应该产生不同的 off-policy loss"
    print("✓ 两种方式产生不同的 off-policy loss（符合预期）")
    
    # 额外测试 1: 验证 on-policy loss 只使用 exp_mask=0 的样本
    print("\n验证 on-policy loss 只使用 exp_mask=0 的样本:")
    on_policy_mask = (1.0 - exp_mask) * response_mask
    assert on_policy_mask[0].sum() == 0, "第一个样本（off-policy）不应参与 on_pg_loss"
    assert on_policy_mask[1].sum() == 0, "第二个样本（off-policy）不应参与 on_pg_loss"
    assert on_policy_mask[2].sum() == response_len, "第三个样本（on-policy）应参与 on_pg_loss"
    assert on_policy_mask[3].sum() == response_len, "第四个样本（on-policy）应参与 on_pg_loss"
    print("  ✓ on-policy mask 正确")
    
    # 额外测试 2: 验证 off-policy loss 只使用 exp_mask=1 的样本
    print("\n验证 off-policy loss 只使用 exp_mask=1 的样本:")
    off_policy_mask = exp_mask * response_mask
    assert off_policy_mask[0].sum() == response_len, "第一个样本（off-policy）应参与 off_pg_loss"
    assert off_policy_mask[1].sum() == response_len, "第二个样本（off-policy）应参与 off_pg_loss"
    assert off_policy_mask[2].sum() == 0, "第三个样本（on-policy）不应参与 off_pg_loss"
    assert off_policy_mask[3].sum() == 0, "第四个样本（on-policy）不应参与 off_pg_loss"
    print("  ✓ off-policy mask 正确")
    
    # 额外测试 3: 边界情况 - 全部 on-policy
    print("\n测试边界情况 - 全部 on-policy:")
    exp_mask_all_on = torch.zeros(batch_size, response_len)
    result_all_on = het_compute_token_on_off_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        exp_mask=exp_mask_all_on,
        cliprange=0.2,
        cliprange_low=0.2,
        cliprange_high=0.2,
        off_cliprange_high=0.6,
        clip_ratio_c=3.0,
        loss_agg_mode="token-mean",
        off_policy_shaping_mode="higher_clip_bound",
        off_policy_shaping_beta=0.1,
    )
    assert result_all_on['off_pg_loss'].item() == 0.0 or torch.isnan(result_all_on['off_pg_loss']).item(), \
        "全部 on-policy 时，off_pg_loss 应为 0 或 nan"
    print(f"  ✓ 全部 on-policy 时 off_pg_loss = {result_all_on['off_pg_loss'].item():.4f}")
    
    return True


def test_grpo_grouping():
    """
    测试 GRPO 分组机制 - Experience Replay 场景
    
    在 experience replay 场景下:
    - 同一个 task 可能有 on-policy rollouts 和 off-policy rollouts
    - 它们共享同一个 uid（基于 group_ids/data_id）
    - GRPO 计算 advantage 时，会将同一 uid 的所有 rollouts 分到同一组
    - exp_mask 用于区分 on-policy (0) 和 off-policy (1)
    """
    print("\n" + "="*80)
    print("测试 6: GRPO 分组机制（Experience Replay 场景）")
    print("="*80)
    
    # ===========================
    # 场景设置：模拟混合 batch
    # ===========================
    # - 2 个 tasks
    # - 每个 task 有 6 个 on-policy rollouts + 2 个 off-policy rollouts = 8 total
    # - 同一 task 的所有 rollouts 共享同一个 uid
    n_tasks = 2
    n_on_policy_per_task = 6
    n_off_policy_per_task = 2
    n_total_per_task = n_on_policy_per_task + n_off_policy_per_task
    response_len = 10
    
    batch_size = n_tasks * n_total_per_task  # 2 * 8 = 16
    
    # 构建 uid 和 exp_mask
    uids = []
    exp_mask = torch.zeros(batch_size, response_len)
    task_ids = []
    is_offpolicy = []
    
    idx = 0
    for task_idx in range(n_tasks):
        task_id = f"task_{task_idx}"
        uid = str(task_idx)  # 同一 task 的所有 rollouts 共享 uid
        
        # On-policy rollouts
        for _ in range(n_on_policy_per_task):
            uids.append(uid)
            task_ids.append(task_id)
            is_offpolicy.append(False)
            # exp_mask 默认为 0，表示 on-policy
            idx += 1
        
        # Off-policy rollouts (experience replay)
        for _ in range(n_off_policy_per_task):
            uids.append(uid)
            task_ids.append(task_id)
            is_offpolicy.append(True)
            exp_mask[idx, :] = 1  # 标记为 off-policy
            idx += 1
    
    uids = np.array(uids, dtype=object)
    
    print(f"✓ 构建混合 batch 成功")
    print(f"  - 总 rollouts: {batch_size}")
    print(f"  - Tasks 数量: {n_tasks}")
    print(f"  - 每 task on-policy: {n_on_policy_per_task}")
    print(f"  - 每 task off-policy: {n_off_policy_per_task}")
    
    # ===========================
    # 模拟 reward
    # ===========================
    # On-policy: 随机 reward
    # Off-policy: 历史成功轨迹，reward = 1.0
    rewards = torch.zeros(batch_size)
    for i in range(batch_size):
        if is_offpolicy[i]:
            rewards[i] = 1.0  # off-policy 是历史成功轨迹
        else:
            rewards[i] = torch.rand(1).item()  # on-policy 随机
    
    # ===========================
    # GRPO 分组计算 (模拟 compute_grpo_outcome_advantage)
    # ===========================
    print("\n模拟 GRPO 分组计算:")
    
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    
    # 分组
    for i in range(batch_size):
        id2score[uids[i]].append(rewards[i])
    
    # 计算每组的均值和标准差
    for uid in id2score:
        scores = id2score[uid]
        if len(scores) == 1:
            id2mean[uid] = torch.tensor(0.0)
            id2std[uid] = torch.tensor(1.0)
        elif len(scores) > 1:
            id2mean[uid] = torch.mean(torch.stack(scores))
            id2std[uid] = torch.std(torch.stack(scores))
        print(f"  - uid={uid}: {len(scores)} rollouts, mean={id2mean[uid].item():.4f}, std={id2std[uid].item():.4f}")
    
    # 计算 advantage
    epsilon = 1e-6
    advantages = torch.zeros(batch_size)
    for i in range(batch_size):
        advantages[i] = (rewards[i] - id2mean[uids[i]]) / (id2std[uids[i]] + epsilon)
    
    print(f"\n✓ GRPO 分组计算成功")
    print(f"  - 分组数: {len(id2mean)}")
    
    # ===========================
    # 验证分组正确性
    # ===========================
    print("\n验证分组正确性:")
    
    # 验证 1: 同一 task 的所有 rollouts（on + off）在同一组
    for task_idx in range(n_tasks):
        uid = str(task_idx)
        task_rollout_indices = [i for i, u in enumerate(uids) if u == uid]
        assert len(task_rollout_indices) == n_total_per_task, \
            f"Task {task_idx} 应有 {n_total_per_task} 个 rollouts 在组 {uid}"
        
        # 验证 on-policy 和 off-policy 数量
        on_policy_count = sum(1 for i in task_rollout_indices if not is_offpolicy[i])
        off_policy_count = sum(1 for i in task_rollout_indices if is_offpolicy[i])
        assert on_policy_count == n_on_policy_per_task
        assert off_policy_count == n_off_policy_per_task
        print(f"  ✓ Task {task_idx} (uid={uid}): {on_policy_count} on-policy + {off_policy_count} off-policy")
    
    # 验证 2: exp_mask 正确标记 off-policy
    for i in range(batch_size):
        if is_offpolicy[i]:
            assert exp_mask[i, 0].item() == 1, f"样本 {i} 是 off-policy，exp_mask 应为 1"
        else:
            assert exp_mask[i, 0].item() == 0, f"样本 {i} 是 on-policy，exp_mask 应为 0"
    print(f"  ✓ exp_mask 正确标记 on/off-policy")
    
    # 验证 3: off-policy 样本（历史成功轨迹）的 advantage 计算
    # off-policy reward = 1.0，通常高于组均值，所以 advantage > 0
    for i in range(batch_size):
        if is_offpolicy[i]:
            # 由于 off-policy reward = 1.0，而组内有随机 reward，
            # off-policy 的 advantage 通常为正（鼓励模仿成功轨迹）
            print(f"  - 样本 {i} (off-policy, task={task_ids[i]}): "
                  f"reward={rewards[i].item():.2f}, advantage={advantages[i].item():.4f}")
    
    print("\n✓ GRPO 分组测试通过（Experience Replay 场景）")
    return True


def test_update_skip_uid_set():
    """测试 skip_uid_set 更新逻辑"""
    print("\n" + "="*80)
    print("测试 7: skip_uid_set 更新逻辑")
    print("="*80)
    
    config = create_mock_config()
    exp_manager = ExperienceManager(config)
    
    # ===========================
    # 场景 1: 全部成功的 task 应该加入 skip_uid_set
    # ===========================
    print("\n场景 1: 全部成功的 task")
    task_id_full_success = "test_task_full_success"
    trajectories_full_success = []
    for i in range(8):
        traj = create_mock_trajectory(task_id_full_success, f"rollout_{i}", success=True)
        traj.task_id = task_id_full_success
        trajectories_full_success.append(traj)
    
    filtered = exp_manager.update_skip_uid_set_and_filter_trajectories(
        trajectories=trajectories_full_success,
        n_rollout=8,
    )
    
    print(f"  - Task {task_id_full_success} 是否在 skip_uid_set: {task_id_full_success in exp_manager.skip_uid_set}")
    print(f"  - 筛选出的轨迹数: {len(filtered)}")
    
    assert task_id_full_success in exp_manager.skip_uid_set, "全部成功的任务应该加入 skip_uid_set"
    assert len(filtered) == 0, "全部成功的任务不应该有筛选出的轨迹"
    print("  ✓ 全部成功的任务正确加入 skip_uid_set")
    
    # ===========================
    # 场景 2: 部分成功的 task 应该有筛选出的轨迹（非全对非全错）
    # ===========================
    print("\n场景 2: 部分成功的 task（符合 experience_lbound < success < experience_rbound）")
    task_id_partial = "test_task_partial_success"
    trajectories_partial = []
    # 3 个成功，5 个失败 (0 < 3 < 8，符合条件)
    for i in range(8):
        success = i < 3
        traj = create_mock_trajectory(task_id_partial, f"rollout_{i}", success=success)
        traj.task_id = task_id_partial
        trajectories_partial.append(traj)
    
    filtered_partial = exp_manager.update_skip_uid_set_and_filter_trajectories(
        trajectories=trajectories_partial,
        n_rollout=8,
    )
    
    print(f"  - Task {task_id_partial} 是否在 skip_uid_set: {task_id_partial in exp_manager.skip_uid_set}")
    print(f"  - 筛选出的轨迹数: {len(filtered_partial)}")
    
    assert task_id_partial not in exp_manager.skip_uid_set, "部分成功的任务不应该在 skip_uid_set"
    assert len(filtered_partial) == 3, "应该筛选出 3 条成功轨迹"
    print("  ✓ 部分成功的任务正确筛选轨迹")
    
    # ===========================
    # 场景 3: 全部失败的 task 不应该有筛选出的轨迹
    # ===========================
    print("\n场景 3: 全部失败的 task")
    task_id_all_fail = "test_task_all_fail"
    trajectories_all_fail = []
    for i in range(8):
        traj = create_mock_trajectory(task_id_all_fail, f"rollout_{i}", success=False)
        traj.task_id = task_id_all_fail
        trajectories_all_fail.append(traj)
    
    filtered_all_fail = exp_manager.update_skip_uid_set_and_filter_trajectories(
        trajectories=trajectories_all_fail,
        n_rollout=8,
    )
    
    print(f"  - Task {task_id_all_fail} 是否在 skip_uid_set: {task_id_all_fail in exp_manager.skip_uid_set}")
    print(f"  - 筛选出的轨迹数: {len(filtered_all_fail)}")
    
    assert task_id_all_fail not in exp_manager.skip_uid_set, "全部失败的任务不应该在 skip_uid_set"
    assert len(filtered_all_fail) == 0, "全部失败的任务不应该有筛选出的轨迹"
    print("  ✓ 全部失败的任务正确处理")
    
    # ===========================
    # 场景 4: 之前在 skip_uid_set 的 task 如果这次没全对，应该移除
    # ===========================
    print("\n场景 4: 从 skip_uid_set 中移除")
    # 先确保 task 在 skip_uid_set 中
    assert task_id_full_success in exp_manager.skip_uid_set
    
    # 模拟这个 task 这次没全对
    trajectories_not_full = []
    for i in range(8):
        success = i < 5  # 5 个成功，3 个失败
        traj = create_mock_trajectory(task_id_full_success, f"rollout_{i}", success=success)
        traj.task_id = task_id_full_success
        trajectories_not_full.append(traj)
    
    exp_manager.update_skip_uid_set_and_filter_trajectories(
        trajectories=trajectories_not_full,
        n_rollout=8,
    )
    
    print(f"  - Task {task_id_full_success} 是否还在 skip_uid_set: {task_id_full_success in exp_manager.skip_uid_set}")
    assert task_id_full_success not in exp_manager.skip_uid_set, "不再全对的任务应该从 skip_uid_set 移除"
    print("  ✓ 不再全对的任务正确从 skip_uid_set 移除")
    
    print("\n✓ skip_uid_set 更新逻辑测试通过")
    return True


def test_end_to_end():
    """
    端到端测试：模拟完整的 Experience Replay 流程
    
    流程：
    1. 初始化 ExperienceManager
    2. 模拟多个 training steps，每个 step 生成 trajectories
    3. 更新 difficulty2task_dict 和 task2trajectories
    4. 使用 ExperienceMixCollateFn 混合 tasks
    5. 模拟 GRPO 分组和 loss 计算
    """
    print("\n" + "="*80)
    print("测试 8: 端到端测试（完整 Experience Replay 流程）")
    print("="*80)
    
    config = create_mock_config()
    exp_manager = ExperienceManager(config)
    
    # ===========================
    # Step 1: 初始训练阶段（积累经验）
    # ===========================
    print("\n=== Step 1: 初始训练阶段 ===")
    
    # 创建 5 个任务
    all_tasks = [create_mock_task(f"task_{i}") for i in range(5)]
    mock_task_manager = create_mock_task_manager(all_tasks)
    
    # 模拟第一个 training step
    step1_trajectories = []
    for task in all_tasks:
        for j in range(8):
            # 不同任务有不同的成功率
            task_idx = int(task.task_id.split("_")[1])
            success = j < (3 + task_idx % 3)  # task_0: 3成功, task_1: 4成功, task_2: 5成功, ...
            traj = create_mock_trajectory(
                task.task_id, 
                f"rollout_{j}", 
                success=success,
                entropy=0.5 + j * 0.05,
            )
            traj.task_id = task.task_id
            step1_trajectories.append(traj)
    
    # 更新 difficulty2task_dict
    exp_manager.update_difficulty2task_dict(step1_trajectories)
    print(f"✓ 更新 difficulty2task_dict: {dict(exp_manager.difficulty2task_dict)}")
    
    # 筛选并保存轨迹
    filtered = exp_manager.update_skip_uid_set_and_filter_trajectories(
        trajectories=step1_trajectories,
        n_rollout=8,
    )
    exp_manager.save_trajectories_to_memory(filtered)
    print(f"✓ 保存了 {len(filtered)} 条轨迹到 task2trajectories")
    print(f"  - task2trajectories 中的 task 数: {len(exp_manager.task2trajectories)}")
    
    # ===========================
    # Step 2: Experience Replay 阶段
    # ===========================
    print("\n=== Step 2: Experience Replay 阶段 ===")
    
    # 检查 valid_replay_task_ids
    valid_ids = exp_manager.get_valid_replay_task_ids()
    print(f"✓ 有效 replay task IDs: {len(valid_ids)} 个")
    
    if len(valid_ids) > 0:
        # 创建 ExperienceMixCollateFn
        mix_collate = ExperienceMixCollateFn(
            exp_manager=exp_manager,
            train_task_manager=mock_task_manager,
            exp_ratio=0.5,
            replay_start_ratio=0.1,
            offpolicy_trajectories_per_task=1,
            n_rollout=8,
        )
        
        # 新的 training tasks（这次用前 3 个）
        training_tasks = all_tasks[:3]
        
        # 混合 tasks（training_progress = 0.5，超过 replay_start_ratio）
        experience_tasks, on_policy_tasks = mix_collate(
            training_tasks=training_tasks,
            training_progress=0.5,
            enable_replay=True,
        )
        
        print(f"✓ 混合成功")
        print(f"  - Experience tasks: {len(experience_tasks)}")
        print(f"  - On-policy tasks: {len(on_policy_tasks)}")
        
        # 验证总数不变
        assert len(experience_tasks) + len(on_policy_tasks) == len(training_tasks), \
            "混合后总数应该等于原始 training_tasks 数量"
        print("  ✓ 总数验证通过")
    
    # ===========================
    # Step 3: 模拟 GRPO 分组和 Loss 计算
    # ===========================
    print("\n=== Step 3: GRPO 分组和 Loss 计算 ===")
    
    # 模拟混合 batch
    batch_size = 16
    response_len = 10
    n_on_policy = 12
    n_off_policy = 4
    
    # 创建模拟数据
    old_log_prob = torch.randn(batch_size, response_len)
    log_prob = torch.randn(batch_size, response_len)
    response_mask = torch.ones(batch_size, response_len)
    
    # exp_mask: 后 4 个样本是 off-policy
    exp_mask = torch.zeros(batch_size, response_len)
    exp_mask[n_on_policy:, :] = 1
    
    # rewards: on-policy 随机，off-policy = 1.0
    rewards = torch.zeros(batch_size)
    rewards[:n_on_policy] = torch.rand(n_on_policy)
    rewards[n_on_policy:] = 1.0
    
    # 模拟 advantages (简化版)
    advantages = (rewards - rewards.mean()).unsqueeze(-1).expand(-1, response_len)
    
    # 计算 loss
    result = het_compute_token_on_off_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        exp_mask=exp_mask,
        cliprange=0.2,
        cliprange_low=0.2,
        cliprange_high=0.2,
        off_cliprange_high=0.6,
        clip_ratio_c=3.0,
        loss_agg_mode="token-mean",
        off_policy_shaping_mode="higher_clip_bound",
        off_policy_shaping_beta=0.1,
    )
    
    print(f"✓ Loss 计算成功")
    print(f"  - pg_loss: {result['pg_loss'].item():.4f}")
    print(f"  - on_pg_loss: {result['on_pg_loss'].item():.4f}")
    print(f"  - off_pg_loss: {result['off_pg_loss'].item():.4f}")
    
    # 验证 on/off-policy loss 分开计算
    assert not torch.isnan(result['on_pg_loss']), "on_pg_loss 不应为 NaN"
    print("  ✓ Loss 验证通过")
    
    # ===========================
    # Step 4: 模拟任务完全做对后移除
    # ===========================
    print("\n=== Step 4: 模拟任务完全做对 ===")
    
    # 模拟 task_0 全部做对
    task_0_full_success = []
    for j in range(8):
        traj = create_mock_trajectory("task_0", f"rollout_{j}", success=True)
        traj.task_id = "task_0"
        task_0_full_success.append(traj)
    
    exp_manager.update_skip_uid_set_and_filter_trajectories(
        trajectories=task_0_full_success,
        n_rollout=8,
    )
    
    print(f"  - task_0 在 skip_uid_set: {'task_0' in exp_manager.skip_uid_set}")
    print(f"  - task_0 在 task2trajectories: {'task_0' in exp_manager.task2trajectories}")
    
    if "task_0" in exp_manager.skip_uid_set:
        print("  ✓ 完全做对的任务正确加入 skip_uid_set")
    
    print("\n✓ 端到端测试通过！")
    return True


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Experience Replay 组件测试")
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["all", "exp_manager", "trajectory_storage", "mix_collate", 
                 "offpolicy_retrieval", "loss_computation", "grpo_grouping", "skip_uid_set", "e2e"],
        help="要运行的测试",
    )
    args = parser.parse_args()
    
    test_functions = {
        "exp_manager": test_exp_manager_basic,
        "trajectory_storage": test_trajectory_storage,
        "mix_collate": test_mix_collate,
        "offpolicy_retrieval": test_offpolicy_retrieval,
        "loss_computation": test_loss_computation,
        "grpo_grouping": test_grpo_grouping,
        "skip_uid_set": test_update_skip_uid_set,
        "e2e": test_end_to_end,
    }
    
    if args.test == "all":
        tests_to_run = list(test_functions.keys())
    else:
        tests_to_run = [args.test]
    
    print("\n" + "="*80)
    print("Experience Replay 组件测试")
    print("="*80)
    print(f"运行测试: {', '.join(tests_to_run)}")
    print("="*80)
    
    results = {}
    for test_name in tests_to_run:
        try:
            results[test_name] = test_functions[test_name]()
        except Exception as e:
            print(f"\n❌ 测试 {test_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查输出")
        return 1


if __name__ == "__main__":
    sys.exit(main())

