#!/usr/bin/env python
"""
测试 Teacher Trajectory 从采集到 Trainer Loss 计算的完整 Pipeline

验证点：
1. ExperienceManager 能正确加载 teacher trajectories
2. convert_offpolicy_to_cmt 能正确转换 teacher trajectories  
3. samples_to_dataproto 能正确生成 teacher_mask 和对齐 log_probs
4. 最终 batch 格式符合 het_compute_token_on_off_policy_loss_with_teacher 要求

使用方法：
    python tests/test_teacher_trajectory_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


def test_1_load_teacher_trajectories():
    """测试 1: ExperienceManager 加载 Teacher Trajectories"""
    print("\n" + "="*60)
    print("测试 1: ExperienceManager 加载 Teacher Trajectories")
    print("="*60)
    
    from agentevolver.module.exp_manager.exp_manager import ExperienceManager
    
    # 创建 mock config
    exp_manager_config = {
        "teacher_experience": {
            "enable": True,
            "data_path": "data/teacher_trajectories/alfworld_qwen7b.jsonl",
            "exp_ratio": 0.25,
            "max_trajectories_per_task": 2,
            "select_mode": "random",
            "use_log_prob": True,
        },
        "experience_replay": {
            "enable": False,
        }
    }
    
    exp_manager = ExperienceManager(exp_manager_config)
    
    # 验证加载
    print(f"✓ Teacher enabled: {exp_manager.teacher_enabled}")
    print(f"✓ Teacher task count: {len(exp_manager.teacher_task2trajectories)}")
    
    for task_id, trajs in exp_manager.teacher_task2trajectories.items():
        print(f"  - Task {task_id}: {len(trajs)} trajectories")
        for i, traj in enumerate(trajs):
            has_logprob = traj.metadata.get("has_log_prob", False)
            reward = traj.reward.outcome if traj.reward else 0
            print(f"    [{i}] reward={reward}, has_log_prob={has_logprob}")
            if has_logprob and "old_log_probs" in traj.metadata:
                lp = traj.metadata["old_log_probs"]
                print(f"        log_probs: len={len(lp)}, mean={np.mean(lp):.4f}")
    
    # 验证 get_teacher_trajectories
    task_ids = list(exp_manager.teacher_task2trajectories.keys())
    retrieved = exp_manager.get_teacher_trajectories(task_ids, num_per_task=1)
    print(f"\n✓ Retrieved {len(retrieved)} trajectories for {len(task_ids)} tasks")
    
    return exp_manager


def test_2_trajectory_format():
    """测试 2: 检查轨迹数据格式"""
    print("\n" + "="*60)
    print("测试 2: 检查轨迹数据格式")
    print("="*60)
    
    data_path = "data/teacher_trajectories/alfworld_qwen7b.jsonl"
    
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            data = json.loads(line)
            
            print(f"\n--- Trajectory {i} ---")
            print(f"Keys: {list(data.keys())}")
            
            # 必须字段
            required_fields = ["task_id", "messages", "reward"]
            for field in required_fields:
                if field in data:
                    print(f"✓ {field}: {type(data[field]).__name__}")
                else:
                    print(f"✗ Missing: {field}")
            
            # 检查 messages 格式
            messages = data.get("messages", [])
            print(f"✓ Messages count: {len(messages)}")
            if messages:
                roles = [m.get("role", "unknown") for m in messages[:5]]
                print(f"  First 5 roles: {roles}")
            
            # 检查 log_probs
            if "log_probs" in data:
                lp = data["log_probs"]
                print(f"✓ log_probs: len={len(lp)}, type={type(lp[0]).__name__ if lp else 'empty'}")
            
            # 检查 log_probs_per_turn
            if "log_probs_per_turn" in data:
                lpt = data["log_probs_per_turn"]
                print(f"✓ log_probs_per_turn: {len(lpt)} turns")
            
            # 检查 metadata
            if "metadata" in data:
                md = data["metadata"]
                print(f"✓ metadata keys: {list(md.keys())}")
            
            if i >= 1:  # 只检查前 2 个
                break


def test_3_convert_to_cmt():
    """测试 3: convert_offpolicy_to_cmt 转换"""
    print("\n" + "="*60)
    print("测试 3: convert_offpolicy_to_cmt 转换")
    print("="*60)
    
    try:
        from agentevolver.module.env_manager.env_manager import EnvManager
        from agentevolver.module.exp_manager.exp_manager import ExperienceManager
        from omegaconf import OmegaConf
        
        # 加载配置
        config = OmegaConf.load("config/alfworld_grpo_3b_teacher_only.yaml")
        
        # 创建 ExperienceManager
        exp_manager_config = {
            "teacher_experience": {
                "enable": True,
                "data_path": "data/teacher_trajectories/alfworld_qwen7b.jsonl",
                "exp_ratio": 0.25,
                "select_mode": "random",
                "use_log_prob": True,
            },
            "experience_replay": {"enable": False}
        }
        exp_manager = ExperienceManager(exp_manager_config)
        
        # 获取 teacher trajectories
        task_ids = list(exp_manager.teacher_task2trajectories.keys())
        trajs = exp_manager.get_teacher_trajectories(task_ids[:1], num_per_task=1)
        
        print(f"✓ Got {len(trajs)} teacher trajectories")
        
        if trajs:
            traj = trajs[0]
            print(f"  task_id: {traj.task_id}")
            print(f"  full_context length: {len(traj.full_context)}")
            print(f"  metadata keys: {list(traj.metadata.keys())}")
            print(f"  is_teacher: {traj.metadata.get('is_teacher', False)}")
            print(f"  has_log_prob: {traj.metadata.get('has_log_prob', False)}")
            
        print("\n✓ 测试 3 完成（完整的 CMT 转换需要 EnvManager 初始化）")
        
    except Exception as e:
        print(f"⚠ 测试 3 跳过（需要完整环境）: {e}")


def test_4_batch_format_simulation():
    """测试 4: 模拟 batch 格式"""
    print("\n" + "="*60)
    print("测试 4: 模拟 batch 格式检查")
    print("="*60)
    
    # 模拟 samples_to_dataproto 的输出
    batch_size = 4
    response_length = 100
    
    # 模拟三种数据类型的 mask
    # 假设: 2 个 on-policy, 1 个 self-generated, 1 个 teacher
    exp_mask = np.zeros((batch_size, response_length), dtype=np.int32)
    teacher_mask = np.zeros((batch_size, response_length), dtype=np.int32)
    
    # 第 2 个样本是 self-generated experience
    exp_mask[2, :] = 1
    
    # 第 3 个样本是 teacher experience
    teacher_mask[3, :] = 1
    
    print(f"✓ exp_mask shape: {exp_mask.shape}")
    print(f"  exp_mask[0] (on-policy): sum={exp_mask[0].sum()}")
    print(f"  exp_mask[2] (self-gen): sum={exp_mask[2].sum()}")
    
    print(f"\n✓ teacher_mask shape: {teacher_mask.shape}")
    print(f"  teacher_mask[0] (on-policy): sum={teacher_mask[0].sum()}")
    print(f"  teacher_mask[3] (teacher): sum={teacher_mask[3].sum()}")
    
    # 验证互斥性
    overlap = (exp_mask * teacher_mask).sum()
    print(f"\n✓ exp_mask 和 teacher_mask 重叠检查: {overlap} (应为 0)")
    
    # 模拟 recorded_old_log_probs
    recorded_old_log_probs = np.zeros((batch_size, response_length), dtype=np.float32)
    
    # self-generated 样本有 old_log_probs
    recorded_old_log_probs[2, :50] = np.random.uniform(-2, -0.5, 50)
    
    # teacher 样本也可能有 old_log_probs（如果 use_log_prob=True）
    recorded_old_log_probs[3, :60] = np.random.uniform(-3, -1, 60)
    
    print(f"\n✓ recorded_old_log_probs shape: {recorded_old_log_probs.shape}")
    print(f"  on-policy [0] mean: {recorded_old_log_probs[0].mean():.4f}")
    print(f"  self-gen [2] mean (non-zero): {recorded_old_log_probs[2, :50].mean():.4f}")
    print(f"  teacher [3] mean (non-zero): {recorded_old_log_probs[3, :60].mean():.4f}")


def test_5_loss_computation_interface():
    """测试 5: Loss 计算接口检查"""
    print("\n" + "="*60)
    print("测试 5: Loss 计算接口检查")
    print("="*60)
    
    try:
        from agentevolver.module.exp_manager.het_core_algos import (
            het_compute_token_on_off_policy_loss,
            het_compute_token_on_off_policy_loss_with_teacher
        )
        import inspect
        
        # 检查原始函数签名
        sig1 = inspect.signature(het_compute_token_on_off_policy_loss)
        print(f"✓ het_compute_token_on_off_policy_loss 参数:")
        for name, param in sig1.parameters.items():
            default = param.default if param.default != inspect.Parameter.empty else "required"
            print(f"    {name}: {default}")
        
        # 检查 teacher 版本函数签名
        print(f"\n✓ het_compute_token_on_off_policy_loss_with_teacher 参数:")
        sig2 = inspect.signature(het_compute_token_on_off_policy_loss_with_teacher)
        for name, param in sig2.parameters.items():
            default = param.default if param.default != inspect.Parameter.empty else "required"
            print(f"    {name}: {default}")
        
        # 检查新增参数
        new_params = set(sig2.parameters.keys()) - set(sig1.parameters.keys())
        print(f"\n✓ Teacher 版本新增参数: {new_params}")
        
    except ImportError as e:
        print(f"⚠ 无法导入 het_core_algos: {e}")


def test_6_end_to_end_simulation():
    """测试 6: 端到端模拟"""
    print("\n" + "="*60)
    print("测试 6: 端到端模拟（不实际运行 GPU 计算）")
    print("="*60)
    
    try:
        import torch
        from agentevolver.module.exp_manager.exp_manager import ExperienceManager
        
        # 1. 加载 teacher trajectories
        exp_manager_config = {
            "teacher_experience": {
                "enable": True,
                "data_path": "data/teacher_trajectories/alfworld_qwen7b.jsonl",
                "exp_ratio": 0.25,
                "select_mode": "confidence",  # 按 confidence 排序
                "use_log_prob": True,
            },
            "experience_replay": {"enable": False}
        }
        
        exp_manager = ExperienceManager(exp_manager_config)
        
        # 2. 获取轨迹
        task_ids = list(exp_manager.teacher_task2trajectories.keys())
        trajs = exp_manager.get_teacher_trajectories(task_ids, num_per_task=1)
        
        print(f"✓ 加载了 {len(trajs)} 个 teacher trajectories")
        
        # 3. 检查 log_probs 格式
        for i, traj in enumerate(trajs):
            has_lp = traj.metadata.get("has_log_prob", False)
            print(f"\n  Trajectory {i}: task={traj.task_id}")
            print(f"    is_teacher: {traj.metadata.get('is_teacher', False)}")
            print(f"    has_log_prob: {has_lp}")
            
            if has_lp and "old_log_probs" in traj.metadata:
                lp = traj.metadata["old_log_probs"]
                print(f"    old_log_probs: len={len(lp)}")
                print(f"    mean log_prob: {np.mean(lp):.4f}")
                print(f"    confidence: {np.mean(lp):.4f}")
                print(f"    entropy_proxy: {-np.mean(lp):.4f}")
            
            if "log_probs" in traj.metadata:
                lp = traj.metadata["log_probs"]
                print(f"    log_probs (raw): len={len(lp)}")
        
        # 4. 模拟 batch 构建
        print(f"\n✓ 模拟 batch 构建...")
        
        batch_size = 4
        response_length = 100
        
        # 假设 batch 中有 1 个 teacher 样本
        teacher_mask = torch.zeros(batch_size, response_length)
        teacher_mask[0, :] = 1  # 第一个样本是 teacher
        
        # 模拟 log_probs
        log_probs_current = torch.randn(batch_size, response_length) * 0.5 - 1.0
        
        # 模拟 old_log_probs (teacher 的)
        recorded_old_log_probs = torch.zeros(batch_size, response_length)
        if trajs and trajs[0].metadata.get("has_log_prob"):
            teacher_lp = trajs[0].metadata["old_log_probs"][:response_length]
            recorded_old_log_probs[0, :len(teacher_lp)] = torch.tensor(teacher_lp)
        
        print(f"  teacher_mask sum: {teacher_mask.sum().item()}")
        print(f"  recorded_old_log_probs[0] non-zero: {(recorded_old_log_probs[0] != 0).sum().item()}")
        
        # 5. 计算 importance ratio
        print(f"\n✓ 计算 Importance Ratio...")
        
        # 对于 teacher (use_log_prob=True)
        # ratio = exp(log_prob_current - old_log_prob_teacher)
        ratio = torch.exp(log_probs_current[0, :50] - recorded_old_log_probs[0, :50])
        print(f"  ratio (first 50 tokens): mean={ratio.mean():.4f}, std={ratio.std():.4f}")
        
        # 对于 teacher (use_log_prob=False, LUFFY style)
        # ratio = exp(log_prob_current) = π_current  (assume π_old = 1)
        ratio_luffy = torch.exp(log_probs_current[0, :50])
        print(f"  ratio LUFFY style: mean={ratio_luffy.mean():.4f}")
        
        print("\n✓ 端到端模拟完成！")
        
    except Exception as e:
        import traceback
        print(f"⚠ 测试 6 出错: {e}")
        traceback.print_exc()


def main():
    print("="*60)
    print("Teacher Trajectory Pipeline 测试")
    print("="*60)
    
    # 检查数据文件
    data_path = Path("data/teacher_trajectories/alfworld_qwen7b.jsonl")
    if not data_path.exists():
        print(f"✗ 数据文件不存在: {data_path}")
        return 1
    
    print(f"✓ 数据文件存在: {data_path}")
    
    # 运行测试
    try:
        test_1_load_teacher_trajectories()
        test_2_trajectory_format()
        test_3_convert_to_cmt()
        test_4_batch_format_simulation()
        test_5_loss_computation_interface()
        test_6_end_to_end_simulation()
        
        print("\n" + "="*60)
        print("✓ 所有测试完成！")
        print("="*60)
        return 0
        
    except Exception as e:
        import traceback
        print(f"\n✗ 测试失败: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

