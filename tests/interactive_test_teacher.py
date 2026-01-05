#!/usr/bin/env python
"""
交互式测试 Teacher Trajectory Pipeline

使用方法：
    cd /home/qisheng/agent/AgentEvolver
    source ~/anaconda3/bin/activate agentevolver
    python -i tests/interactive_test_teacher.py
    
然后在交互式 Python 中：
    >>> test_all()  # 运行所有测试
    >>> exp_manager  # 查看 exp_manager 对象
    >>> trajs[0].metadata  # 查看第一个轨迹的 metadata
"""

import sys
import os
sys.path.insert(0, '.')

import json
import numpy as np
from omegaconf import OmegaConf

# ============================================================
# Step 1: 检查 JSONL 文件格式
# ============================================================
def test_jsonl_format():
    """检查 JSONL 文件格式"""
    print("\n" + "="*60)
    print("Step 1: 检查 JSONL 文件格式")
    print("="*60)
    
    data_path = 'data/teacher_trajectories/alfworld_qwen7b.jsonl'
    
    with open(data_path, 'r') as f:
        lines = [l for l in f.readlines() if l.strip()]
    
    print(f"✓ 共 {len(lines)} 条轨迹")
    
    for i, line in enumerate(lines):
        d = json.loads(line)
        
        # 检查必要字段
        assert 'task_id' in d, f"Traj {i}: 缺少 task_id"
        assert 'messages' in d, f"Traj {i}: 缺少 messages"
        assert 'reward' in d, f"Traj {i}: 缺少 reward"
        
        # 检查 metadata
        metadata = d.get('metadata', {})
        is_teacher = metadata.get('is_teacher', False)
        has_log_prob = metadata.get('has_log_prob', False)
        
        # 检查 log_probs
        log_probs = d.get('log_probs', [])
        
        print(f"\n  Traj {i}: task_id={d['task_id']}")
        print(f"    reward={d['reward']}, success={d.get('success', 'N/A')}")
        print(f"    is_teacher={is_teacher}, has_log_prob={has_log_prob}")
        print(f"    messages: {len(d['messages'])} 条")
        
        if log_probs:
            print(f"    log_probs: len={len(log_probs)}, mean={np.mean(log_probs):.4f}")
            print(f"    confidence: {np.mean(log_probs):.4f}")
            print(f"    entropy_proxy: {-np.mean(log_probs):.4f}")
        
        if 'log_probs_per_turn' in d:
            print(f"    log_probs_per_turn: {len(d['log_probs_per_turn'])} turns")
    
    print("\n✓ JSONL 格式检查通过")
    return True


# ============================================================
# Step 2: 测试 ExperienceManager 加载
# ============================================================
exp_manager = None
trajs = None

def create_test_config():
    """创建测试用的完整配置"""
    # 加载基础配置
    base_config = OmegaConf.load("config/alfworld_grpo_3b_teacher_only.yaml")
    
    # 覆盖 teacher_experience 配置
    base_config.exp_manager.teacher_experience = OmegaConf.create({
        "enable": True,
        "data_path": "data/teacher_trajectories/alfworld_qwen7b.jsonl",
        "exp_ratio": 0.25,
        "max_trajectories_per_task": 2,
        "select_mode": "confidence",
        "use_log_prob": True,
    })
    
    return base_config

def test_exp_manager_load():
    """测试 ExperienceManager 加载"""
    global exp_manager, trajs
    
    print("\n" + "="*60)
    print("Step 2: 测试 ExperienceManager 加载")
    print("="*60)
    
    from agentevolver.module.exp_manager.exp_manager import ExperienceManager
    
    # 使用完整配置
    config = create_test_config()
    print(f"✓ 加载配置: config/alfworld_grpo_3b_teacher_only.yaml")
    
    exp_manager = ExperienceManager(config)
    
    print(f"✓ teacher_enabled: {exp_manager.teacher_enabled}")
    print(f"✓ teacher_select_mode: {exp_manager.teacher_select_mode}")
    print(f"✓ teacher_use_log_prob: {exp_manager.teacher_use_log_prob}")
    print(f"✓ 加载的 task 数量: {len(exp_manager.teacher_task2trajectories)}")
    
    for tid, task_trajs in exp_manager.teacher_task2trajectories.items():
        print(f"\n  Task {tid}: {len(task_trajs)} 条轨迹")
        for j, t in enumerate(task_trajs):
            reward = t.reward.outcome if t.reward else 0
            has_lp = t.metadata.get('has_log_prob', False)
            is_teacher = t.metadata.get('is_teacher', False)
            print(f"    [{j}] reward={reward}, is_teacher={is_teacher}, has_log_prob={has_lp}")
    
    # 测试获取轨迹
    task_ids = list(exp_manager.teacher_task2trajectories.keys())
    trajs = exp_manager.get_teacher_trajectories(task_ids, num_per_task=1)
    
    print(f"\n✓ get_teacher_trajectories 返回 {len(trajs)} 条轨迹")
    
    return True


# ============================================================
# Step 3: 测试不同的 select_mode
# ============================================================
def test_select_modes():
    """测试不同的 select_mode"""
    global exp_manager
    
    print("\n" + "="*60)
    print("Step 3: 测试不同的 select_mode")
    print("="*60)
    
    if exp_manager is None:
        test_exp_manager_load()
    
    task_ids = list(exp_manager.teacher_task2trajectories.keys())
    
    # 测试 random 模式
    exp_manager.teacher_select_mode = "random"
    trajs_random = exp_manager.get_teacher_trajectories(task_ids, num_per_task=1)
    print(f"✓ random 模式: 获取 {len(trajs_random)} 条轨迹")
    
    # 测试 confidence 模式
    exp_manager.teacher_select_mode = "confidence"
    trajs_conf = exp_manager.get_teacher_trajectories(task_ids, num_per_task=1)
    print(f"✓ confidence 模式: 获取 {len(trajs_conf)} 条轨迹")
    for t in trajs_conf:
        if t.metadata.get('old_log_probs'):
            lp = t.metadata['old_log_probs']
            print(f"    task={t.task_id}, confidence={np.mean(lp):.4f}")
    
    # 测试 entropy 模式
    exp_manager.teacher_select_mode = "entropy"
    trajs_ent = exp_manager.get_teacher_trajectories(task_ids, num_per_task=1)
    print(f"✓ entropy 模式: 获取 {len(trajs_ent)} 条轨迹")
    for t in trajs_ent:
        if t.metadata.get('old_log_probs'):
            lp = t.metadata['old_log_probs']
            print(f"    task={t.task_id}, entropy_proxy={-np.mean(lp):.4f}")
    
    return True


# ============================================================
# Step 4: 检查 log_probs 对齐所需的字段
# ============================================================
def test_log_probs_alignment():
    """检查 log_probs 对齐所需的字段"""
    global trajs
    
    print("\n" + "="*60)
    print("Step 4: 检查 log_probs 对齐所需的字段")
    print("="*60)
    
    if trajs is None:
        test_exp_manager_load()
    
    for i, traj in enumerate(trajs):
        print(f"\nTrajectory {i}: task_id={traj.task_id}")
        
        # 检查必要的 metadata
        md = traj.metadata
        print(f"  is_teacher: {md.get('is_teacher', False)}")
        print(f"  has_log_prob: {md.get('has_log_prob', False)}")
        
        # 检查 log_probs
        log_probs = md.get('log_probs') or md.get('old_log_probs')
        if log_probs:
            print(f"  log_probs 长度: {len(log_probs)}")
            print(f"  log_probs 均值: {np.mean(log_probs):.4f}")
            
            # 这些 log_probs 将在 env_manager._align_teacher_log_probs() 中
            # 与 tokenized response 的 loss_mask 对齐
            print(f"  → 这些 log_probs 将与 loss_mask=1 的位置对齐")
        else:
            print(f"  ⚠ 没有 log_probs，将使用 LUFFY 风格 (π_old = 1)")
        
        # 检查 log_probs_per_turn
        lpt = md.get('log_probs_per_turn')
        if lpt:
            print(f"  log_probs_per_turn: {len(lpt)} turns")
            for turn in lpt[:3]:  # 只显示前 3 个 turn
                print(f"    turn {turn['turn_idx']}: {len(turn['log_probs'])} tokens")
    
    print("\n✓ log_probs 对齐字段检查完成")
    return True


# ============================================================
# Step 5: 模拟 importance ratio 计算
# ============================================================
def test_importance_ratio():
    """模拟 importance ratio 计算"""
    global trajs
    
    print("\n" + "="*60)
    print("Step 5: 模拟 importance ratio 计算")
    print("="*60)
    
    if trajs is None:
        test_exp_manager_load()
    
    import torch
    
    for i, traj in enumerate(trajs):
        log_probs = traj.metadata.get('log_probs') or traj.metadata.get('old_log_probs')
        
        if not log_probs:
            print(f"\nTrajectory {i}: 无 log_probs，跳过")
            continue
        
        print(f"\nTrajectory {i}: task_id={traj.task_id}")
        
        # 模拟 current policy 的 log_probs（随机生成）
        n_tokens = min(len(log_probs), 50)  # 取前 50 个 token
        old_log_probs = torch.tensor(log_probs[:n_tokens])
        current_log_probs = torch.randn(n_tokens) * 0.5 - 1.0  # 模拟当前 policy
        
        # 方式 1: 标准 importance ratio (use_log_prob=True)
        # ratio = exp(log_prob_current - log_prob_teacher)
        ratio_standard = torch.exp(current_log_probs - old_log_probs)
        print(f"  标准 ratio (use_log_prob=True):")
        print(f"    mean={ratio_standard.mean():.4f}, std={ratio_standard.std():.4f}")
        print(f"    min={ratio_standard.min():.4f}, max={ratio_standard.max():.4f}")
        
        # 方式 2: LUFFY 风格 (use_log_prob=False, assume π_old = 1)
        # ratio = exp(log_prob_current)
        ratio_luffy = torch.exp(current_log_probs)
        print(f"  LUFFY ratio (use_log_prob=False):")
        print(f"    mean={ratio_luffy.mean():.4f}, std={ratio_luffy.std():.4f}")
    
    print("\n✓ importance ratio 计算模拟完成")
    return True


# ============================================================
# 运行所有测试
# ============================================================
def test_all():
    """运行所有测试"""
    results = {}
    
    try:
        results['jsonl_format'] = test_jsonl_format()
    except Exception as e:
        print(f"✗ test_jsonl_format 失败: {e}")
        results['jsonl_format'] = False
    
    try:
        results['exp_manager_load'] = test_exp_manager_load()
    except Exception as e:
        print(f"✗ test_exp_manager_load 失败: {e}")
        import traceback
        traceback.print_exc()
        results['exp_manager_load'] = False
    
    try:
        results['select_modes'] = test_select_modes()
    except Exception as e:
        print(f"✗ test_select_modes 失败: {e}")
        import traceback
        traceback.print_exc()
        results['select_modes'] = False
    
    try:
        results['log_probs_alignment'] = test_log_probs_alignment()
    except Exception as e:
        print(f"✗ test_log_probs_alignment 失败: {e}")
        import traceback
        traceback.print_exc()
        results['log_probs_alignment'] = False
    
    try:
        results['importance_ratio'] = test_importance_ratio()
    except Exception as e:
        print(f"✗ test_importance_ratio 失败: {e}")
        import traceback
        traceback.print_exc()
        results['importance_ratio'] = False
    
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        all_passed = all_passed and passed
    
    if all_passed:
        print("\n✓ 所有测试通过！")
    else:
        print("\n✗ 部分测试失败")
    
    return all_passed


if __name__ == "__main__":
    print("="*60)
    print("Teacher Trajectory Pipeline 交互式测试")
    print("="*60)
    print("\n可用命令:")
    print("  test_all()              - 运行所有测试")
    print("  test_jsonl_format()     - 测试 JSONL 格式")
    print("  test_exp_manager_load() - 测试 ExperienceManager 加载")
    print("  test_select_modes()     - 测试不同的 select_mode")
    print("  test_log_probs_alignment() - 检查 log_probs 对齐")
    print("  test_importance_ratio() - 模拟 importance ratio 计算")
    print("\n全局变量:")
    print("  exp_manager - ExperienceManager 实例")
    print("  trajs       - 获取的 teacher trajectories 列表")
    print("")
    
    # 自动运行所有测试
    test_all()

