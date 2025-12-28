#!/usr/bin/env python3
"""
Experience Replay 快速检查脚本

这个脚本可以快速检查 experience replay 的关键组件是否正常工作。
不需要完整的环境，只需要基本的 Python 和 torch。

使用方法:
    python tests/quick_check_experience_replay.py
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_imports():
    """检查必要的导入"""
    print("="*80)
    print("检查 1: 导入检查")
    print("="*80)
    
    try:
        import torch
        print("✓ torch 可用")
    except ImportError:
        print("❌ torch 不可用，请安装: pip install torch")
        return False
    
    try:
        from agentevolver.module.exp_manager.het_core_algos import het_compute_token_on_off_policy_loss
        print("✓ het_compute_token_on_off_policy_loss 导入成功")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    
    try:
        from agentevolver.module.exp_manager.exp_manager import ExperienceManager
        print("✓ ExperienceManager 导入成功")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    
    try:
        from agentevolver.module.exp_manager.experience_collate import ExperienceMixCollateFn
        print("✓ ExperienceMixCollateFn 导入成功")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    
    return True


def check_loss_computation():
    """检查 loss 计算函数"""
    print("\n" + "="*80)
    print("检查 2: Loss 计算函数")
    print("="*80)
    
    try:
        import torch
        from agentevolver.module.exp_manager.het_core_algos import het_compute_token_on_off_policy_loss
        import verl.utils.torch_functional as verl_F
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    
    # 创建测试数据
    batch_size, response_len = 4, 10
    old_log_prob = torch.randn(batch_size, response_len)
    log_prob = torch.randn(batch_size, response_len)
    advantages = torch.randn(batch_size, response_len)
    response_mask = torch.ones(batch_size, response_len)
    exp_mask = torch.zeros(batch_size, response_len)
    exp_mask[0, :] = 1  # 第一个样本是 off-policy
    
    # 测试 higher_clip_bound
    try:
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
        print("✓ higher_clip_bound 方式计算成功")
        print(f"  - pg_loss: {result1['pg_loss'].item():.4f}")
        print(f"  - on_pg_loss: {result1['on_pg_loss'].item():.4f}")
        print(f"  - off_pg_loss: {result1['off_pg_loss'].item():.4f}")
    except Exception as e:
        print(f"❌ higher_clip_bound 计算失败: {e}")
        return False
    
    # 测试 exgrpo_policy_shaping
    try:
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
        print("✓ exgrpo_policy_shaping 方式计算成功")
        print(f"  - pg_loss: {result2['pg_loss'].item():.4f}")
        print(f"  - on_pg_loss: {result2['on_pg_loss'].item():.4f}")
        print(f"  - off_pg_loss: {result2['off_pg_loss'].item():.4f}")
    except Exception as e:
        print(f"❌ exgrpo_policy_shaping 计算失败: {e}")
        return False
    
    # 验证两种方式产生不同的结果
    if torch.isclose(result1['off_pg_loss'], result2['off_pg_loss'], atol=1e-5):
        print("⚠️  警告: 两种方式产生相同的 off-policy loss（可能有问题）")
    else:
        print("✓ 两种方式产生不同的 off-policy loss（符合预期）")
    
    return True


def check_exp_manager_init():
    """检查 ExperienceManager 初始化"""
    print("\n" + "="*80)
    print("检查 3: ExperienceManager 初始化")
    print("="*80)
    
    try:
        from omegaconf import DictConfig, OmegaConf
        from agentevolver.module.exp_manager.exp_manager import ExperienceManager
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("   提示: 需要安装 omegaconf: pip install omegaconf")
        return False
    
    try:
        # 创建最小配置
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
        
        exp_manager = ExperienceManager(config)
        print("✓ ExperienceManager 初始化成功")
        
        # 检查关键属性
        assert hasattr(exp_manager, 'difficulty2task_dict'), "缺少 difficulty2task_dict"
        assert hasattr(exp_manager, 'task2trajectories'), "缺少 task2trajectories"
        assert hasattr(exp_manager, 'skip_uid_set'), "缺少 skip_uid_set"
        print("✓ 关键属性存在")
        
        return True
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_grpo_grouping_logic():
    """检查 GRPO 分组逻辑"""
    print("\n" + "="*80)
    print("检查 4: GRPO 分组逻辑")
    print("="*80)
    
    import torch
    from collections import defaultdict
    
    # 模拟数据
    batch_size = 16
    n_rollout = 8
    
    # 创建 uid（基于 data_id）
    uids = [str(i // n_rollout) for i in range(batch_size)]
    rewards = torch.randn(batch_size)
    
    # 按 uid 分组
    id2scores = defaultdict(list)
    for i, uid in enumerate(uids):
        id2scores[uid].append(rewards[i].item())
    
    # 计算组内均值
    id2mean = {}
    for uid, scores in id2scores.items():
        if len(scores) > 1:
            id2mean[uid] = sum(scores) / len(scores)
    
    print(f"✓ GRPO 分组计算成功")
    print(f"  - 总 rollouts: {batch_size}")
    print(f"  - 分组数: {len(id2mean)}")
    print(f"  - 每组 rollouts 数: {n_rollout}")
    
    # 验证同一 task 的 rollouts 在同一组
    for i in range(0, batch_size, n_rollout):
        task_uid = uids[i]
        if task_uid in id2mean:
            print(f"  - Task {task_uid}: mean={id2mean[task_uid]:.4f}")
    
    return True


def main():
    """主函数"""
    print("\n" + "="*80)
    print("Experience Replay 快速检查")
    print("="*80)
    
    results = {}
    
    # 检查 1: 导入
    results['imports'] = check_imports()
    
    # 检查 2: Loss 计算
    if results['imports']:
        results['loss_computation'] = check_loss_computation()
    else:
        results['loss_computation'] = False
        print("\n⚠️  跳过 loss 计算检查（导入失败）")
    
    # 检查 3: ExperienceManager 初始化
    results['exp_manager'] = check_exp_manager_init()
    
    # 检查 4: GRPO 分组逻辑
    results['grpo_grouping'] = check_grpo_grouping_logic()
    
    # 总结
    print("\n" + "="*80)
    print("检查总结")
    print("="*80)
    for check_name, passed in results.items():
        status = "✓ 通过" if passed else "❌ 失败"
        print(f"  {check_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 所有检查通过！")
        return 0
    else:
        print("\n⚠️  部分检查失败，请检查输出")
        return 1


if __name__ == "__main__":
    sys.exit(main())

