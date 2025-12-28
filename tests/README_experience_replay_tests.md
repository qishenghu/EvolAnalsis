# Experience Replay 组件测试指南

## 概述

`test_experience_replay_components.py` 提供了快速测试 experience replay 流程各个组件的功能，无需完整训练流程。

## 使用方法

### 1. 运行所有测试

```bash
cd /home/qisheng/agent/AgentEvolver
python tests/test_experience_replay_components.py --test all
```

### 2. 运行特定测试

```bash
# 测试 ExperienceManager 基本功能
python tests/test_experience_replay_components.py --test exp_manager

# 测试轨迹存储和检索
python tests/test_experience_replay_components.py --test trajectory_storage

# 测试 ExperienceMixCollateFn
python tests/test_experience_replay_components.py --test mix_collate

# 测试 off-policy 轨迹获取
python tests/test_experience_replay_components.py --test offpolicy_retrieval

# 测试 loss 计算（两种 policy shaping 方式）
python tests/test_experience_replay_components.py --test loss_computation

# 测试 GRPO 分组机制
python tests/test_experience_replay_components.py --test grpo_grouping

# 测试 skip_uid_set 更新逻辑
python tests/test_experience_replay_components.py --test skip_uid_set
```

## 测试项说明

### 1. exp_manager: ExperienceManager 基本功能
- 测试 ExperienceManager 初始化
- 测试 `update_difficulty2task_dict` 更新难度分组
- 测试 `get_valid_replay_task_ids` 获取有效 replay task IDs

### 2. trajectory_storage: 轨迹存储和检索
- 测试 `save_trajectories_to_memory` 保存轨迹
- 测试 `get_offpolicy_trajectories_from_memory` 检索轨迹
- 测试 `max_trajectories_per_task` 限制
- 验证选择的是 entropy 最低的轨迹

### 3. mix_collate: ExperienceMixCollateFn
- 测试任务混合逻辑
- 验证 experience tasks 和 on-policy tasks 的比例
- 验证总数正确

### 4. offpolicy_retrieval: Off-policy 轨迹获取
- 测试 `get_offpolicy_batch` 批量获取
- 测试 `get_all_candidates_batch` 获取所有候选轨迹
- 验证轨迹选择逻辑

### 5. loss_computation: Loss 计算
- 测试 `higher_clip_bound` 方式
- 测试 `exgrpo_policy_shaping` 方式
- 验证两种方式产生不同的 off-policy loss
- 验证 on-policy 和 off-policy loss 分别计算

### 6. grpo_grouping: GRPO 分组机制
- 测试按 `uid`（基于 `data_id`）分组
- 验证同一 task 的 rollouts 在同一组
- 验证组内均值计算

### 7. skip_uid_set: skip_uid_set 更新逻辑
- 测试完全成功的任务被加入 `skip_uid_set`
- 验证轨迹筛选逻辑

## 快速验证清单

如果你想快速验证 experience replay 是否正常工作，可以按以下顺序运行测试：

```bash
# 1. 基本功能
python tests/test_experience_replay_components.py --test exp_manager

# 2. 存储和检索
python tests/test_experience_replay_components.py --test trajectory_storage

# 3. Loss 计算（最重要）
python tests/test_experience_replay_components.py --test loss_computation

# 4. 分组机制
python tests/test_experience_replay_components.py --test grpo_grouping
```

## 常见问题

### Q: 测试失败，提示 ModuleNotFoundError
A: 确保在正确的 Python 环境中运行，并安装了所有依赖：
```bash
pip install omegaconf torch verl
```

### Q: 如何调试特定组件？
A: 可以修改测试脚本，添加更多调试输出，或者直接调用相应的函数进行测试。

### Q: 测试数据是模拟的，如何测试真实数据？
A: 测试脚本使用模拟数据是为了快速验证逻辑。要测试真实数据，需要：
1. 准备真实的 Trajectory 和 Task 对象
2. 修改 `create_mock_trajectory` 和 `create_mock_task` 函数
3. 或者直接使用训练流程中的真实数据

## 扩展测试

如果你想添加新的测试项，可以：

1. 在 `test_functions` 字典中添加新的测试函数
2. 测试函数应该返回 `True`（成功）或 `False`（失败）
3. 使用 `print` 输出测试进度和结果

示例：
```python
def test_new_feature():
    """测试新功能"""
    print("\n" + "="*80)
    print("测试 X: 新功能")
    print("="*80)
    
    # 测试逻辑
    result = some_function()
    assert result is not None, "结果不应该为空"
    print("✓ 测试通过")
    
    return True
```

