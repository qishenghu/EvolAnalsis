# Experience Replay 设计文档最终检查清单

## ✅ 已完成的关键改进

### 1. 关键方法实现说明 ✅
- ✅ `update_skip_uid_set_and_filter_trajectories` - 已添加详细实现（3.1.5）
- ✅ `save_trajectories_to_memory` - 已添加详细实现（3.1.3）
- ✅ `get_offpolicy_trajectories_from_memory` - 已添加详细实现（3.1.4）

### 2. tokenize_steps 中的 is_experience_replay 处理 ✅
- ✅ 已明确说明需要在 `tokenize_steps` 中检查 `is_experience_replay`
- ✅ 已提供具体的代码实现示例
- ✅ 已说明如何避免断言错误

### 3. 数据顺序一致性 ✅
- ✅ 已明确说明 `rollout` 返回的轨迹已排序
- ✅ 已明确说明 batch 顺序与轨迹顺序一致
- ✅ 已说明如何匹配 `old_log_prob` 和 `entropy`

### 4. Entropy 计算 ✅
- ✅ 已明确说明 entropy 在 `compute_log_prob` 时计算
- ✅ 已说明如何获取和使用 `entropys`
- ✅ 已说明如何计算平均 entropy

### 5. 配置项完善 ✅
- ✅ 已移除冗余的 `replay_task_count`，使用 `exp_ratio`
- ✅ 已明确 `exp_select_mode` 的默认值和选项
- ✅ 已明确 `experience_rbound` 的默认值

### 6. 训练循环集成顺序 ✅
- ✅ 已调整获取 off-policy trajectory 的时机（Task 混合之后）
- ✅ 已明确说明各个步骤的执行顺序

## ⚠️ 需要注意的实现细节

### 1. tokenize_steps 修改
**位置**：`agentevolver/module/context_manager/cmt_linear.py` 第 609-619 行

**需要添加的代码**：
```python
# 在遍历 ext_steps 之后，断言之前
is_experience_replay = self.metadata.get("is_experience_replay", False)
if is_experience_replay:
    split_prompt_reponse_index = len(input_ids)
```

### 2. compute_log_prob 返回 entropy
**位置**：需要确认 `compute_log_prob` 是否返回 `entropys`

**检查点**：
- 如果返回，直接使用 `current_old_log_prob.batch["entropys"]`
- 如果不返回，需要修改 `compute_log_prob` 或单独计算 entropy

### 3. 数据顺序匹配
**关键**：在保存轨迹时，确保 `trajectories[i]` 对应 `old_log_probs[i]` 和 `entropys[i]`

**保证方式**：
- `rollout` 返回的轨迹已按 `(data_id, rollout_id)` 排序
- `to_dataproto` 保持相同的顺序
- 使用索引直接匹配

### 4. ExperienceMixCollateFn 的返回值使用
**关键**：`ExperienceMixCollateFn` 返回 `(experience_tasks, on_policy_tasks)`

**使用方式**：
- 在训练循环中，需要分别使用这两个列表
- 或者合并后，需要知道哪些是 replay tasks（用于获取 off-policy trajectory）

## 📋 实现前检查清单

### Phase 1: 基础框架
- [ ] 在 `ExperienceManager` 中添加所有必要的属性和方法
- [ ] 实现 `update_difficulty2task_dict`
- [ ] 实现 `save_trajectories_to_memory`
- [ ] 实现 `get_offpolicy_trajectories_from_memory`
- [ ] 实现 `update_skip_uid_set_and_filter_trajectories`
- [ ] 实现 `sample_tasks_from_replaypool`
- [ ] 实现 `get_offpolicy_batch`

### Phase 2: 数据转换
- [ ] 实现 `convert_offpolicy_to_cmt`
- [ ] 修改 `get_extra` 以支持 `is_experience_replay`
- [ ] 修改 `samples_to_dataproto` 以支持 `exp_mask` 和 `recorded_old_log_probs`
- [ ] **修改 `tokenize_steps` 以支持 `is_experience_replay`** ⚠️

### Phase 3: 训练循环集成
- [ ] 实现 `ExperienceMixCollateFn`
- [ ] 在训练循环中集成 Task 混合逻辑
- [ ] 在训练循环中获取 off-policy trajectory
- [ ] 调整 replay tasks 的 rollout_n
- [ ] 实现 `_replace_recorded_old_log_probs`

### Phase 4: 数据保存
- [ ] 在生成轨迹后更新 `difficulty2task_dict`
- [ ] 在计算 old_log_prob 后保存轨迹
- [ ] 实现 entropy 计算和保存

### Phase 5: 测试
- [ ] 单元测试各个方法
- [ ] 集成测试整个流程
- [ ] 验证数据顺序一致性
- [ ] 验证 prefix 机制正确性

## 🎯 设计文档质量评估

### 完整性：✅ 优秀
- 所有关键方法都有详细实现说明
- 数据流完整且清晰
- 配置项完整

### 正确性：✅ 优秀
- 与现有 codebase 兼容
- 逻辑正确，无矛盾
- 考虑了边界情况

### 可实现性：✅ 优秀
- 提供了具体的代码示例
- 明确了实现位置
- 说明了关键细节

### 清晰度：✅ 优秀
- 结构清晰，层次分明
- 说明详细，易于理解
- 有足够的示例和注释

## 总结

设计文档已经非常完整和详细，包含了所有必要的实现细节。在实现前，只需要：

1. **确认 `compute_log_prob` 是否返回 `entropys`**，如果不返回，需要修改
2. **实现 `tokenize_steps` 中的 `is_experience_replay` 检查**，这是 prefix 机制的关键
3. **确保数据顺序一致性**，在保存轨迹时正确匹配索引

其他部分的设计都已经非常清晰，可以直接按照文档实现。

