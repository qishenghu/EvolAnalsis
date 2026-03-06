# FRC-Lite 开发清单

> 状态：`2026-03-06` 已按当前仓库实现完成主链接入，本文档同步记录“实际实现口径”，不再只是待办清单。

本文档对应 `docs/FRC_DESIGN.md` 的工程落地版本，只记录实现目标、代码落点与验证顺序，不重复设计推导。

## 目标

完成一个可在现有 AgentEvolver 主训练链中运行的 `FRC-lite` 版本，使得：

1. replay 单位从 whole trajectory 扩展为 frontier-conditioned continuation cell；
2. teacher trajectory 可在启动时转成 frontier cells 作为初始 replay 池；
3. 训练时可以在**当前 batch 内**选择 frontier task，采样 frontier cells，并把 continuation 转成训练样本；
4. replay continuation 优先在 GRPO 中按 `cell_id` 分组，并与 projected on-policy continuation 对齐；
5. repair 支持通过配置切换 `none / similarity_gated / mixture / dr3_local`；
6. 提供可直接跑 AlfWorld 的配置和脚本。

## 实现分解

### Phase 1: Schema 与存储

- 新增 `agentevolver/schema/frontier_cell.py`
- 定义：
  - `FrontierCell`
  - `CellContinuation`
  - `CellStats`
  - `FrontierReplaySample`
- 在 `ExperienceManager` 中新增：
  - `frontier_task2cells`
  - `frontier_cell_id2cell`
  - frontier 配置解析
  - teacher 初始化建池

### Phase 2: Cell 构建与采样

- 在 `ExperienceManager` 中实现：
  - frontier 候选点提取
  - frontier hash 构造
  - trajectory -> cells
  - 当前 batch 内 task 的 cell 选择
  - continuation flatten 成 replay sample
  - on-policy trajectory -> frontier continuation projection
- 保守约束：
  - FRC-lite 默认不复用 whole-trajectory 的 recorded old log prob 对齐
  - cell replay 默认用 `exp_is_correct=false`
  - teacher-only 且当前 step 未匹配到 on-policy continuation 的 cell，允许 task-level fallback grouping

### Phase 3: 样本转换与 GRPO 接入

- 在 `EnvManager` 中新增 `convert_cell_to_cmt()`
- 在 `Linear_CMT.tokenize_steps()` 中支持：
  - `frontier_response_start_index`
  - prefix 作为 prompt、suffix 作为 response
- 在 `samples_to_dataproto()` 中允许使用 frontier 自定义 `group_id`

### Phase 4: Trainer 混入与 Repair

- 在 `ae_ray_trainer.py` 中新增：
  - `frontier_replay` 配置分支
  - `FrontierExperienceMixCollateFn`
  - frontier replay 样本注入
  - rollout 后 projected on-policy frontier continuation 注入
  - replay 后的 `cell-level group_ids`
  - advantage 级别 repair scaling
- repair 实现位置：
  - `agentevolver/module/exp_manager/cell_repair.py`

### Phase 5: 训练入口与验证

- 新增 `config/paper_alfworld_frc_lite.yaml`
- 新增 `scripts/run_alfworld_frc_lite.sh`
- 脚本当前会自动带 `--with-alfworld`
- 验证顺序：
  1. `python -m py_compile` 过核心文件
  2. `ReadLints` 检查编辑文件
  3. 在 `agentevolver` conda 环境下跑最小启动检查

## 当前实现口径

- 当前版本的训练 schedule 是：
  - 从当前 mini-batch 的 task 中选择 frontier task
  - 这些 task 的 `n_offpolicy_trajectories` 会减少对应的 on-policy rollout 数
  - rollout 后把当前 on-policy trajectory 投影成 frontier-conditioned continuation
  - projected on-policy continuation 与 replay continuation 一起进入 GRPO

- 当前版本的 frontier abstraction 是：
  - 基于“当前 observation 主体 + 最近 1-2 个 action”构造 `frontier_hash`
  - 会显式去掉 `Thought` 与 action hints
  - 不是显式 inventory / object-state abstraction

- 当前版本的 `dr3_local` 是 lightweight local-ratio proxy：
  - 在 cell 内按 group 做局部归一化；
  - 不引入新的判别器训练环；
  - 目标是先把 FRC-lite 主链打通并支持稳定实验切换。

- 如果后续要追求与设计文档中更完整的 `DR3-local discriminator` 完全一致，可以在此基础上继续把 local discriminator 接到 actor loss 前。
