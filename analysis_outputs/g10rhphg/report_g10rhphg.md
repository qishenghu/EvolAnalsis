# g10rhphg（DR³ v3_aug + reward-gap gate）分析报告

## 背景与目标

本报告分析 W&B run **`g10rhphg`**（配置来自 `config/alfworld_grpo_3b_dr3_hidden.yaml`，其中 `feature_mode: v3_aug` + `gap_gate_enable: true`），并与两条历史对照 run 做定量/机制对比：

- **DR³ v2（含 advantage leakage 的旧特征）**：`b2zkg9r1`
- **DR³ v5_hidden（hidden 特征，判别器更强但更容易“维持 teacher 影响”）**：`1dq5mzmv`

目标是回答：**v3_aug + gap_gate 是否修复了 v5_hidden 后期 reward 掉队，并在不牺牲稳定性的前提下提升训练 reward。**

---

## 数据来源与对齐方式

### W&B 数据（全量 time series）

使用 `wandb.Api()` 拉取每个 run 的 `scan_history()`（本次 3 条 run 均为 **98 steps**，state 都是 `crashed` 但足够对齐比较）。

W&B entity / project：

- **entity**：`qisheng001-nanyang-technological-university-singapore`
- **project**：`agentevolver`

### 本地 Trajectory 数据（更细粒度、可解释）

本地轨迹路径（用户提供）：

- `checkpoints/agentevolver/alfworld_3b_grpo_dr3_v3aug_teacher72b_bz8_ntr1_gap_gate/Trajectory`

其中包含两类文件（按 step 落盘）：

- `batch_diag_step_*.json`：每 step 的汇总诊断指标（例如 reward gap、teacher ratio 等）
- `trajectories_step_*.jsonl`：每条 rollout 的记录，单条包含 `success`、`reward.outcome`，以及 `diag.is_teacher` 等字段  
  注：单行 JSON 很大，不适合直接文本预览，建议用 Python 解析抽取字段再统计。

### “reward=success rate”的一致性校验

对 `g10rhphg` 做了 W&B onpolicy reward 与本地轨迹 onpolicy success 的对齐验证：

- `corr(wandb critic/rewards_onpolicy/mean, local succ_on) ≈ 1.0`
- mean absolute diff \( \approx 1e^{-8} \)

因此在本任务里，`critic/rewards_onpolicy/mean` 可以直接当作 **on-policy 成功率**理解，便于解释机制。

---

## 配置要点（g10rhphg 关键开关）

来自 `config/alfworld_grpo_3b_dr3_hidden.yaml` 的关键信息（与本次分析相关）：

- **DR³ 特征**：`feature_mode: v3_aug`（去掉 advantage，加入更丰富的 logprob/kl 形状统计）
- **reward-gap gate**：`gap_gate_enable: true`
- **warmup**：`apply_warmup_steps: 10`
- **buffer 与训练阈值**：
  - `buffer_size: 1024`
  - `apply_min_buf_size: 512`
  - `disc_train_min_buf_size: 256`
- **判别器训练**：
  - `disc_hidden: 64`
  - `disc_lr: 3e-4`
  - `disc_steps_per_call: 2`
- **稳定性约束**：
  - `clip_max: 10.0`
  - `dual_enable: true`，`ess_target_ratio: 0.5`
  - `ratio_shaping_mode: auto`

---

## 关键里程碑（g10rhphg）

从 W&B 的 DR³ 指标自动识别到的“首次发生 step”：

- **DR³ apply_ready 首次为真**：step **10**  
  - 与 `apply_warmup_steps: 10` 一致（warmup 前 DR³ 不 apply 修正）
- **disc_train_ready 首次为真**：step **5**  
  - 说明 buffer 达到 `disc_train_min_buf_size` 后判别器就开始训练，但在 step 10 前不会用于 apply
- **ratio_shaping_enabled 首次为真**：step **12**

解释：

- step 1~9：收集 buffer + 可能训练判别器（当满足训练阈值）  
- step 10 起：DR³ 开始在 actor loss 中 apply（对 teacher/offpolicy 部分进行 repair）  
- step 12 起：自动 ratio shaping 介入（更保守/稳定地使用权重）

---

## 核心结论（定量对比）

### 1) g10rhphg 在本段训练（1~98 steps）整体优于 v2 / v5_hidden

以 W&B 的 `critic/rewards_onpolicy/mean` 为主指标（≈ onpolicy success rate）：

| run | id | steps | reward_last | reward_max | reward_mean | reward_auc（sum over steps） |
|---|---:|---:|---:|---:|---:|---:|
| dr3_v2 | b2zkg9r1 | 98 | 0.7321 | 0.9821 | 0.5647 | 55.3371 |
| dr3_v5_hidden | 1dq5mzmv | 98 | 0.6429 | 0.9464 | 0.5361 | 52.5423 |
| **v3aug_gap_gate** | **g10rhphg** | 98 | **0.7679** | 0.9107 | **0.5701** | **55.8678** |

结论：**v3_aug + gap_gate 在这段训练里同时拿到了更高的 final、mean、AUC。**

### 2) g10rhphg 的 DR³ 稳定性健康（且不依赖 dual 救火）

最后一步（step 98）关键 DR³ 指标快照：

| run | reward_last | reward_gap_last | disc_acc_last | disc_loss_last | ess_off_last | w_off_mean_last | w_clipfrac_off_last | dual_lambda_last |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dr3_v2 (b2zkg9r1) | 0.7321 | 0.2679 | 0.9507 | 0.1213 | 16.8832 | 0.2401 | 0.0625 | 3.7945 |
| dr3_v5_hidden (1dq5mzmv) | 0.6429 | 0.3571 | 1.0000 | 0.1997 | 31.9231 | 0.5874 | 0.0000 | 0.0000 |
| **g10rhphg** | **0.7679** | **0.2321** | 0.8843 | 0.3390 | 26.4246 | 0.7476 | 0.0000 | 0.0000 |

解读（高层、与我们之前讨论一致）：

- **v2**：把 teacher/offpolicy 压得很狠（`w_off_mean` 很低），而且开始撞 clip，并触发 `dual_lambda` 兜底（说明进入“救火模式”）。
- **v5_hidden**：判别器极度自信（acc≈1.0），但 teacher/offpolicy 权重并未被系统性淡出，后期 reward 相对更差（符合之前“teacher 影响太持久”的推断）。
- **g10rhphg**：不撞 clip、dual 不介入，ESS 也稳定；在这种更“温和但可控”的修正下拿到了更高 reward。

---

## gap_gate 是否真的在“闭环淡出 teacher”？

### 1) gate 与 reward gap 正相关（符合设计）

在 `g10rhphg`，计算得到：

- `corr(dr3/gap_gate_mean, diag/group_teacher_minus_on_reward_mean) = 0.6287`

含义：

- teacher - onpolicy 的 gap 大（teacher 明显更强）→ gate 倾向更大  
- gap 变小（onpolicy 追上 teacher）→ gate 倾向下降  

### 2) 一个可操作的“有效 teacher 注入强度”指标

为方便观测 teacher 实际影响强弱，定义一个近似量：

- **effective_teacher_multiplier ≈ `w_off_mean × gap_gate_mean`**

对 `g10rhphg` 统计：

- mean ≈ **0.6393**
- last ≈ **0.4602**
- 早期常见 ~0.8-0.9，后期常见 ~0.2-0.5（整体下降趋势，非严格单调）

解释：这说明本次确实形成了“随 gap 缩小而逐步衰减 teacher 注入”的闭环行为；这正是我们此前认为 v5_hidden 需要补齐的关键组件。

---

## 机制解释：为什么这次能超过 v2 / v5_hidden？

### 相比 v5_hidden：补上了显式 fade-out 回路

此前 v5_hidden 的核心问题可以概括为：

- hidden 特征使判别器过强、过自信；
- teacher 的注入缺少明确的衰减机制；
- 导致后期模型更难“摆脱 teacher 牵引”，表现不如一种“意外抑制 teacher”的 v2。

而 `g10rhphg` 做到：

- 使用 `v3_aug` 避免 advantage leakage，同时特征更丰富（比 v3 更强）；
- 通过 `gap_gate` 把 teacher 影响从“固定强度”改成“闭环强度”，在 gap 变小时自动衰减；
- 结果是后期 reward 更好，同时 DR³ 稳定性指标没有恶化（不靠 dual 救火）。

### 相比 v2：不是“把 teacher 打死”，而是“该用才用”

从 step 98 的对比可见：

- v2 的 `w_off_mean` 极低 + 撞 clip + dual 大 → 更像系统被迫强行压 teacher，再由 dual 兜底 ESS。
- g10rhphg 的 `w_off_mean` 不低，但 **effective_teacher_multiplier** 因为 gate 在后期下降而降低 → 更像“可控淡出”，而不是“硬压 + 救火”。

这更符合我们希望的叙事：**分布修正与 teacher 使用是可解释、可控的。**

---

## 建议优先观测的指标（建议你跑后续实验时保留/对齐）

### 第一优先级（直接解释后期强弱）

- **`w_off_mean × gap_gate_mean`**（teacher 实际注入强度近似）
- **`diag/group_teacher_minus_on_reward_mean` vs `dr3/gap_gate_mean`**（gate 是否过慢/过快）

### 第二优先级（DR³ 稳定性健康度）

- `dr3/ess_off_window`
- `dr3/w_clipfrac_off`
- `dr3/dual_lambda`

经验判断：

- 如果 `dual_lambda` 经常 > 0 且 `w_clipfrac_off` 上升，说明进入“救火模式”，需要调小修正强度或更保守的 shaping。

---

## 可复现产物与本地输出文件

本次分析生成的文件目录：

- `analysis_outputs/g10rhphg/`

文件说明：

- `wandb_g10rhphg_history.csv`：W&B 全量 history（g10rhphg）
- `local_batch_diag.csv`：本地 `batch_diag_step_*.json` 汇总拼接
- `local_traj_agg.csv`：本地 `trajectories_step_*.jsonl` 按 step 聚合（teacher/onpolicy success 等）
- `g10rhphg_reward_gap_gate.png`：reward + gap + gate 曲线
- `g10rhphg_disc.png`：disc_acc / disc_loss 曲线
- `g10rhphphg_dr3_stability.png`：ESS / clipfrac / dual 曲线
- `reward_compare_runs.png`：三 run 的 onpolicy reward 曲线对比
- `local_success_teacher_vs_on.png`：本地 teacher vs onpolicy success 对比

---

## 复现方式（在本机 conda 环境下）

你可以直接用以下方式验证/再生成上述 CSV（以及图）：

1) 进入环境：

```bash
conda activate agentevolver
```

2) 读取并检查生成文件是否存在：

```bash
ls -lh /home/qisheng/agent/AgentEvolver/analysis_outputs/g10rhphg/
```

3) 如果你要重新拉数/再画图：建议直接复用本次会话里执行过的脚本逻辑（W&B + trajectory 解析），并输出到同目录。

