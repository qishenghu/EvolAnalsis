# CATALYST 训练侧集成 — M1 实现规格(IMPL SPEC)

状态:**已实现**(2026-08-08 审定放行:D1-D5 按默认;修正 A1 data_id 撞号断言、
A2 补 T9 混合批 dataproto 测试、A3 退休×TTL 语义写入 Governor docstring)。
实现落点与本规格 §4 一致;与代码现实的偏差逐条见交付报告。测试:
tests/test_catalyst_*.py(70 用例)。日期:2026-08-08。范围:M1(提示臂 rollout /
分臂基线 / 去提示重放 BC / 治理层 / 遥测 / 素材构建)。M2(entry-k 状态课程、
门控 BC 兜底)不在本文。

前置阅读:`CATALYST_论文总设计_2026-08-08.md`、`SIEVE_算法提案_v1` §4、
`CATALYST_试点预注册` §6.5、CLAUDE.md 张量制式节。

---

## 0. 不变式与硬约束(实现必须满足)

1. **零模仿**:教师 token 在任何配置下不进任何损失。M1 训练批中根本不存在教师
   token(`teacher_experience.enable` 必须为 false,启动时断言);唯一的教师影响
   通道是 prompt 里的 think 摘要(不参与损失)与由它转写出的**学生自写** token。
2. **纯增量**:所有新行为藏在 `catalyst.*` 键后,默认全关。默认关闭时:
   * 不新增任何 batch tensor key、不改任何现有 tensor 的值;
   * uid 构造、advantage 计算、损失分派逐字节走现有路径;
   * 有单元测试作证(见 §8 T1)。
3. **张量制式**:新增 mask `catalyst_replay_mask` 采用 full-sequence 制式
   (`(bs, prompt_len+response_len)`,与 `teacher_mask` 同构);actor 内一律
   `[:, -response_length:]` 切片后使用。arm 标记走 non_tensor(numpy object)。
4. **不碰**:`train_h200/` 下 grpo_smoke 与 chord/luffy 草稿、
   `scripts/convert_teacher_v2_to_training.py`、duet/vllm2/duet-train 环境、在跑作业。
   (catalyst smoke 配置是**新文件**,fork 自 grpo_smoke,不改原文件。)

---

## 1. 精读得出的机制事实(约束设计的地基)

这些是写规格前逐一验证过的事实,后文改动点全部建立在其上:

* **F1 · per-rollout prompt 差异是结构上支持的**。
  `env_manager.rollout()`(env_manager.py L481-622)为每个 rollout 槽位构造独立的
  `TrajExpConfig`(L544-545),`AgentFlow.execute()`(agent_flow.py L87)拿到它后在
  `save_init_input` 之前有现成的 init 消息改写点(L119-128,experience 注入就在
  这里)。**不存在"rollout 生成路径不支持 per-rollout prompt 差异"的结构性障碍**。
* **F2 · 快照训练制式**。ICLR2027 链路 `snapshot_training: true`:每条 rollout 只产
  **1 个训练样本** = token 加权随机选中的一个 decision 快照
  (cmt_linear.py `_select_decision_snapshot` L949-978、
  `_group_tokenize_decision_snapshot` L980-1062),response = vLLM 原始采样 token id
  (`_capture_decision_snapshot` L374-448,精确 id + per-token logprob)。
  重放 BC 样本必须做成同形状(单 decision 快照式样本)。
* **F3 · uid=GRPO 分组键的唯一产生点**在 trainer fit
  (ae_ray_trainer.py L3941-3950):`uid = str(int(group_ids[i]))`,group_ids 来自
  sample.data_id。分臂基线只需在这一个点改 uid 语义。
* **F4 · 行为 logprob 恒等门会豁免重放行**。`use_rollout_log_probs_as_old: true`
  (smoke 配置)下,`_expected_onpolicy_behavior_mask`(L186-222)对
  `extras.is_experience_replay=True` 的行不要求 vLLM logprob;但
  `samples_to_dataproto` L1481-1488 要求 `snapshot_training` extras 为 False 的样本
  才允许缺 `rollout_log_probs`。→ 重放样本 extras 必须:
  `is_experience_replay=True, is_catalyst_replay=True, snapshot_training=False,
  rollout_log_probs=None`。
* **F5 · 单样本 GRPO 组的优势不是 0**。`compute_grpo_outcome_advantage`
  L1314-1316:组大小 1 时 mean=0/std=1 → adv = 原始 score。重放样本自成一组时
  **必须显式清零 advantage**,不能指望分组机制。
* **F6 · 零优势跳更新守卫已兼容重放**。`_should_skip_zero_advantage_grpo_actor_update`
  (L724-798)把 exp_mask token 视为 auxiliary → 有重放样本的批不会被
  `skip_zero_advantage_grpo_update` 跳过(BC 损失不会丢)。无需改动。
* **F7 · union 按 task_id 回拼**。`union_gen_batch_via_task_id`(L1252-1269)对不在
  当前 batch 任务列表里的 task_id 直接 KeyError → **重放注入只能取当前批任务**
  (与 LUFFY/RePO 现行 replay 语义一致)。
* **F8 · 注入模板与剥除逻辑已有单一事实源**。
  `scripts/collect_student_rollouts_hinted.py` L951-955 的 `HINT_TEMPLATE`
  (sha256=`e72d043eb44793852cf2b342697ffe899f6c60aed72b409e25199f1f1074efb0`,
  已与试点 manifest `/contract/hint_injection/hint_template_sha256` 核对一致);
  注入位置 = init 消息中**第一条 user 消息末尾**(HintedAgentFlow L1029-1050);
  剥除逻辑 = `analysis/catalyst_purity_score.py::strip_hint_messages` L132-158
  (prefix..suffix 整段剥除 + 恰剥 1 条 + 全局无 marker 残留断言)。
  训练侧一律 **ast 静态提取** HINT_TEMPLATE(照抄 purity 脚本 L82-96 的做法,
  避免 import 采集器的重依赖)并断言 sha 等于上述 pin 值。
* **F9 · 试点 hint 清洗管线已被逐字节复现**(本次精读中实证,230/230 全对):
  对 teacher v2 记录按 decision 顺序取 `<think>…</think>` 内文(strip)、`'\n'.join`、
  然后依次:
  1. 去 flash 草稿块:`re.sub(r'response<action>.*?</action>', '', t, flags=DOTALL)`
  2. 修粘连边界:`re.sub(r'\.(?=[A-Z])', '. ', t)`
  3. 截断:`t[:5000]`
  对照 `/projects_vol/gp_wangwy/qisheng/duet_h200/catalyst_pilots/data/
  {alfworld,webshop}_hints.json`:AF 120/120、WS 110/110 逐字节相等。
  (试点当时的提取脚本没有归档进 repo——这是本次发现的缺口,
  `build_catalyst_hints.py` 即其正式归档,并带回归 fixture。)
* **F10 · 训练用教师素材源**:
  `data/teacher_trajectories/iclr2027_flash/{alfworld,webshop}_dsv4flash_success_dedup.jsonl`
  (openrouter_teacher_trajectory_v2,一任务一条成功轨迹,自带 manifest)。
* **F11 · actor 的 entropy 只在 `entropy_coeff != 0` 时才算**
  (het_actor.py L1457-1460)。φ=logπ+H 需要 per-token 全词表熵 → catalyst 重放 BC
  开启时必须强制 `calculate_entropy=True`(有额外算力成本,见 §7 R7)。
* **F12 · 成功轨迹的消息形状**:`remove_last_context`(cmt_linear.py L190-203)弹掉
  末尾 env 消息 → messages = init 前缀 + (assistant,user)*(T-1) + assistant,
  与 `cll_teacher_profile.reconstruct_ext_messages`(L142-195)假设一致。重放池
  直接按 full_context 的 author 标签存消息,连推断都省掉。

---

## 2. 新增 config 键清单(全部默认关闭/无副作用)

落点 `config/agentevolver.yaml`(默认值,给所有实验兜底)+ 新 smoke 配置显式覆盖。

```yaml
# ---- exp_manager 侧(trainer/exp_manager 可见) ----
exp_manager:
  catalyst:
    enable: false                  # 总开关;false 时下面一切无效且零副作用
    hints:
      file: null                   # data/catalyst_hints/{env}_{teacher}.json;enable 时必填,缺失 fail-fast
      require_manifest: true       # 校验旁挂 manifest(源 jsonl sha、清洗版本、条数)
    governance:
      s_hi: 0.8                    # ρ 公式的 s*(SR_bare 达标线)
      rho_max: 0.5                 # ρ 上限
      delta_u: 0.0                 # δ:U>δ 才开提示臂
      ema_alpha: 0.2               # SR_bare/SR_hint EMA 步长(按"该任务出现在批里"为一次更新)
      u_bootstrap_min_obs: 8       # 该任务提示臂累计 rollout 数 < 此值时 U 门直接放行(冷启动自举)
      min_hint_rollouts: 2         # k_hint 量化下限(0 或 ≥2;防单样本臂,见 §4.2)
      max_hint_rollouts: -1        # -1 = n-2(保裸臂 ≥2)
      retire_windows: 3            # U≤δ 连续 N 个更新窗 → 该任务退休(教师素材停用)
      state_dir: null              # 默认 {default_local_dir}/catalyst_state;断点续训持久化
    arm_baseline:
      enable: true                 # 分臂基线(仅 catalyst.enable=true 时生效)
      std_fallback_to_group: true  # 臂内样本 <2 时 std 回退整组(防御;正常被 min_hint_rollouts 挡住)
    replay:
      enable: false                # 重放通道子开关(依赖 catalyst.enable)
      per_task: 1                  # 每个在批任务每步注入的重放 BC 样本数上限
      pool_max_per_task: 4         # 池内每任务保留条数(新胜旧,FIFO)
      ttl_steps: 20                # 窗口 TTL:inserted_step 距今 > ttl 即淘汰(防陈旧)
      audit_on_insert: true        # 入池时带提示重渲染,断言逐 decision prompt ids == 快照 ids(军规自检)
    thermostat:
      enable: false                # 熵恒温器
      h_ref: 0.35                  # H_ref(nats/token;smoke 时按裸跑实测 Ĥ 校准后再定)
      eta: 0.01                    # λ 步长
      lambda_init: 0.0
      lambda_max: 1.0              # λ 上限(防失控)

# ---- actor 侧(het_actor 只能看到 actor_rollout_ref.actor.*) ----
actor_rollout_ref:
  actor:
    catalyst:
      replay_bc:
        enable: false              # 与 exp_manager.catalyst.replay.enable 同开同关(trainer 启动断言一致)
        beta: 0.1                  # β:全局重放 BC 系数
        w_cap: 1.0                 # w = min(w_cap, exp(φ/tau))
        phi_tau: 1.0               # τ(消融用;默认 1)
        exclude_replay_from_entropy_kl: true   # 熵奖励/KL 项的 mask 是否剔除重放 token(见 §4.3)
```

互斥断言(trainer `__init__` 或 fit 开头,fail-fast):`catalyst.enable=true` 时要求
`teacher_experience.enable=false`、`experience_replay.enable=false`、`repo.enable=false`、
`state_channel.enable=false`、`use_chord=false`、`use_dr3=false`、`use_dapo=false`、
`algorithm.grpo.teacher_baseline_separation.enable=false`(原开关原样保留不动,只是
与 catalyst 不同时开)。`replay_bc.enable != replay.enable` 报错。

---

## 3. 字段/张量流转图(M1 全景)

```
teacher v2 jsonl ──scripts/build_catalyst_hints.py──> data/catalyst_hints/{env}_{teacher}.json (+manifest)
                                                            │ exp_manager 启动加载 (fail-fast)
                                                            ▼
fit() 每步:
  tasks ──CatalystGovernor.plan_arms(tasks, n)──> 每任务 k_hint (ρ_task 量化)
     │        (读 SR_bare/SR_hint EMA、U、退休集)
     ▼
  TaskExpConfig.catalyst_hint_slots: List[Optional[str]]   # 长度 n;槽位含 hint 文本或 None
     ▼ env_manager.rollout() 按 rollout_id 分发
  TrajExpConfig.catalyst_hint_text / catalyst_arm ("hint"/"bare")
     ▼ AgentFlow.execute() 顶部(manage_rollout_context 之前,与试点 HintedAgentFlow 同序)
  init_messages 第一条 user 消息末尾 += HINT_TEMPLATE.format(hint=...)   # 逐字节同模板
  cmt.metadata["catalyst_arm"], ["catalyst_hint_sha256"]
     ▼ rollout 完成
  ├─ CatalystGovernor.update(trajectories, step)     # 双臂 EMA / U / 退休
  ├─ CatalystReplayPool.insert(成功的 hint 臂 CMT)   # 存 author 标签消息 + 各 decision 采样 ids(+入池军规审计)
  └─ CatalystReplayPool.build_replay_samples(在批 tasks, step)
        · strip_hint_messages(消息副本)              # purity 同款剥除+断言
        · StructuredContextPolicy.build 逐 decision 重渲染 prompt(无提示)
        · 选 1 个 decision(token 加权,seed 掺 global_step)→ Sample
        · extras: is_experience_replay=T, is_catalyst_replay=T, snapshot_training=F
        · data_id = 100000+i(自成 uid 组,不入活 GRPO 组)
     ▼
  env_manager.to_dataproto(all_trajectories, optimizer_batch=True, extra_samples=replay_samples)
     · samples_to_dataproto:若批内存在 is_catalyst_replay → 新增 full-seq tensor
       catalyst_replay_mask(response 段=response_loss_mask,prompt 段=0);默认关闭时该 key 不存在
     · non_tensor: extras[i]["catalyst_arm"]
     ▼ trainer
  uid 构造点:catalyst.arm_baseline.enable 时 uid = f"{gid}|h" (hint 臂) / str(gid)(其余)
     ▼ compute_advantage(现有 GRPO 函数,自动实现 (task,arm) 分组均值/方差)
  catalyst 后处理:
     ① advantages[replay 行] = 0                    # F5
     ② thermostat: A += λ·(−old_log_probs − b)·mask(非重放行);λ ← clip[λ+η(H_ref−Ĥ)]₊
     ▼ update_actor(batch 含 catalyst_replay_mask)
  het_actor:
     · select_keys += catalyst_replay_mask(若存在)
     · replay_bc.enable → calculate_entropy=True
     · pg/entropy/kl 的 mask 剔除重放 token;BC:w=sg[min(1,exp((logπ+H)/τ))],
       L += β·agg(w·(−logπ), replay_mask)
     ▼ wandb: catalyst/* 全套遥测
```

---

## 4. 六个工作项 → 改动点明细(文件:函数:行号)

### 4.0 新模块 `agentevolver/module/exp_manager/catalyst.py`(NEW,~500 行)

一切 CATALYST 逻辑集中于此,老文件只加"细钩子"。内容:

| 对象 | 职责 | 关键实现来源 |
|---|---|---|
| `load_hint_template()` | ast 提取 HINT_TEMPLATE + sha pin 断言(F8) | 照抄 catalyst_purity_score.py L82-99 |
| `build_hint_from_v2_record(record) -> str` | F9 清洗管线(与脚本共用,单一事实源) | 本文 F9 三条规则 |
| `strip_hint_messages(messages) -> None` | 就地剥除 + 3 断言 | 移植 catalyst_purity_score.py L132-158(语义逐条保持) |
| `CatalystHintBook` | 加载 hints json + manifest 校验;`get(task_id)` | collect_student_rollouts_hinted.load_hints 的裁剪版 |
| `CatalystGovernor` | per-task EMA(SR_bare/SR_hint/U/obs 计数)、ρ、退休集、`plan_arms()`、`update()`、`telemetry()`、`save/load_state()` | 新写;公式见 §4.4 |
| `CatalystReplayPool` | `insert()`(含入池军规审计)、TTL/容量淘汰、`build_replay_samples()` | 渲染链 = reconstruct(author 直存,免推断)+ `StructuredContextPolicy.build`(context_policy.py L316);Sample 构造对齐 `_group_tokenize_decision_snapshot`(cmt_linear.py L980-1062) |

### 4.1 工作项① 提示臂 rollout(在线转写)

| # | 文件:位置 | 改动 | 
|---|---|---|
| 1a | `exp_manager.py::TaskExpConfig`(L19-22) | 加字段 `catalyst_hint_slots: Optional[List[Optional[str]]] = None`、`catalyst_hint_sha256: Optional[str] = None`(dataclass 默认 None,纯增量) |
| 1b | `exp_manager.py::TrajExpConfig`(L24-33) | 加字段 `catalyst_hint_text: Optional[str] = None`、`catalyst_arm: str = "bare"` |
| 1c | `exp_manager.py::ExperienceManager.__init__`(L39-129 末尾) | `catalyst.enable` 时构造 `self.catalyst = CatalystGovernor(cfg, hint_book, replay_pool)`;hints 文件缺失/manifest 不符 → raise(对齐教师加载 fail-fast L97-103 风格) |
| 1d | `ae_ray_trainer.py::fit`(tasks 建好后、L3794 `get_complete_exp_configs` 之后) | catalyst on:`self.exp_manager.catalyst.plan_arms(tasks, task_exp_configs, n_rollout, global_step)` 就地往每个 TaskExpConfig 填 hint_slots(槽位顺序:前 k_hint 个 rollout_id 为 hint 臂——确定性,便于测试) |
| 1e | `env_manager.py::rollout`(L539-550) | 构造 TrajExpConfig 时透传:`catalyst_hint_text = slots[rollout_id]`、`catalyst_arm = "hint" if slots[rollout_id] else "bare"`(slots 为 None 时零改动路径) |
| 1f | `agent_flow.py::AgentFlow.execute`(L109 之后、L119 `manage_rollout_context` 之前) | 注入块(≈12 行):`if getattr(traj_exp_config,"catalyst_hint_text",None):` deepcopy init_messages → 第一条 user 消息 content += `HINT_TEMPLATE.format(hint=...)`;无 user 消息 raise。随后 `self.cmt.metadata["catalyst_arm"]`、`["catalyst_hint_sha256"]` 落 metadata。**位置与试点 HintedAgentFlow 同序(save_init_input/上下文策略之前)、同目标消息(首条 user)、同模板字节** |
| 1g | `env_manager.py::get_extra`(extras dict L1052-1114) | extras 增加 `"catalyst_arm": cmt.metadata.get("catalyst_arm", "bare")`(恒定加键;值默认 "bare",不影响默认路径的行为消费方——现有代码不读该键) |

验证模式(mode="validate")天然不注入:validate 的 TaskExpConfig 不经过 plan_arms。

### 4.2 工作项② 分臂基线

| # | 文件:位置 | 改动 |
|---|---|---|
| 2a | `ae_ray_trainer.py::fit` uid 构造点(L3941-3950) | catalyst.arm_baseline on 时:`uid = f"{gid}|h"` 若该样本 extras `catalyst_arm=="hint"`,否则 `str(gid)`。重放样本 data_id=100000+i 自然自成组。**这一处即完成 (task,arm) 分组的均值/方差分离**——`compute_grpo_outcome_advantage`(L1272-1332)按 uid 分组,零数学新代码 |
| 2b | `CatalystGovernor.plan_arms` | k_hint 量化:`k = round(ρ_task·n)`;`k<min_hint_rollouts → 0`;`k > n−2 → n−2`(裸臂恒 ≥2)。保证两臂组内样本数都 ≥2,避开 F5 单样本组病态(std=1、mean=0) |
| 2c | 保留原机制 | `teacher_baseline_separation`(L1336-1434 与 L1488-1519)**一行不动**,开关不动;catalyst 配置里它保持 false(§2 互斥断言) |

风险:uid 后缀会同时改变 DAPO 动态采样(L5048-5061)与 `duet/group_*`、
`compute_teacher_effect_metrics` 的分组视角。DAPO 在 catalyst 配置强制关(§2);
诊断指标把两臂看成两组,属于可接受的语义漂移(catalyst/ 遥测另有专表)。
`trajectories_to_samples` 的纯 on-policy 完整性检查(env_manager.py L1177-1223)用的
是 data_id 而非 uid,臂拆分不影响"每 UID 组恰 n 条"校验。

### 4.3 工作项③ 去提示重放 BC 通道

**池维护(trainer 侧)**

| # | 文件:位置 | 改动 |
|---|---|---|
| 3a | `ae_ray_trainer.py::fit`(L3886-3887 `else: all_trajectories = trajectories` 分支后,插 catalyst 块) | ① `exp_manager.catalyst.update(trajectories, self.global_steps)`(治理 EMA,见 4.4);② 对成功(`reward.success_rate>0`)且 `catalyst_arm=="hint"` 且未 discarded、未 length-truncation 的 CMT:`replay_pool.insert(...)` 存 `{task_id, rollout_id, msgs:[(author,role,content)...from full_context 原文], decisions:[{step_index, completion_content, completion_token_ids(采样原 ids), assistant_content}], inserted_step}`;③ TTL/容量淘汰 |
| 3b | `CatalystReplayPool.insert` 军规审计(audit_on_insert) | 带提示原样(不剥)逐 decision `policy.build(prefix)`,断言 prompt ids == 快照 `prompt_token_ids`(F2 快照里有)。**比 purity 脚本更强**:那边只能验 completion sha,这里能验证整条重构链在本训练栈下 100% 保真;然后剥提示的渲染差异就只剩 hint 文本本身。失败 → 该条不入池 + `catalyst/replay_audit_failures` 计数(预期恒 0) |

**样本注入(渲染 → 批)**

| # | 文件:位置 | 改动 |
|---|---|---|
| 3c | `CatalystReplayPool.build_replay_samples(tasks, policy, tokenizer, step)` | 只取**在批任务**(F7)。每条:副本上 `strip_hint_messages`(同款断言);对选中 decision(token 加权,digest 掺 `global_step` → 同一 episode 跨步训练不同 decision)`policy.build(去提示前缀)` 得 prompt ids;response = **原采样 completion_token_ids**(完全零重分词漂移;军规 completion 恒等由构造保证);拼 Sample(字段对齐 `_group_tokenize_decision_snapshot` L1030-1061:loss_mask prompt 全 0/response 全 1);`data_id=str(100000+i)`、extras 见 F4 + `is_catalyst_replay=True` + `catalyst_replay_inserted_step`。prompt 超限(去提示只会更短,理论不可能)→ 丢弃+计数 |
| 3d | `env_manager.py::to_dataproto`(L626-646)与 `trajectories_to_samples`(L1118-1235) | 新增可选参 `extra_samples: Optional[List[Sample]] = None`(默认 None=零改动);在 Step1 转换后 `sample_arr_final += extra_samples`,一起走对齐/padding。对齐 DP(L105-158)把每个重放样本当独立完整 uid 组处理,可被整组裁剪(可接受) |
| 3e | `env_manager.py::samples_to_dataproto`(teacher_mask 段 L1400-1473 旁) | **仅当批内存在 `is_catalyst_replay` 样本时**构造 full-seq `catalyst_replay_mask`(response 段= `response_loss_mask`,prompt 段=0,与 teacher_mask_full 同构),加入 batch_fields。默认关闭 → key 不存在 → 字节等价(§0.2) |
| 3f | `ae_ray_trainer.py::fit` compute_advantage 之后(L5219 后插 catalyst 块) | `advantages[replay_rows] = 0`(F5;replay_rows 由 catalyst_replay_mask 行和 >0 判定)。同块做 thermostat(4.4) |

**BC 损失(actor 侧)**

| # | 文件:位置 | 改动 |
|---|---|---|
| 3g | `het_actor.py::update_policy` select_keys(L1072-1086) | `if "catalyst_replay_mask" in data.batch: select_keys.append(...)`(仿 teacher_mask L1078-1081) |
| 3h | `het_actor.py` micro 循环 L1453-1460 | `catalyst_bc = self.config.get("catalyst",{}).get("replay_bc",{})`;enable 时 `calculate_entropy=True`(F11) |
| 3i | `het_actor.py` L2685 默认分支前后 | catalyst on 时:`replay_mask = catalyst_replay_mask[:, -response_length:] * response_mask`;`pg_mask = response_mask * (1 - replay_row_expand)`;把 `pg_mask` 传入 `het_compute_token_on_off_policy_loss`(替换其 response_mask 实参;函数本身不动)。熵奖励(L2730-2734)与 KL(L2738-2744)的 agg mask 亦用 `pg_mask`(`exclude_replay_from_entropy_kl: true`) |
| 3j | 同上,ret_dict 之后 policy_loss 组装前 | `with torch.no_grad(): phi = (log_prob + entropy) / tau; w = torch.clamp(torch.exp(phi), max=w_cap)`(**w 全程 no_grad = stop-grad**);`bc_losses = w * (-log_prob)`;`bc_loss = agg_loss(bc_losses, replay_mask, loss_agg_mode)`(het_core_algos.agg_loss L17);`policy_loss = policy_loss + beta * bc_loss`。**断言零模仿**:`assert not has_teacher_data`(catalyst 配置下 teacher_mask 恒空,防御性双保险)。NaN 防御:replay_mask 全 0 时 bc_loss=0(agg_loss 对空 mask 返回 nan → 仿 compute_chord_sft_loss L1823-1824 置 0) |
| 3k | 同文件 metrics 段(L2820-2842 旁) | `catalyst/w_mean, w_p10, w_p50, w_p90, bc_loss, bc_weighted_loss, replay_token_count`(仅 catalyst on 时追加) |

要点:φ 从**同一次 forward** 的 `log_prob`/`entropy` 就地取(零额外 forward);
`old_log_probs` 对重放行等于当前策略重算值(trainer L3980),ratio≈1,但反正
pg_mask 已剔除,重放行不进 PG/熵/KL 任何一项——**只走 BC**。

### 4.4 工作项④ 治理层

全部在 `CatalystGovernor`(exp_manager 持有,trainer 调两个钩子)。

**状态**(per task):`sr_bare_ema, sr_hint_ema, n_bare_obs, n_hint_obs,
u_low_streak, retired`。持久化:`save_state()/load_state()` json 落
`{default_local_dir}/catalyst_state/step_{n}.json`,fit 的 checkpoint 时机同步调
(对齐 experience_pool 的续训语义,L3515-3527)。

**update(trajectories, step)**(4.3-3a 调用):按 task 聚合本步 rollout,
`sr_arm ← (1-α)·sr_arm + α·batch_arm_sr`(该臂本步有样本才更新;obs 计数累加);
`U = sr_hint_ema − sr_bare_ema`;若该步该任务两臂都有读数:`U ≤ delta_u` 则
`u_low_streak += 1` 否则清零;`u_low_streak ≥ retire_windows → retired=True`
(M1 内永久;记录 step)。

**plan_arms(tasks, ...)**(4.1-1d 调用),对每个任务:
```
R0: 无 hint 素材 / retired / sr_bare_ema ≥ s_hi        → k_hint = 0
R1: 其余,且 (n_hint_obs < u_bootstrap_min_obs 或 U > delta_u):
      ρ = clip(1 − sr_bare_ema / s_hi, 0, rho_max)
      k_hint = quantize(round(ρ·n))   # §4.2-2b:{0} ∪ [min_hint_rollouts, n−2]
```
冷启动:任务首次见面 `sr_bare_ema` 初始化为 0(未证明会做 → 视为难)→ ρ=ρ_max;
U 门凭 `u_bootstrap_min_obs` 放行(否则提示臂永远没有第一批读数,鸡生蛋死锁)。

**熵恒温器**(trainer 侧,4.3-3f 同块):
```
mask   = response_mask ∧ loss_mask ∧ ¬replay_rows          # response-only 制式
H_hat  = masked_mean(entropys, mask)                        # L3981 已算好的 entropys,零成本
b      = masked_mean(−old_log_probs, mask)                  # 平移零均值基线
advantages[mask] += λ · (−old_log_probs − b)[mask]          # A′ = A + λ(−logπ−b)
λ ← clip(λ + eta·(h_ref − H_hat), 0, lambda_max)            # 对偶上升,投影非负
```
`old_log_probs` 在 `use_rollout_log_probs_as_old` 下已是行为策略 logprob
(L4595-4600 替换后),语义正确(采样时刻的 logπ);全部 detached,无梯度泄漏。
执行顺序:先清零重放行 → 再平移(平移 mask 排除重放行,保证重放行 adv 恒 0)。

### 4.5 工作项⑤ 遥测(wandb,前缀 `catalyst/`)

trainer 侧(metrics dict,L5982 `tracker.log` 统一出口),每步:

| 指标 | 含义(F4 图证据) |
|---|---|
| `catalyst/rho_mean`, `rho_max_task` | 本批 R1 任务 ρ 均值/最大 |
| `catalyst/tasks_r0`, `tasks_r1`, `tasks_retired_total` | 路由分布与退休累计(退休动力学) |
| `catalyst/hint_rollouts`, `hint_rollout_frac` | 实际提示臂 rollout 数/占比 |
| `catalyst/sr_bare_ema_mean`, `sr_hint_ema_mean`, `u_ema_mean`, `u_pos_frac` | 双臂 SR 与因果效用 U(批内任务均值) |
| `catalyst/sr_bare_batch`, `sr_hint_batch` | 本步两臂原始 SR(未平滑,可对照) |
| `catalyst/replay_pool_entries`, `replay_pool_tasks`, `replay_pool_age_mean`, `replay_pool_age_max` | 池大小与年龄(TTL 观测) |
| `catalyst/replay_samples_in_batch`, `replay_render_drops`, `replay_audit_failures` | 注入量与军规健康度(后两者预期 0) |
| `catalyst/lambda`, `h_hat`, `h_ref`, `adv_shift_abs_mean` | 恒温器闭环 |
| actor 侧(3k):`catalyst/w_mean, w_p10, w_p50, w_p90, bc_loss, bc_weighted_loss, replay_token_count` | w 分布分位数与 BC 力度 |

另:每步把 per-task `{task_id: sr_bare, sr_hint, u, rho, retired}` dump 到
`rollout_data_dir/catalyst_gov_step_{n}.json`(对齐 batch_diag 先例 L5804-5824),
供 F4 逐任务曲线离线出图。

### 4.6 工作项⑥ `scripts/build_catalyst_hints.py`(NEW)

薄脚本(~150 行),核心清洗函数 import 自 `catalyst.py::build_hint_from_v2_record`
(单一事实源,训练加载端与构建端共享)。

* CLI:`--input <v2 jsonl>... --env alfworld --teacher dsv4flash --output-dir data/catalyst_hints/`
* 行为:逐条 v2 记录(复用 `_dict_to_teacher_trajectory` 的结构校验思路但纯只读)
  → F9 三步清洗 → `{task_id: {"raw": hint}}`(与试点 hints 文件同构;`raw` 键名
  兼容试点/HintBook 两端)→ `{env}_{teacher}.json` + `{env}_{teacher}.json.manifest.json`
  (源文件 sha256、记录数、任务数、清洗规则版本号 `catalyst_hint_clean/1.0.0`、
  HINT 无关——注入模板 sha 不在此,归 HintBook 侧核)。
* fail-fast:重复 task_id、think 段缺失(计数并列出;缺 think 的任务**不产 hint**,
  该任务自然落 R0)、空清洗产物。
* 回归锚:tests 内置 3 组真实 (v2 record → 试点 hint) fixture(AF 2 + WS 1,含
  草稿块与粘连案例),逐字节断言(F9 的 230/230 已验证此管线正确)。

---

## 5. 与现有机制的交互汇总(为什么不踩)

| 现有机制 | 交互 | 结论 |
|---|---|---|
| 纯 on-policy GRPO 完整性检查(env_manager L1156-1235) | 重放样本 `is_experience_replay=True` → 批判定为 mixed,走宽松整组对齐 | 与 LUFFY 混批现状同款行为 |
| 行为 logprob 恒等门(L4208-4600) | 重放行被 F4 豁免;hint 臂是真采样行,正常受门 | 无须改门 |
| `skip_zero_advantage_grpo_update` | F6:exp_mask=auxiliary → 不误跳 BC 批 | 无改动 |
| `_replace_recorded_old_log_probs`(L2221) | 仅 `experience_replay.enable` 下运行,catalyst 配置强制关 | 不相交 |
| SC/DR3/CHORD/DAPO/LUFFY | §2 互斥断言,M1 不同开 | 不相交 |
| 难度追踪 `update_difficulty2task_dict` | 只被 experience_replay/LUFFY 路径调用(L3815/3866),catalyst 配置下二者关闭、该函数不执行;治理层用**自己的**双臂 EMA | 无污染 |
| validate/`initialize_exp_pool` | 不过 plan_arms → 永不注入 hint | 保证验证纯净 |
| checkpoint 续训 | Governor/Pool state 落盘+加载(4.4);不落也只是冷启动重自举,无正确性问题 | 低风险 |

---

## 6. 默认关闭 = 逐字节等价的实现纪律

* 所有新字段:dataclass `=None` 默认;所有新函数参数:`=None` 默认;
* `catalyst_replay_mask`/arm-uid/优势后处理/BC 分支/遥测,全部包在
  `catalyst.enable`(或批内标志存在)判定内;
* `get_extra` 新增 `"catalyst_arm"` 键是唯一"默认路径也会出现"的改动
  (值恒 "bare",无消费者)——若评审认为违反字节等价洁癖,可降级为
  "仅 catalyst on 时加键"(实现按此执行,测试按严格版写);
* AgentFlow 注入块:`getattr(traj_exp_config, "catalyst_hint_text", None)` 为 None
  即零行为(连 deepcopy 都不做)。

---

## 7. 风险清单(逐项)

| # | 风险 | 缓解 |
|---|---|---|
| R1 | 臂内样本过少 → 基线噪声大(std 病态) | k_hint 量化 {0}∪[2, n−2];`std_fallback_to_group` 兜底 |
| R2 | uid 后缀波及以 uid 分组的诊断指标/DAPO | DAPO 互斥关闭;诊断漂移记录在案(§4.2 风险段);catalyst/ 专表替代 |
| R3 | 重放样本把训练 SR 类指标(reward_scores 聚合)略推高 | 重放行 exp_mask=1,现有 on/off 拆分指标已把它归 off 侧;catalyst/replay_samples_in_batch 提供扣除依据 |
| R4 | hint 使 prompt 变长 → 22528 预算挤压 | hint ≤5000 字符(≈1.2-1.5K token);AF/WS raw prompt 常态 ~3-8K,余量充足;`check_context_token_num_safe` 存量护栏兜底(超限即 context_overflow 终止,计入失败——会压低 SR_hint,属自我调节而非崩溃)。遥测加 `catalyst/hint_ctx_overflow`(从 episode_end_reason 统计 hint 臂 overflow 数)监控 |
| R5 | 重放样本渲染在 CPU 主循环,增加步时 | 每步 ≤ batch_size×per_task 条、每条 1 次 build+组装;试点级测量 ~毫秒级/decision;audit_on_insert 是主要成本,仅入池时一次 |
| R6 | 恒温器 h_ref 拍脑袋 → λ 乱飙 | lambda_max 封顶;M1 冒烟先 thermostat.enable=false 跑通,再用裸跑 Ĥ 实测值定 h_ref(遥测 h_hat 已备) |
| R7 | replay_bc 强制 calculate_entropy=True,4B/32K 下熵计算显存/耗时上升 | 仅 catalyst 批;fp32_temperature 精确路径(het_actor L410-538)本就 micro_bsz=1,entropy 是 logits 复用,增量可控;冒烟实测列入"上 GPU 前清单" |
| R8 | 试点清洗管线之前没归档,规则是逆向出来的 | F9 已 230/230 字节验证;build 脚本带真实 fixture 回归;manifest 记 `catalyst_hint_clean/1.0.0` |
| R9 | 入池军规审计若因上下文策略演化失配 → 池饿死 | audit 失败计数暴露(`replay_audit_failures`),非静默;可配置关(但默认开) |
| R10 | 提示臂成功轨迹的 hint 泄漏进重放样本(剥除失败) | strip 的 3 断言(恰 1 条、无 marker 残留)+ 单元测试 T5;失败即弃样计数 |

---

## 8. 单元测试计划(pytest,纯 CPU,tests/test_catalyst_*.py)

| # | 测试 | 断言 |
|---|---|---|
| T1 | `test_catalyst_default_off_equivalence` | 构造合成 samples/config(catalyst 键缺失 vs 显式 false):`samples_to_dataproto` 输出 tensor keys 与逐 tensor 值完全相等;uid 构造分支输出相等;`TrajExpConfig()` 默认字段不改变现有构造;AgentFlow 注入块对 hint=None 的 init_messages 全等(id 级不 copy) |
| T2 | `test_catalyst_arm_baseline` | 合成 1 任务 8 rollout(4 hint/4 bare,已知 reward):uid 后缀后 `compute_grpo_outcome_advantage` 的每样本 adv == 手算 (task,arm) 组内 (r−mean)/(std+ε);关掉 arm_baseline 时 == 整组基线 |
| T3 | `test_catalyst_w_stopgrad` | 玩具 logits 走 BC 公式:`w.requires_grad==False`;`d(bc_loss)/d(logits)` 数值 == 手算 `−β·w·∂logπ`(w 不贡献二阶项);w==min(w_cap, exp((logπ+H)/τ)) 数值正确 |
| T4 | `test_catalyst_rho_controller` | ρ=clip(1−SR/s*,0,ρ_max) 边界(SR=0→ρ_max;SR≥s*→0);U 门(δ、bootstrap 放行);量化 {0}∪[2,n−2];EMA 演化;连续 N 窗退休、且退休不可逆;state save/load 往返相等 |
| T5 | `test_catalyst_replay_render` | 手工构造带提示 episode(init[system,assistant,user+hint] + 2 decisions):strip 后无 marker、恰剥 1 条;`policy.build` 重渲染 prompt 不含 hint;response ids === 原采样 ids(对象相等);extras 契约(F4 四元组);带提示审计路径 prompt ids == 快照 ids |
| T6 | `test_build_catalyst_hints` | 3 组真实 fixture 逐字节;草稿块/粘连/5000 截断各一合成极端例;缺 think → 不产 hint |
| T7 | `test_catalyst_thermostat` | λ 非负投影与 lambda_max 封顶;平移后 masked adv == A+λ(−logπ−b);重放行 adv 恒 0(先清零后平移的顺序性) |
| T8 | `test_catalyst_replay_advantage_zero` | 单样本 uid 组过 GRPO 得非零 adv(F5 再现)→ catalyst 后处理后为 0(防回归) |

模板/剥除一致性:T5/T6 均通过 `load_hint_template()` 取模板并断言 pin sha,
杜绝三处(采集器/训练注入/剥除)漂移。

---

## 9. 二阶段交付物(放行后)

1. 实现 diff(§4 全部改动点;新文件 2:`catalyst.py`、`build_catalyst_hints.py`);
2. §8 全部 pytest(目标:catalyst 新增测试全绿 + 现有 `pytest tests/` 无新失败);
3. `data/catalyst_hints/alfworld_dsv4flash.json`(+manifest)实际构建产物;
4. catalyst smoke 配置草稿:`config/duet_paper_experiments_configs/iclr2027/train_h200/alfworld_qwen35_4b_catalyst_smoke.yaml`
   (**新文件**,fork grpo_smoke,20 步:catalyst.enable=true, replay.enable=true,
   thermostat.enable=false 首跑;experiment_name/workspace_id 独立);
5. "上 GPU 冒烟前还差什么"清单(预填:vllm2 服务 4 端口拉起;AF 8081 环境;
   hints 文件构建+审计;R7 熵计算耗时实测;λ/h_ref 校准流程;wandb 面板分组)。

---

## 9.5 上 GPU 冒烟前清单(交付物 5;登录节点即可核的项已打勾)

- [x] hint 素材:`data/catalyst_hints/alfworld_dsv4flash.json`(1437 任务)+
      `webshop_dsv4flash.json`(599 任务)已构建,manifest 齐,试点回归
      AF 120/120 / WS 110/110 字节全等;
- [x] 配置可组:catalyst smoke 与 grpo smoke 双双 hydra compose 通过,
      catalyst 键合并正确(默认关/显式开);
- [x] 单元测试:catalyst 70 用例全绿;受影响模块回归
      (grpo_group_integrity / actor_metric_aggregation / teacher_v2_ingestion /
      rollout_drift / telemetry / qwen35_experiment_contracts / experience_replay
      共 79 用例)全绿;
- [ ] 节点侧:vllm2 外部 rollout 服务 4 端口(8201-8204,MAX_NUM_SEQS=1,
      GPU_MEM_UTIL=0.25)+ AF env 8081 拉起(沿用 grpo smoke 的
      start_rollout_servers.sh / start_env_alfworld.sh 流程);
- [ ] R7 实测:catalyst 批强制 calculate_entropy=True 的 update_actor 步时/显存
      对照 grpo smoke(预算内则常开;超预算再谈熵分块);
- [ ] 首跑观测(20 步):`catalyst/hint_rollout_frac`≈0.5·R1 占比、
      `catalyst/replay_audit_failures_total`==0、`catalyst/replay_render_drops`==0、
      `catalyst/hint_ctx_overflow` 低位、`catalyst/w_p50`≈1(自写数据近地板)、
      `training/actor_update_applied` 正常;
- [ ] λ 校准(D5):首跑记录 `catalyst/h_hat` 分布 → 定 `thermostat.h_ref`
      → 第二次冒烟再开 `thermostat.enable`;
- [ ] wandb 面板:catalyst/ 分组 + F4 素材(rollout_log/catalyst_gov_step_*.json
      离线拉曲线脚本,可后补)。

## 10. 需主会话拍板的开放决策(不阻塞规格,均已给默认)

| # | 决策 | 本规格默认 | 备选 |
|---|---|---|---|
| D1 | 分臂基线实现形态 | uid 后缀(零数学新代码,复用现 GRPO 分组) | 新写 `compute_grpo_outcome_advantage_arm_separated`(隔离更彻底,代码更多) |
| D2 | 重放样本每次取 1 个 decision(快照式) | 是(与在线制式同构、批形状稳定) | 整 episode 全 decision 入批(信号多但形状/预算复杂,留 M2 消融) |
| D3 | `get_extra` 恒加 `catalyst_arm` 键 vs 仅开启时加 | 仅开启时加(严格字节等价) | 恒加(diff 更小) |
| D4 | 退休永久 vs 可复活 | M1 永久 | U 回升复活(治理更优雅,M2) |
| D5 | thermostat 首跑即开 | 否(先测 Ĥ 再定 h_ref) | 直接开(冒险) |
