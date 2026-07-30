# DUET → ICLR 2027 转投改进方案 (v1.1)

**制定日期**: 2026-07-31 · **Abstract 截止**: 2026-09-18 (AOE) · **全文截止**: 2026-09-25 (AOE) · **剩余时间: 7 周**

v1.1 说明:v1.0 经三路对抗评审(可行性 / 模拟 ICLR 审稿人 / 事实核查)修订。主要变更:①算力预算按"冻结日前的日历窗口"重算,降档方案改为默认基线;②双 lane 策略(Qwen3-4B 现栈先行 + Qwen3.5 升级并行冲刺);③"API teacher"从问题设定降级为无额外假设属性(原表述与公开 rebuttal 矛盾);④新增判别器退化 Gate 判据;⑤补齐缺失工作项与预算行。

信息来源:NeurIPS 三份审稿(4/3/3)+ rebuttal 全档 + 内部 forensics(`rebuttal/paper_corrections.md`、`DECISION_*.md`、`H200_HANDOFF_*.md`)+ 实验资产盘点 + 基础设施调研(均为 2026-07-31 多 agent 调研,事实已逐项核查)。

---

## 0. 战略总纲:三条主线,一个原则

NeurIPS 失分的根因不是方法,而是**证据形态**:单 seed headline、baseline 欠复现、数学与代码不符、评测面窄。ICLR 版的原则只有一条:

> **所有数字提交前可从 raw log 按统一协议复算;所有 claim 在 baseline 也做了 multi-seed 之后仍然成立;所有与 NeurIPS OpenReview 公开记录相关的表述保持一致(审稿人可能就是原班人马,且能读到原帖)。**

| 主线 | 内容 | 性质 |
|---|---|---|
| **A. 诚信手术** | distribution-first 主表、修 Eq.8/9、超参如实分列、provenance 修复、兑现 rebuttal 承诺清单 | 与换模型无关,无论如何都要做 |
| **B. 模型现代化** | 推理学生(目标 Qwen3.5-2B/4B,de-risk lane Qwen3-4B)+ DeepSeek-v4 / Qwen3.5-122B-A10B 教师 | 用户指定方向 + 叙事升级载体 |
| **C. 广度与机制补强** | 第三环境、teacher 轴、BC-only 对照、off-policy 对照、probe/noise 实验在新栈复现后入正文 | 直接回应三位审稿人的共同质疑 |

---

## 1. 叙事升级(修订版)

### 1.1 API teacher:是"无额外假设"属性,不是问题设定

⚠ v1.0 的表述("LUFFY/CHORD 假设 teacher log-prob 可得")**是错的**:本 codebase 的 LUFFY/CHORD 实现本就不用 teacher log-prob,且我们的 NeurIPS rebuttal 公开写过"teacher likelihood is unavailable in the first place"。ICLR 版正确写法:

- **正面表**:一张 per-method 需求表(tokens / log-probs / weights / online access),说明全部方法(含 DUET)只需冻结文本 cache——teacher 可以是任何 API 模型;DR3 在无 likelihood 下仍给出有界的分布修正,是该设定下的"修正上限"。
- **如果要让设定咬人**:增加一个真正需要 teacher likelihood 的对照(exact importance sampling 或 KL-to-teacher 正则),展示它在 API teacher 下**不可实现**(而非仅仅缺席)。P2 优先级。
- 跨 tokenizer 兼容(teacher 文本用 student tokenizer 重分词)照写,这是真实且可验证的属性。

### 1.2 推理学生的新研究问题(增量贡献)

- **Think-BC 消融**:teacher `<think>` 保留 vs 剥离作 BC target(推理风格蒸馏 vs 纯动作克隆)。
- **判别器在格式漂移下的行为**是新的科学问题(也是新的风险,见 §3.4):DeepSeek-think 文本 vs Qwen 学生 rollout 可能第 0 步即可分。如实研究并报告,好于假装不存在。

### 1.3 主表叙事押注(multi-seed 后可幸存的证据)

1. **corrections 是承重墙**(去 BS → 弱学生处崩溃)——**前提是保证至少一列 GRPO <30% 的弱学生列**(见 §2.1 注 3,评审指出强推理学生可能让该证据蒸发);
2. **teacher 自动退场**:强学生处 LUFFY 变有害而 DUET 无害;
3. **后半程动态**:在**新预算协议下重算**(旧 100 步斜率 claim 不能直接搬到 150–200 步协议)。

### 1.4 机制证据同栈规则(硬约束)

**任何进入正文的机制结果(probe、noise 2×2、escape-step、attribution timing、shuffled map)必须在 ≥1 个 headline 学生 × 两环境 × ≥2 seeds 的新协议下复现**;Qwen2.5-1.5B 时代的版本一律进附录并标注 legacy。否则"headline 一套模型、机制证据另一套模型"会被 y9x6 型审稿人一枪打穿。

---

## 2. 实验计划

### 2.1 新主表(P0)

| 维度 | 取值 |
|---|---|
| 环境 | ALFWorld、WebShop(WebShop 150–200 步预算,§2.4) |
| 学生 | 目标 **Qwen3.5-4B + Qwen3.5-2B**;de-risk lane **Qwen3-4B**(见 §3.2 双 lane,8/15 定 headline) |
| 教师 | **DeepSeek-v4-flash**(主,API;WebShop 必须真实 rollout 采集,替换旧 gold-action 构造) |
| 方法 | GRPO、LUFFY、CHORD、SFT→RL(task-matched)、**BC-only(μ≡1.0+BS,作为 contributed baseline 正面主张)**、DUET |
| Seeds | 基线 3/cell;**决定性三角(DUET / BC-only / GRPO)在至少一个环境 ≥5 seeds**(1.5B 时代 sd=±6.6 的教训:n=3 分不出小差距);混合 seed 数必须在 caption 标注 |
| 统计 | mean±sd + task×seed 层级 bootstrap(替代单 seed McNemar)+ 后半程斜率(新协议) |

注:
1. BC-only 入表是**主动贡献**("我们提出并复现了最强对照"),不是被动披露。
2. **Adaptivity 对比预注册**:在跑 regime 实验前,论文先写机制预测("constant μ 应恰在 teacher 信号变有害处受伤:强学生/长预算/弱 teacher"),然后**报告全部测过的 regime 含 null 结果**——事后挑格子会被定性 HARKing。配套直接对照:teacher-in-PG+DR3 vs teacher-excluded+BC-only。若 parity 依旧,诚实贡献 = "同性能、自动调度、强学生端可证安全"。
3. **弱学生列保证**:pilot 时先测 GRPO 裸基线;若 4B GRPO 在 ALFWorld >60%,弱学生叙事移至 2B/难 split/SciWorld,保证正文至少一列 GRPO <30%,并把"bias 严重度随 student–teacher gap 变化"做成图(scope 声明 + 证据,替代祈祷 0.0% 崩溃复现)。

**体量**:2 env × 2 学生 × 6 方法 × 3 seeds = 72 runs + 决定性三角加密(+6~12 runs)+ 每学生每环境 SFT stage。

### 2.2 广度列(P1,优先级已按评审调整)

- **SciWorld 完整一列**(1 学生 × 6 方法 × 3 seeds):评审明确 **SciWorld 全列 > teacher 轴加档**——2 个 2023 文本环境在 ICLR 2027 的 agentic-RL 竞争中已显窄旧,第三环境是刚需,降档时最后砍。现状核查:`sciworld/` 仅有 3B 学生 4 方法 config,**SFT→RL、BC-only 及新学生 config 需新写**(工作项,S);先 3 天冒烟(env service + teacher 采集成功率)。
- **跨家族学生**:Llama-3.2-3B 用 auto-β 重跑 DUET/GRPO ×2 seeds;稳定→正文小表,不稳→附录 honest-negative + β-scaling 规则。**无论结果如何,step-100 曲线必须如实报告**(rebuttal 已公开 step-50 "stable throughout",隐瞒后续会构成 cherry-picking 实锤)。

### 2.3 机制与消融(P1)

a. 四组件消融 × 2 env × ≥2 seeds + shuffled-map 行(paired-seed)。
b. Attribution 三行(μ=0 / μ≡1.0 / adaptive)× 2 env + timing 图。
c. SC robustness 2×2 在 **headline 学生**上重做(§1.4 规则)+ soft matcher 设默认。
d. **auto-β**(SC 系数按学生 reward 量级归一化):工程 S,消一个手调超参 + 修 Llama 失败模式。
e. Adaptivity regime 实验(预注册,见 2.1 注 2):长预算 / 弱 teacher / 强学生 × 2 seeds。
f. Think-BC 消融 × 1 env × 2 seeds。
g. Off-policy 对照(AWAC 式或 imputed-likelihood V-trace)× 1 env × 2 seeds + related-work 段。
h. Teacher 轴:flash vs 122B(v4-pro 仅在算力盈余时加)+ cache 供给曲线(full/10%/1%)。
i. Probe 实验在新栈重算 + **online discriminator 版本**。
j. **Sub-goal-indexed Φ:必须至少交付 rebuttal 承诺的 ALFWorld 一格修复**(公开记录写了 "the fix is direct...the revision will include this analysis",不可静默降级);全列重跑视算力。
k. **共享默认超参 run**:两环境用同一套 μ 默认值跑 1–2 seeds——有了它才能写"默认值可迁移、调优值见附录";没有它,诚实的分列超参表反而单方面武装审稿人(rebuttal 里 "only d_floor differs" 的公开表述已被 C7 证伪,ICLR 版一个字都不能重复)。

### 2.4 协议升级(全部新 run 第一天生效)

1. WebShop 150–200 步;主指标 strict SR 分布 + 辅列 mean reward + escape-step;正文解释 0.95 悬崖。
2. `task_seed` 与 run seed 分离;**新建 `scripts/verify_curriculum.py`**(v1.0 误写为已存在,W1 工作项 S)逐 run 校验 curriculum。
3. **训练内验证即最终协议**(固定测试集、固定采样参数、config 断言)——省掉 600–1,150 GPU·h 的期末重评测;run 结束即归档 `validation_log/*.jsonl` + config 快照,建 run→config→log 对照表。
4. 表格数字一律脚本从 provenance 目录再生;禁止引用无版本号 config。

### 2.5 预算与供给(v1.1 重算)

**需求**(含 v1.0 漏项):

| 块 | GPU·h |
|---|---|
| P0 主表 72 runs + 三角加密 | 4,500–7,400 |
| P1(SciWorld+消融+机制,~70–90 runs) | 2,500–3,500 |
| μ sweep(每学生≤10 run,**半预算跑**) | 600–1,000 |
| 重跑税(新栈 ~15%) | 1,000–1,600 |
| **小计 × 1.0(协议内验证已并入 run)** | **8,600–13,500** |

**供给**(按日历窗口算,不是满载天数):Gate-P0(8/15)→ 冻结(9/12)= 28 天。
- 单机 A100:28 × 192 × 0.8 利用率 ≈ **4,300** —— **连下限一半都不够,单机方案不成立**。
- 双机(A100 + H200):≈ **8,600–9,700**(0.8–0.9 利用率)—— 覆盖需求下限,**H200 是前提条件而非备选杠杆,必须 8/1 前确认**。

**因此:v1.0 的"降档预案"即 v1.1 的默认 scope**:
- LUFFY/CHORD 2 seeds(DUET/BC-only/GRPO 保 3–5);
- teacher 轴只跑 flash vs 122B;v4-pro 不排;
- 8/25 前若进度超前,再恢复满 scope(顺序:LUFFY/CHORD 第 3 seed → v4-pro → sub-goal Φ 全列)。
- W2 pilot 的首要任务之一:**实测新学生每 run 成本,8/15 用实测数重排全表**(v1.0 的 ×1.5–3 系数与 2048→4-8K response 上调不自洽,可能低估;seeds 数以实测定)。

---

## 3. 基础设施改造(W1–W2 全部内容)

### 3.1 教师侧(立即启动,不占 GPU)

- OpenRouter 通道改造(S–M):api_base、`reasoning`/`reasoning_content` 合并为 `<think>` 文本、retry/限速、`--strip_think` 双版本落盘。格式约束:messages 前 3 条 preamble 保持(`state_progress.py:568` `skip_initial=3`);API teacher 强制 `use_log_prob=false` 校验。
- **目标量(v1.0 未定,现明确)**:AF ≥8K 成功轨迹、WS ≥10K(旧 cache 19K/26K;progress map 覆盖率是硬约束,Gate-T 以 hit-rate 验收而非条数)。**8/1 起 1 天试点(50 任务)→ 立即高并发全量**,不等 pilot 训练。
- **预算重定价**:$60–210 是 v1.0 的乐观估计,试点后按实测 token/轨迹重算;**申请授权上限 $500**(flash 全量 + 余量),若需 v4-pro 另批。
- 本地 Qwen3.5-122B(第二教师):独立 conda env `teacher35`(vllm≥0.17),TP=8 约 31GB/卡权重,**预订硬日期窗口**(建议 8/9–8/10,当前 Llama run 结束后)而非"排在其后"。
- Progress-map smoke test:构建后打印 per-task key 数与 on-policy hit-rate(复用 `validate_teacher_for_training.py`,新增 D1 小项)。

### 3.2 学生侧:双 lane(v1.1 关键变更)

**Lane-1(现栈,W1 第 1 天启动)**:**Qwen3-4B 在现有 duet 栈直接开跑**——vllm 0.8.5/transformers 4.53 支持 Qwen3,config 模板已在 repo(注意:现有 6 个模板是 duet/luffy/chord/onpolicy/state_channel/action_channel,**SFT→RL 与 BC-only 两行需新写 config**,S)。它同时充当:teacher 数据/协议/成本的全链路 pilot + Gate-1 失败时的现成 headline。W1 每一天都在产出,不押注升级成功。

**Lane-2(升级冲刺,并行 timebox)**:新建 `duet2` env,vllm≥0.17 + transformers≥5.2 + 配套 verl,**重移植 fork**(ae_ray_trainer.py ~3,500 行 + het_actor.py 的 DR3/exp_mask/SC 管线跨 ~4 个上游版本)。工程量 2–3 人周,**8/8 的 10 步冒烟 gate 不现实**,改为:
- **Gate-S(8/15)**:Qwen3.5-2B 通过**强化验收**——100 步训练 + 一次 checkpoint resume + 一次 n=200 验证 + 无 OOM/泄漏。
- 通过 → Qwen3.5 为 headline,Lane-1 已产出的 teacher cache/协议/成本数据全部复用;
- 未通过 → **Qwen3-4B(+Qwen3-1.7B 作弱学生列)为 headline**,Qwen3.5 转为 camera-ready 前的加分项。叙事损失小(同为 hybrid-thinking 推理模型)。

Reasoning 学生四项必改(两 lane 通用,file:line 已核实):
1. **A1 (M)**:`Linear_CMT` 历史轮 `<think>` 剥离(复用 `_compress_old_context`/`_retoken`,`cmt_linear.py:461,530-573`);
2. **A2 (S,阻塞 think-BC)**:验证模板对离线 assistant think 的静默剥离(`cmt_base.py:250-254` round-trip);
3. **E1 (M)**:thinking 控制走 `chat_template_kwargs` 透传(`env_manager.py:118-123`);
4. **F1/F2**:长度四处同步上调,**`max_model_len` 上限封顶 ~32–40K 并写下 KV 显存算术**(262K native 上下文默认值会直接 OOM);`stop_sequences: </action>` think 误停评估。

### 3.3 v1.0 缺失的关键路径工作项(全部列入 W1–W2)

| # | 项 | 工作量 |
|---|---|---|
| M1 | FSDP→HF merge 接线 + smoke(`scripts/merge_fsdp_checkpoint.py` 已有,SFT→RL 行与 SFT-stage 评测都靠它) | S |
| M2 | 存储预算 + 自动清理策略(~157 runs × 4B 权重+optimizer,数 TB;W4 磁盘写满会静默杀 run) | S |
| M3 | **每台训练机各起一套 env service** + 2× 会话时长压测(WebShop 共享 Ray actor 是串行瓶颈;36001/36003 在 ephemeral 区的 kill-by-port 风险双 lane 加倍) | M |
| M4 | `scripts/verify_curriculum.py` 新建 | S |
| M5 | SciWorld / SFT→RL / BC-only 缺失 config 编写 | S |

### 3.4 判别器退化 Gate(评审新增,科学风险)

跨家族 reasoning teacher(DeepSeek think 文本)vs 学生 rollout 可能**第 0 步即线性可分**→ disc_acc 立即饱和 → ŵ 钉死 w_min、μ 钉死 valley → "自适应"全线空转,fade-out 叙事变假。
- **Gate-P(随 pilot,8/17)**:检查 disc_acc/μ/ŵ 轨迹;**若 step 5 内 disc_acc>0.9,停 P0**,先做特征重设计(格式剥离、content-matched 特征)或 μ 驱动改造;
- 论文加一段:support mismatch 下 ŵ_α 的 bias 界与诊断方法。发现于 W2 是设计输入,发现于 W5 是灾难。

---

## 4. 时间线(v1.1,双 lane + 拆分 gates)

| 周 | 日期 | Lane-1(现栈,A100) | Lane-2 + 教师 + 论文 | Gate |
|---|---|---|---|---|
| W1 | 8/1–8/7 | Qwen3-4B GRPO/DUET 首 run(现栈);A1/F1 改造;M1–M5 工作项;NeurIPS discussion 收尾 | `duet2` 搭建开始;**8/1 试点采集→8/2 起全量**(flash);**H200 确认(前提条件)**;Eq.8/9 重写与 §3 改稿启动 | — |
| W2 | 8/8–8/14 | GRPO 12 runs 先行(无 teacher 依赖,吃满空闲卡);4B pilot 全流程 | teacher 全量完成+过滤+map smoke;122B 窗口 8/9–8/10;SciWorld 冒烟 | **Gate-T (8/13)**:teacher 质量/hit-rate |
| W3 | 8/15–8/21 | **前 3 天:μ sweep(半预算)+ GRPO/BC-only/SFT→RL 回填**;第 4 天起 DUET/CHORD seeds | **Gate-S (8/15)**:Qwen3.5 强化验收→定 headline;**Gate-P (8/17)**:pilot 健康 + 判别器轨迹 + 实测成本→重排全表 | Gate-S/P |
| W4 | 8/22–8/28 | P0 全速(A100 弱学生档 / H200 4B 档) | 8/25 进度评估:超前则恢复满 scope | — |
| W5 | 8/29–9/4 | P0 收尾 + P1 启动(消融/SciWorld/attribution) | 机制实验(probe/noise/2×2)在 headline 学生上重做 | **Gate-3 (9/1)**:P0<60% 则再降档 |
| W6 | 9/5–9/11 | P1 收尾;adaptivity regime;off-policy 对照;think-BC | 图表全部 provenance 脚本再生;全文成稿 | — |
| W7 | 9/12–9/18 | **9/12 实验冻结** | 内部红队(模拟原三审稿人 + 读 OpenReview 原帖一致性检查);9/18 abstract | Gate-4 |
| +1 | 9/19–9/25 | — | 打磨、页数、Reproducibility Statement;9/25 全文 | — |

写作不等实验:§3 数学、related work、协议附录、legacy 素材收编从 W1 开始。

---

## 5. 论文改版要点

1. **Eq.8**:有界 α-relative ratio `ŵ_α = r̂/((1−α)r̂+α) ≤ 1/(1−α)≈1.13`,正面写"只降权、不放大"。
2. **Eq.9**:imputed-behavior-policy 替换式。**telescoping 推导已被三次 code audit 证伪,绝不可写入**(`evidence_eq9_dr3.md` §VERDICT)。⚠ 新增工作项:**以代码为准核定最终形式并与 rebuttal 公开形式(log π̂_β := log π_old − log ŵ)写一句 reconciliation**——rebuttal 帖与论文式若有出入,审稿人会 diff。WebShop policy-shaping 变体入附录。
3. **超参**:附录并排分列(AF 0.3/0.05/0.4/0.5 vs WS 0.3/0.10/0.6/0.2)+ §2.3-k 的共享默认 run 撑起"默认可迁移"表述;**"only d_floor differs" 的说法永不再写**。
4. 主表 distribution-first;"principled" 收窄到两个 bias 的诊断修正;SC 明示 heuristic + 接口化(Φ + 可替换 lookup)。
5. **Rebuttal 承诺清单(公开记录,逐项兑现或书面说明原因)**:sub-goal Φ 的 ALFWorld 修复、cache-size 供给曲线、Llama 100 步完整曲线、SFT 训练曲线。缺一条都是可引用的失信。
6. 收编素材(全部过 §1.4 同栈规则后):noise 2×2 → §4;probe(含 online 版)→ §4;escape-step → §5;attribution timing → §4;Pick-Two 重算 → 附录;Llama → 视结果。
7. **C0 清账**:3B validation log 找回(仅用户可做);找不回则 3B 列从主表移除、由新模型列替代。
8. ICLR 格式:10 页、natbib、删 checklist、Reproducibility Statement(素材=provenance 对照表)、LLM 使用声明;清 `\wenya{}` 批注;表格单一来源;`build.sh pages` 纪律不变。
9. WebShop teacher 若最终回落到 gold-action+rationale 构造:cache-distribution correction 的框架**保持与 rebuttal 一致**,并如实写明构造方式(bDeY-Q3 教训)。

---

## 6. 需要用户确认/行动(按紧急度)

1. **【8/1 前,阻塞】H200 机器使用权确认**——双机是本方案成立的前提(单机供给 4,300 vs 需求下限 8,600);拿不到则 scope 需砍半重排,请立即告知。
2. **【W1 内】3B validation logs 找回**(远端 L20X/H100)——C0 阻塞项,找不回则 3B 列移出主表。
3. **【8/1 前】OpenRouter 预算授权:上限 $500**(flash 试点+全量+余量;v4-pro 另批)。
4. **【知情】双 lane 策略确认**:Qwen3.5 仍是目标 headline,但 Qwen3-4B 现栈先行 de-risk;若 8/15 Gate-S 未过,headline 落在 Qwen3-4B(+1.7B 弱学生列)。如你坚持 Qwen3.5-only,请明示(相应地 W1–W2 全押升级,风险自负)。
5. **【知情】NeurIPS 对冲**:notification(~9/25)与 ICLR 全文截止重叠;默认 ICLR 版照做到底,NeurIPS 意外录取则弃投。

---

## 7. 风险登记册(v1.1)

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| H200 拿不到 → 供给砍半 | ? | 致命 | §6-1 立即确认;失败则 scope 重排(单学生 headline + SciWorld 降附录) |
| Qwen3.5 栈升级失败/延期 | **高**(fork 重移植 2–3 人周) | 中(已 de-risk) | 双 lane;Gate-S 强化验收;Qwen3-4B headline fallback |
| 判别器对 API teacher 第 0 步饱和 → 自适应叙事空转 | 中 | 高 | Gate-P 判据 + 特征重设计预案(§3.4);论文写 support-mismatch bias 界 |
| 强推理学生令 GRPO 基线过强 → corrections 叙事蒸发 | 中 | 高 | 保证一列 GRPO<30%(2B/难 split/SciWorld);gap-scaling 图(§2.1 注 3) |
| BC-only 在新模型仍 parity | 中 | 高 | 预注册 regime 预测 + 全量报告(含 null);退守叙事已写好(§2.1 注 2) |
| thinking 学生实测成本 > ×3 | 中 | 高 | Gate-P 实测重排;A1 剥离历史 think;thinking budget 上限;seeds 以实测定 |
| DeepSeek 采集成功率低(WS 尤其) | 中 | 中 | 8/1 试点;备选 122B(窗口已预订);WS 保底构造但如实披露 |
| WebShop 新栈依旧高方差 | 中 | 中 | 150–200 步 + 双指标 + 分布报告内建 |
| SciWorld 不稳 | 中 | 中(评审升级:第三环境是刚需) | 3 天冒烟;降档顺位排最后(先砍 teacher 轴) |
| 存储写满 / env-service 瓶颈 / 端口冲突 | 中 | 中 | M2/M3 工作项(每机独立 env service + 压测 + 清理策略) |

---

*维护约定:本文件为 ICLR 转投总纲,每周日更新进度与 Gate 状态。三份评审原文见 session 工作流输出(可追溯)。实验协议细则与 provenance 规范落地后在 `docs/` 单独成文。*
