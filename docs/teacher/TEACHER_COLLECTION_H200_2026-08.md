# 教师轨迹采集战役全记录(GaaS H200 集群,2026-08-04 ~ 08-05)

状态:**已恢复且提速,全量+补采运行中**(2026-08-05 10:50 提速重启)。flash 双环境经 **v2 驱动**在登录节点接力(`run_flash_driver_v2_{env}.sh`,一期 finalize → 16-worker 补采);122B 一期 PBS **42568**(服务器并发 16),补采 **42569**(并发 32、16 workers/env)挂 `afterok:42568`。旧 42540/42541 已撤销(数据零损失,账本续接)。
目标:Qwen3.5-122B-A10B(同家族,本地 vLLM)与 DeepSeek-v4-flash(跨家族,OpenRouter API)两位教师,在 ALFWorld/WebShop 各自的 **1600 任务 seed-2026 课程**上采集教师轨迹;硬性要求:①每轮 reasoning+action 完整;②师生同上下文契约;③WebShop 环境完全可重放;④尝试上限 8 次/任务(2026-08-05 用户批准,两段式实施)。

---

## 1. 采集协议(两位教师完全一致,保证家族轴逐字可比)

| 项 | 值 |
|---|---|
| 采集器 | `scripts/collect_openrouter_teacher_trajectories.py`(122B)/ `_dsv4.py` fork(flash)/ `_topup.py`(补采,两者通用) |
| 学生契约 | v5:32K=22528 prompt+10240 response、`native_qwen35`、max_steps 30、react_tags;上下文压缩 AF recent_turns 2/历史观测 160tok、WS 4/512tok;当前观测无损 |
| 学生 config | `config/duet_paper_experiments_configs/iclr2027/collect_h200/{alfworld,webshop}_qwen35_4b_collect_h200.yaml`(= lane-B v5/s200 生产基线全量拷贝,仅改 model.path 与 experiment_name) |
| 课程 | `data/{env}/task_ids_1600_seed2026.txt`,由采集器 `expected_curriculum`(池 shuffle seed2026 取前缀)生成;AF 池 2420、WS 池 6710;ordered sha:AF `38373eb2…`、WS `bd235d35…` |
| 采样语义 | **stop-on-success**(非 pass@1):每任务 1 成功槽;一期上限 4 次尝试,补采对未覆盖任务 +4 次(总计 8);温度 AF 0.9 / WS 0.6;成功=reward 1.0 且过校验 |
| 契约与账本 | 每战役一份 manifest(全部实现/tokenizer/课程 sha);attempts ledger 逐尝试记账(跨断点累计预算);成功轨迹逐条 fsync |
| 学生 tokenizer | `/projects_vol/gp_wangwy/models/Qwen3.5-4B`(本集群 stock;chat template 语义=A100 的 -think patch:thinking 默认开+历史 think 剥离;tokenizer.json 与 A100 契约逐字节一致;template/config 哈希已重钉,跨机需重新对账) |

## 2. 基建工件(本战役新建/修改)

**环境**(均在 `/projects_vol/gp_wangwy/qisheng/duet_h200/conda/envs/`):
- `vllm2`:vLLM **0.21.0**(0.20.x 无二进制 wheel;驱动 580.65/CUDA13 兼容;支持 `Qwen3_5MoeForConditionalGeneration`)。构建:`conda create -p …/vllm2 python=3.12` + `pip install --only-binary=:all: vllm==0.21.0`
- `agentenv-webshop`:py3.8+faiss+openjdk11+pyserini,索引已建(`AgentGym/agentenv-webshop/webshop/search_engine/indexes_1k`,3.0M)。构建脚本:`$SCRATCH/build_webshop_env.sh`

**脚本/配置**(repo 内):
- `env_config.sh`:GaaS 集群适配(CONDA_PATH=$SCRATCH/conda、duet_activate、短 RAY_TMPDIR、ALFWORLD_DATA、CONDA_ENV_VLLM2)
- `start_env_alfworld.sh`:取回 rebuttal 期集群版(duet_activate + 共享节点 kill 属主保护)
- `start_env_webshop.sh`:+PYTHONPATH(vendored web_agent_site 非 pip 安装)、+kill 属主保护
- `scripts/collect_openrouter_teacher_trajectories.py`:py3.11 f-string 兼容;tokenizer 契约重钉本集群工件
- `agentevolver/module/teacher/openai_teacher_llm.py`:无 reasoning 字段时重建 `<think>` 开标签(thinking 模板把开标签放在 prompt 里)
- `scripts/collect_openrouter_teacher_trajectories_dsv4.py`(flash 专用 fork,共享模块零改动):强制 `reasoning:{enabled:true}`+OpenRouter 解析缺陷修复(全输出进 reasoning/content 为空 → 按最后一个完整 `<action>` 块确定性切分)+decision 级 reasoning 缺失重试(≤2)+成功轨迹逐 decision think 硬校验
- `scripts/collect_openrouter_teacher_trajectories_topup.py`(补采 fork):`--skip-tasks-covered-by`(契约记录 skip 文件 sha)+`--force-openrouter-reasoning` 门控(flash 开/122B 关)
- `scripts/accept_teacher_pilot.py`:验收门(traces:逐 decision think/action 完整性+slot SR;webshop-determinism:同 boot+跨重启指纹)
- `scripts/h200_node_preflight.sh`:自老 repo 复制
- `run_teacher122b_collect.pbs`:122B 一期作业(gpu_as 4×H200;流水线 vLLM→环境栈→确定性门→试点→验收门→全量;幂等可重提)
- `run_teacher122b_topup.pbs`:122B 补采作业(42431 结束后提交)
- `analysis_outputs/teacher_122b_pilot_cases.md`:试点抽样人工审查文档(已交用户)

## 3. 时间线与事件

**08-04 21:47 作业 42420(122B 一期,首发)**:vLLM 加载 57.3GB/卡(427s)后引擎在多 rank 编译期出现 shm broadcast 等待。**操作失误**:qstat 的 CPU 时间列(08:18 为四 worker 累计)被误读为 8 小时运行时长,判为死锁误杀(实际仅运行 ~17 分钟,大概率正常慢启动)。弯路收益=三个真实缺口修复:①看门狗 curl 无 `--max-time`(端口通但接口挂时会永久卡死);②torch.compile 缓存默认在 NFS `~/.cache`(多 rank 文件锁死锁风险)→ 改节点本地 /tmp;③新增 enforce-eager 回退档 + 监控 60 分钟停滞心跳。

**08-04 22:06 作业 42431(重提)**:vLLM 带 `--reasoning-parser qwen3` 一档启动成功(~13 分钟);计算节点上确定性门 a/b 再次 PASS;**23:02 试点两门 PASS**(AF slot SR 80%、WS ~34%)→ 自动转 1600×2 全量。

**flash 三轮试点迭代**:
- v1(50×2):AF 50/50 全败——登录节点 TMPDIR 在 NFS,TextWorld 清理临时目录 ENOTEMPTY(silly-rename)→ 修复:服务启动 `TMPDIR=/tmp/duet_tmp_$USER`;WS 0/106 成功(policy_failure,当时归因"能力",后证不完全)
- v2:AF 出轨迹但 **501 decision 仅 18 个带 reasoning**——flash 多轮下默认不走 reasoning 通道 → 强制 reasoning 后又暴露 OpenRouter 解析缺陷(整个输出含 `<action>`+EOS 进 reasoning、content 空)→ fork 内确定性切分修复;微试点 4/4 任务成功、58/60 decision 带 think → 加 think 硬校验 + decision 级重试
- v3(50×2):**AF slot SR 84%、WS 32%(含 1.0 满分),两门 PASS** → 全量启动

**08-05**:用户批准尝试上限 8 → 两段式设计落地(topup fork+PBS 备好,未提交);**用户指令全停** → 采集进程精确终止、42431 撤销、服务与监控关闭。

**08-05 01:15 恢复采集(用户指令,8 次两段式)**:登录节点重启双环境栈(TMPDIR 本地化)→ flash AF/WS 驱动脚本后台接力(一期→补采);122B 一期 qsub 42540、补采 42541 挂 `afterok` 依赖。实现要点:**一期 cap 必须保持 4**——`max_attempts_per_rollout` 参与 manifest 的 `contract_sha256`,直接改 8 触发 "resume manifest contract mismatch",且成功记录/账本扫描按契约哈希过滤(改哈希 = 已成功任务被视为未做);"每任务 8 次"由一期 4 + 补采 4 两段叠加实现,语义等价。`.lock` 文件为 `fcntl.flock` 内核锁,进程退出自动释放,陈文件无害。

**08-05 10:50 提速重启(用户指令"全力加速")**,三项变更:
1. **rc=2 语义修正(关键 bug)**:采集器 rc=2 = "incomplete but safely resumable"(有任务在尝试预算内未覆盖)= 全量采集的**正常结局**(rc=0 要求 1600 全覆盖,现实不可能)。v1 flash 驱动误判 rc=2 为失败 → AF 一期 07:26 完成(覆盖 1361/1600)后补采未启动,空转 3 小时;122B 两个 PBS 的出口逻辑同病 → **afterok 补采链永远不会放行**。修复:v2 驱动与两个 PBS 均改为 rc∈{0,2} 视为成功,仅 rc=1(异常)失败。
2. **服务器并发提升**:vLLM 实测 `Running: 8, Waiting: 6-8, KV cache 1%`——瓶颈是 `TEACH_MAX_NUM_SEQS=8` 自设上限,一半客户端 worker 在排队,GPU 算力大量闲置。一期重提为并发 16(=契约锁定的 8+8 客户端并发),补采 32。经 `qsub -v` 传入,不触碰采集契约。
3. **补采 worker 16/env**:补采契约在首次启动时才创建,worker 数此刻可自由设定(设后即锁);一期契约锁 8 不可动。flash 补采与 122B 补采(PBS 已改)均用 16。
作业更替:42540/42541 → **42568**(一期,cap16)+ **42569**(补采,cap32,afterok)。qdel 仅损失在途尝试(账本/成功记录逐条 fsync,零数据损失)。

## 4. 核心发现

1. **WebShop 可重放性钉死**(用户点名风险):根因=原版 princeton 代码 SimServer init 的**无 seed goal shuffle**(每次重启 goal↔session 映射漂移="同一 task 目标商品参数变");repo vendored 副本带 `random.seed(233)` 修复("HBY: Fix",`web_agent_text_env.py:306`)。实测:5 任务 × 同 boot 两次 + 跨全栈重启,初始状态指纹全一致;登录节点与计算节点各验一遍;验收门固化进 PBS 作业。
2. **上下文无罪核验**(用户质疑"goal 是否被压缩丢掉"):①WS 每页观测天然含 Instruction(26/27);②全部 WS 成功轨迹 `dropped_turns=0`、prompt 峰值 10218≪22528,压缩仅为历史观测 512tok 截断;③AF 用 60 个真实存盘 prompt 验证 goal 100% 在场(25 步轨迹第 24 步仍在);④122B 同管线 WS ~34% 是决定性交叉证据。
3. **教师 reasoning 开关的因果翻转(论文素材)**:flash 的 WS 严格成功 **0/106(reasoning 关)↔ 32% slot SR(reasoning 开)**,AF 80%→84%;同协议 122B WS ~34%。教师侧 thinking 对 WebShop 属性精确对齐有决定性作用——"teacher 怎么思考值得学"的直接证据。
4. **flash 失败签名**:reasoning 关时 12/106 拿部分分、最高 0.95(悬崖),买近似商品——"看得见目标但对不齐属性",非 goal-blind。
5. **NFS 两坑**(集群运维):TextWorld TMPDIR 与 torch.compile 缓存都不能放 NFS。
6. **qstat 教训**:默认输出第 4 列是 **CPU 时间**(多进程累计),不是运行时长;判死锁前先对时间戳。

7. **格式检查(2026-08-05,四文件全量扫描)**:①decision 数与带 `<action>` 的 assistant 轮 **100% 对齐**(messages 第二条是开场 ack assistant 轮,无 action,属 react_tags 正常前导);②think 覆盖:flash 双环境与 122B/WS 100%,122B/AF 2855/2856,think 字符占比 90-97%;③prompt 原文未存盘(`store_prompt_messages` 关),每 decision 有 `prompt_messages_sha256`/`prompt_token_ids_sha256`/token 数——分析走"同管线重建 prompt + sha 校验"路径,可精确复现教师所见上下文;④**块外残留**:flash/WS 6%(DeepSeek 把推理同时写进 reasoning 字段与正文,正文多一段 ``` 围栏重复 thinking)、122B/WS 2.5%(散落围栏/多余闭标签)、flash/AF 0.3%(块外 deliberation + 9 个多 `<action>` 块)、122B/AF 0——**转换阶段统一规范化**(按采集同一解析规则重抽 think + 被执行 action,残留丢弃;r1 原文件保留作 provenance);CLL 分析中残留计作 "other" 段。

8. **WS 成功判据(2026-08-05 与用户对齐:接受)**:采集 success = 环境 success 标志——WebShop 对 **score≥0.9 即报 success_rate=1.0**,故两个 WS 文件各约半数记录 reward∈{0.9, 0.95}(近满分示范:买对商品、次要属性小偏差),该任务成功槽即被占、不再追求 1.0。决定:**维持现状**;reward 逐条在案,训练转换期可按阈值过滤;论文如实披露 "teacher demos = env-success (score≥0.9)"(bDeY-Q3 教训)。若未来需要 1.0-only 教师集须改判据重采 WS——已评估,不做。

## 5. 数据资产(截至暂停,全部完好)

`$SCRATCH = /projects_vol/gp_wangwy/qisheng/duet_h200`

| 路径 | 内容 |
|---|---|
| `$SCRATCH/teacher_data/qwen35_122b/alfworld_…_t1600_r1.jsonl` | **190** 条成功轨迹(全量一期,进行中暂停) |
| `$SCRATCH/teacher_data/qwen35_122b/webshop_…_t1600_r1.jsonl` | **32** 条 |
| `$SCRATCH/teacher_data/qwen35_122b/pilot/` | 试点 40(AF)+20(WS)条 + ledger + manifest |
| `$SCRATCH/teacher_data/deepseek_v4_flash/alfworld_…_t1600_r1.jsonl` | **199** 条 |
| `$SCRATCH/teacher_data/deepseek_v4_flash/webshop_…_t1600_r1.jsonl` | **48** 条 |
| `$SCRATCH/teacher_data/deepseek_v4_flash/pilot/` | v3 试点 36(AF)+9(WS)条 |
| `…/pilot_attempt1_infra_failed/`、`…/pilot_attempt2_no_reasoning/`、`…/micro_*` | 失败试点 ledger 归档(证据链,勿删) |
| `…/webshop_det_fingerprints.json` | WS 确定性指纹(登录+计算节点各一) |
| 每个输出旁 `.attempts.jsonl` / `.manifest.json` | 尝试账本(跨断点预算)/ 契约快照 |

每条成功记录含:`messages`(完整多轮原文)、`decision_trace`(逐 decision 的 completion、prompt token 数/sha、context_stats、API 元数据)、reward、契约/课程/tokenizer sha。

**成本**:flash 累计 **$6.66**(1779 次尝试,95.6M token;其中全量已花 $4.89);122B 为本地 GPU,无 API 成本。全量+补采完成的 flash 总成本粗估 $40-80。

**覆盖率预期**(8 次上限后):122B AF ~99.9%/WS ~90%+;flash AF ~99%/WS ~65-70%。WS 空缺格=教师 8 次做不出的硬任务(guided-p≈0 角点素材)。

## 6. 验收门(全量前必须全绿,均已固化为脚本)

1. 契约+live-profile:`collect…py --contract-only`(课程与环境 train split 逐字比对)
2. WS 确定性:`accept_teacher_pilot.py webshop-determinism --phase a/b`(同 boot+跨重启)
3. 轨迹完整性+SR:`accept_teacher_pilot.py traces --min-sr {AF 0.40|WS 0.20}`(逐 decision think/action;messages action 轮数==decision 数)
4. flash 附加:fork 内逐 decision think 硬校验(缺 think 的成功轨迹拒收重试)

## 7. 常用命令

```bash
# 环境栈(登录节点必须带本地 TMPDIR;计算节点 PBS 自动设)
TMPDIR=/tmp/duet_tmp_$(id -un) bash start_env_alfworld.sh   # :36001+:8081
bash start_env_webshop.sh                                    # :36003+:8083
bash start_env_{alfworld,webshop}.sh stop

# 课程文件再生(与学生课程算法逐字一致)
python - <<'P'
import sys; sys.path.insert(0,'.')
from scripts.collect_openrouter_teacher_trajectories import expected_curriculum
for env in ("alfworld","webshop"):
    c = expected_curriculum(env, 2026, 1600)
    open(f"data/{env}/task_ids_1600_seed2026.txt","w").write("\n".join(c["task_ids"])+"\n")
P

# 122B 作业
qsub run_teacher122b_collect.pbs          # 一期(幂等重提即续采)
tail -f $SCRATCH/logs/teach122b.live.log
touch $SCRATCH/logs/TEACH122B_STOP        # 阶段间优雅停
qsub run_teacher122b_topup.pbs            # 补采(一期结束后)

# flash 全量(登录节点;续采)
python scripts/collect_openrouter_teacher_trajectories_dsv4.py \
  --config config/duet_paper_experiments_configs/iclr2027/collect_h200/alfworld_qwen35_4b_collect_h200.yaml \
  --env-url http://127.0.0.1:8081 --task-file data/alfworld/task_ids_1600_seed2026.txt \
  --output $SCRATCH/teacher_data/deepseek_v4_flash/alfworld_dsv4flash_t1600_r1.jsonl \
  --model deepseek/deepseek-v4-flash --api-base https://openrouter.ai/api/v1 \
  --api-key-source /home/qisheng001/DUET_H200/test_openrouter.py \
  --rollouts-per-task 1 --max-attempts-per-rollout 4 --max-workers 8 --resume \
  --wandb-run-name dsv4flash_alfworld_full_t1600
# (webshop 同式:端口 8083、任务文件与输出换 webshop)

# flash 补采(一期完成后;+4 次=总 8)
python scripts/collect_openrouter_teacher_trajectories_topup.py \
  … --max-attempts-per-rollout 4 --force-openrouter-reasoning \
  --skip-tasks-covered-by $SCRATCH/teacher_data/deepseek_v4_flash/{env}_dsv4flash_t1600_r1.jsonl \
  --output $SCRATCH/teacher_data/deepseek_v4_flash/{env}_dsv4flash_t1600_topup8.jsonl

# 抽样渲染人工审查文档
# → analysis_outputs/teacher_122b_pilot_cases.md 的生成逻辑(读 jsonl 渲染逐轮观测/think/action)
```

## 8. 恢复手册(从当前暂停态)

1. `qsub run_teacher122b_collect.pbs` —— 122B 一期续跑(自动过确定性门、跳过试点已完成 slot、续全量)
2. 登录节点起两栈(注意 TMPDIR)+ 按 §7 的 flash 全量命令原样重跑(`--resume` 从 ledger 接续)
3. 一期完成后:`qsub run_teacher122b_topup.pbs`;flash 跑 topup 命令
4. 最终合并:教师集 = `*_t1600_r1.jsonl ∪ *_t1600_topup8.jsonl`(过滤/转换阶段一并处理)

## 9. 遗留事项

- 122B 全量+补采运行中(42540 → afterok → 42541);flash 全量+补采由驱动脚本接力运行中
- 转换阶段待办:块外残留规范化 + reward 阈值过滤开关(§4-7/8)
- 最终统计(覆盖率、逐轮 reasoning 覆盖、token 分布、两教师对比)待采集完成后出
- 跨机 tokenizer 契约对账:若最终训练在 A100 侧,需重钉 template/config 哈希(tokenizer.json 已逐字节一致)
- wandb runs:`teach122b_*`、`dsv4flash_*`(project agentevolver,全部 online 有账)
