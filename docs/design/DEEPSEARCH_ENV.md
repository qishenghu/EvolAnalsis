# DeepSearch 环境设计(Musique 3-4 hop × wiki-18 BM25)

日期:2026-08-05 · 状态:已实现,待冒烟三件套(见 §6)
定位:ICLR 2027 第四域(时间盒制:冒烟不过即砍,不伤 AF/WS/SciWorld 核心盘)。
用户指令(2026-08-05):训练数据全部用难题(3-4 hop);划分 train/test;搭建环境;定上下文管理。

## 1. 组件与端口

| 组件 | 实现 | 端口 |
|---|---|---|
| 语料 | wiki-18(FlashRAG `wiki18_100w`,21M 段落,14.4GB jsonl) | — |
| 检索 | BM25(Pyserini/Lucene,`agentenv-webshop` env,storeRaw) | **25011** |
| 环境 | `env_service/environments/deepsearch/`(自含,无 AgentGym 后端) | **8086** |
| 启动 | `start_env_deepsearch.sh`(检索→env_service,归档日志、按端口停、属主保护) | — |

索引:`$SCRATCH/deepsearch/bm25_wiki18`;原始数据:`$SCRATCH/deepsearch/raw/`。
确定性:固定索引 + BM25 + docid 决胜 → 同 query 恒同结果,**重放契约由构造保证**(优于 WS 的 seed 修补)。

## 2. 任务与划分(`scripts/build_deepsearch_splits.py`,manifest 落盘)

- 源:FlashRAG 规范化 MuSiQue-Ans(train 19,938 / dev 2,417;全部 answerable)。
- 池:train 中 hop∈{3,4} 共 **5,562**(3hop 4,387 / 4hop 1,175);hop = `len(question_decomposition)`。
- 划分:**与采集器约定逐字一致**(sorted ids → `Random(2026).shuffle` → 前缀):
  train = [:1600](3hop 1246 / 4hop 354),val = [1600:1800](159/41),两者构造性不重叠;
  test = dev 3-4 hop 全量 1,165(760/405),冻结。sha 见 `data/deepsearch/SPLIT_MANIFEST.json`。
- id 命名空间:`musique_train_*` / `musique_dev_*`。
- ⚠ 集成注意:采集器 `expected_curriculum` 的 sorted_membership 排序键是 `int(item)`,
  deepsearch 的字符串 id 需在接入采集时给该函数打一个 per-env 排序键补丁(S)。

## 3. 环境协议

- 动作:`search[query]`(检索 top-3,返回 `[Doc i] title\ntext` 观测)| `answer[text]`(终局)。
- 格式:react_tags(`<think>…</think>\n<action>…</action>`),与 AF/WS/SciWorld 同款;
  前导消息约定 system + assistant-ack + user(兼容 `state_progress` 的 `skip_initial=3`)。
- 奖励(预注册):主指标 = `answer[...]` 内文本的**严格 EM**(SQuAD 归一化:小写、去标点、去冠词);
  辅指标 F1 记入 info 不作奖励;只评 answer 块内文本(防"长文包含答案"式 hacking)。
- 无效动作:回显纠错提示,不终局(与 SciWorld 同模式);步数上限由 agent_flow 的 max_steps 管,
  **设 20**(2026-08-05 用户质疑后上调,原 10):理想路径 3-hop 4-5 轮 / 4-hop 5-6 轮,但冷启动学生
  需要富余轮次改写查询、消化无用文档——上限过紧会直接压低成功率、加深燃料沙漠(与 AF 给 30 步同理)。
  不再上调至 30 的理由:失败 episode 付满上限的钱,且 30 轮上下文 ≈ 22K 贴 22,528 天花板,无损承诺破。
  **预注册调整规则**:试点记录成功 episode 轮数分布(师生各一);成功贴上限 → 上调;95 分位 ≪ 20 → 维持。

## 4. 上下文管理决策(本环境为什么可以"无损")

预算算术(32K 契约 = 22,528 prompt + 10,240 response 不变):

```
观测 = top-3 × ~150-200 tok/段 ≈ 450-700 tok
episode ≤ 20 轮:system(~350) + 问题(~50) + 20×(动作~30 + 观测~700) ≈ 15K < 22,528
(最坏情形每轮满额观测 ≈ 16K,仍无损;30 轮 ≈ 22K 会贴天花板,故上限不设 30)
```

**决策:整个 episode 在 32K 契约内全量无损**——不启用 AF(recent 2/160tok)/WS(4/512tok)式的
历史观测压缩。理由:多跳检索的历史证据是**累积性**的(hop-1 找到的实体是 hop-2 查询的输入,
终答需要跨轮证据),截断历史观测的伤害模型与具身环境不同;而预算上根本用不着截。
配置表达:`recent_turns: 10`、历史观测上限 1024 tok(仅作病态兜底,正常永不触发)、当前观测无损。
历史轮 think 剥离循 Qwen3.5 模板默认(与其他环境一致);师生同上下文军规自动满足
(教师采集用同一 config,且无压缩 = 无对齐风险)。

## 5. 与研究问题的接口

- **跨家族 off-policyness 第二面貌**:教师轨迹 = think + 检索调用,CLL 染色/G1-G2 分解在
  信息检索域复现 = 泛化证据;
- **hop 数 = 免费难度轴**:guided-p、角点素材按 3/4 hop 分层;
- **entry 轴已知偏弱**(episode 短),该域主要检验 hint/语义通道与角点判据——如实预注册。

## 6. 冒烟三件套(立项 gate,数字定去留)

1. env 全回路(检索服务起后):create → search → answer,重放两遍观测逐字一致;
2. 学生 closed-book 污染预检:Qwen3.5-4B 禁检索直答 val200,EM 应显著低(<10%),
   高则说明参数记忆旁路,该域降级;
3. 教师试点:DeepSeek flash 50 题(同上下文契约),slot SR 与逐 decision think 占比;
   SR < 0.2 则教师数据太薄,评估 122B 或砍域。

## 7. 待办

- [ ] BM25 索引构建完成(进行中,8 线程)
- [ ] 冒烟三件套(§6)
- [ ] `collect_h200/deepsearch_qwen35_4b_collect_h200.yaml`(过 gate 后写)
- [ ] `expected_curriculum` 排序键 per-env 补丁(接采集时)
- [ ] GRPO baseline config(iclr2027/deepsearch/)
