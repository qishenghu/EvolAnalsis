# DeepSearch 稠密检索索引构建方案(异机执行版,2026-08-08)

**目的**:为 DeepSearch 域准备 e5 稠密检索索引("DeepSearch-dense"变体的期权)。
**执行环境**:另一台机器(本方案自包含,不依赖 GaaS 集群状态)。
**决策背景**:BM25 下 flash 教师覆盖 32%,失败探针显示 92% 失败源于检索
(52% 检不到证据未作答 + 40% 检到错证据答错),EM/格式假说均被数据排除
(见 $SCRATCH/diag/ 探针分析)。ICLR 主线保持 BM25 不变;本索引为附录/
camera-ready 增强的前置物料。**建成后不要接入现有环境**——检索器属于环境
契约,切换需重采教师数据并重冻结契约(约 1–2 天,须与主线排期协调)。

## 1. 输入

- 语料:FlashRAG wiki-18(2100 万段落)。本集群副本:
  `/projects_vol/gp_wangwy/qisheng/duet_h200/deepsearch/`(jsonl,字段
  `id` / `contents`,contents 首行为 `"标题"`,余为正文——与 BM25 索引同源;
  异机可从 FlashRAG_datasets(huggingface `RUC-NLPIR/FlashRAG_datasets`,
  wiki18_100w… 使用与本仓库 build 时相同的 wiki-18 全量版本)重新获取,
  **务必与 BM25 用同一份语料**,否则 dense/BM25 对照失去意义。
- 编码器:`intfloat/e5-base-v2`(768 维)。约定:段落侧加前缀 `"passage: "`,
  查询侧加 `"query: "`(e5 训练约定,漏加会掉 5–10 个点)。

## 2. 构建步骤

### 2.1 编码(GPU,一次性)
- fp16,batch 512–1024,max_length 256(段落截断);单张 80G+ 卡约 4–8 h。
- 分片落盘:`emb_shard_{i:04d}.npy`(fp16)+ 同序 `ids_shard_{i:04d}.json`;
  段落顺序 = 语料文件行序(id 对齐军规:最终索引的第 j 行必须能回查原始 id)。
- 伪代码:
  ```python
  model = AutoModel.from_pretrained("intfloat/e5-base-v2").half().cuda().eval()
  for batch in corpus:                       # "passage: " + title + "\n" + text
      out = model(**tok(batch, truncation=True, max_length=256, ...))
      emb = mean_pool(out, mask); emb = F.normalize(emb, dim=-1)   # 单位化!
  ```
  余弦相似度 = 内积(向量已单位化),FAISS 用 `METRIC_INNER_PRODUCT`。

### 2.2 FAISS 索引
- **GPU 服务方案(推荐)**:`IndexFlatIP`(fp16 存储 ≈32 GB),单卡可载,
  精确检索无调参——军规友好(确定性最强)。
- **CPU 服务方案(备选)**:`IVF16384,PQ64`,nprobe=64;RAM ≈6 GB,延迟
  10–30 ms;注意 IVF 检索结果对 add 顺序敏感,构建后**冻结索引文件**,
  以文件 sha256 保证可复现。
- 落盘:`e5_wiki18.faiss` + `meta.json`(编码器名/版本、faiss 版本、语料
  sha256、单位化声明、构建日期)。

### 2.3 检索服务(drop-in 兼容现有 BM25 服务)
必须与 `env_service/launch_script/retrieval_server.py` **同一 HTTP 契约**,
让 deepsearch_env 零改动切换:
- `GET /health` → 200;
- `POST /search`,body `{"query": str, "k": int}`(k 服务端 clamp 到 ≤10),
  响应 schema 逐字段对照 retrieval_server.py 现实现(title 从 contents 首行
  拆出——见其 `_split_title`;分数字段名保持一致)。
- 端口约定:**25012**(与 BM25 的 25011 并存,便于 A/B)。
- 查询前缀 `"query: "` 在服务端加,环境侧无感知。

## 3. 验收判据(先冻结再测)

1. **确定性(军规)**:同一 query 连发 100 次,响应逐字节一致;服务重启后
   再测一次仍一致(GPU Flat 天然满足;IVF 需验证);
2. **质量 sanity**:val200(`data/deepsearch/task_ids_val200_seed2026.txt`)
   上做检索命中检查:金标答案字符串出现在 top-3 段落内的比例,dense 应显著
   高于 BM25(参考:BM25 下教师 32% 覆盖,期望 dense 命中率 ≥1.5×);
3. **吞吐**:≥32 qps(16 workers × 2 环境余量);
4. 交付物清单:emb 分片、faiss 索引 + meta.json、服务脚本、验收报告
   (三项判据数字)、依赖版本清单(pip freeze)。拷回集群路径:
   `$SCRATCH/deepsearch/e5_wiki18/`。

## 4. 明确不做的事

- 不接入现役 deepsearch 环境(8086),不动 BM25 索引与 25011 服务;
- 不重采教师数据(那是接入决策之后、与主线排期协调的独立动作);
- 不改 `deepsearch_env.py`(切换时只换 retrieval URL/端口即可)。
