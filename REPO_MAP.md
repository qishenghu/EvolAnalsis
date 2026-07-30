# REPO_MAP — EvolAnalsis 目录导览(2026-07-31 重组后)

> 重组说明:根目录从 180 项精简到 ~42 项;历史 run 脚本入 `run_scripts/`,NeurIPS 时代杂物入 `archive/`,文档归入 `docs/` 分区。逐文件迁移记录:`archive/MIGRATION_MANIFEST.tsv`。**位置 load-bearing 的路径一律未动**(见文末)。

## 当前主线

| 路径 | 说明 |
|---|---|
| `ICLR2027_PLAN.md` | **ICLR 2027 转投总纲(v1.1)** — 当前所有工作的驱动文档 |
| `EXPERIMENT_LOG.md` | 跨 run 实验总账(持续更新) |
| `CLAUDE.md` | Claude Code 项目指令 |
| `NeurIPS_2026_Latex/` | NeurIPS 论文 + rebuttal 工作区(**discussion 期冻结**,含 `rebuttal/paper_corrections.md` 等 ICLR 改版依据) |
| `neurips_reviews/` | 三份原始审稿(被 rebuttal README 引用,discussion 结束后归档) |

## 代码与配置(全部原地未动)

| 路径 | 说明 |
|---|---|
| `agentevolver/` | 训练框架核心(trainer / exp_manager / env_manager) |
| `env_service/` + `AgentGym/` | 环境服务与 vendored 环境后端(路径被启动脚本写死) |
| `config/` | Hydra 配置树(`duet_paper_experiments_configs/` 按 env/model/algorithm 组织) |
| `external/config_fallback/` | veRL PPO 默认配置(Hydra searchpath 硬依赖) |
| `cookbook/env_profiles/` | 被 352+ 个 yaml 相对路径引用,不可动 |
| `scripts/` | 工具脚本(teacher 采集/过滤/校验、FSDP merge、监控);新增 `check_correctness.py`、`check_val_reward.py` |
| `tests/` | pytest(依赖与 `config/` 的 sibling 关系) |
| `launcher.py` / `env_config.sh` / `.env` / `start_env_*.sh` / `setup_*.sh` / `watchdog_agentgym.sh` | 入口与基础设施(约定 cwd = repo root) |
| `run_a100_queue_driver.sh` | 通用 file-driven 队列驱动(读 `logs/queue_list.txt`),ICLR 复用 |

## 数据与运行产物(原地未动;磁盘大头)

| 路径 | 大小 | 说明 |
|---|---|---|
| `checkpoints/` | 656G | 203 个 run 的 FSDP checkpoint(可剪枝池,须对照 EXPERIMENT_LOG 决策) |
| `experiments/` | 261G | 各 run 轨迹/validation dump |
| `wandb/` | 85G | 本地 wandb 缓存(单个 52G 巨型 run;已同步云端即可清) |
| `logs/` | 18G | 训练/env 日志 + `REFERENCE_RUNS.md`(正确 reference run 指南) |
| `data/` | 5.8G | teacher 轨迹缓存(qwen72b/qwen14b/sub 采样)、任务 splits |
| `launcher_record/` | 847M | launcher 每次运行的代码快照(自动追加) |
| `.git` | 12G | 含 ~13.5G unreachable blobs(历史误 commit,`git gc --prune` 可回收 ~11.5G) |

## 历史归档(2026-07-31 新建)

| 路径 | 说明 |
|---|---|
| `run_scripts/{00..60}/` | 94 个历史 run 队列脚本按时代分组(详见 `run_scripts/README.md`) |
| `archive/` | NeurIPS 时代杂物:分析脚本、参考仓库、上游遗留、tmp checkpoints(详见 `archive/README.md`) |
| `docs/neurips2026/` | 交接文档(handoffs)、早期笔记(notes)、方法开发设计稿(dev-notes)、样例(samples) |

## 文档分区

| 路径 | 说明 |
|---|---|
| `docs/teacher/` | teacher 采集指南(QWEN3_TEACHER_COLLECTION.md 等,ICLR 期沿用) |
| `docs/design/DUET_Report.md` | DUET 方法理论蓝本 |
| `docs/papers/` | 参考论文 PDF |
| `docs/{index.md,tutorial/,guidelines/,img/}` | 上游框架 mkdocs 文档站(nav 依赖,原地保留) |
| `analysis/` `analysis_outputs/` `analysis_reports/` | 分析脚本库与产物(原地保留,历史报告引用其内路径) |

## ⚠ 不可动清单(位置 load-bearing)

1. **仓库根目录名 `EvolAnalsis` 不可改**(`env_service/launch_script/alfworld.sh` 等硬编码)。
2. `env_config.sh`、`launcher.py`、`.env` 必须在根(72 个脚本 source / cwd-relative 约定)。
3. `agentevolver/` 与 `config/` 必须互为 sibling(Hydra `config_path="../config"`);`tests/` 同理。
4. `cookbook/env_profiles/`、`AgentGym/`、`external/config_fallback/`、`env_service/` 不可移。
5. `data/teacher_trajectories/` 被 378 处 config 相对路径引用。
6. 所有训练/分析假定 **cwd = repo root**。
