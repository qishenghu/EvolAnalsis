# run_scripts/ — 历史实验队列脚本归档(2026-07-31 重组)

原先散在 repo 根目录的 94 个 `run_*.sh` 按时代/用途归入子目录。它们是**历史执行记录**(nohup + launcher.py 队列),与 `EXPERIMENT_LOG.md`、`logs/`、`launcher_record/` 互为印证;`archive/MIGRATION_MANIFEST.tsv` 有逐文件旧→新路径映射。

| 子目录 | 内容 | 时代 |
|---|---|---|
| `00_early_dev/` | 早期框架开发与单实验 launcher wrapper | 2025-12 ~ 2026-04 |
| `10_teacher_data/` | Teacher 轨迹采集/合成/过滤队列 | 2026-03 |
| `20_main_paper/` | NeurIPS 主表队列(1.5B/3B/7B/Llama、SFT→RL)。`EXPERIMENT_LOG.md` 引用的 `run_qwen{1.5b,3b}_{alfworld,webshop}.sh` 在此 | 2026-04 |
| `30_method_dev_rounds/` | 方法开发 round5–12、v38–v41 迭代(adaptive-μ 定型于 round7 v39) | 2026-04 |
| `40_sweeps/` | 超参 sweep、SOTA hunt、L20X velocity、链式 orchestrator | 2026-04 ~ 05 |
| `50_ablations_neurips/` | NeurIPS 消融队列 | 2026-05 |
| `60_rebuttal/` | Rebuttal 与 discussion 期补实验队列(A100/H200) | 2026-07 |

仍留在 repo 根目录的:`run_a100_queue_driver.sh`(通用 file-driven 队列驱动,读 `logs/queue_list.txt`,ICLR 迭代可直接复用)、`start_env_*.sh`、`setup_envs.sh`、`setup_new_server.sh`、`watchdog_agentgym.sh`、`env_config.sh`。

## 重跑注意

这些脚本假定 **cwd = repo root**(相对引用 `python launcher.py`、`config/...`、`logs/...`),过半还要求 `env_config.sh` 与脚本同目录(已在每个子目录放了 `env_config.sh -> ../../env_config.sh` symlink 兜底)。但少数脚本(主要是 `60_rebuttal/` 里带 `cd "$SCRIPT_DIR"` 的)移动后相对路径会错位。**如需重跑:把脚本拷回 repo root 执行;ICLR 迭代请写新脚本,不要在归档里原地改。**

旧文档(EXPERIMENT_LOG、HANDOFF、analysis_reports、rebuttal 材料)按旧文件名提及这些脚本属历史记录,不再逐一修改——查 `archive/MIGRATION_MANIFEST.tsv` 即可定位新路径。
