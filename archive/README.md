# archive/ — NeurIPS 时代归档区(2026-07-31 重组,只进不出)

重组原则:**只移动、不删除**。逐文件旧→新路径映射见 `MIGRATION_MANIFEST.tsv`(第三列 `git` = 用 git mv 保留历史,`fs` = untracked 直接 mv)。回滚方法:按 manifest 逆向 mv。

| 子目录 | 内容 |
|---|---|
| `neurips2026/` | 投稿期产物:`duet_anonymous_release/`(匿名代码快照)、`nips.zip`(投稿 LaTeX 打包)、`tmp_ckpts/`(**14G**,v12/v24 step100 旧探针 checkpoint,确认无用后可删)、`tmp_scripts/`(WebShop option-click 探针脚本与中间数据) |
| `analysis_root_scripts/` | 原 repo 根目录的 8 个一次性 `analysis_*.py`(结论已写入 `analysis_reports/`,无任何代码引用) |
| `reference/` | `Search-R1/`(第三方参考仓库 clone,嵌套 .git,主库零引用) |
| `legacy_upstream/` | 上游 AgentEvolver 遗留:`research/`(CuES)、`examples/`、`analysis_early_plots/`(原 `.analysis/`)、两个手测 notebook |
| `tools/` | 一次性运维脚本(copy/download/upload/clean 等);`gpu_reserve/` 是占卡工具(`rebuttal.py` + `rebuttal` symlink,symlink 指向 duet env 的 python3.11,绝对路径仍有效) |
| `junk/` | 确认无价值的残留(`nohup.out`、`analysis_test.txt`),按"不删除"原则暂存 |

**不在本目录**但同属 NeurIPS 归档性质、因引用关系原地保留的:`NeurIPS_2026_Latex/`(discussion 期冻结)、`neurips_reviews/`(被 rebuttal README 引用)、`launcher_record/`(launcher 自动代码备份)、`logs/`、`checkpoints/`、`experiments/`。discussion 期结束后可将 `neurips_reviews/` 移入此处。
