# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

DUET (DUal Expert Trajectory utilization) — a principled experience-replay framework for training LLM agents with GRPO in interactive environments (NeurIPS 2026 submission). Built on the AgentEvolver framework with veRL (distributed RL) and vLLM (inference) backends.

The paper framing is **correct-then-extract**. Mixing teacher trajectories into GRPO rollout batches induces two biases; DUET first corrects them, then extracts teacher signal through two channels:

- **Stage 1 — Correct:**
  - **Baseline Separation (BS)**: GRPO uses separate mean/std for teacher vs on-policy samples, so successful teacher rollouts don't contaminate the group baseline and suppress on-policy advantages.
  - **DR3 (Density-Ratio Repair)**: A discriminator estimates `ŵ = π_θ/π_teacher = D/(1-D)` to correct importance weights on teacher samples — works without teacher log-probs and self-attenuates as the student improves.
- **Stage 2 — Extract:**
  - **Action Channel (BC)**: Token-level behavior cloning on teacher tokens with **adaptive weight μ driven by the DR3 discriminator's accuracy**: `μ_t = clamp(μ_max·(1−EMA(disc_acc))/(1−d_floor), μ_min, μ_max)`. One discriminator serves both density-ratio correction and the BC schedule — no hand-tuned decay.
  - **State Channel (SC)**: Expert progress map `Φ(s)` gives dense reward shaping `β·P(τ)` plus step-level deltas `η·[Φ(s_{t+1})−Φ(s_t)]` to on-policy samples only.

Baselines: GRPO (on-policy), LUFFY (teacher mixing + policy shaping), CHORD (GRPO + weighted SFT with manual μ schedule), SFT→RL.

This repo is three things at once: the training framework (`agentevolver/`, `env_service/`), the experiment archive (`config/duet_paper_experiments_configs/`, `run_*.sh`, `EXPERIMENT_LOG.md`), and the paper source (`NeurIPS_2026_Latex/`).

## Common Commands

```bash
# Environment (conda env name is 'duet' on current servers; configured in env_config.sh)
conda activate duet

# ALL shell scripts source env_config.sh — the single place for conda envs, paths,
# ports, TMPDIR/RAY_TMPDIR, and wandb key. On a new server, edit ONLY that file.

# Environment services (must run before training)
bash start_env_alfworld.sh          # AgentGym :36001 + env_service :8081
bash start_env_alfworld.sh stop
bash start_env_webshop.sh           # AgentGym :36003 + env_service :8083 (uses 'agentenv-webshop' conda env + Java)
bash start_env_webshop.sh stop

# Run one experiment
python launcher.py --conf config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_duet.yaml
python launcher.py --conf <config> --kill   # kill lingering Ray processes before re-run

# Tests
pytest tests/ -v
pytest tests/test_task_manager.py::test_specific -v

# GPU selection
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Paper build (tectonic, dedicated 'tex' conda env — handled inside build.sh)
cd NeurIPS_2026_Latex && bash build.sh        # full build + page breakdown
bash build.sh pages                            # page count per section
bash build.sh clean
```

**Process safety**: never kill training with broad patterns (`pkill -f python`, `pkill -f ray`). Multiple long-running trainings may share the machine. Kill environment services by port (the `start_env_*.sh stop` scripts do this), and Ray training via `launcher.py --kill`.

## Architecture

### Training Flow

```
launcher.py (arg parse, backup code to launcher_record/, launch env services)
  └─> python -m agentevolver.main_ppo (Hydra config: config/script_config.yaml)
      └─> TaskRunner.run() [Ray remote actor]
          └─> AgentEvolverRayPPOTrainer.fit() [ae_ray_trainer.py]
              ├── Rollout generation (vLLM async + environment interaction)
              ├── LUFFY mixer: merge teacher trajectories into batch
              ├── State Channel: β·P(τ) bonus on on-policy samples
              ├── Step-level deltas: η·[Φ(s_{t+1})-Φ(s_t)] per step
              ├── compute_advantage() (GRPO, teacher/on-policy baseline separation)
              ├── PPO update via het_actor.py (DR3 ratio + adaptive-μ BC loss)
              └── Checkpoint + wandb logging
```

### Key Files

| File | What it does |
|------|-------------|
| `agentevolver/module/trainer/ae_ray_trainer.py` | Core training loop. State Channel injection, advantage computation, all DUET metrics logging. ~3500+ lines, most critical file. |
| `agentevolver/module/exp_manager/het_actor.py` | Actor policy update. DR3 discriminator training, density ratio application, **adaptive BC μ schedules** (`chord_mu_adaptive_mode`: `disc_acc` is the paper's method; `gap`/`nll`/`va`/`ess_ratio`/`kl_lagrangian`/`disc_acc_velocity` are explored alternatives), loss dispatch (GRPO/LUFFY/CHORD/DR3). |
| `agentevolver/module/exp_manager/het_core_algos.py` | Loss functions: `het_compute_teacher_aware_loss()`, `repo_compute_token_loss()`, `compute_chord_sft_loss()`, DAPO loss. |
| `agentevolver/module/exp_manager/state_progress.py` | `ExpertProgressMap`: hash maps from teacher observations to progress values. `compute_trajectory_progress()`, `compute_step_deltas()`. |
| `agentevolver/module/exp_manager/dr3_ratio.py` | `DR3RatioEstimator`: discriminator-based density ratio estimation with dual ESS clipping. |
| `agentevolver/module/exp_manager/exp_manager.py` | Experience management: teacher trajectory loading (fail-fast if missing), replay buffer, difficulty tracking. |
| `agentevolver/module/exp_manager/experience_collate.py` | LUFFY mixing: `mix_trajectories()` merges teacher + on-policy rollouts per task group. |
| `agentevolver/module/env_manager/env_manager.py` | Parallel environment orchestration, trajectory tokenization, teacher log_prob alignment. |
| `launcher.py` | CLI entry: parses args, backs up code to `launcher_record/`, launches env services (`--with-alfworld` etc.), invokes `main_ppo.py`. |

### Configuration System

Hydra-based. Entry: `@hydra.main(config_path="../config", config_name="script_config")` in `main_ppo.py`.

Config hierarchy (later overrides earlier):
1. `external/config_fallback/ppo_trainer.yaml` — veRL PPO defaults
2. `config/agentevolver.yaml` — algorithm defaults (GRPO, DAPO, DR3, SC)
3. Experiment-specific YAML under `config/duet_paper_experiments_configs/`

Key DUET config knobs (in `actor_rollout_ref.actor`): `use_dr3`, `dr3.*` (disc_temperature, w_min, gap_gate_enable), `use_chord`, `chord_mu_adaptive: true` + `chord_mu_adaptive_mode: "disc_acc"` + `chord_mu_peak/valley/d_floor/d_ema_alpha` (实际主表取值按环境不同: ALFWorld 0.3/0.05/0.4/0.5, WebShop 0.3/0.10/0.6/0.2 — 见 rebuttal/paper_corrections.md C7). SC lives under `exp_manager.state_channel.*`.

### Experiment Config Layout

`config/duet_paper_experiments_configs/{env}/{env}_{model}_{algorithm}[_version].yaml` for alfworld/webshop/sciworld. Models: qwen1.5b, qwen3b (aka 3b), 7b, qwen3_4b, llama3b.

Caveats when picking a config:
- Many **versioned variants** (v24, v39, v39b, …) exist from the method-development sweeps; `disc_acc` adaptive-μ (v39b lineage) is the published method. Check `EXPERIMENT_LOG.md` and the newest `run_*.sh` scripts to see which config a result actually came from — don't assume the unversioned file is canonical.
- `ablations_neurips/{alfworld,webshop}/` holds the paper's ablation grid (`duet_minus_bc`, `duet_minus_sc`, etc.).
- Historical `run_*.sh` experiment queues (nohup + launcher.py per config) are archived under `run_scripts/{00_early_dev..60_rebuttal}/` grouped by era (see `run_scripts/README.md`); they double as the record of what was run in what order. They assume cwd = repo root — copy back to root to rerun. Only `run_a100_queue_driver.sh` (generic file-driven queue) stays at root for reuse.

### Tensor Shape Convention

Two length regimes coexist in the batch:
- **Full sequence**: `(bs, prompt_len + response_len)` — `input_ids`, `attention_mask`, `exp_mask`, `teacher_mask`, `step_ids`
- **Response only**: `(bs, response_len)` — `token_level_rewards`, `advantages`, `responses`

When indexing across regimes (e.g., using `step_ids` to mask `token_level_rewards`), always slice the full-sequence tensor to `[:, -response_len:]`. This is a recurring source of shape mismatch bugs.

### DUET-Specific Design Decisions

- **SC excludes teacher samples** (`state_channel.exclude_teacher: true`): Teacher trajectories have high progress by definition (~0.85). Adding SC bonus to them inflates GRPO advantages and fights DR3's natural fade-out.
- **DR3 w_hat floor** (`dr3.w_min: 0.01`): Numerical safety only — not meant to force minimum teacher influence. The natural fade-out is a feature.
- **Teacher baseline separation** (`algorithm.grpo.teacher_baseline_separation.enable: true`): This is Stage-1 Bias-1 correction — ablating it collapses 1.5B runs to ~0%.
- **BC and DR3 share one discriminator**: μ is computed from `dr3/disc_acc` (EMA-smoothed), so BC needs no manual schedule. Empirically ŵ_τ falls ~1.0 → ~0.67 over training (≈33% teacher-gradient down-weighting), not to zero.
- **Temporary batch keys** (`_sc_progress`, `_sc_bonus`, etc.): Stored in `batch.batch` for trajectory saving, then cleaned up before `update_actor()` to avoid FSDP serialization issues. Non-tensor data goes in `batch.non_tensor_batch`.

### Wandb Metrics to Monitor

| Metric | Healthy range | What it means |
|--------|--------------|---------------|
| `critic/success_onpolicy/mean` | Increasing | Primary performance metric (but training-time SR ≠ val SR) |
| `diag/teacher_sample_ratio` | ~0.125 | Teacher mix ratio (1/8 with n=8, n_teacher=1) |
| `duet/teacher_gradient_share` | Decreasing | DR3 teacher down-weighting |
| `chord/mu` | 0.3 → 0.05 | Adaptive BC weight (tracks 1−disc_acc) |
| `dr3/disc_acc` | 0.5 → 0.95+ | Discriminator separates teacher vs policy |
| `state_channel/bonus_vs_reward_ratio` | <0.15 | SC bonus proportional to task reward |
| `actor/kl_loss` | <0.5 | Policy stability |

### Environment Service

FastAPI + Ray Actor Pool (`env_service/`). Each environment (ALFWorld, WebShop, etc.) implements `BaseEnv` with `create/step/evaluate/release` endpoints. Environments run as separate processes — start via `start_env_{alfworld,webshop}.sh` (preferred; they source `env_config.sh` and manage ports) or `launcher.py --with-{env}`. WebShop uses a shared Ray actor internally (serialized access — a 2026-04 deadlock fix); don't revert it to per-request actors.

## Paper (`NeurIPS_2026_Latex/`)

- Build with `bash build.sh` (uses tectonic in the `tex` conda env). **After every edit to the paper, recompile and check the page count** (`build.sh pages`) — the main paper sits right at the 9-page limit.
- Sections live in `sections/`, tables in `tables/` (ablation cells are updated in-place as runs finish), figures rendered by `figures/make_figures.py`.
- Planning/context docs inside the folder: `PAPER_PLAN.md`, `STATUS_*.md`, `narrative.md`, `ABLATION_PLAN.md`, `mentor_comments/`, `pseudo_reviews/`. Read the newest STATUS file before touching the paper.

## Server Handoff Docs

`docs/neurips2026/handoffs/HANDOFF_{3B,L20X}_SERVER.md` capture per-server context (what's running, baseline numbers, narrative constraints) for work split across machines. `EXPERIMENT_LOG.md` is the cross-run results log. Analysis scripts (`analysis/`; one-off NeurIPS-era ones archived in `archive/analysis_root_scripts/`) write to `analysis_outputs/` and `analysis_reports/`. See `REPO_MAP.md` for the full directory guide after the 2026-07-31 reorg (`archive/MIGRATION_MANIFEST.tsv` maps old→new paths).

## Teacher Data

### Qwen2.5-72B (main teacher for paper experiments)
- ALFWorld: `data/teacher_trajectories/qwen72b/alfworld_qwen72b_filtered_react_tags.pkl` (19K trajectories, react_tags format)
- WebShop: `data/teacher_trajectories/qwen72b/webshop_qwen72b_filtered.pkl` (26K trajectories)

### Qwen3-30B-A3B-Thinking-2507 (newer teacher, for Qwen3 students)
- Collection: `bash scripts/collect_qwen3_teacher.sh {alfworld|webshop|sciworld|all}` → `data/teacher_trajectories/qwen3_30b/{env}_qwen3_30b.jsonl`; guide in `docs/teacher/QWEN3_TEACHER_COLLECTION.md`.

### General
- Format conversion: `python scripts/convert_alfworld_react_to_tags.py`; filtering: `python scripts/filter_teacher_trajectories.py --input <raw.jsonl> --output <filtered.jsonl>`
- If `teacher_experience.enable: true` but the data file is missing, `ExperienceManager` raises `FileNotFoundError` (fail-fast).
