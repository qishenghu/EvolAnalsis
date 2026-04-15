# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

DUET (DUal Expert Trajectory utilization) — an off-policy integrated GRPO algorithm for training LLM agents in interactive environments. Built on the AgentEvolver framework with veRL (distributed RL) and vLLM (inference) backends.

Two orthogonal channels leverage expert demonstrations:
- **Action Channel (DR3)**: Density-ratio discriminator corrects importance weights for teacher samples. Provides data-driven teacher fade-out — no manual schedule needed.
- **State Channel (SC)**: Expert progress map provides dense reward shaping for on-policy samples only (teacher samples excluded by design).

Baselines: GRPO (on-policy), LUFFY (teacher mixing + policy shaping), CHORD (GRPO + weighted SFT).

## Common Commands

```bash
# Setup
bash install.sh                    # Creates conda env 'agentevolver'
conda activate agentevolver

# Run experiments
python launcher.py --conf config/duet_paper_experiments_configs/alfworld/alfworld_3b_duet.yaml
bash run_launcher.sh               # ALFWorld: DUET + LUFFY + CHORD
bash run_launcher_webshop_duet_paper.sh  # WebShop: DUET + LUFFY + CHORD

# Tests
pytest tests/ -v
pytest tests/test_task_manager.py::test_specific -v

# GPU selection
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Kill lingering Ray processes before re-run
python launcher.py --conf <config> --kill
```

## Architecture

### Training Flow

```
launcher.py (arg parse, backup code, launch env services)
  └─> python -m agentevolver.main_ppo (Hydra config: config/script_config.yaml)
      └─> TaskRunner.run() [Ray remote actor]
          └─> AgentEvolverRayPPOTrainer.fit() [ae_ray_trainer.py]
              ├── Rollout generation (vLLM async + environment interaction)
              ├── LUFFY mixer: merge teacher trajectories into batch
              ├── State Channel: β·P(τ) bonus on on-policy samples
              ├── Step-level deltas: η·[Φ(s_{t+1})-Φ(s_t)] per step
              ├── compute_advantage() (GRPO: group-relative normalization)
              ├── PPO update via het_actor.py (DR3 corrects old_log_prob)
              └── Checkpoint + wandb logging
```

### Key Files

| File | What it does |
|------|-------------|
| `agentevolver/module/trainer/ae_ray_trainer.py` | Core training loop. State Channel injection, advantage computation, all DUET metrics logging. ~3500+ lines, most critical file. |
| `agentevolver/module/exp_manager/het_actor.py` | Actor policy update. DR3 discriminator training, density ratio application, loss computation dispatch (GRPO/LUFFY/CHORD/DR3). |
| `agentevolver/module/exp_manager/het_core_algos.py` | Loss functions: `het_compute_teacher_aware_loss()`, `repo_compute_token_loss()`, `compute_chord_sft_loss()`, DAPO loss. |
| `agentevolver/module/exp_manager/state_progress.py` | `ExpertProgressMap`: builds hash maps from teacher observations to progress values. `compute_trajectory_progress()`, `compute_step_deltas()`. |
| `agentevolver/module/exp_manager/dr3_ratio.py` | `DR3RatioEstimator`: discriminator-based density ratio estimation with dual ESS clipping. |
| `agentevolver/module/exp_manager/exp_manager.py` | Experience management: teacher trajectory loading (fail-fast if missing), experience replay buffer, difficulty tracking. |
| `agentevolver/module/exp_manager/experience_collate.py` | LUFFY mixing: `mix_trajectories()` merges teacher + on-policy rollouts per task group. |
| `agentevolver/module/env_manager/env_manager.py` | Parallel environment orchestration, trajectory tokenization, teacher log_prob alignment. |
| `launcher.py` | CLI entry point: parses args, backs up code to `launcher_record/`, launches env services, invokes `main_ppo.py`. |

### Configuration System

Hydra-based. Entry: `@hydra.main(config_path="../config", config_name="script_config")` in `main_ppo.py`.

Config hierarchy (later overrides earlier):
1. `external/config_fallback/ppo_trainer.yaml` — veRL PPO defaults
2. `config/agentevolver.yaml` — algorithm defaults (GRPO, DAPO, DR3, SC)
3. Experiment-specific YAML (e.g., `config/duet_paper_experiments_configs/alfworld/alfworld_3b_duet.yaml`)

All experiment configs live under `config/duet_paper_experiments_configs/{env}/{env}_{size}_{algorithm}.yaml`.

### Tensor Shape Convention

Two length regimes coexist in the batch:
- **Full sequence**: `(bs, prompt_len + response_len)` — `input_ids`, `attention_mask`, `exp_mask`, `teacher_mask`, `step_ids`
- **Response only**: `(bs, response_len)` — `token_level_rewards`, `advantages`, `responses`

When indexing across regimes (e.g., using `step_ids` to mask `token_level_rewards`), always slice the full-sequence tensor to `[:, -response_len:]`. This is a recurring source of shape mismatch bugs.

### DUET-Specific Design Decisions

- **SC excludes teacher samples** (`state_channel.exclude_teacher: true`): Teacher trajectories have high progress by definition (~0.85). Adding SC bonus to them inflates GRPO advantages and fights DR3's natural fade-out.
- **DR3 w_hat floor** (`dr3.w_min: 0.01`): Numerical safety only — not meant to force minimum teacher influence. The natural fade-out is a feature.
- **Teacher baseline separation** (`algorithm.grpo.teacher_baseline_separation.enable: true`): GRPO uses separate mean/std for teacher vs on-policy samples to prevent teacher rewards from dominating normalization.
- **Temporary batch keys** (`_sc_progress`, `_sc_bonus`, etc.): Stored in `batch.batch` for trajectory saving, then cleaned up before `update_actor()` to avoid FSDP serialization issues. Non-tensor data goes in `batch.non_tensor_batch`.

### Wandb Metrics to Monitor

| Metric | Healthy range | What it means |
|--------|--------------|---------------|
| `critic/success_onpolicy/mean` | Increasing | Primary performance metric |
| `diag/teacher_sample_ratio` | ~0.125 | Teacher mix ratio (1/8 with n=8, n_teacher=1) |
| `duet/teacher_gradient_share` | 50%→5% over training | DR3 closed-form teacher curriculum |
| `state_channel/bonus_vs_reward_ratio` | <0.15 | SC bonus proportional to task reward |
| `dr3/disc_acc` | 0→0.95+ | Discriminator learns to separate distributions |
| `actor/kl_loss` | <0.5 | Policy stability |

### Environment Service

FastAPI + Ray Actor Pool (`env_service/`). Each environment (ALFWorld, WebShop, etc.) implements `BaseEnv` with `create/step/evaluate/release` endpoints. Environments run as separate processes, launched by `launcher.py --with-{env}`.

## Teacher Data

### Qwen2.5-72B (existing)
- ALFWorld: `data/teacher_trajectories/qwen72b/alfworld_qwen72b_filtered_react_tags.pkl` (19K trajectories, react_tags format)
- WebShop: `data/teacher_trajectories/qwen72b/webshop_qwen72b_filtered.pkl` (26K trajectories, react_tags format)

### Qwen3-30B-A3B-Thinking-2507 (new teacher)
- Model: `models/Qwen/Qwen3-30B-A3B-Thinking-2507` (MoE, qwen3_moe, ~30B total / ~3B active, BF16 ~61GB)
- Collection: `bash scripts/collect_qwen3_teacher.sh {alfworld|webshop|sciworld|all}`
- Output: `data/teacher_trajectories/qwen3_30b/{env}_qwen3_30b.jsonl`
- Guide: `docs/QWEN3_TEACHER_COLLECTION.md`
- Student: `Qwen3-1.7B` (same vocab/tokenizer, log_probs directly reusable)
- Training context: 32K (prompt=28672 + response=4096), fits 4×A100-80GB for 1.7B

### General
- Format conversion: `python scripts/convert_alfworld_react_to_tags.py`
- Filtering: `python scripts/filter_teacher_trajectories.py --input <raw.jsonl> --output <filtered.jsonl>`
- If `teacher_experience.enable: true` but data file missing, `ExperienceManager` raises `FileNotFoundError` (fail-fast).
