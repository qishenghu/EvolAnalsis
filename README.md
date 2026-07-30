<p align="center">
 <img src="docs/img/logo.png" alt="AgentEvolver Logo" width="70%">
</p>
<h2 align="center">DUET: Dual-Channel Expert Trajectory Utilization for Off-Policy Integrated GRPO</h2>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11+-blue" alt="Python Version"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-black" alt="License"></a>
  <a href="https://arxiv.org/abs/2511.10395"><img src="https://img.shields.io/badge/arXiv-2511.10395-b31b1b.svg" alt="arXiv"></a>
</p>

**DUET** (DUal Expert Trajectory utilization) is a principled off-policy integrated GRPO algorithm for training LLM agents in interactive environments. It introduces two orthogonal channels — **Action Channel** and **State Channel** — that synergistically leverage expert demonstrations to accelerate on-policy learning while preserving asymptotic exploration capability.

Built on the [AgentEvolver](https://github.com/modelscope/AgentEvolver) framework with [veRL](https://github.com/volcengine/verl) distributed RL backend.

## Key Idea

Off-policy teacher data accelerates early RL training but can hurt asymptotic performance if not properly controlled. Existing approaches use heuristic schedules (CHORD's manually-tuned &mu; decay) or lack principled off-policy correction (LUFFY). DUET addresses both issues:

| Component | Role | Mechanism |
|-----------|------|-----------|
| **Action Channel (DR3)** | Off-policy correction | Density-ratio discriminator estimates `w(τ) = π_θ(τ) / π_teacher(τ)`, corrects importance weights. Provides **data-driven teacher fade-out** — teacher influence naturally decreases as the policy improves. |
| **State Channel (SC)** | Dense reward shaping | Expert progress map `Φ(s)` measures how far on-policy trajectories follow expert states. Adds `β·P(τ)` as shaped reward to combat reward sparsity. **Only applied to on-policy samples** — teacher trajectories don't need shaping. |

The two channels form a **closed-form teacher curriculum**: DR3 controls *gradient weight* (importance sampling), SC improves *reward quality* (dense shaping). Both naturally let the teacher guide early training and fade out for autonomous exploration in later stages — without hand-crafted schedules.

## Algorithm Comparison

| | GRPO | LUFFY | CHORD | **DUET** |
|---|---|---|---|---|
| Teacher data | None | Mixed into rollouts | SFT loss on teacher | Mixed + DR3 correction |
| Off-policy correction | N/A | Policy shaping (heuristic) | None (SFT avoids PG) | **DR3 density ratio** (principled) |
| Teacher fade-out | N/A | None | Manual &mu; schedule | **Data-driven** (closed-form) |
| Dense reward | No | No | No | **State Channel** |
| Reward shaping target | N/A | N/A | N/A | **On-policy only** (excludes teacher) |

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │          DUET Training Loop                 │
                    │                                             │
  ┌─────────┐      │  ┌──────────────┐    ┌──────────────────┐   │
  │ Teacher  │──────┼─▶│  LUFFY Mixer │───▶│  token_level_    │   │
  │  Data    │      │  │ (rollout-lvl)│    │  rewards         │   │
  └─────────┘      │  └──────────────┘    └────────┬─────────┘   │
                    │                              │              │
                    │  ┌──────────────┐            │              │
                    │  │State Channel │    β·P(τ)  │ on-policy    │
                    │  │ Progress Map │───────────▶│ samples only │
                    │  └──────────────┘            │              │
                    │                              ▼              │
                    │                     ┌────────────────┐      │
                    │                     │compute_advantage│      │
                    │                     │   (GRPO)       │      │
                    │                     └────────┬───────┘      │
                    │                              │              │
                    │  ┌──────────────┐            ▼              │
                    │  │Action Channel│   ┌────────────────┐      │
                    │  │  DR3 Density │──▶│  PPO Update    │      │
                    │  │  Ratio Repair│   │ (corrected     │      │
                    │  └──────────────┘   │  old_log_prob) │      │
                    │                     └────────────────┘      │
                    └─────────────────────────────────────────────┘
```

## Quick Start

### 1. Environment Setup

```bash
# Install dependencies
bash install.sh

# Set up ALFWorld environment
cd env_service/environments/alfworld && bash setup.sh && cd ../../..

# Activate
conda activate agentevolver
```

### 2. Prepare Teacher Data

Teacher trajectories should be collected from a strong model (e.g., Qwen-72B) and stored as pickle files:

```bash
# ALFWorld: convert to react_tags format if needed
python scripts/convert_alfworld_react_to_tags.py

# Data locations:
# ALFWorld: data/teacher_trajectories/qwen72b/alfworld_qwen72b_filtered_react_tags.pkl
# WebShop:  data/teacher_trajectories/qwen72b/webshop_qwen72b_filtered.pkl
```

### 3. Run Experiments

```bash
# ALFWorld 3B: DUET vs baselines (LUFFY, CHORD, GRPO)
bash run_scripts/00_early_dev/run_launcher.sh  # archived; copy to repo root to run

# WebShop 3B: DUET vs baselines
bash run_launcher_webshop_duet_paper.sh
```

Or run individual experiments:

```bash
# DUET
python launcher.py --conf config/duet_paper_experiments_configs/alfworld/alfworld_3b_duet.yaml

# LUFFY baseline
python launcher.py --conf config/duet_paper_experiments_configs/alfworld/alfworld_3b_luffy.yaml

# CHORD baseline
python launcher.py --conf config/duet_paper_experiments_configs/alfworld/alfworld_3b_chord.yaml
```

### 4. Monitor on wandb

Key metrics to track during DUET training:

| Metric | What it tells you |
|--------|-------------------|
| `critic/success_onpolicy/mean` | On-policy success rate (primary metric) |
| `duet/teacher_gradient_share` | Teacher influence fraction (should naturally decrease) |
| `state_channel/bonus_vs_reward_ratio` | SC bonus relative to task reward (should be <15%) |
| `dr3/disc_acc` | DR3 discriminator accuracy |
| `state_channel/progress_onpolicy_mean` | On-policy expert state coverage |
| `actor/kl_loss` | Policy divergence from reference |

## Configuration

### DUET-specific settings

```yaml
actor_rollout_ref:
  actor:
    use_dr3: true                    # Enable Action Channel
    kl_loss_coef: 0.005              # KL regularization strength
    dr3:
      enable: true
      disc_temperature: 1.5          # Discriminator softness (higher = softer ratios)
      disc_label_smoothing: 0.1      # Prevent over-confident discriminator
      w_min: 0.01                    # Minimum density ratio (numerical safety)
      gap_gate_enable: true          # Reward-gap based teacher gating

exp_manager:
  teacher_experience:
    enable: true
    mix_mode: rollout_level          # LUFFY-style rollout mixing
    n_teacher_rollouts_per_task: 1   # 1 teacher per 7 on-policy per group
    data_path: data/teacher_trajectories/qwen72b/alfworld_qwen72b_filtered_react_tags.pkl

  state_channel:
    enable: true                     # Enable State Channel
    exclude_teacher: true            # Only shape on-policy rewards (key design choice)
    beta: 0.2                        # Progress bonus coefficient
    beta_decay: true                 # Decay beta as performance improves
    beta_decay_target: 0.3           # Target reward for full decay
    step_level:
      enable: true                   # Per-step progress deltas
      eta: 0.05                      # Step-level coefficient
```

### Baseline configurations

All configs are in `config/duet_paper_experiments_configs/`:

```
alfworld/
  alfworld_3b_duet.yaml          # DUET (ours)
  alfworld_3b_luffy.yaml         # LUFFY baseline
  alfworld_3b_chord.yaml         # CHORD baseline
  alfworld_3b_onpolicy.yaml      # Vanilla GRPO
  alfworld_3b_state_channel.yaml # Ablation: SC only
  alfworld_3b_action_channel.yaml # Ablation: DR3 only
webshop/
  (same structure)
```

## Supported Environments

| Environment | Reward | Action Format | Tasks |
|-------------|--------|---------------|-------|
| ALFWorld | Binary {0,1} | react_tags | Household tasks |
| WebShop | Continuous + penalty | search/click | E-commerce shopping |
| ScienceWorld | Continuous [0,1] | Text commands | Science experiments |
| AppWorld | Binary | Python code | Multi-app API calls |
| BFCL | Binary | tool_call JSON | Function calling |

## Project Structure

```
├── agentevolver/
│   ├── module/
│   │   ├── trainer/
│   │   │   └── ae_ray_trainer.py      # Core training loop (State Channel, advantage computation)
│   │   ├── exp_manager/
│   │   │   ├── het_actor.py           # Actor with DR3 integration (Action Channel)
│   │   │   ├── het_core_algos.py      # Loss functions (GRPO/LUFFY/CHORD/DR3)
│   │   │   ├── state_progress.py      # Expert progress map (State Channel)
│   │   │   ├── dr3_ratio.py           # Density ratio estimator
│   │   │   ├── exp_manager.py         # Experience management + teacher loading
│   │   │   └── experience_collate.py  # LUFFY mixing logic
│   │   ├── env_manager/               # Parallel environment orchestration
│   │   └── task_manager/              # Task lifecycle + data loading
│   └── schema/                        # Data models (Trajectory, Sample, Task)
├── env_service/                       # FastAPI environment service
│   └── environments/                  # ALFWorld, WebShop, SciWorld, etc.
├── config/                            # Hydra YAML configs
│   └── duet_paper_experiments_configs/ # All DUET paper experiment configs
├── scripts/                           # Data collection & processing
├── data/teacher_trajectories/         # Expert demonstration data
├── launcher.py                        # Main launcher (config + services + training)
└── run_scripts/                       # Archived experiment launch queues (see run_scripts/README.md)
```

## Acknowledgements

Built upon:
- [AgentEvolver](https://github.com/modelscope/AgentEvolver) — self-evolving agent training framework
- [veRL](https://github.com/volcengine/verl) — distributed RL training
- [vLLM](https://github.com/vllm-project/vllm) — fast LLM inference

## Citation

```bibtex
@misc{AgentEvolver2025,
  title         = {AgentEvolver: Towards Efficient Self-Evolving Agent System},
  author        = {Yunpeng Zhai and Shuchang Tao and Cheng Chen and Anni Zou and Ziqian Chen and Qingxu Fu and Shinji Mai and Li Yu and Jiaji Deng and Zouying Cao and Zhaoyang Liu and Bolin Ding and Jingren Zhou},
  year          = {2025},
  eprint        = {2511.10395},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2511.10395}
}
```
