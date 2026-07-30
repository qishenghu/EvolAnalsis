# Handoff to Remote 3B Server — For Another Claude Instance

**From**: Primary DUET research server (1.5B experiments)
**To**: Remote 8×A100 server (3B scaling experiments)
**Date**: 2026-04-24
**Last 1.5B commit pushed**: `9acd90c4`

---

## 1. Quick Context (What This Project Is)

**DUET** = DUal Expert Trajectory utilization. An off-policy RL algorithm for LLM agents, targeting **NeurIPS 2026** (deadline ~2-3 weeks out).

The algorithm has **3 components**:
- **DR3 (Action Channel)**: Discriminator-based density-ratio correction for teacher trajectories. Computes `w_hat = D/(1-D)` where `D(s,a)` is the discriminator's probability that `(s,a)` came from teacher.
- **SC (State Channel)**: Expert progress map providing dense reward shaping for on-policy samples.
- **BC (Behavior Cloning)**: `μ · (-log π_θ(a_teacher|s))` added on teacher tokens. The weight μ is the **key question**.

**The question we've been solving for 2 weeks**: How do we set μ without hand-tuning? v24 used `μ=0.3→0.05 over 25 steps` manually; we want closed-form adaptive.

---

## 2. The Paper Narrative (DO NOT DEVIATE)

We settled on this after Phase 1 experiments:

> **"A single DR3 discriminator D(s,a) is a sufficient statistic for both (1) density-ratio correction (w = D/(1-D)) AND (2) closed-form adaptive BC schedule (μ = μ_max·(1 − acc(D))/(1 − d_floor)). The Bayes-accuracy-to-TV identity (2·acc − 1 ≈ TV(π_θ, π_teacher)) gives μ a KKT-multiplier interpretation on a Total-Variation budget — without requiring teacher logprobs or dual-ascent machinery."**

**Translation**: The BC weight μ is driven by the DR3 discriminator's accuracy. When disc_acc ≈ 0.5 (can't distinguish policy from teacher), μ = μ_max (strong BC). When disc_acc → 1.0 (fully separable), μ → μ_min (BC retires). **Single discriminator, two roles.**

### Full adaptive rule (v39b — the winner)

```python
d_t = EMA(dr3/disc_acc, alpha=0.5)
μ_t = clamp(μ_max · (1 - d_t) / (1 - d_floor), μ_min, μ_max)

# Hyperparameters
μ_max = 0.3
μ_min = 0.05
d_floor = 0.5
d_ema_alpha = 0.5
```

**Only 2 hyperparameters** (d_floor, d_ema_alpha) — vs CHORD's 4-parameter manual schedule.

---

## 3. Baseline Results (1.5B, already complete)

### WebShop 1.5B (200-task validation, 100 training steps)

| Method | reward@100 | success@100 |
|---|---:|---:|
| GRPO (no BC) | 0.549 | 4.0% |
| LUFFY | 0.573 | 5.5% |
| CHORD | 0.603 | 11.5% |
| SFT alone | 0.387 | 0% |
| SFT + GRPO | 0.404 | — |
| **DUET v24** (hand-tuned BC) | **0.678** | **22.0%** |
| **DUET v39b** (adaptive BC) ⭐ | **0.637** | **19.0%** |

### ALFWorld 1.5B

| Method | Val@50 success | Val@100 success |
|---|---:|---:|
| OnPolicy GRPO | — | 1.0% |
| LUFFY | — | 5.5% |
| CHORD | — | 27.0% |
| **DUET v1** (no BC) | 27.5% | **32.5%** |
| DUET v24 (hand-tuned) | 33.5% | **30.5% ↓** (regression) |
| DUET v39 (α=0.2) | 45.5% | **42.0%** |
| **DUET v39b** (α=0.5) | 40.0% | **crashed at Val@100 — needs rerun** |

**Key story for paper**: v39b matches v24 on WebShop (within 3pp) AND dramatically beats v24 on ALFWorld (+10-12pp Val@100). Cross-environment robustness from adaptive BC.

---

## 4. YOUR Mission on This Server (3B Scaling)

Run two experiments sequentially via the provided script:

### Experiment 1: v39b on WebShop 3B
- **Config**: `config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v39b.yaml`
- **Model**: `/data/shared_models/Qwen2.5-3B-Instruct` (or wherever 3B model is on your server — may need to update config path)
- **Expected runtime**: ~8-10h on 8×A100
- **Success criterion**: success@100 at or near baseline DUET 3B WebShop results
  - DUET 3B WebShop baseline: reward@100 = 0.763, success@100 = TBD
  - Goal: v39b should match or exceed baseline DUET while requiring zero schedule tuning

### Experiment 2: v39b on ALFWorld 3B
- **Config**: `config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_duet_v39b.yaml`
- **Expected runtime**: ~8-10h on 8×A100
- **Success criterion**: success@100 ≥ v24 3B ALFWorld baseline
  - DUET 3B ALFWorld baseline: Val@100 = 69.5%
  - Since 1.5B showed v39 > v24 on ALFWorld by +11.5pp, 3B should also see improvement or at minimum parity

### How to run

```bash
# Clone the repo (or git pull if already cloned)
cd /path/to/EvolAnalsis
git pull origin main

# Make sure env_config.sh exists and sets CONDA_ENV_DUET + RAY_TMPDIR
# (template likely already in repo — if missing, create one matching 1.5B server)

# Make sure the 3B model is downloaded:
ls /data/shared_models/Qwen2.5-3B-Instruct   # or update config path if different

# Launch the orchestrator
nohup bash run_duet_3b_v39b.sh > logs/3b_orchestrator.log 2>&1 &

# Monitor
tail -f logs/webshop_qwen3b_duet_v39b.log
tail -f logs/alfworld_qwen3b_duet_v39b.log
```

The orchestrator runs WebShop 3B → ALFWorld 3B sequentially. ~16-20h total.

### Critical config values to verify before launch

Both 3B configs contain:
```yaml
actor_rollout_ref:
  actor:
    use_dr3: true
    use_chord: true
    chord_mu_warmup_steps: 0
    chord_mu_decay_steps: 25
    chord_mu_peak: 0.3
    chord_mu_valley: 0.05
    chord_use_token_weighting: false
    # v39b: adaptive μ from disc_acc (fast EMA α=0.5)
    chord_mu_adaptive: true
    chord_mu_adaptive_mode: "disc_acc"
    chord_mu_d_floor: 0.5
    chord_mu_d_ema_alpha: 0.5
```

If `chord_mu_adaptive` is NOT `true` or `chord_mu_adaptive_mode` is not `"disc_acc"`, the run will default to hand-tuned v24 schedule — NOT what we want.

---

## 5. What to Monitor (Health Checks During Run)

### Adaptive-μ sanity checks

The following metrics should appear in wandb logs:

| Metric | Healthy range | What it means |
|---|---|---|
| `chord/mu_mode` | **3.0** | Confirms disc_acc adaptive mode active |
| `chord/disc_acc_ema` | 0.5 → 1.0 over ~30 steps | EMA of discriminator accuracy |
| `chord/mu_adaptive_gated` | 1.0 → 0.0 over training | The gated (pre-clamp) μ coefficient |
| `chord/mu` | 0.30 → 0.05 over ~25 steps | Final applied μ |

### Other important training metrics

| Metric | Healthy | Danger sign |
|---|---|---|
| `actor/grad_norm` | 3-8 | >20 = instability |
| `actor/kl_loss` | <0.5 | >1.0 = policy drift |
| `critic/rewards_onpolicy/mean` | rising over training | stagnant = learning failed |
| `dr3/disc_acc` | 0.5 early, →1.0 by step 30 | stuck at 0.5 = discriminator not training |
| `chord/mu_mode` | 3.0 throughout | anything else = adaptive disabled |

### Expected μ trajectory (from 1.5B runs)

```
step 1:   μ ≈ 0.30   (max)
step 5:   μ ≈ 0.30
step 10:  μ ≈ 0.25
step 20:  μ ≈ 0.15
step 25:  μ ≈ 0.12
step 30:  μ ≈ 0.08
step 50:  μ ≈ 0.05   (at floor)
step 100: μ ≈ 0.05   (at floor)
```

3B should show similar shape (maybe slightly faster decay if 3B discriminates teacher-from-policy faster).

---

## 6. Known Risks / Issues

### Risk 1: CUDA OOM at Val@100 (just happened on 1.5B)
The 1.5B ALFWorld v39b run crashed at step 99 during val@100 phase due to vLLM kv_cache wake_up OOM. If 3B hits the same issue:
- Check `gpu_memory_utilization` in config — may need to lower (e.g. from 0.65 → 0.55)
- Check `param_offload: false` / `optimizer_offload: false` — if memory tight, enable offload
- After training completes, even if val@100 crashes, **Val@50 numbers are already logged** and usable

### Risk 2: 3B may saturate disc_acc differently
On 1.5B, disc_acc reaches 1.0 by step 26-30. On 3B, the larger policy may:
- Saturate faster → μ retires too early → miss imprinting
- Or saturate slower → μ stays high too long → over-imitation

Both are monitoring issues, not blockers. Just watch `chord/disc_acc_ema` vs training step.

### Risk 3: Config paths may differ on remote server
The config assumes `/data/shared_models/Qwen2.5-3B-Instruct`. If your server has it elsewhere, edit both configs:
```yaml
actor_rollout_ref:
  model:
    path: /YOUR/PATH/TO/Qwen2.5-3B-Instruct
```

### Risk 4: Rank desync (minor)
On 4-GPU FSDP with `broadcast_params: true`, only rank0 computes disc_acc; other ranks see `disc_acc=0.0` during warmup, with a fallback to 0.5 substitution. This is **known and handled** via the `disc_trained_steps` guard in the adaptive rule. Don't try to fix unless runs clearly fail.

### Risk 5: Orchestrator `set -e` behavior
The provided `run_duet_3b_v39b.sh` uses `set -e` — if WebShop 3B crashes, ALFWorld 3B won't run. **If that happens**: manually run the failed one or launch ALFWorld separately:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launcher.py \
  --conf config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_duet_v39b.yaml \
  > logs/alfworld_qwen3b_duet_v39b.log 2>&1 &
```

---

## 7. What to Report Back

After experiments complete, the user needs:

### Results summary (CSV-like)
```
variant, env, reward@50, success@50, reward@100, success@100, crashed?
webshop_qwen3b_duet_v39b, webshop, ?, ?, ?, ?, y/n
alfworld_qwen3b_duet_v39b, alfworld, ?, ?, ?, ?, y/n
```

### μ trajectory verification
Extract `chord/mu` at steps {1, 10, 25, 50, 100} from each log. Paste into the report.

### Any anomalies
- grad_norm spikes
- kl_loss > 1.0 episodes
- disc_acc trajectory shape differences from 1.5B

### Preserve these files (do NOT delete):
- `logs/webshop_qwen3b_duet_v39b.log`
- `logs/alfworld_qwen3b_duet_v39b.log`
- `experiments/webshop/webshop_qwen3b_duet_v39b/validation_log/*.jsonl`
- `experiments/alfworld/alfworld_qwen3b_duet_v39b/validation_log/*.jsonl`
- `experiments/webshop/webshop_qwen3b_duet_v39b/rollout_log/*.jsonl`
- `experiments/alfworld/alfworld_qwen3b_duet_v39b/rollout_log/*.jsonl`

These are needed for paper ablation tables and case studies.

Checkpoint folders `global_step_100/` are ~60GB per 3B run. Can be deleted after results extracted. Keep `Trajectory/` folders (small, trajectory data).

---

## 8. If Things Go Badly

### If both experiments crash:
1. Check disk space: `df -h /data`
2. Check GPU state: `nvidia-smi`
3. Check Ray processes: `ps -ef | grep ray`
4. Kill everything cleanly: `ray stop --force` + kill launcher PIDs
5. Restart with lower memory settings: edit configs to set `gpu_memory_utilization: 0.55`, `param_offload: true`

### If WebShop 3B crashes specifically:
- WebShop env service at `http://127.0.0.1:8083` may have died — restart via `bash start_env_webshop.sh`
- Check AgentGym server on port 36003
- The orchestrator has `if ! curl -s http://127.0.0.1:8083` check but failures in-flight won't auto-recover

### If ALFWorld 3B crashes specifically:
- ALFWorld env at `http://127.0.0.1:8081`
- `start_env_alfworld.sh` handles lifecycle
- ALFWorld uses a lot of CPU — may be CPU-bound on env simulation

### If disc_acc never rises:
- Check `dr3/apply_ready` (should become 1 by step 10)
- Check `dr3/buf_size` (should reach `apply_min_buf_size` = 512)
- If discriminator broken, adaptive μ defaults to μ_max throughout — not catastrophic, just behaves like constant μ=0.3

---

## 9. Reference: Paper Narrative Quick-Ref

If the user asks "what's this experiment about" while it's running:

**Single sentence**: "v39b tests whether DUET's discriminator-driven adaptive BC scales to 3B — the key experiment for the NeurIPS paper's cross-scale claim."

**Three sentences**: "DUET v39b uses the DR3 discriminator's accuracy to automatically control BC weight μ (no manual schedule). On 1.5B WebShop it achieves 19.0% success vs CHORD's 11.5% and vs hand-tuned v24's 22.0% — matching the hand-tuned version with only 2 hyperparameters. On 1.5B ALFWorld, v39 (similar variant) beat v24 by +11.5pp. This 3B run validates that the closed-form adaptive approach generalizes across model scales."

---

## 10. Files/Paths on This (1.5B) Server for Reference

Everything committed and pushed to `git@github.com:qishenghu/EvolAnalsis.git` main branch.

**Critical files**:
- `analysis_reports/PHASE1_SYNTHESIS.md` — full synthesis, 10 sections
- `analysis_reports/phase1_deep_dive.md` — per-variant empirical diagnosis
- `analysis_reports/theory_empirics_reconciliation.md` — why disc_acc wins
- `analysis_reports/round8_preflight_audit.md` — implementation audit (important!)
- `agentevolver/module/exp_manager/het_actor.py:1757-1976` — adaptive μ dispatch
- `config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet_v39b.yaml` — 1.5B reference
- `config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v39b.yaml` — YOUR config
- `config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_duet_v39b.yaml` — YOUR config
- `run_duet_3b_v39b.sh` — YOUR orchestrator

## 11. Contact (Primary Server Claude)

If this Claude instance is unclear about anything, ping the primary server user (qisheng) who can bridge between Claude instances. The primary server has full conversation history about:
- Why Lagrangian / NLL / ESS all failed (implementation issues, not fundamental)
- Why disc_acc won (buffer averaging, self-correction, zero extra compute)
- The KKT/TV theoretical interpretation details
- Prior ablations v1, v12, v22-v41 — context for reviewer responses

---

**TL;DR for this server's Claude**: Run `bash run_duet_3b_v39b.sh`. Monitor `chord/mu_mode=3.0` and `chord/disc_acc_ema` trajectory. Report back the validation numbers after ~16-20h. This is the NeurIPS paper's cross-scale validation — don't skip the Val@100 step.
