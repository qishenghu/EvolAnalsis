# Handoff to 4×L20X Server — DUET v39 Series (3B)

**From**: 3B baselines server (4×H100-80G) — now running SFT / SFT+RL / CHORD rerun to fill the 3B baseline matrix
**To**: 4×L20X server — assigned the **DUET v39 series (3B)** experiments
**Date**: 2026-04-24
**Latest commit on main**: `853e3ff8`
**Paper deadline**: NeurIPS 2026 — 2026-05-07 (~2 weeks)

---

## 1. Why Two Servers

Paper's 3B main table needs: OnPolicy, LUFFY, CHORD, SFT+RL, **DUET (ours)**, all on both WebShop and ALFWorld. The 4×H100 primary server is taking the **baseline补齐** track (SFT, SFT+RL, optional CHORD rerun). **You (4×L20X) own the DUET side**.

| Server | Track | Experiments | ETA |
|---|---|---|---|
| 4×H100 (primary) | Baselines补齐 | `webshop_qwen3b_sft` → `webshop_qwen3b_sft_rl` → `alfworld_qwen3b_sft` → `alfworld_qwen3b_sft_rl` (+ optional ALFWorld CHORD rerun) | ~42h (sequential) |
| **4×L20X (you)** | **DUET v39 series** | **`webshop_qwen3b_duet_v39b` → `alfworld_qwen3b_duet_v39b`** | **~18–22h (sequential)** |

Don't touch the baselines; we've got them. Don't run other DUET variants (v24, v39c, etc.) without coordinating — only v39b at 3B is on the plan.

---

## 2. What DUET v39b Actually Is (One Paragraph)

DUET = DR3 (action-channel density-ratio correction on teacher samples) + State Channel (progress shaping on on-policy samples) + adaptive BC weight μ. **v39b's key move**: μ is driven by the DR3 discriminator's accuracy via
```
d_t = EMA(disc_acc, α=0.5)
μ_t = clamp(μ_max · (1 − d_t) / (1 − d_floor), μ_min, μ_max)
# μ_max=0.3, μ_min=0.05, d_floor=0.5
```
→ **one discriminator does two jobs**: density-ratio correction (`w = D/(1−D)`) AND closed-form adaptive BC schedule. Only 2 new hyperparameters (`d_floor`, `α`). This is v39b's thesis. Paper claim: this scales from 1.5B to 3B without tuning.

**1.5B v39b reference numbers** (already completed elsewhere):
- WebShop: reward@100 = 0.637, success@100 = 19.0% (vs CHORD 11.5%, v24 hand-tuned 22.0%)
- ALFWorld: Val@50 = 40.0% (Val@100 crashed — vLLM OOM during validation)

---

## 3. Your Mission — Two Experiments, Sequential

### Experiment 1 — WebShop 3B
- Config: `config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v39b.yaml`
- Wall clock: ~9–11h on 4×L20X
- Success criterion: **reward@100 ≥ 0.763** (matches the existing DUET-v1 3B reference) and **success@100 ≥ 32%** (existing v1 record). Beating v1 would be nice but not required — the paper narrative is "v39b scales; adaptive BC works at 3B without schedule tuning."

### Experiment 2 — ALFWorld 3B
- Config: `config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_duet_v39b.yaml`
- Wall clock: ~9–11h on 4×L20X
- Success criterion: **Val@100 ≥ 0.695** (existing DUET-v1 3B reference). 1.5B showed v39 > v24 on ALFWorld by +11.5pp, so 3B likely improves or matches v1.

Orchestrator script is already in the repo: `run_duet_3b_v39b.sh`. It handles env-service lifecycle (WebShop up → exp1 → WebShop down → ALFWorld up → exp2 → ALFWorld down) automatically.

---

## 4. Server Setup Checklist (Edits You Must Make)

Repo paths are written for my (4×H100) server. You need to patch for your L20X paths.

### 4.1 Clone + checkout
```bash
git clone https://github.com/qishenghu/EvolAnalsis.git
cd EvolAnalsis
git checkout main
git log -1 --oneline   # should show 853e3ff8 or later
```

### 4.2 Model path (BOTH configs)
Configs currently hardcode `/data/shared_models/Qwen2.5-3B-Instruct`. Change to wherever your server has Qwen2.5-3B-Instruct:

```bash
# Replace <YOUR_PATH> with e.g. /models/Qwen2.5-3B-Instruct
sed -i 's|/data/shared_models/Qwen2.5-3B-Instruct|<YOUR_PATH>|' \
    config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v39b.yaml \
    config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_duet_v39b.yaml
```

Verify:
```bash
grep "model:" -A1 config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v39b.yaml
grep "model:" -A1 config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_duet_v39b.yaml
```

### 4.3 `env_config.sh`
Edit these lines to match your server:
```bash
# Conda
export CONDA_PATH="/path/to/anaconda3"          # e.g. /opt/conda
export CONDA_ENV_DUET="duet"                    # or whatever you named the training env
export CONDA_ENV_WEBSHOP="agentenv-webshop"     # name of the WebShop env
export CONDA_ENV_ALFWORLD="alfworld"            # name of the ALFWorld env

# Ray tmp — must have 50GB+ free; avoid /tmp if small
export RAY_TMPDIR="/path/to/big/disk/ray"

# ALFWorld
export ALFWORLD_DATA="/path/to/alfworld_data"   # directory containing json_2.1.1/, logic/, detectors/
```

You also need the **ALFWorld cache symlink** (the `env_wrapper.py` forcibly resets `ALFWORLD_DATA` to `~/.cache/alfworld` at import time — so we symlink instead):
```bash
rm -rf ~/.cache/alfworld
mkdir -p ~/.cache
ln -sfn "$ALFWORLD_DATA" ~/.cache/alfworld
ls ~/.cache/alfworld/json_2.1.1/train | wc -l   # should print 2420+
```

### 4.4 GPU count in orchestrator
Script already defaults to `0,1,2,3` after the recent commit. If your L20X indices differ, override at launch:
```bash
CUDA_GPUS=0,1,2,3 bash run_duet_3b_v39b.sh
```

### 4.5 wandb
```bash
wandb login       # paste API key
wandb status      # api_key must be non-null before launching
```

### 4.6 lsof / fuser
The env-startup scripts use `lsof` to detect bound ports. If not installed:
```bash
apt-get install -y lsof      # or: yum install -y lsof
```

### 4.7 Teacher trajectories
Both configs reference `data/teacher_trajectories/qwen72b/{webshop,alfworld}_qwen72b_filtered*.pkl`. These **are not in git** (too large). Copy from the primary server or from whatever artifact store you use:
```bash
ls data/teacher_trajectories/qwen72b/webshop_qwen72b_filtered.pkl
ls data/teacher_trajectories/qwen72b/alfworld_qwen72b_filtered_react_tags.pkl
```
If missing, `ExperienceManager` will fail fast at startup (by design).

### 4.8 L20X-specific memory tuning (likely required)
L20X has ~48GB per GPU vs H100's 80GB. Current configs may OOM. Start with these tweaks:

```yaml
actor_rollout_ref:
  rollout:
    gpu_memory_utilization: 0.55    # was 0.65 — give vLLM less
  actor:
    fsdp_config:
      param_offload: false          # try false first
      optimizer_offload: true       # enable if OOM during training step
    ppo_micro_batch_size_per_gpu: 1 # drop from 2 if OOM during forward
```

**Rollout phase** is the main OOM risk (vLLM + training weights co-resident). If WebShop exp crashes within first 5 minutes with `CUDA out of memory`, lower `gpu_memory_utilization` further (0.55 → 0.45) before touching offload.

---

## 5. Launch Sequence

Once 4.1–4.8 are done:
```bash
source /path/to/anaconda3/etc/profile.d/conda.sh
conda activate duet

# Dry check: env vars resolve
source env_config.sh
echo "ALFWORLD_BIN=$ALFWORLD_BIN"
echo "ALFWORLD_DATA=$ALFWORLD_DATA"
ls $ALFWORLD_BIN $ALFWORLD_DATA   # both must exist

# Launch (the orchestrator starts WebShop env, runs exp1, stops WebShop, starts ALFWorld, runs exp2, stops ALFWorld)
mkdir -p logs
nohup bash run_duet_3b_v39b.sh > logs/3b_orchestrator.log 2>&1 &
disown
echo $! > logs/3b_orchestrator.pid

# Monitor
tail -f logs/3b_orchestrator.log                    # high-level progress
tail -f logs/webshop_qwen3b_duet_v39b.log           # exp1 detail (first ~9-11h)
tail -f logs/alfworld_qwen3b_duet_v39b.log          # exp2 detail (after exp1)
```

The orchestrator uses `set -e`, so **if WebShop crashes, ALFWorld never starts**. If that happens, launch ALFWorld manually:
```bash
bash start_env_alfworld.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python launcher.py \
  --conf config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_duet_v39b.yaml \
  > logs/alfworld_qwen3b_duet_v39b.log 2>&1 &
```

---

## 6. What to Monitor (Live Health)

### Must-have wandb metrics

| Metric | Healthy | Danger |
|---|---|---|
| `chord/mu_mode` | **3.0** constant | anything else ⇒ adaptive μ OFF (config wrong) |
| `chord/disc_acc_ema` | 0.5 at step 1 → ~1.0 by step 30 | stuck at 0.5 ⇒ discriminator not training |
| `chord/mu_adaptive_gated` | 1.0 early → 0.0 late | flat ⇒ EMA broken |
| `chord/mu` | 0.30 → 0.05 over ~25 steps | stuck at 0.3 throughout ⇒ adaptive disabled |
| `actor/grad_norm` | 3–8 | >20 = instability |
| `actor/kl_loss` | <0.5 | >1.0 = policy drift |
| `critic/rewards_onpolicy/mean` | rising | stagnant = training failed |
| `dr3/disc_acc` | → 0.95+ by step 30 | stuck near 0.5 |
| `duet/teacher_gradient_share` | ~50% early → ~5% by step 100 | flat = DR3 broken |

### Expected μ trajectory (from 1.5B empirical; 3B should look similar)
```
step 1:   μ ≈ 0.30
step 5:   μ ≈ 0.30
step 10:  μ ≈ 0.25
step 20:  μ ≈ 0.15
step 25:  μ ≈ 0.12
step 30:  μ ≈ 0.08
step 50:  μ ≈ 0.05
step 100: μ ≈ 0.05
```
3B may decay slightly faster (stronger model → discriminator saturates earlier) — that's fine as long as `disc_acc_ema` is rising.

---

## 7. Known Risks

### R1 — L20X OOM (most likely issue)
See §4.8. If both rollout and training fit on 80GB H100 with `gpu_mem_util=0.65`, 48GB L20X needs ~30% headroom. **Start conservative** (`gpu_mem_util=0.55`, `optimizer_offload=true`); relax if runs fine by step 5.

### R2 — vLLM wake_up OOM at validation
On 1.5B, ALFWorld v39b crashed at step 99 during Val@100 phase (vLLM kv_cache wake_up OOM). Same risk here. If it happens, **Val@50 is already logged and paper-usable**. Don't retry from scratch — just lower `gpu_memory_utilization` by 0.05 on the next ALFWorld attempt, or set `test_freq` to only validate at step 50.

### R3 — 3B discriminator may saturate differently than 1.5B
On 1.5B, `disc_acc` reaches 1.0 by step 26–30. On 3B it may saturate faster (larger policy ⇒ teacher discrimination is easier) → μ retires earlier → less BC imprinting early on. If you see `disc_acc > 0.95` by step 10 and `μ` already at floor by step 15, note it but **do not** intervene — the data-driven fade-out is v39b's intended behavior.

### R4 — `chord/mu_mode ≠ 3.0`
If you see `mu_mode = 0` or `1` or missing, the config is reverting to hand-tuned schedule. Verify:
```bash
grep -E "chord_mu_adaptive|chord_mu_adaptive_mode" config/duet_paper_experiments_configs/*/webshop_qwen3b_duet_v39b.yaml config/duet_paper_experiments_configs/*/alfworld_qwen3b_duet_v39b.yaml
```
Should print `chord_mu_adaptive: true` and `chord_mu_adaptive_mode: "disc_acc"` for both.

### R5 — Rank desync (informational, not a blocker)
Only rank0 computes `disc_acc`; other ranks see `disc_acc=0.0` during warmup, with `0.5` substituted as fallback via a `disc_trained_steps` guard. This is **known and handled**. If you see rank-0-only warning lines, ignore.

---

## 8. What to Report Back

When both experiments finish, post (via git push or message):

### 8.1 Results table
```
experiment                        | reward@50 | success@50 | reward@100 | success@100 | crashed_at
webshop_qwen3b_duet_v39b          | ?         | ?          | ?          | ?           | none|step_N
alfworld_qwen3b_duet_v39b         | ?         | ?          | ?          | ?           | none|step_N
```
Extract from: `experiments/{webshop,alfworld}/{exp_name}/validation_log/50.jsonl` and `100.jsonl`
(each line is a task with a `reward` field; mean of rewards = `reward@N`; count of `reward>=1.0` / total = `success@N`).

### 8.2 μ trajectory verification
Extract `chord/mu` at steps {1, 10, 25, 50, 100} from wandb for both runs. Paste as a small table.

### 8.3 Anomaly log
- grad_norm spikes (record step & magnitude)
- kl_loss > 1.0 episodes
- disc_acc shape differences from the 1.5B curve above
- any OOM / crash and how it was recovered

### 8.4 wandb run IDs
Paste both run URLs (e.g., `https://wandb.ai/<entity>/agentevolver/runs/<run_id>`).

### 8.5 Do NOT delete (need for paper case studies)
```
logs/webshop_qwen3b_duet_v39b.log
logs/alfworld_qwen3b_duet_v39b.log
experiments/webshop/webshop_qwen3b_duet_v39b/validation_log/*.jsonl
experiments/webshop/webshop_qwen3b_duet_v39b/rollout_log/*.jsonl
experiments/alfworld/alfworld_qwen3b_duet_v39b/validation_log/*.jsonl
experiments/alfworld/alfworld_qwen3b_duet_v39b/rollout_log/*.jsonl
checkpoints/agentevolver/webshop_qwen3b_duet_v39b/Trajectory/      # small, keep
checkpoints/agentevolver/alfworld_qwen3b_duet_v39b/Trajectory/     # small, keep
```
The `global_step_100/` checkpoint folders (~36GB each) can be deleted after results extracted.

---

## 9. Reference Numbers — Targets & Context

### Existing 3B DUET (v1, no adaptive BC) — your floor
| Env | Val@50 | Val@100 | Source |
|---|---:|---:|---|
| WebShop | 0.599 | **0.763** | `webshop_3b_duet_0409_ema` (wandb `v1df0dep`) |
| ALFWorld | 0.480 | **0.695** | `alfworld_3b_duet_0329` (wandb `9ryexv2i`) |

### Existing 3B other baselines (for context — **don't re-run**; primary server owns these)
| Env | OnPolicy | LUFFY | CHORD | SFT+RL |
|---|---:|---:|---:|---:|
| WebShop | 0.402 | 0.753 | 0.728 | ⏳ primary running |
| ALFWorld | 0.585 | 0.615 | 0.545¹ | ⏳ primary running |

¹ ALFWorld 3B CHORD = 0.545 is suspiciously low (below OnPolicy); primary may rerun with updated config. Don't worry about it from your side.

### 1.5B v39b (for trajectory-shape reference)
- WebShop 1.5B: reward@100 = 0.637, success@100 = 19.0%
- ALFWorld 1.5B: Val@50 = 40.0%, Val@100 crashed

---

## 10. File/Path Reference (commits are `git log` off `main`)

| File | Purpose |
|---|---|
| `run_duet_3b_v39b.sh` | Orchestrator (WebShop → ALFWorld) |
| `config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v39b.yaml` | Exp 1 config |
| `config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_duet_v39b.yaml` | Exp 2 config |
| `env_config.sh` | Central path config (edit for your server) |
| `start_env_webshop.sh` / `start_env_alfworld.sh` | Env lifecycle |
| `launcher.py` | Training entry point |
| `agentevolver/module/exp_manager/het_actor.py:1757-1976` | Adaptive μ dispatch (read-only; for debugging) |
| `analysis_reports/PHASE1_SYNTHESIS.md` | Full v39b theory & empirical justification |
| `HANDOFF_3B_SERVER.md` | Previous handoff (1.5B → this H100); overlapping context |

---

## 11. Coordination with Primary Server

- Primary server (this one) is running **SFT → SFT+RL → (optional CHORD rerun)** in sequence on 4×H100.
- Estimated completion: ~42h total for baselines (so ~2 days from now).
- We'll aggregate the 3B main table once both tracks finish. Final table is:
  ```
  | Env       | OnPolicy | LUFFY | CHORD | SFT+RL | DUET-v39b (you) |
  | WebShop   |   ✓      |  ✓    |  ✓    |   ⏳   |   ⏳ (you)      |
  | ALFWorld  |   ✓      |  ✓    |  ⏳¹   |   ⏳   |   ⏳ (you)      |
  ```
- Ping back on finish so we close out the baseline matrix.

---

## 12. If Something Goes Badly

| Symptom | First debug step |
|---|---|
| OOM at rollout start | Lower `rollout.gpu_memory_utilization` 0.65 → 0.55 → 0.45 |
| OOM at training step | Enable `optimizer_offload: true` in both configs |
| OOM at val@100 only | Accept val@50 numbers; don't rerun |
| `chord/mu` stuck at 0.3 | Check §R4 — adaptive rule not engaged |
| `dr3/disc_acc` stuck at 0.5 | Check `dr3/buf_size` reaches `apply_min_buf_size=512` by ~step 15; if not, discriminator broken |
| WebShop env crashes mid-run | Restart env: `bash start_env_webshop.sh stop; bash start_env_webshop.sh`. Resume training is not automatic — would need to restart from last checkpoint (save_freq=100, so if crash before step 100 you restart from scratch). |
| `nvidia-smi` shows zombie processes | `ray stop --force` then kill any `python launcher.py` PIDs |

---

**TL;DR**: Clone, patch paths (§4), `wandb login`, `nohup bash run_duet_3b_v39b.sh`. Watch `chord/mu_mode=3.0` and the μ trajectory. Two runs, ~20h total. Report back the Val@50/Val@100 numbers + μ trajectory. If L20X OOMs, lower `gpu_memory_utilization` first.
