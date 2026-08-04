# Forensics: does `data.seed` change WHICH tasks are trained / evaluated on?

Scope: WebShop Qwen2.5-1.5B DUET, seed 2026 (`webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06`)
vs seed 2025 (`webshop_qwen1.5b_duet_a100_seed2025`).
Everything below is read-only analysis of committed code, saved logs, saved trajectories, and the
teacher pickle. All paths are absolute-from-repo-root `/data/home/qisheng/EvolAnalsis`.

## TL;DR

| Question | Answer |
|---|---|
| (a) Does `data.seed` permute/select the TRAIN subset? | **Yes.** It selects which 800 of 6710 WebShop train tasks are used, and their visiting order. |
| (b) Does `data.seed` touch the VAL set? | **No.** Val is loaded with `shuffle=False` and `seed=None`; the same fixed 200 tasks in the same order. |
| Actual train-set overlap between the two runs | **89 / 800 tasks (11.1%); 89% of each run's training data is disjoint from the other's.** |
| Is that overlap anomalous? | No — hypergeometric expectation is 95.4, sd ≈ 8.6. 89 is a normal random draw. |
| Are the two train subsets systematically different in difficulty or teacher support? | **No measurable difference** beyond sampling noise (see §3). |
| Are the two runs evaluated on the same 200 val tasks? | **Yes** (see §4). The 7 "missing" rows are degenerate generations, not different/absent tasks. |

Verdict for this angle: **task sampling is real and large in the TRAIN set, but it is not a
confound that can explain the val gap** — the val set is provably identical, and the two train
subsets are statistically indistinguishable on every difficulty/teacher-support proxy measured.

---

## 1. Code path: `data.seed` → task selection

`data.seed` is read in **exactly two places** in the whole repo
(`grep -rn "data\.seed\|data_config.get(\"seed\"" external/ agentevolver/` → 1 hit;
`grep -rn "seed" agentevolver/module/trainer/ae_ray_trainer.py` → the two below). It is *not* used
for model init, dropout, vLLM sampling, or anything else.

### 1a. TRAIN subset selection — `agentevolver/module/trainer/ae_ray_trainer.py:886-904`

```python
if self.config.data.train_files is not None:   # <- null in both configs, so ELSE branch runs
    ...
else:
    max_train_tasks = self.config.data.get("max_train_tasks", None)   # 800
    shuffle         = self.config.data.get("shuffle", True)           # true
    seed            = self.config.data.get("seed", 2026)              # 2026 vs 2025
    self.train_task_manager.load_tasks_from_environment(
        env_client, env_type=env_type, split="train",
        max_tasks=max_train_tasks, shuffle=shuffle, seed=seed,
    )
```

`agentevolver/module/task_manager/task_manager.py:140-169`:

```python
response = env.get_env_profile(env_type, split, params)   # ordered list of 6710 id strings
if seed is not None:
    random.seed(seed)          # global RNG
if shuffle:
    random.shuffle(response)
if max_tasks is not None and max_tasks > 0:
    response = response[:max_tasks]        # top-800 of a seed-dependent permutation
```

So the seed **is** the subset selector. `env_service/environments/webshop/webshop_env.py:557-585`
(`get_query_list`) returns the ids from `env_service/environments/webshop/webshop_train.json`
in file order — a deterministic 6710-element list, so the permutation is fully determined by
the seed.

### 1b. TRAIN visiting order — `agentevolver/module/trainer/ae_ray_trainer.py:356-361`

```python
if data_config.shuffle:
    train_dataloader_generator = torch.Generator()
    train_dataloader_generator.manual_seed(data_config.get("seed", 1))
    sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
```

Second effect: the order in which the selected 800 are visited also changes with the seed.

### 1c. VAL — `agentevolver/module/trainer/ae_ray_trainer.py:906-948`

`val_files` is null, `env_type != "alfworld"`, `val_type: val`, so the loop over `["val", "dev"]`
runs and breaks on the first success:

```python
loaded = self.val_task_manager.load_tasks_from_environment(
    env_client, env_type=env_type, split=split,
    shuffle=False,          # <- no shuffle
)                           # <- seed argument NOT passed => seed=None => random.seed() NOT called
```

`get_query_list("val")` returns `[str(i) for i in eval_idxs]` — the 200 ids of
`env_service/environments/webshop/webshop_test.json`, in file order.
**No seed, no shuffle, no max_tasks. The val set is bit-identical across seeds by construction.**

Note also that the `UnifiedMixtureStrategy` uses a hard-coded `seed=42` in both runs (visible in
both logs), so it introduces no divergence either.

### 1d. Runtime confirmation from the run logs

`logs/webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06.log:367,373-377`
```
Limiting tasks from 6710 to 800 (max_tasks=800)
📌 DATASET SIZES (env=webshop)
  - train: 800 tasks (unique task_id=800)  source=env(split=train, max_train_tasks=800, shuffle=True, seed=2026)
  -   val: 200 tasks (unique task_id=200)  source=env(split=val)
  - train∩val overlap (by task_id): 0
```
`logs/webshop_qwen1.5b_duet_a100_seed2025.log:372,378-382` — identical except `seed=2025`.

Config diff confirmed (3 lines): `experiment_name`, `data.seed: 2026 -> 2025`, `workspace_id`, between
`config/duet_paper_experiments_configs/webshop/sweep_1.5b/webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06.yaml`
and `config/duet_paper_experiments_configs/rebuttal_neurips/webshop/webshop_qwen1.5b_duet_a100_seed2025.yaml`.
Relevant data block: `train_files: null`, `val_files: null`, `val_type: val`,
`max_train_tasks: 800`, `shuffle: true`.

---

## 2. Empirical confirmation from saved trajectories

Union of `task_id` over all 100 `checkpoints/agentevolver/<run>/Trajectory/trajectories_step_*.jsonl`:

| Run | steps | unique train task_ids |
|---|---|---|
| seed 2026 | 100 | 800 |
| seed 2025 | 100 | 800 |

- **Overlap = 89 task_ids. Jaccard = 0.0589.** 711/800 (88.9%) of each run's train tasks are unique to it.
- Offline reproduction (`random.seed(s); random.shuffle(ids_from_webshop_train.json); [:800]`)
  reproduces both observed sets **exactly** — symmetric difference 0 for both seeds. So the
  mechanism is fully understood and deterministic.
- Random-draw expectation for the overlap: 800·800/6710 = 95.4, sd ≈ 8.6 → 89 is ~0.7 sd below the
  mean, i.e. a completely ordinary draw. Nothing pathological about the specific seed-2025 subset.
- Each run consumes exactly its 800 tasks once over 100 steps (8 prompts/step × 100 = 800), i.e.
  one epoch, no repeats — so the disjointness is *fully* realised, not partially.
- `train ∩ test = 0` at the id level for both runs (webshop_train.json and webshop_test.json ids are
  disjoint), so there is no leakage difference either.

---

## 3. Do the two train subsets differ in teacher support or intrinsic difficulty?

Teacher cache `data/teacher_trajectories/qwen72b/webshop_qwen72b_filtered.pkl`:
26,178 demos over 5,691 distinct task_ids (4.600 demos/task). 84.81% of the 6710-task train pool
has ≥1 demo.

| Metric (over the run's own 800 train tasks) | seed 2026 | seed 2025 |
|---|---|---|
| fraction with ≥1 teacher demo | 0.8575 (686/800) | 0.8350 (668/800) |
| mean demos available per task | 3.973 | 3.824 |
| median demos per task | 5 | 5 |
| mean teacher demo reward (covered tasks) | 1.0000 | 1.0000 |
| fraction of covered tasks with a reward-1.0 demo | 1.000 | 1.000 |
| mean teacher demo `num_turns` (covered tasks) | 7.363 | 7.322 |
| instruction chars / words / commas | 242.8 / 37.50 / 4.228 | 243.5 / 37.65 / 4.251 |
| instruction contains a price constraint | 0.996 | 0.996 |

The 0.8575 / 0.8350 numbers independently reproduce the previously reported "0.857 vs 0.834 teacher
rollouts per prompt" — that quantity *is* the teacher-coverage fraction of the sampled subset.

**Is 0.8350 unusual?** Monte-Carlo over 3000 random 800-subsets of the same pool:
mean 0.8479, sd 0.0120, 95% interval [0.8237, 0.8700]. seed 2026 sits at the 76th percentile,
seed 2025 at the 14th. Both are ordinary; the 2.25 pp difference is ~1.8 binomial SE.

**Does coverage even matter?** Splitting each run's own on-policy rollouts (steps 61-100) by whether
the task has teacher demos:

| Run | covered mean / strict | uncovered mean / strict |
|---|---|---|
| seed 2026 | 0.5341 / 0.0681 (n=1925) | 0.5645 / 0.0167 (n=360) |
| seed 2025 | 0.5103 / 0.0469 (n=1897) | 0.4801 / 0.0587 (n=392) |

The sign of the covered-vs-uncovered effect **flips between runs**, i.e. it is noise at this sample
size. A 2.25 pp difference in coverage cannot plausibly move val strict success by 32 points.

**Difficulty measured behaviourally.** On-policy rollout reward (teacher rows excluded via
`diag.is_teacher`), aggregated over step windows:

| window | seed 2026 mean / strict(=1.0) / band[0.75,1) | seed 2025 mean / strict / band |
|---|---|---|
| 1-20   | 0.1051 / 0.0087 / 0.0341 | 0.1058 / 0.0130 / 0.0513 |
| 41-60  | 0.4618 / 0.0158 / 0.0825 | 0.4311 / 0.0253 / 0.1013 |
| 81-100 | 0.5670 / 0.0962 / 0.2448 | 0.5460 / 0.0654 / 0.2871 |

At steps 1-20 the policies are still near-identical initialisations, and the two subsets give the
*same* mean reward (0.1051 vs 0.1058) — direct evidence that the two 800-task subsets are equally
hard at the start. Any later divergence is policy-driven, not task-driven.

**Magnitude mismatch.** Even at the end of training the *train* strict-success gap is 9.6% vs 6.5%
(≈1.5×), while the *val* strict gap is 35.5% vs 3.5% (≈10×). Train-set difficulty is not a
sufficient explanation for the val gap.

---

## 4. The VAL set: settling the 7 unparsed rows

Headline numbers reproduced from `experiments/webshop/<run>/validation_log/{50,100}.jsonl`:

| run | step | n | mean score | strict (=1.0) | fraction score<0 |
|---|---|---|---|---|---|
| seed 2026 | 50  | 200 | 0.5219 | 0.0100 | 0.0150 |
| seed 2026 | 100 | 200 | 0.7057 | **0.3550** | 0.0500 |
| seed 2025 | 50  | 200 | 0.4390 | 0.0150 | 0.1600 |
| seed 2025 | 100 | 200 | 0.5212 | **0.0350** | 0.1100 |

Both logs have exactly 200 rows at every step — consistent with the code (200 fixed val tasks).

Instruction recovery via `re.search(r"Instruction:\s*\[SEP\]\s*(.+?)\s*\[SEP\]", row["output"])`:

| run | step | unique instructions | unparsed rows | scores of unparsed rows |
|---|---|---|---|---|
| seed 2026 | 50  | 199 | 1 | all −0.1 |
| seed 2026 | 100 | 200 | 0 | — |
| seed 2025 | 50  | 194 | 6 | all −0.1 |
| seed 2025 | 100 | 193 | 7 | all −0.1 |

Three independent pieces of evidence that the val task set is identical:

1. **Code**: val load path passes `shuffle=False` and no seed (§1c). There is no code path by which
   `data.seed` can reach the val list.
2. **Union test**: the union of instruction sets over all four val logs is **exactly 200** — never
   201. Every parsed instruction seen in seed 2025 also appears in seed 2026's step-100 log, and
   seed 2026's step-100 log alone covers all 200. If the seeds selected different val tasks, the
   union would exceed 200.
3. **Cause of the unparsed rows**: they are not missing tasks and not env faults. Their `output`
   field contains a single degenerate assistant turn — a repetition loop that exhausts the response
   budget before any environment observation is returned, e.g.
   `search[yellow heather men's dress shirts youth small small small small …]` (4986 chars).
   With no environment turn, no `Instruction: [SEP] … [SEP]` block ever appears in the transcript,
   hence "unparsed". Every such row scores exactly −0.1, the invalid-action penalty.
   Both runs have this failure mode; seed 2025 has 7× more of it at step 100 (7 vs 0) and 6× at
   step 50 (6 vs 1). Mean environment turns per val episode at step 100: 4.31 (seed 2026) vs 3.80
   (seed 2025); zero-turn rows: 0 in both.

The `input` field of the val logs contains only the system prompt (no instruction), so
output-side regex is the only recovery route; this was verified by printing a full `input`.

Row ordering is *not* preserved between runs (only 1 of 193 positions matched), so positional
alignment cannot be used — async rollout completion order. The union test above is the sound
substitute.

---

## 5. What this angle rules out and what it leaves open

Ruled out:
- A different val set. Impossible by code and contradicted by the union-of-200 test.
- The 7 unparsed rows being lost/absent/extra tasks. They are degenerate repetition-loop
  generations on tasks that are in the shared 200.
- A systematically harder or teacher-poorer train subset for seed 2025. Every proxy measured
  (coverage, demos/task, demo reward, demo length, instruction complexity, and — most directly —
  early-training on-policy reward at steps 1-20) is equal within noise.
- The specific seed-2025 subset being an outlier draw. It sits at the 14th percentile of the
  coverage distribution; the overlap with seed 2026 is within 1 sd of the hypergeometric mean.

Confirmed and worth stating in a rebuttal:
- `data.seed` genuinely resamples 89% of the training data. **These two runs are not "same data,
  different noise" — they are different 800-task training sets.** Any claim of "identical setup
  except the seed" must be phrased as "identical configuration except the seed, which also
  resamples the training subset", because a reviewer who checks will find this.
- The proper framing is therefore: seed variance in this setup bundles (i) training-subset
  resampling, (ii) task ordering, and (iii) stochastic rollout/optimisation noise. Reporting
  seed variance is still legitimate; claiming the two runs share training data is not.

Open:
- The residual 10× val gap is not attributable to task sampling. Given that (a) train on-policy
  reward differs by only ~0.02 at step 100, (b) both runs' val strict success was ~1% at step 50
  and seed 2026 jumped to 35.5% by step 100 while seed 2025 reached only 3.5%, and (c) seed 2025
  shows 2-3× the rate of degenerate repetition-loop generations at val time, the gap looks like a
  late, sharp phase transition that seed 2026 crossed within the 100-step budget and seed 2025 did
  not. Whether that is genuine seed sensitivity or an environment/decoding fault is the subject of
  the other forensic angles; this angle cannot discriminate between them.
- Not checked here: whether validation decoding parameters (temperature/greedy) were identical at
  runtime — only the configs were diffed, which showed no difference, but no runtime assertion was
  found.

---

## Reproduction commands

```bash
cd /data/home/qisheng/EvolAnalsis
PY=/data/home/qisheng/miniconda3/envs/duet/bin/python

# train subsets from seed
$PY - <<'EOF'
import json, random
ids=[str(int(d['item_id'].split('_')[1]))
     for d in json.load(open('env_service/environments/webshop/webshop_train.json'))]
def sub(seed,n=800):
    r=list(ids); random.seed(seed); random.shuffle(r); return set(r[:n])
a,b=sub(2026),sub(2025)
print(len(a),len(b),len(a&b))          # 800 800 89
EOF

# observed train sets from saved trajectories
$PY - <<'EOF'
import json,glob
def s(run):
    out=set()
    for p in glob.glob(f'checkpoints/agentevolver/{run}/Trajectory/trajectories_step_*.jsonl'):
        out |= {str(json.loads(l)['task_id']) for l in open(p)}
    return out
A=s('webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06')
B=s('webshop_qwen1.5b_duet_a100_seed2025')
print(len(A),len(B),len(A&B))          # 800 800 89
EOF
```
