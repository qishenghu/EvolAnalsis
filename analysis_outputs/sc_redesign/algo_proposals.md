# SC Redesign Proposals for WebShop

**Author**: algo-engineer
**Date**: 2026-04-01
**Status**: Proposal

---

## 1. Current SC Implementation: Deep Dive

### What is hashed?

The full normalized observation string. `normalize_observation()` in `state_progress.py:29-64` strips dynamic suffixes:
- ALFWorld: removes `\nAVAILABLE ACTIONS: ...`
- WebShop: removes `\n\nYou can use: ...`, `\nClickable elements: ...`, `\nAvailable actions: ...`

After stripping, the **entire remaining text** is used as a dictionary key (Python string hash). No further normalization is done.

### How is Phi(s) computed?

`ExpertProgressMap.__init__()` at `state_progress.py:146-161`:
```python
for j, obs in enumerate(obs_list):
    progress = j / max(T - 1, 1)   # j/(T-1), so last obs = 1.0
    progress_map[obs] = max(progress_map.get(obs, 0.0), progress)
```

So `Phi(s) = j / (T-1)` where `j` is the 0-indexed position of observation `s` in a teacher trajectory of length `T`. If the same observation appears in multiple teacher trajectories for the same task, the **maximum** progress index is kept.

### How is P(tau) computed?

`compute_trajectory_progress()` at `state_progress.py:187-194`:
```python
P(tau) = (1/T) * sum(Phi(s_t) for s_t in observations)
```
Simple mean of all per-step potentials.

### How are step-level deltas computed?

`compute_step_deltas()` at `state_progress.py:196-207`:
```
potentials = [Phi(s_0), Phi(s_1), ...]
deltas = [Phi(s_1) - Phi(s_0), Phi(s_2) - Phi(s_1), ...]
```

### Where is SC injected?

`ae_ray_trainer.py:3254-3580`:
1. **Trajectory-level** (lines 3329-3380): For each sample, compute `P(tau)`, distribute `beta * P(tau)` uniformly across valid response tokens in `token_level_rewards`.
2. **Step-level** (lines 3491-3580): For each step, compute `eta * (Phi(s_{t+1}) - Phi(s_t))`, distribute across tokens belonging to that step (using `step_ids`).
3. Both exclude teacher samples (`exclude_teacher: true`).
4. Both happen BEFORE `compute_advantage()`, so SC bonuses participate in GRPO normalization.

### Current ExpertProgressMap stats (WebShop)
- Built from 26,178 teacher trajectories across 5,691 tasks
- 191,951 total expert observations, 43,836 unique state keys
- Teacher self-coverage: 100% (trivially, since keys come from teacher data)
- **On-policy coverage: ~0%** (different search queries produce different observation text)

---

## 2. WebShop Observation Structure Analysis

### Observation format
WebShop observations are `[SEP]`-delimited strings with a consistent structure:

**Page types** (4 distinct types):

| Page Type | Structure | Example prefix |
|-----------|-----------|---------------|
| `search_home` | `WebShop [SEP] Instruction: [SEP] ... [SEP] Search` | Shown on homepage/after "back to search" |
| `search_results` | `Instruction: [SEP] ... [SEP] Back to Search [SEP] Page N (Total results: M) [SEP] Next > [SEP] ASIN1 [SEP] Title1 [SEP] Price1 ...` | Search results listing |
| `product_detail` | `Instruction: [SEP] ... [SEP] Back to Search [SEP] < Prev [SEP] option_type [SEP] val1 [SEP] val2 ... [SEP] ProductTitle [SEP] Price [SEP] Rating [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now` | Product detail page |
| `purchase_complete` | `Thank you for shopping with us! [SEP] ... [SEP] Purchased [SEP] asin [SEP] ... [SEP] Reward [SEP] score` | Post-purchase |

### What is INVARIANT vs VARIANT?

| Component | Invariant? | Notes |
|-----------|-----------|-------|
| Instruction text | YES | Same across all sessions for same task_id |
| Page type | YES | Deterministic based on page visited |
| Search results (product listings) | **NO** | Different queries produce different product listings |
| Product detail (for SAME product) | YES | Same ASIN always shows same options/price |
| Product detail (for DIFFERENT product) | **NO** | Different products have different options |
| Option types on product detail | PARTIALLY | Same product always has same option types |
| Purchase confirmation (same product) | YES | Deterministic per ASIN + options |

### Key finding: Product detail pages don't change after option selection
WebShop returns the **same observation** before and after clicking an option (verified on tasks 0, 1, 2). The available actions also remain the same. This means we **cannot** distinguish "viewing product" from "selected option" using observations alone.

### Why hash matching fails on WebShop

Teacher (Qwen-72B) and on-policy (3B) models use **different search queries** for the same task. Even though the same BM25 engine processes both queries, the different keywords produce different product listings. Since the full search results text differs, the hash won't match.

Specifically:
- All teacher trajectories for the **same task** reach the **same product** (verified: 200/200 tasks have consistent ASIN)
- Teacher trajectories for the same task share 4/6 observations on average (product detail + purchase page + search_home + some search results)
- Teacher-to-teacher hash coverage: ~67% within same task
- On-policy-to-teacher hash coverage: **~0%** because search results pages differ

---

## 3. Proposed SC Redesigns

### Approach A: Procedural Stage Progress

**Concept**: Map each observation to a discrete shopping stage based on page type. Progress = stage_value, with fixed values calibrated to the shopping pipeline.

**Stage mapping**:
```python
WEBSHOP_STAGE_PROGRESS = {
    'search_home':       0.0,    # Landed on homepage, haven't searched yet
    'search_results':    0.2,    # Viewing search results
    'product_detail':    0.5,    # Found and viewing a specific product
    'purchase_complete': 1.0,    # Bought successfully
    'invalid_action':    0.0,    # Error state
}
```

**Page type classifier** (regex-based, O(1) per observation):
```python
def classify_webshop_page(obs: str) -> str:
    parts = obs.split(' [SEP] ')
    if parts[0] == 'WebShop':
        return 'search_home'
    if parts[0].startswith('Thank you for shopping'):
        return 'purchase_complete'
    if 'Page ' in obs and 'Back to Search' in obs:
        return 'search_results'
    if '< Prev' in obs and ('Buy Now' in obs or 'Description' in obs):
        return 'product_detail'
    return 'search_home'  # fallback
```

**Implementation sketch**:
- Add `classify_webshop_page()` and `WEBSHOP_STAGE_PROGRESS` to `state_progress.py`
- Modify `ExpertProgressMap.get_potential()`:
  ```python
  def get_potential(self, task_id, observation):
      if self.match_mode == "stage":
          page_type = classify_webshop_page(observation)
          return WEBSHOP_STAGE_PROGRESS.get(page_type, 0.0)
      # existing hash logic
      pmap = self.progress_maps.get(task_id)
      ...
  ```
- Config: `state_channel.match_mode: stage`

**Complexity**: ~30 lines added to `state_progress.py`, ~0 lines changed in `ae_ray_trainer.py`

**Expected coverage**: **100%** on WebShop. Every observation maps to a stage.

**Teacher P(tau) distribution** (measured on 1000 teacher trajectories):
- mean=0.466, std=0.048, range=[0.375, 0.583]
- All >0 (100% nonzero ratio)

**Step-level delta analysis**:
- Positive deltas: search_home→search_results (+0.2), search_results→product_detail (+0.3), product_detail→purchase (+0.5)
- Zero deltas: same-stage transitions (e.g., browsing multiple search result pages)
- Negative deltas: product_detail→search_home (-0.5) when going back to search

**Risk**:
- Coarse granularity: only 4 distinct stage values. Two trajectories that both reach product_detail get the same progress, regardless of whether one found the right product.
- Does not capture "quality" of progress — being on ANY product page counts the same as being on the RIGHT product page.
- P(tau) variance is low (std=0.048), which means SC bonus is nearly uniform across all trajectories that reach similar stages. This limits GRPO's ability to differentiate.

**Compatibility**: Fully backward-compatible. ALFWorld continues using `match_mode: hash`. Only WebShop switches to `match_mode: stage`. No changes to `ae_ray_trainer.py`.

---

### Approach B: Semantic-Key Extraction

**Concept**: Parse the `[SEP]`-delimited observation to extract structural features. Hash on `(page_type, option_types)` instead of the full text. This is more fine-grained than Approach A because different products have different option configurations.

**Semantic key function**:
```python
def extract_webshop_semantic_key(obs: str) -> str:
    parts = obs.split(' [SEP] ')
    page_type = classify_webshop_page(obs)

    if page_type == 'search_results':
        # Extract page number
        for p in parts:
            m = re.match(r'Page (\d+)', p)
            if m:
                return f'search_results_p{m.group(1)}'
        return 'search_results'

    if page_type == 'product_detail':
        # Extract option types (size, color, style, ...)
        option_kw = {'size','color','style','pattern','material','flavor','scent','count','design','type'}
        options = sorted(p.strip().lower() for p in parts if p.strip().lower() in option_kw)
        return f'product_detail|opts={"_".join(options)}'

    return page_type
```

**Key count**: Only 12 unique semantic keys across 1000 teacher trajectories (vs 1654 hash keys):
```
product_detail|opts=
product_detail|opts=color
product_detail|opts=color_size
product_detail|opts=color_style
product_detail|opts=size
product_detail|opts=style
purchase_complete
search_home
search_results_p1
search_results_p2
search_results_p3
search_results_p4
```

**Progress computation**: Build the progress map using semantic keys instead of full text hashes. Each key maps to the max teacher progress value where that key type appears.

**Implementation**:
- Add `extract_webshop_semantic_key()` to `state_progress.py`
- New `match_mode: "semantic_key"` in ExpertProgressMap
- ~50 lines added to `state_progress.py`

**Expected coverage**: **100%** on WebShop for same option-type products.

**Risk**:
- Slightly more complex than A but still simple
- Keys are task-independent: `product_detail|opts=size` maps to the same progress for ALL tasks. Loses task-specific teacher calibration.
- Only marginally more informative than Approach A — the option types don't significantly change the progress signal.
- Edge case: if on-policy finds a product with different option types than teacher, the semantic key won't match the teacher's progress. But since we use max across all teachers, this is unlikely to be 0.

**Compatibility**: Same as Approach A. ALFWorld uses `hash`, WebShop uses `semantic_key`.

---

### Approach C: Reward-as-Progress

**Concept**: WebShop gives a continuous reward in [0, 1] that measures how well the purchased item matches the instruction. Use this reward directly as the progress signal, bypassing observation matching entirely.

**Implementation**:
```python
# In ae_ray_trainer.py SC section:
if match_mode == "reward_as_progress":
    # Use final reward as P(tau) — no hash matching needed
    _sc_P = float(batch.batch["token_level_rewards"][_sc_idx].sum().item())
    _sc_P = max(0.0, min(1.0, _sc_P))  # clip to [0,1]
```

**Expected coverage**: **100%** (no matching needed).

**Risk**:
- **Circular signal**: SC bonus is `beta * R(tau)`, meaning we add a scaled copy of the reward to itself. This doesn't break new ground — it's equivalent to scaling the reward by `(1 + beta)`.
- **No intermediate progress**: Reward is 0 for all steps until the final purchase. Failed trajectories (didn't buy) get P(tau) = 0, no matter how close they got.
- **Theoretical issue**: SC's value comes from providing a DIFFERENT signal than the raw reward. Using reward-as-progress collapses the two signals.

**Verdict**: **NOT RECOMMENDED**. The SC bonus degenerates to reward scaling. Does not solve the core problem (providing progress signal to incomplete trajectories).

**Compatibility**: Trivially compatible with all environments.

---

### Approach D: Embedding Similarity

**Concept**: Use a lightweight text encoder (e.g., `all-MiniLM-L6-v2`, 80MB) to compute embeddings of observations. Match on-policy observations to the nearest teacher observation by cosine similarity. Use the matched teacher observation's progress value, weighted by similarity.

**Implementation sketch**:
```python
class EmbeddingProgressMap:
    def __init__(self, teacher_data, env_type, model_name='all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(model_name)

        # Pre-compute teacher embeddings per task
        self.task_embeddings = {}  # task_id -> (embeddings_matrix, progress_values)
        for task_id, trajs in teacher_data.items():
            obs_list, progress_list = [], []
            for traj in trajs:
                obs = extract_observations_from_steps(traj.steps, env_type)
                for j, o in enumerate(obs):
                    obs_list.append(o)
                    progress_list.append(j / max(len(obs)-1, 1))
            embeddings = self.encoder.encode(obs_list, normalize_embeddings=True)
            self.task_embeddings[task_id] = (embeddings, progress_list)

    def get_potential(self, task_id, observation, threshold=0.8):
        if task_id not in self.task_embeddings:
            return 0.0
        emb = self.encoder.encode([observation], normalize_embeddings=True)
        teacher_embs, teacher_progress = self.task_embeddings[task_id]
        sims = emb @ teacher_embs.T  # (1, N)
        max_sim = sims.max()
        if max_sim < threshold:
            return 0.0
        best_idx = sims.argmax()
        return teacher_progress[best_idx] * float(max_sim)
```

**Complexity**:
- ~100 lines in `state_progress.py` for the new class
- Dependency on `sentence-transformers` package
- Pre-computation of 43,836 embeddings (one-time, ~30 seconds)
- Per-batch inference: ~0.5s for encoding 64 observations (may slow training)

**Expected coverage**: **50-80%** with threshold=0.7, **80-95%** with threshold=0.5. Soft matching can match search results pages with overlapping products even when exact product lists differ.

**Risk**:
- **Latency**: Embedding computation at training time adds ~0.5-1s per batch
- **Memory**: Storing embeddings for 43K teacher observations (~100MB)
- **False matches**: Two search results pages with low similarity might still match above threshold, giving misleading progress values
- **Dependency**: Requires `sentence-transformers` and a model download
- **Overkill**: The WebShop observation structure is so regular that a simple parser achieves similar coverage with zero cost

**Compatibility**: Works with any environment (env-agnostic).

---

### Approach E: Action-Pattern Matching

**Concept**: Instead of matching observations, match on the sequence of ACTION TYPES taken. Define action types: `search`, `click_product` (click ASIN), `click_option` (click size/color/etc.), `click_buy`, `click_nav` (back to search, next page, etc.).

**Implementation sketch**:
```python
def classify_action(action_text: str, available_actions: dict) -> str:
    if action_text.startswith('search['):
        return 'search'
    if action_text == 'click[buy now]':
        return 'click_buy'
    if action_text in ('click[back to search]', 'click[< prev]', 'click[next >]'):
        return 'click_nav'
    # ASIN pattern
    if re.match(r'click\[b[0-9a-z]{8,}\]', action_text, re.I):
        return 'click_product'
    return 'click_option'
```

**Progress from action pattern**:
- `search` → 0.1
- `click_product` → 0.4 (found a product to investigate)
- `click_option` → 0.6 (selecting options → close to buying)
- `click_buy` → 1.0

**Expected coverage**: ~100% (every action maps to a type).

**Risk**:
- Requires parsing ASSISTANT messages in addition to user observations
- Current SC code only processes observations (user messages), not actions
- More fragile: action parsing depends on format (`react_tags` vs `react`)
- Progress is based on what the agent DID, not what it SAW. This is philosophically different from the original SC design.
- Doesn't capture "quality" of action (clicking a bad product vs. a good one)

**Complexity**: ~60 lines in `state_progress.py`, plus changes to extraction functions to also parse assistant messages.

**Compatibility**: WebShop-specific. ALFWorld would need its own action classifier.

---

## 4. Recommendation: Approach A (Procedural Stage Progress)

### Why Approach A wins

| Criterion | A: Stage | B: Semantic-Key | C: Reward | D: Embedding | E: Action |
|-----------|----------|----------------|-----------|-------------|-----------|
| Coverage on WebShop | 100% | 100% | 100% | 50-95% | 100% |
| Implementation effort | ~30 LOC | ~50 LOC | ~10 LOC | ~100 LOC + dep | ~60 LOC |
| Runtime cost | Zero | Zero | Zero | ~0.5s/batch | Zero |
| New dependencies | None | None | None | sentence-transformers | None |
| Theoretically sound | Yes | Yes | **No** (circular) | Yes | Partial |
| Works on ALFWorld | N/A (keep hash) | N/A | N/A | Yes | N/A |
| Captures intermediate progress | **Yes** | Yes | **No** | Yes | Yes |
| Differentiates trajectories | Moderate | Moderate | Poor | Good | Moderate |
| Step-level deltas meaningful | **Yes** | Yes | No | Yes | Yes |

**Approach A is the clear winner because**:
1. **Simplest implementation** — ~30 LOC, zero changes to the trainer, zero new dependencies
2. **100% coverage** — every observation maps to a stage
3. **Correct incentive gradient** — searching < finding product < buying
4. **Zero runtime cost** — string matching is negligible vs. hash lookup
5. **Fully backward-compatible** — ALFWorld continues using hash mode unchanged
6. **Meaningful step deltas** — advancing stages gives positive delta, staying gives zero
7. **Minimal risk** — no external models, no threshold tuning, deterministic

### Known limitation: Low P(tau) variance

With only 4 stage values, P(tau) has low variance across trajectories (std=0.048). This means the SC bonus is relatively uniform. However:
- The PRIMARY goal is to provide ANY signal where there was previously NONE (0% coverage → 100%)
- Even a uniform bonus helps break reward sparsity for trajectories that don't complete purchase
- The step-level deltas (eta term) add finer-grained token-level signal

### Enhancement for future work (Approach A+)

If Approach A proves too coarse, the natural upgrade is to **combine stage progress with reward information**:

```python
def get_potential(self, task_id, observation, current_reward=None):
    page_type = classify_webshop_page(observation)
    base_progress = WEBSHOP_STAGE_PROGRESS[page_type]

    # If on product_detail and we know the partial reward:
    if page_type == 'product_detail' and current_reward is not None:
        # Interpolate between 0.5 and 1.0 based on how good the product match is
        base_progress = 0.5 + 0.5 * current_reward

    return base_progress
```

This is NOT Approach C (which uses reward directly), because:
- It only applies reward refinement WITHIN the product_detail stage
- Trajectories that never reach product_detail still get progress 0.0-0.2
- Trajectories on product_detail get differentiated by product quality

### Implementation plan

**Files to modify**: `agentevolver/module/exp_manager/state_progress.py` only

**Changes**:
1. Add `WEBSHOP_STAGE_PROGRESS` constant
2. Add `classify_webshop_page()` function
3. Modify `ExpertProgressMap.get_potential()` to branch on `match_mode`
4. Modify `ExpertProgressMap.compute_trajectory_progress()` to work with stage mode
5. Modify `ExpertProgressMap.compute_step_deltas()` to work with stage mode
6. Modify `ExpertProgressMap.get_coverage_stats()` to work with stage mode

**Config change** (in experiment YAML): `state_channel.match_mode: stage`

**No changes needed to**:
- `ae_ray_trainer.py` (the SC injection code calls `get_potential()` and `compute_trajectory_progress()`, which are the methods we modify)
- `het_actor.py`, `het_core_algos.py` (SC is injected before these run)
- Any other file

**Estimated effort**: 1-2 hours for implementation + testing.
