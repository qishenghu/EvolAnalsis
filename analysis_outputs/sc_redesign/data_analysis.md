# WebShop SC Redesign - Observation Data Analysis
**Date**: 2026-04-01
**Total teacher trajectories**: 26178
**All rewards = 1.0**: Yes (pre-filtered)

## 1. Observation Structure (5 Sample Trajectories)

### Traj 1: task_id=0, msgs=15

| Step | Role | Page Type | Length | First 120 chars |
|------|------|-----------|--------|-----------------|
| 0 | sys | - | 865 | (system prompt) |
| 1 | act | - | 76 | OK. I'll help you find and purchase the item according to yo |
| 2 | obs | **initial** (stage=0) | 323 | WebShop [SEP] Instruction: [SEP] Find me double sided, machine washable decorative pillows with printing technology with |
| 3 | act | - | 263 | search[decorative pillows] |
| 4 | obs | **search_results** (stage=1) | 2752 | Instruction: [SEP] Find me double sided, machine washable decorative pillows with printing technology with size: 28" x 2 |
| 5 | act | - | 309 | click[back to search] |
| 6 | obs | **initial** (stage=0) | 323 | WebShop [SEP] Instruction: [SEP] Find me double sided, machine washable decorative pillows with printing technology with |
| 7 | act | - | 509 | search[Ambesonne Abstract Throw Pillow Cushion Cover, Geomet |
| 8 | obs | **search_results** (stage=1) | 2905 | Instruction: [SEP] Find me double sided, machine washable decorative pillows with printing technology with size: 28" x 2 |
| 9 | act | - | 307 | click[b0743jkhbv] |
| 10 | obs | **product_detail** (stage=2) | 1374 | Instruction: [SEP] Find me double sided, machine washable decorative pillows with printing technology with size: 28" x 2 |
| 11 | act | - | 225 | click[28" x 28"] |
| 12 | obs | **product_detail** (stage=2) | 1374 | Instruction: [SEP] Find me double sided, machine washable decorative pillows with printing technology with size: 28" x 2 |
| 13 | act | - | 284 | click[buy now] |
| 14 | obs | **confirmation** (stage=3) | 624 | Thank you for shopping with us! [SEP] Your code: [SEP] None [SEP] (Paste it in your MTurk interface.) [SEP] Purchased [S |

### Traj 2: task_id=1, msgs=17

| Step | Role | Page Type | Length | First 120 chars |
|------|------|-----------|--------|-----------------|
| 0 | sys | - | 865 | (system prompt) |
| 1 | act | - | 76 | OK. I'll help you find and purchase the item according to yo |
| 2 | obs | **initial** (stage=0) | 338 | WebShop [SEP] Instruction: [SEP] Find me butt lifting, light weight women's shorts with high waist, tummy control with c |
| 3 | act | - | 270 | search[women's shorts] |
| 4 | obs | **search_results** (stage=1) | 2587 | Instruction: [SEP] Find me butt lifting, light weight women's shorts with high waist, tummy control with color: black, a |
| 5 | act | - | 263 | click[back to search] |
| 6 | obs | **initial** (stage=0) | 338 | WebShop [SEP] Instruction: [SEP] Find me butt lifting, light weight women's shorts with high waist, tummy control with c |
| 7 | act | - | 494 | search[LWLW High Waist Yoga Shorts for Women,Tummy Control B |
| 8 | obs | **search_results** (stage=1) | 2413 | Instruction: [SEP] Find me butt lifting, light weight women's shorts with high waist, tummy control with color: black, a |
| 9 | act | - | 369 | click[b09qcp4579] |
| 10 | obs | **product_detail** (stage=2) | 1274 | Instruction: [SEP] Find me butt lifting, light weight women's shorts with high waist, tummy control with color: black, a |
| 11 | act | - | 239 | click[black] |
| 12 | obs | **product_detail** (stage=2) | 1274 | Instruction: [SEP] Find me butt lifting, light weight women's shorts with high waist, tummy control with color: black, a |
| 13 | act | - | 260 | click[3x-large] |
| 14 | obs | **product_detail** (stage=2) | 1274 | Instruction: [SEP] Find me butt lifting, light weight women's shorts with high waist, tummy control with color: black, a |
| 15 | act | - | 262 | click[buy now] |
| 16 | obs | **confirmation** (stage=3) | 639 | Thank you for shopping with us! [SEP] Your code: [SEP] None [SEP] (Paste it in your MTurk interface.) [SEP] Purchased [S |

### Traj 3: task_id=2, msgs=17

| Step | Role | Page Type | Length | First 120 chars |
|------|------|-----------|--------|-----------------|
| 0 | sys | - | 865 | (system prompt) |
| 1 | act | - | 76 | OK. I'll help you find and purchase the item according to yo |
| 2 | obs | **initial** (stage=0) | 348 | WebShop [SEP] Instruction: [SEP] Find me ready hang wall art with solid wood for living room with color: turtles and afr |
| 3 | act | - | 280 | search[wall art] |
| 4 | obs | **search_results** (stage=1) | 2771 | Instruction: [SEP] Find me ready hang wall art with solid wood for living room with color: turtles and african american  |
| 5 | act | - | 341 | click[back to search] |
| 6 | obs | **initial** (stage=0) | 348 | WebShop [SEP] Instruction: [SEP] Find me ready hang wall art with solid wood for living room with color: turtles and afr |
| 7 | act | - | 659 | search[African American Wall Art Get Naked Fashion Black Gir |
| 8 | obs | **search_results** (stage=1) | 2979 | Instruction: [SEP] Find me ready hang wall art with solid wood for living room with color: turtles and african american  |
| 9 | act | - | 439 | click[b099wh1rtm] |
| 10 | obs | **product_detail** (stage=2) | 1643 | Instruction: [SEP] Find me ready hang wall art with solid wood for living room with color: turtles and african american  |
| 11 | act | - | 314 | click[turtles and african american couple] |
| 12 | obs | **product_detail** (stage=2) | 1643 | Instruction: [SEP] Find me ready hang wall art with solid wood for living room with color: turtles and african american  |
| 13 | act | - | 292 | click[8x10 inch] |
| 14 | obs | **product_detail** (stage=2) | 1643 | Instruction: [SEP] Find me ready hang wall art with solid wood for living room with color: turtles and african american  |
| 15 | act | - | 274 | click[buy now] |
| 16 | obs | **confirmation** (stage=3) | 670 | Thank you for shopping with us! [SEP] Your code: [SEP] None [SEP] (Paste it in your MTurk interface.) [SEP] Purchased [S |

### Traj 4: task_id=3, msgs=15

| Step | Role | Page Type | Length | First 120 chars |
|------|------|-----------|--------|-----------------|
| 0 | sys | - | 865 | (system prompt) |
| 1 | act | - | 76 | OK. I'll help you find and purchase the item according to yo |
| 2 | obs | **initial** (stage=0) | 277 | WebShop [SEP] Instruction: [SEP] Find me dual band streaming media players with quad core, and price lower than 350.00 d |
| 3 | act | - | 321 | search[streaming media players] |
| 4 | obs | **search_results** (stage=1) | 2215 | Instruction: [SEP] Find me dual band streaming media players with quad core, and price lower than 350.00 dollars [SEP] B |
| 5 | act | - | 264 | click[next >] |
| 6 | obs | **search_results** (stage=1) | 2634 | Instruction: [SEP] Find me dual band streaming media players with quad core, and price lower than 350.00 dollars [SEP] B |
| 7 | act | - | 288 | click[back to search] |
| 8 | obs | **initial** (stage=0) | 277 | WebShop [SEP] Instruction: [SEP] Find me dual band streaming media players with quad core, and price lower than 350.00 d |
| 9 | act | - | 404 | search[Superbox S3 Pro Dual Band Wi-Fi 2.4Ghz 5Ghz Supports  |
| 10 | obs | **search_results** (stage=1) | 2554 | Instruction: [SEP] Find me dual band streaming media players with quad core, and price lower than 350.00 dollars [SEP] B |
| 11 | act | - | 331 | click[b09lskqf8c] |
| 12 | obs | **product_detail** (stage=2) | 625 | Instruction: [SEP] Find me dual band streaming media players with quad core, and price lower than 350.00 dollars [SEP] B |
| 13 | act | - | 283 | click[buy now] |
| 14 | obs | **confirmation** (stage=3) | 603 | Thank you for shopping with us! [SEP] Your code: [SEP] None [SEP] (Paste it in your MTurk interface.) [SEP] Purchased [S |

### Traj 5: task_id=4, msgs=19

| Step | Role | Page Type | Length | First 120 chars |
|------|------|-----------|--------|-----------------|
| 0 | sys | - | 865 | (system prompt) |
| 1 | act | - | 76 | OK. I'll help you find and purchase the item according to yo |
| 2 | obs | **initial** (stage=0) | 383 | WebShop [SEP] Instruction: [SEP] Find me hand wash women's sweaters with long sleeve, stretch fabric, polyester spandex  |
| 3 | act | - | 277 | search[women's sweaters] |
| 4 | obs | **search_results** (stage=1) | 2414 | Instruction: [SEP] Find me hand wash women's sweaters with long sleeve, stretch fabric, polyester spandex for teen girls |
| 5 | act | - | 357 | click[next >] |
| 6 | obs | **search_results** (stage=1) | 2655 | Instruction: [SEP] Find me hand wash women's sweaters with long sleeve, stretch fabric, polyester spandex for teen girls |
| 7 | act | - | 389 | click[back to search] |
| 8 | obs | **initial** (stage=0) | 383 | WebShop [SEP] Instruction: [SEP] Find me hand wash women's sweaters with long sleeve, stretch fabric, polyester spandex  |
| 9 | act | - | 545 | search[YALFJV Women Long Sleeve Crew Neck Side Button T Shir |
| 10 | obs | **search_results** (stage=1) | 2446 | Instruction: [SEP] Find me hand wash women's sweaters with long sleeve, stretch fabric, polyester spandex for teen girls |
| 11 | act | - | 508 | click[b09m63b87v] |
| 12 | obs | **product_detail** (stage=2) | 3352 | Instruction: [SEP] Find me hand wash women's sweaters with long sleeve, stretch fabric, polyester spandex for teen girls |
| 13 | act | - | 240 | click[xnj-tshirt334-gray] |
| 14 | obs | **product_detail** (stage=2) | 3352 | Instruction: [SEP] Find me hand wash women's sweaters with long sleeve, stretch fabric, polyester spandex for teen girls |
| 15 | act | - | 215 | click[x-large] |
| 16 | obs | **product_detail** (stage=2) | 3352 | Instruction: [SEP] Find me hand wash women's sweaters with long sleeve, stretch fabric, polyester spandex for teen girls |
| 17 | act | - | 302 | click[buy now] |
| 18 | obs | **confirmation** (stage=3) | 651 | Thank you for shopping with us! [SEP] Your code: [SEP] None [SEP] (Paste it in your MTurk interface.) [SEP] Purchased [S |

## 2. Action & Stage Progression Analysis

### Page type distribution (100 trajectories)

| Page Type | Count | Fraction |
|-----------|-------|----------|
| product_detail | 300 | 0.362 |
| search_results | 228 | 0.275 |
| initial | 200 | 0.242 |
| confirmation | 100 | 0.121 |

### Stage distribution

| Stage | Count | Fraction |
|-------|-------|----------|
| 0 (Initial) | 200 | 0.242 |
| 1 (Search Results) | 228 | 0.275 |
| 2 (Product Detail) | 300 | 0.362 |
| 3 (Confirmation) | 100 | 0.121 |

### Stage sequences (first 20 trajectories)

| # | task_id | stage_sequence |
|---|---------|----------------|
| 1 | 0 | 0(initial) → 1(search_results) → 0(initial) → 1(search_results) → 2(product_detail) → 2(product_detail) → 3(confirmation) |
| 2 | 0 | 0(initial) → 1(search_results) → 0(initial) → 1(search_results) → 2(product_detail) → 2(product_detail) → 3(confirmation) |
| 3 | 0 | 0(initial) → 1(search_results) → 0(initial) → 1(search_results) → 2(product_detail) → 2(product_detail) → 3(confirmation) |
| 4 | 0 | 0(initial) → 1(search_results) → 0(initial) → 1(search_results) → 0(initial) → 1(search_results) → 2(product_detail) → 2(product_detail) → 3(confirmation) |
| 5 | 0 | 0(initial) → 1(search_results) → 2(product_detail) → 2(product_detail) → 3(confirmation) |
| 6 | 1 | 0(initial) → 1(search_results) → 0(initial) → 1(search_results) → 2(product_detail) → 2(product_detail) → 2(product_detail) → 3(confirmation) |
| 7 | 1 | 0(initial) → 1(search_results) → 0(initial) → 1(search_results) → 2(product_detail) → 2(product_detail) → 2(product_detail) → 3(confirmation) |
| 8 | 1 | 0(initial) → 1(search_results) → 0(initial) → 1(search_results) → 2(product_detail) → 2(product_detail) → 2(product_detail) → 3(confirmation) |
| 9 | 1 | 0(initial) → 1(search_results) → 0(initial) → 1(search_results) → 0(initial) → 1(search_results) → 2(product_detail) → 2(product_detail) → 2(product_detail) → 3(confirmation) |
| 10 | 1 | 0(initial) → 1(search_results) → 2(product_detail) → 2(product_detail) → 2(product_detail) → 3(confirmation) |
| 11 | 2 | 0(initial) → 1(search_results) → 0(initial) → 1(search_results) → 2(product_detail) → 2(product_detail) → 2(product_detail) → 3(confirmation) |
| 12 | 2 | 0(initial) → 1(search_results) → 0(initial) → 1(search_results) → 2(product_detail) → 2(product_detail) → 2(product_detail) → 3(confirmation) |
| 13 | 2 | 0(initial) → 1(search_results) → 0(initial) → 1(search_results) → 2(product_detail) → 2(product_detail) → 2(product_detail) → 3(confirmation) |
| 14 | 2 | 0(initial) → 1(search_results) → 0(initial) → 1(search_results) → 0(initial) → 1(search_results) → 2(product_detail) → 2(product_detail) → 2(product_detail) → 3(confirmation) |
| 15 | 2 | 0(initial) → 1(search_results) → 2(product_detail) → 2(product_detail) → 2(product_detail) → 3(confirmation) |
| 16 | 3 | 0(initial) → 1(search_results) → 1(search_results) → 0(initial) → 1(search_results) → 2(product_detail) → 3(confirmation) |
| 17 | 3 | 0(initial) → 1(search_results) → 0(initial) → 1(search_results) → 2(product_detail) → 3(confirmation) |
| 18 | 3 | 0(initial) → 1(search_results) → 0(initial) → 1(search_results) → 2(product_detail) → 3(confirmation) |
| 19 | 3 | 0(initial) → 1(search_results) → 1(search_results) → 0(initial) → 1(search_results) → 0(initial) → 1(search_results) → 2(product_detail) → 3(confirmation) |
| 20 | 3 | 0(initial) → 1(search_results) → 2(product_detail) → 3(confirmation) |

**Monotonic stage progression: 20/100 = 20.0%**

### Simplified flow patterns

| Pattern | Count | % |
|---------|-------|----|
| initial → search_results → initial → search_results → product_detail → confirmation | 60 | 60% |
| initial → search_results → initial → search_results → initial → search_results → product_detail → confirmation | 20 | 20% |
| initial → search_results → product_detail → confirmation | 20 | 20% |

## 3. Stage-Based Progress Coverage (All Teacher Trajectories)

**Total observations across all teacher trajectories: 218129**
**Avg obs per trajectory: 8.3 (std=1.7)**

### Stage distribution (ALL teacher data)

| Stage | Name | Count | Fraction |
|-------|------|-------|----------|
| 0 | Initial | 52063 | 0.239 |
| 1 | Search/Browse | 59494 | 0.273 |
| 2 | Product Detail | 80394 | 0.369 |
| 3 | Confirmation | 26178 | 0.120 |

### Page type distribution (ALL teacher data)

| Page Type | Count | Fraction |
|-----------|-------|----------|
| product_detail | 80394 | 0.369 |
| search_results | 59494 | 0.273 |
| initial | 52063 | 0.239 |
| confirmation | 26178 | 0.120 |

### Proposed stage → progress mapping

| Stage | Progress Value | Teacher obs count | Notes |
|-------|---------------|-------------------|-------|
| 0 (Initial) | 0.00 | 52063 | Before any search |
| 1 (Search/Browse) | 0.33 | 59494 | Found search results |
| 2 (Product Detail) | 0.67 | 80394 | Viewing/configuring product |
| 3 (Confirmation) | 1.00 | 26178 | Purchase complete |
| -1 (Unknown) | N/A | 0 | Classification failure |

## 4. Reward Distribution

| Metric | Value |
|--------|-------|
| Mean | 1.0000 |
| Std | 0.0000 |
| Min | 1.0000 |
| Max | 1.0000 |
| All = 1.0 | **YES** |

**All 26,178 teacher trajectories have reward = 1.0 (pre-filtered).**

WebShop reward is TRAJECTORY-LEVEL (given at buy), not step-level.
Intermediate observations get reward = 0 during training.
SC provides value by giving STEP-LEVEL progress signal.

## 5. Observation Matching & Coverage Analysis

### 5a. Unique teacher observations at different matching granularities

| Matching Method | Unique Fingerprints | Compression vs Full |
|----------------|--------------------|--------------------|
| Full text | 43872 | 1.0x |
| First 200 chars | 16549 | 2.7x |
| First 100 chars | 1319 | 33.3x |
| First 50 chars | 490 | 89.5x |
| Page type only | 4 | 10968x |

**Key insight**: At 100 chars, the first 100 chars are dominated by the task instruction.
Observations from the SAME TASK collide (good for same-task matching).
Different tasks diverge after ~50 chars (task-specific instructions).

### 5b. Cross-task observation overlap

Sampled 500 tasks. For each task, what fraction of its
100-char observation fingerprints appear in OTHER tasks?

| Metric | Value |
|--------|-------|
| Mean cross-task overlap | 0.860 |
| Median | 1.000 |
| Std | 0.256 |
| Min | 0.333 |
| Max | 1.000 |

**Interpretation**: High overlap at 100 chars means the first 100 chars are too generic
(dominated by common prefixes like 'instruction: [sep] find me...').
This is NOT useful for SC - we need task-specific matching.

### 5c. Same-task matching depth

Tasks with 2+ teacher trajectories: 5527

**Within-task observation overlap (Jaccard similarity between different teacher rollouts for same task)**:

| Matching | Mean Jaccard | Median | Std |
|----------|-------------|--------|-----|
| Full text | 0.692 | 0.667 | 0.114 |
| 200 chars | 0.978 | 1.000 | 0.069 |

**This measures**: if the teacher solves task X twice, how similar are the observations?
Low overlap → teacher takes different paths → on-policy agent unlikely to see same obs.
High overlap → observations are deterministic given the task → SC can match.

### 5d. Observation commonality analysis

Total unique 200-char fingerprints: 16549

| # Tasks Sharing Fingerprint | # Fingerprints | % |
|----------------------------|----------------|---|
| 1 (unique) | 15853 | 95.8% |
| 2-5 | 410 | 2.5% |
| 6-20 | 259 | 1.6% |
| 21-100 | 27 | 0.2% |
| 100+ | 0 | 0.0% |

Most-shared 200-char fingerprints:

- Shared by **30 tasks**: `thank you for shopping with us! [sep] your code: [sep] none [sep] (paste it in your mturk interface....`
- Shared by **29 tasks**: `thank you for shopping with us! [sep] your code: [sep] none [sep] (paste it in your mturk interface....`
- Shared by **29 tasks**: `webshop [sep] instruction: [sep] find me officially licensed, machine wash men's t-shirts with polye...`
- Shared by **28 tasks**: `webshop [sep] instruction: [sep] find me wash cold, machine wash men's shirts with polyester heather...`
- Shared by **28 tasks**: `thank you for shopping with us! [sep] your code: [sep] none [sep] (paste it in your mturk interface....`

### 5e. Stage-based coverage (page type matching)

If SC matches on **page type** instead of exact observation text:

| Stage set covered | # Tasks | % |
|-------------------|---------|---|
| {0, 1, 2, 3} | 5691 | 100.0% |

**With stage-based matching, every on-policy trajectory that reaches any of these stages
would get a progress value. Coverage is ~100% for all tasks.**

## 6. Current SC Implementation - What Gets Hashed?

**Source**: `agentevolver/module/exp_manager/state_progress.py`

The current `ExpertProgressMap` implementation:
1. **Matching**: Exact string hash on normalized observation text (`progress_map[obs]`)
2. **Per-task**: Separate hash map per `task_id` — only matches within same task
3. **Normalization**: Strips trailing "Available Actions", "Clickable elements" etc.
4. **Progress value**: `j / (T-1)` where j = position index, T = total observations in teacher traj
5. **Aggregation**: Takes max progress across all teacher rollouts for the same task

**WebShop-specific implications**:
- SC only works for on-policy trajectories with the SAME task_id as a teacher traj
- 5,691 unique task IDs have teacher data (out of ~12K training tasks)
- **~50% of training tasks have ZERO teacher data → zero SC coverage**
- Even for tasks WITH teacher data, within-task Jaccard is 0.692 (full text)
- This means ~30% of observations from on-policy rollouts won't match teacher observations
- **Effective coverage estimate: 50% (task overlap) × 70% (obs overlap) ≈ 35%**

## 7. Summary & Recommendations

### Critical Findings

| Finding | Implication |
|---------|-------------|
| All teacher rewards = 1.0 | Reward-as-progress gives no variation |
| 43872 unique full observations | Exact matching → near-zero cross-task coverage |
| 88% obs classified as unknown → FIXED: now properly classified | Page type classifier needed [SEP] format handling |
| 52063 initial + 59494 search + 80394 product + 26178 confirm | Clear stage structure exists |
| Within-task Jaccard = 0.692 (full), 0.978 (200ch) | High same-task overlap |
| 20/100 trajectories have monotonic stage progression | Stage NOT monotonic (search→back loops) |
| Effective SC coverage ≈ 35% | 50% task overlap × 70% obs overlap |

### Recommendation: Action-Stage Progress Map

The strongest SC redesign option for WebShop is **stage-based progress**:

```python
def webshop_stage_progress(observation):
    if 'Thank you for shopping' in observation:
        return 1.0  # Buy complete
    if '< Prev' in observation and 'Back to Search' in observation:
        return 0.67  # Product detail / options
    if re.search(r'Page \d+.*Total results', observation):
        return 0.33  # Search results
    if 'WebShop' in observation and 'Search' in observation:
        return 0.0   # Initial page
    return 0.0  # Default
```

**Why this works**:
1. 100% coverage - every observation maps to a stage (vs ~35% with hash-based)
2. No need for observation hashing or matching — purely structural
3. Environment-specific but simple to implement
4. Provides step-level signal that WebShop's trajectory-level reward cannot

**Caveat — Non-monotonicity**:
- Only 20% of teacher trajectories are strictly monotonic (stage never decreases)
- 80% have search→back_to_search loops (stage 1→0 regression)
- This means step_deltas will have NEGATIVE values during search refinement loops
- Solutions: (a) use trajectory-level P(τ) = max_stage_reached / 3 instead of step deltas,
  or (b) use max(stage_so_far) to enforce monotonicity, or (c) accept negative deltas
  as legitimate signal (going backward IS less progress)

### Why NOT use existing SC (hash-based matching)

1. WebShop observations embed task-specific content (product names, prices)
2. Even same-task teacher rollouts may visit different products
3. Cross-task observation overlap is entirely in generic page structure, not content
4. Hash-based SC would give 0 coverage for tasks not in teacher set

### Alternative: Instruction-Aware Stage Progress

A refined version could extract stage from observation structure + partial match on instruction:
- Same instruction prefix → same task family
- Stage 0-3 from page structure
- Progress = stage/3 (linear) or use non-linear mapping based on teacher stage distribution
