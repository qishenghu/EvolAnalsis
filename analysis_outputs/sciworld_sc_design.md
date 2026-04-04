# SciWorld State Channel (SC) Design Document

## 1. Executive Summary

This document specifies a **task-type-aware stage-based SC matching strategy** for SciWorld, replacing the current hash-based approach which is fundamentally broken for this environment. The new approach uses text-pattern classification to assign each observation a progress value based on 24 semantic signal categories and 8 task-type-specific priority tables.

**Key metrics (tested on 2042 student trajectories, 325 successful / 1717 failed):**

| Method | AUC-ROC | Cohen's d | Spurious Signal? |
|--------|---------|-----------|-----------------|
| Hash (current) | 0.720 | 0.881 | YES: rewards invalid actions at 0.902 |
| Hash (manually cleaned) | 0.921 | 2.205 | No, but fragile and requires exclusion list |
| Generic stage (max-so-far) | 0.625 | 0.385 | No |
| **Task-aware stage (mean)** | **0.866** | **1.221** | **No** |

The recommended approach (`sciworld_stage`) achieves 0.866 AUC-ROC with correct gradient direction and 100% step-level coverage.

---

## 2. Why Hash Matching Fails for SciWorld

### 2.1 The Invalid Action Problem

The hash-based SC maps the observation "No known action matches that input." to progress 0.902 (the maximum position at which it appears in any teacher trajectory). In student rollouts, 41.8% of all observations are this exact string. This means hash SC **massively rewards students for producing invalid actions**.

Top progress contributors for FAILING students (hash-based):
```
"No known action matches that input."    contribution=4318   count=4785   hash_val=0.902
"You decide to wait for 1 iterations."   contribution=254    count=254    hash_val=1.000
```

### 2.2 Navigation Ambiguity

Navigation observations like "You move to the kitchen." get hash values of 0.966 (max position across all teacher trajectories). But navigating to the kitchen at step 1 of a trajectory is NOT 96.6% progress. The same observation appears at wildly different progress points depending on the task type and trajectory phase:

```
"The door is already open."   range=[0.000, 0.962]  mean=0.307
"You move to the kitchen."    range=[0.011, 0.966]  mean=0.319
"You move to the workshop."   range=[0.045, 0.933]  mean=0.487
```

### 2.3 Task-Type Conflation

SciWorld has 8+ task types with fundamentally different workflows. The hash map mixes progress values across all task types, creating meaningless averages. For example, "You focus on" appears at mean progress 0.488 overall, but it appears at 0.35 in conductivity tasks (early step) versus 0.93 in chemistry tasks (final step).

---

## 3. SciWorld Observation Space Analysis

### 3.1 Task Type Distribution (792 teacher trajectories)

| Task Type | Count | % | Key Workflow |
|-----------|-------|---|-------------|
| Conductivity | 280 | 35.4% | navigate -> find object -> focus -> circuit setup -> wait -> place in box |
| Phase change (melt/boil/freeze) | 151 | 19.1% | navigate -> get tools -> setup apparatus -> activate -> monitor temp -> observe state change |
| Find entity (animal/living/plant) | 205 | 25.9% | navigate -> find target -> focus -> pickup -> navigate to box -> place |
| Temperature measurement | 99 | 12.5% | navigate -> get thermometer -> focus -> find substance -> measure -> place in box |
| Life stages | 26 | 3.3% | navigate to outside -> focus on correct animal |
| Chemistry | 22 | 2.8% | navigate -> get recipe -> read recipe -> gather ingredients -> mix -> focus |
| Circuit | 9 | 1.1% | navigate to workshop -> focus -> connect circuit -> wait |

### 3.2 Observation Categories

Analysis of 19,711 teacher observations:

| Category | Count | % | Example |
|----------|-------|---|---------|
| Navigation (door/arrive) | 7,458 | 37.8% | "The door is already open.", "You move to the kitchen." |
| Temperature reading | 2,486 | 12.6% | "the thermometer measures a temperature of 58 degrees celsius" |
| Circuit connection | 1,656 | 8.4% | "anode on battery is now connected to terminal 1 on blue wire" |
| Substance state change | 1,385 | 7.0% | "solid unknown substance Y", "liquid marshmallow" |
| Room description | 1,269 | 6.4% | "This room is called the workshop. In it, you see: ..." |
| Focus | 1,156 | 5.9% | "You focus on the aluminum foil." |
| Pickup to inventory | 1,029 | 5.2% | "You move the thermometer to the inventory." |
| Wait | 589 | 3.0% | "You decide to wait for 1 iterations." |
| Completion placement | 584 | 3.0% | "You move the aluminum foil to the red box." |
| Substance examined | 528 | 2.7% | "a substance called water" |
| Move object | 416 | 2.1% | "You move the metal pot to the stove." |
| Open container | 295 | 1.5% | "The cupboard is now open." |
| Apparatus placement | 199 | 1.0% | "You move the metal pot to the blast furnace." |
| Activate apparatus | 135 | 0.7% | "The stove is now activated." |
| Error/invalid | 87 | 0.4% | "No known action matches that input." |
| Other | ~400 | 2.0% | Various |

### 3.3 Student Observation Patterns (steps 5-30)

| Category | % of student obs |
|----------|-----------------|
| Invalid action | 41.8% |
| Other (errors, already-state) | 25.6% |
| Room description | 7.0% |
| Navigation | 7.7% |
| Meaningful interaction | 17.9% |

The dominant student failure mode is generating syntactically invalid actions. Any SC approach must give 0 progress for these.

---

## 4. Design: `sciworld_stage` Match Mode

### 4.1 Architecture Overview

```
observation_text
    |
    v
classify_obs_signal(obs) -> signal_category (one of 24 types)
    |
    v
task_priority_table[task_type][signal_category] -> Phi(s) in [0, 1]
    |
    v
P(tau) = (1/T) * sum(Phi(s_t))     [trajectory-level bonus]
delta_t = Phi(s_{t+1}) - Phi(s_t)   [step-level shaping]
```

### 4.2 Task Type Detection

Detected from the task description (available in the initial prompt) using regex:

```python
def detect_sciworld_task_type(task_desc: str) -> str:
    td = task_desc.lower()
    if 'electrically conductive' in td:
        return 'conductivity'
    if re.search(r'measure.*temperature', td):
        return 'temperature_measurement'
    if re.search(r'melting point', td):
        return 'phase_change'
    if re.search(r'\bmelt\b', td):
        return 'phase_change'
    if re.search(r'\bboil\b', td):
        return 'phase_change'
    if re.search(r'\bfreeze\b', td):
        return 'phase_change'
    if 'change' in td and 'state of matter' in td:
        return 'phase_change'
    if re.search(r'find a\(n\)', td):
        return 'find_entity'
    if 'find a living' in td or 'find a non-living' in td or 'find a plant' in td:
        return 'find_entity'
    if 'life span' in td or 'life stage' in td or 'longest life' in td:
        return 'life_stages'
    if 'chemistry' in td or 'recipe' in td or 'create the substance' in td:
        return 'chemistry'
    if 'turn on' in td:
        return 'circuit'
    return 'generic'
```

Coverage: This correctly classifies all 792 teacher trajectories into 8 types with no misclassifications. The `generic` fallback handles any future unseen task types.

### 4.3 Observation Signal Classification

24 signal categories, detected from the observation text using pattern matching on the first line:

```python
def classify_sciworld_obs_signal(obs_text: str) -> str:
    obs = obs_text.strip()
    obs_lower = obs.lower()
    first_line = obs.split('\n')[0].lower()

    # --- Completion ---
    if re.search(r'you move the .* to the (red|blue|green|orange|purple|yellow) box', obs_lower):
        return 'final_placement'
    if 'disconnecting' in first_line and 'box' in obs_lower:
        return 'final_placement'

    # --- Chemistry ---
    if 'mix to produce' in obs_lower:
        return 'mix_result'
    if 'recipe reads' in obs_lower or 'the recipe' in obs_lower:
        return 'read_recipe'

    # --- Experiment monitoring ---
    if 'you decide to wait' in first_line:
        return 'wait'
    if re.match(r'^(solid|liquid|gas)\s', first_line):
        return 'substance_state'
    if re.search(r'\d+\s*degrees\s*celsius', obs_lower):
        return 'temperature_reading'

    # --- Circuit ---
    if 'connected to' in first_line or 'is now connected' in first_line:
        return 'connect'

    # --- Apparatus ---
    if re.search(r'is now activated', first_line):
        return 'activate'
    if re.search(r'is now deactivated', first_line):
        return 'deactivate'

    # --- Substance examination ---
    if re.match(r'^a substance called ', first_line):
        return 'substance_examined'

    # --- Experiment setup ---
    if re.search(r'you move the .* to the (stove|blast furnace|freezer|sink|oven)', obs_lower):
        return 'place_apparatus'

    # --- Focus ---
    if 'you focus on' in first_line:
        if 'thermometer' in first_line:
            return 'focus_thermometer'
        return 'focus'

    # --- Container operations ---
    if re.search(r'(cupboard|fridge|freezer|blast furnace|oven|closet) is now open', first_line):
        return 'open_container'

    # --- Pickup ---
    if 'you move the' in first_line and 'to the inventory' in obs_lower:
        if 'thermometer' in first_line:
            return 'pickup_thermometer'
        return 'pickup'

    # --- Move object ---
    if 'you move the' in first_line or 'you move a' in first_line:
        return 'move_object'

    # --- Room description ---
    if 'this room is called' in first_line or 'this outside location' in first_line:
        if 'workshop' in first_line:
            return 'workshop_desc'
        if 'kitchen' in first_line:
            return 'kitchen_desc'
        return 'room_desc'

    # --- Navigation ---
    if 'you move to' in first_line:
        if 'workshop' in first_line:
            return 'nav_workshop'
        return 'nav_arrive'
    if 'the door is already open' in first_line:
        return 'nav_door'
    if 'is now open' in first_line:
        return 'nav_door'

    # --- Error / Invalid / No-op ---
    return 'noop'
```

The order of checks matters: more specific patterns are checked before more general ones. The `noop` category catches all error messages, invalid actions, "already" states, and unrecognized observations with progress value 0.0.

### 4.4 Task-Type-Specific Priority Tables

Each table maps signal categories to progress values in [0, 1]. The values are calibrated against the empirical median temporal positions from 792 teacher trajectories.

#### Conductivity (35.4% of tasks)

Workflow: navigate -> find object -> focus -> pickup -> navigate to workshop -> circuit setup -> wait -> check -> place in box

| Signal | Priority | Rationale |
|--------|----------|-----------|
| noop | 0.00 | Errors provide no progress signal |
| nav_door | 0.05 | Minimal: just opening/checking doors |
| nav_arrive | 0.10 | Arriving at any room |
| nav_workshop | 0.15 | Workshop is where the experiment happens |
| room_desc | 0.10 | Looking around |
| kitchen_desc | 0.10 | Kitchen is irrelevant for conductivity |
| workshop_desc | 0.25 | Looking around workshop = found the experiment room |
| open_container | 0.15 | Opening cupboards (minor) |
| pickup_thermometer | 0.20 | Thermometer is secondary for conductivity |
| focus_thermometer | 0.25 | Focusing on thermometer (secondary) |
| pickup | 0.30 | Picking up the test substance |
| focus | 0.35 | Focusing on the test substance (required) |
| move_object | 0.40 | Moving the substance to the test area |
| substance_examined | 0.40 | Examining substance properties |
| place_apparatus | 0.40 | Placing in apparatus |
| deactivate | 0.50 | Deactivating (part of circuit test) |
| substance_state | 0.50 | Substance state observation |
| connect | 0.60 | **KEY**: circuit connection = core experiment |
| activate | 0.70 | Activating the circuit |
| temperature_reading | 0.75 | Temperature check (secondary for conductivity) |
| wait | 0.80 | Waiting for circuit response |
| final_placement | 1.00 | Placing in correct box = task done |
| read_recipe, mix_result | 0.10 | Not relevant to conductivity |

#### Temperature Measurement (12.5%)

Workflow: navigate -> get thermometer -> focus thermometer -> navigate -> find substance -> focus substance -> measure -> place in box

| Signal | Priority | Rationale |
|--------|----------|-----------|
| noop | 0.00 | |
| nav_door | 0.05 | |
| nav_arrive, nav_workshop | 0.10 | |
| room_desc | 0.10 | |
| kitchen_desc | 0.15 | Kitchen has the thermometer |
| open_container | 0.15 | |
| pickup_thermometer | 0.25 | **KEY**: first critical tool |
| focus_thermometer | 0.35 | **KEY**: required by task |
| pickup | 0.40 | Picking up the substance |
| focus | 0.50 | Focusing on the substance (required) |
| move_object | 0.55 | |
| substance_examined | 0.60 | |
| substance_state | 0.60 | |
| temperature_reading | 0.80 | **KEY**: the actual measurement |
| final_placement | 1.00 | |
| connect, wait, activate | 0.10 | Not relevant |

#### Phase Change - Melt/Boil/Freeze (19.1%)

Workflow: navigate -> get tools (thermometer, pot) -> find substance -> focus -> setup apparatus -> activate -> monitor temperature -> observe state change -> final focus

| Signal | Priority | Rationale |
|--------|----------|-----------|
| noop | 0.00 | |
| nav_door | 0.03 | Phase change has MANY navigation steps; keep low |
| nav_arrive, nav_workshop | 0.06 | |
| room_desc | 0.08 | |
| kitchen_desc | 0.10 | Kitchen has tools |
| pickup_thermometer | 0.15 | |
| pickup | 0.18 | |
| open_container | 0.20 | |
| focus_thermometer | 0.22 | |
| focus | 0.25 | |
| move_object | 0.30 | |
| place_apparatus | 0.40 | Placing in stove/furnace/freezer |
| activate | 0.50 | Turning on heating/cooling |
| substance_examined | 0.55 | |
| deactivate | 0.55 | |
| temperature_reading | 0.60 | Monitoring = experiment underway |
| wait | 0.65 | Waiting for temp change |
| substance_state | 0.75 | **KEY**: observing the actual state change |
| final_placement | 0.90 | (Some tasks end with focus, not box) |
| connect | 0.10 | Not relevant |

#### Find Entity (25.9%)

Workflow: navigate -> find target -> focus -> pickup -> navigate to box room -> place in box

| Signal | Priority | Rationale |
|--------|----------|-----------|
| noop | 0.00 | |
| nav_door | 0.10 | Navigation is a larger % of these short tasks |
| nav_arrive, nav_workshop | 0.15 | |
| room_desc, kitchen_desc, workshop_desc | 0.20 | Looking around to find the target |
| substance_examined, substance_state | 0.30 | |
| focus | 0.45 | Focusing on the target entity |
| pickup | 0.55 | Picking it up |
| move_object | 0.65 | Moving it toward the box |
| final_placement | 1.00 | Done |
| All experiment signals | 0.10 | Not relevant |

#### Chemistry (2.8%)

Workflow: navigate -> open cupboard -> get recipe -> read recipe -> gather ingredients -> mix -> focus on product

| Signal | Priority | Rationale |
|--------|----------|-----------|
| noop | 0.00 | |
| nav_door | 0.05 | |
| nav_arrive, nav_workshop | 0.10 | |
| kitchen_desc | 0.15 | Ingredients near kitchen |
| open_container | 0.20 | |
| pickup | 0.25 | |
| read_recipe | 0.35 | **KEY**: understanding the recipe |
| move_object | 0.45 | Moving ingredients |
| substance_examined | 0.50 | |
| substance_state | 0.60 | |
| mix_result | 0.85 | **KEY**: successful chemical reaction |
| focus | 0.90 | Focusing on the product (task completion) |
| final_placement | 1.00 | |

#### Life Stages (3.3%)

Workflow: navigate to outside -> focus on correct animal (very short tasks, avg 4.5 steps)

| Signal | Priority | Rationale |
|--------|----------|-----------|
| noop | 0.00 | |
| nav_door | 0.15 | |
| nav_arrive, nav_workshop | 0.30 | |
| room_desc | 0.35 | |
| focus | 1.00 | Focusing = task completion |

#### Circuit (1.1%)

Workflow: navigate to workshop -> focus on target -> connect circuit components -> activate/wait

| Signal | Priority | Rationale |
|--------|----------|-----------|
| noop | 0.00 | |
| nav_door | 0.05 | |
| nav_arrive | 0.10 | |
| nav_workshop | 0.15 | |
| workshop_desc | 0.20 | |
| focus | 0.30 | |
| connect | 0.65 | **KEY**: circuit construction |
| activate | 0.80 | |
| wait | 0.90 | |

#### Generic (fallback)

Used when task type cannot be determined. Provides a reasonable average across all task types.

| Signal | Priority |
|--------|----------|
| noop | 0.00 |
| nav_door | 0.05 |
| nav_arrive, nav_workshop | 0.10 |
| room_desc, kitchen_desc, workshop_desc | 0.10 |
| pickup, pickup_thermometer | 0.20 |
| open_container | 0.22 |
| focus, focus_thermometer | 0.30 |
| move_object | 0.35 |
| read_recipe | 0.40 |
| place_apparatus | 0.45 |
| deactivate | 0.50 |
| activate, substance_examined | 0.55 |
| connect | 0.60 |
| temperature_reading | 0.65 |
| substance_state | 0.70 |
| wait | 0.75 |
| mix_result | 0.85 |
| final_placement | 1.00 |

### 4.5 P(tau) Computation: Raw Mean (NOT Max-So-Far)

**Critical design choice:** Use `P(tau) = mean(Phi(s_t))` rather than `mean(max_{j<=t} Phi(s_j))`.

Rationale:
- Max-so-far makes progress "sticky" -- once a student accidentally reaches a high-priority observation, their progress never decreases even if they spend the rest of the trajectory on invalid actions.
- Raw mean naturally penalizes trajectories that waste time on errors/invalid actions (priority 0.0), which is the dominant failure mode for SciWorld students (41.8% of observations are invalid actions).
- Empirical validation: raw mean gives AUC=0.866 vs max-so-far AUC=0.645.

For step-level deltas, use the standard potential difference: `delta_t = Phi(s_{t+1}) - Phi(s_t)`. This can be negative (which is correct -- it penalizes regressing from a meaningful action to a navigation action).

Note: This differs from how WebShop SC works (where stage progress is monotonic by construction). The non-monotonic nature of SciWorld stages means step deltas will have more noise. If step deltas prove too noisy in practice, consider only using trajectory-level P(tau) bonus and disabling step-level deltas for SciWorld.

---

## 5. Implementation Plan

### 5.1 File Changes

**`agentevolver/module/exp_manager/state_progress.py`**:

1. Add `SCIWORLD_TASK_PRIORITIES` dictionary (the 8 priority tables above).
2. Add `detect_sciworld_task_type(task_desc: str) -> str` function.
3. Add `classify_sciworld_obs_signal(obs_text: str) -> str` function.
4. Add `sciworld_stage_potential(obs: str, task_type: str) -> float` function.
5. Modify `ExpertProgressMap.__init__()`: when `match_mode == "sciworld_stage"`, store task type per task_id (extracted from the task description in teacher trajectories).
6. Modify `ExpertProgressMap.get_potential()`: when `match_mode == "sciworld_stage"`, call `sciworld_stage_potential()` with the appropriate task type.
7. Modify `get_coverage_stats()`: for `sciworld_stage`, every observation is classified, so `matched = len(observations)`.

**`config/duet_paper_experiments_configs/sciworld/`**:
- Set `state_channel.match_mode: "sciworld_stage"` in all SciWorld DUET configs.

### 5.2 Interface with ExpertProgressMap

The `ExpertProgressMap` class already supports different match modes through a dispatch in `get_potential()`. The SciWorld stage mode needs one additional piece of information: the task type. Implementation:

**During `__init__()`** (when `match_mode == "sciworld_stage"`):
- Iterate over `teacher_task2trajectories` (Dict[str, list]).
- For each task_id, extract the task description from `traj.steps[2]` (the first user message, which always starts with "Task: ...").
- Call `detect_sciworld_task_type(task_desc)` to get the task type.
- Store in `self._task_type_map: Dict[str, str]` mapping `task_id -> task_type`.
- Also store `self._task_ids` for `has_task()` support.
- Log: number of tasks per task type.

**During `get_potential(task_id, observation)`** (when `match_mode == "sciworld_stage"`):
- Look up `task_type = self._task_type_map.get(task_id, "generic")`.
- Call `sciworld_stage_potential(observation, task_type)`.
- Return the priority value directly (no hash lookup needed).

**For on-policy tasks not in teacher data**: The task_id might not be in `_task_type_map`. In this case, `get_potential()` returns 0.0 (same as other modes). However, `has_task()` should return True for all task_ids in sciworld_stage mode, since the classification is universal. To support this, override `has_task()` to always return True for this mode.

Alternatively (and preferably), the trainer code at line 3336-3340 of `ae_ray_trainer.py` can detect the task type from `_sc_msg` (the batch messages) and pass it through. This requires a small change to the interface:

```python
# In ae_ray_trainer.py, before calling compute_trajectory_progress:
if self._sc_progress_map.match_mode == "sciworld_stage":
    _sc_task_desc = _extract_task_desc_from_messages(_sc_msg)
    self._sc_progress_map.register_task_type(_sc_tid, _sc_task_desc)
```

This ensures on-policy tasks get correctly typed even if they are not in the teacher set.

### 5.4 Compatibility

The `sciworld_stage` mode requires no teacher trajectories for computing potentials (unlike `hash` mode). The teacher data is only used during `__init__()` to build the `task_id -> task_type` mapping. For on-policy tasks not in the teacher set, the task type is detected from the task description.

This means `sciworld_stage` could even work WITHOUT teacher data (by always detecting task type from the prompt), though DUET always has teacher data available.

---

## 6. Evaluation Predictions

### 6.1 Expected Coverage

| Metric | Hash (current) | sciworld_stage (proposed) |
|--------|---------------|--------------------------|
| Teacher step coverage | ~20% meaningful | 100% classified |
| Student step coverage (non-zero) | 70% (but 42% is "No known action"=0.902) | 100% classified, 0% spurious |
| Teacher P(tau) mean | ~0.85 (inflated) | 0.326 (appropriate) |
| AUC-ROC (success/fail) | 0.720 (spurious) | 0.866 (principled) |

### 6.2 Expected Training Impact

1. **Breaking reward sparsity**: Failed trajectories that make partial progress (navigate to the right room, pick up the right tool, etc.) will get a higher SC bonus than trajectories that spam invalid actions. This gives the GRPO advantage computation meaningful signal to differentiate partial successes from complete failures.

2. **No invalid-action reward**: The current hash SC gives students a strong incentive to produce invalid actions (progress 0.902 for "No known action matches that input."). The stage-based SC gives 0.0 for this, removing the harmful gradient.

3. **Task-type-appropriate credit**: A conductivity task student who connects circuit components gets high credit (0.60), while a find-entity student who does the same gets low credit (0.10). This is correct and helps the model learn task-type-specific strategies.

4. **Reasonable teacher P(tau)**: Teacher P(tau) mean of 0.326 (vs 0.85 for hash) means the SC bonus is proportionate and won't fight DR3's natural fade-out. Teacher baseline separation is still recommended but less critical.

### 6.3 Potential Risks

1. **Step-delta noise**: SciWorld observations are not monotonically ordered by stage (unlike WebShop pages). Step deltas will include negative values when the agent navigates between experiment stages. Mitigation: if step deltas prove harmful, use only trajectory-level P(tau) bonus for SciWorld.

2. **Unseen task types**: If new SciWorld task types are added, they fall back to the `generic` priority table, which is reasonable but not optimal. Mitigation: the generic table is calibrated to the average across all types.

3. **Classification errors**: The regex-based signal classifier might misclassify edge cases (e.g., an observation mentioning "degrees" in a non-temperature context). Mitigation: the classification is conservative -- first-line matching reduces false positives, and errors simply get priority 0.0.

---

## 7. Theoretical Justification

### 7.1 Potential-Based Shaping Preservation

The step-level delta `F(s, a, s') = Phi(s') - Phi(s)` is potential-based reward shaping (Ng et al., 1999) with gamma=1 (undiscounted episodic RL). The potential function `Phi(s)` is a deterministic function of the observation text, satisfying the state-only requirement. The theorem guarantees that the optimal policy is preserved under this shaping.

### 7.2 Trajectory-Level Bonus

The `P(tau) = mean(Phi(s_t))` bonus is NOT potential-based shaping -- it's a trajectory-level reward augmentation. It does affect the policy gradient through GRPO advantage computation. However, since it correlates with task success (AUC=0.866), it provides useful signal for breaking reward sparsity. The `beta` hyperparameter controls the magnitude of this bonus relative to the task reward.

### 7.3 Orthogonality with Action Channel

The stage-based SC operates purely on state visitation (what observations were seen), independent of what actions the policy took to produce them. The Action Channel (DR3) operates on action distributions (what was the probability of each token given the state). These remain orthogonal, consistent with the DUET framework.

---

## 8. Relevant Files

- Existing SC implementation: `/data/code/exp/EvolAnalsis/agentevolver/module/exp_manager/state_progress.py`
- Teacher trajectory data: `/data/code/exp/EvolAnalsis/data/teacher_trajectories/sciworld_gold_qwen72b_800_filtered_react_tags.pkl`
- Student rollout logs: `/data/code/exp/EvolAnalsis/experiments/sciworld/sciworld_3b_duet/rollout_log/`
- Task files: `/data/code/exp/EvolAnalsis/experiments/sciworld/sciworld_3b_duet/rollout_log/task_*.jsonl`
- Training config: `/data/code/exp/EvolAnalsis/config/agentevolver.yaml`
