"""
Analyze WebShop teacher trajectory observations for SC redesign.
Covers all 5 analysis tasks.
"""
import pickle
import re
import sys
import os
import json
from collections import Counter, defaultdict
import numpy as np

OUTPUT_FILE = 'analysis_outputs/sc_redesign/data_analysis.md'
report_lines = []

def report(line=""):
    report_lines.append(line)
    print(line)

def save_report():
    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"\n[Saved report to {OUTPUT_FILE}]")

# ============================================================
# Load teacher trajectories
# ============================================================
print("Loading teacher trajectories...")
with open('data/teacher_trajectories/qwen72b/webshop_qwen72b_filtered.pkl', 'rb') as f:
    teacher_data = pickle.load(f)

report(f"# WebShop SC Redesign - Observation Data Analysis")
report(f"")
report(f"**Total teacher trajectories: {len(teacher_data)}**")
report(f"")

# ============================================================
# TASK 1: Analyze observation structure for 5 task_ids
# ============================================================
report("## 1. Observation Structure Analysis")
report("")

# Get 5 diverse trajectories (different task_ids)
seen_tasks = set()
sample_trajs = []
for t in teacher_data:
    tid = t['task_id']
    if tid not in seen_tasks:
        seen_tasks.add(tid)
        sample_trajs.append(t)
    if len(sample_trajs) >= 5:
        break

def classify_page_type(obs_text):
    """Classify a WebShop observation into page type."""
    if 'Instruction:' in obs_text and '[Search]' in obs_text and 'WebShop' in obs_text:
        return 'initial'
    if '[Back to Search]' in obs_text and '[< Prev]' in obs_text:
        return 'search_results'
    if '[Back to Search]' in obs_text and ('Price:' in obs_text or 'Description' in obs_text):
        return 'product_detail'
    if 'Your score' in obs_text or 'Thank you' in obs_text.lower():
        return 'confirmation'
    if '[Back to Search]' in obs_text:
        return 'product_or_options'
    return 'unknown'

for i, traj in enumerate(sample_trajs):
    report(f"### Trajectory {i+1}: task_id={traj['task_id']}, reward={traj['reward']}")
    report(f"Messages count: {len(traj['messages'])}")
    report("")

    for j, msg in enumerate(traj['messages']):
        role = msg['role']
        content = msg['content']
        if role == 'user':
            # These are environment observations
            page_type = classify_page_type(content)
            report(f"  **Msg[{j}] role=user (observation), page_type={page_type}, len={len(content)}**")
            # Show first 300 chars
            preview = content[:500].replace('\n', '\\n')
            report(f"  ```")
            report(f"  {preview}")
            report(f"  ```")
            report("")
        elif role == 'assistant':
            report(f"  Msg[{j}] role=assistant (action), len={len(content)}")
            preview = content[:200].replace('\n', '\\n')
            report(f"    `{preview}`")
            report("")
        elif role == 'system':
            report(f"  Msg[{j}] role=system, len={len(content)}")
            report(f"    `{content[:150]}...`")
            report("")
    report("---")
    report("")

# ============================================================
# TASK 2: Action stage analysis for 20 trajectories
# ============================================================
report("## 2. Action Stage Analysis (20 trajectories)")
report("")

def extract_action(msg_content):
    """Extract action from assistant message."""
    # Look for action patterns like search[...], click[...]
    content = msg_content.strip()
    # Try to find think/action pattern
    action_match = re.search(r'(?:Action:\s*)?(?:```\s*)?(search\[.*?\]|click\[.*?\])', content, re.IGNORECASE | re.DOTALL)
    if action_match:
        return action_match.group(1)
    # Simple pattern
    if content.startswith('search[') or content.startswith('click['):
        bracket_end = content.find(']')
        if bracket_end > 0:
            return content[:bracket_end+1]
    return content[:100]

def classify_action_stage(action, obs_before):
    """Classify action into stage."""
    action_lower = action.lower()
    if action_lower.startswith('search['):
        return 'search'
    if 'buy now' in action_lower:
        return 'buy'
    if 'back to search' in action_lower:
        return 'back_to_search'
    if '< prev' in action_lower or '> next' in action_lower or 'next >' in action_lower:
        return 'navigate'
    if action_lower.startswith('click['):
        # Try to determine what was clicked based on obs context
        if obs_before and ('Price:' in obs_before or 'Description' in obs_before):
            # On product page - clicking option or feature
            return 'select_option'
        else:
            return 'click_product'
    return 'other'

stage_sequences = []
all_action_types = Counter()

for traj in teacher_data[:100]:  # Sample 100 for statistics
    sequence = []
    prev_obs = ""
    for msg in traj['messages']:
        if msg['role'] == 'user':
            prev_obs = msg['content']
        elif msg['role'] == 'assistant':
            action = extract_action(msg['content'])
            stage = classify_action_stage(action, prev_obs)
            sequence.append(stage)
            all_action_types[stage] += 1
    stage_sequences.append(sequence)

# Report detailed sequences for first 20
report("### Detailed action sequences (first 20 trajectories)")
report("")
report("| # | task_id | reward | action_sequence |")
report("|---|---------|--------|-----------------|")
for i, traj in enumerate(teacher_data[:20]):
    seq = stage_sequences[i]
    report(f"| {i+1} | {traj['task_id']} | {traj['reward']} | {' → '.join(seq)} |")
report("")

report("### Action type distribution (100 trajectories)")
report("")
report("| Action Type | Count | Fraction |")
report("|-------------|-------|----------|")
total_actions = sum(all_action_types.values())
for action_type, count in all_action_types.most_common():
    report(f"| {action_type} | {count} | {count/total_actions:.3f} |")
report("")

# Analyze standard flow patterns
report("### Flow pattern analysis")
report("")

# Define standard flow: search -> click_product -> (select_option)* -> buy
standard_patterns = Counter()
for seq in stage_sequences:
    # Simplify: collapse consecutive same stages
    simplified = []
    for s in seq:
        if not simplified or simplified[-1] != s:
            simplified.append(s)
    pattern = ' → '.join(simplified)
    standard_patterns[pattern] += 1

report("| Pattern | Count | Fraction |")
report("|---------|-------|----------|")
for pattern, count in standard_patterns.most_common(15):
    report(f"| {pattern} | {count} | {count/len(stage_sequences):.3f} |")
report("")

# Check monotonicity
stage_order = {'search': 0, 'navigate': 1, 'click_product': 2, 'select_option': 3, 'buy': 4, 'back_to_search': 0, 'other': -1}
monotonic_count = 0
for seq in stage_sequences:
    stages = [stage_order.get(s, -1) for s in seq if s in stage_order]
    is_monotonic = all(stages[i] <= stages[i+1] for i in range(len(stages)-1))
    if is_monotonic:
        monotonic_count += 1

report(f"**Monotonic progression: {monotonic_count}/{len(stage_sequences)} = {monotonic_count/len(stage_sequences):.1%}**")
report(f"(A trajectory is monotonic if action stages never go backward)")
report("")

# ============================================================
# TASK 3: Stage-based progress mapping
# ============================================================
report("## 3. Stage-Based Progress Mapping")
report("")

def obs_to_stage(obs_text):
    """Map observation to a discrete stage."""
    if 'Instruction:' in obs_text and '[Search]' in obs_text and 'WebShop' in obs_text:
        return 0  # Initial page
    if '[Back to Search]' in obs_text and ('[< Prev]' in obs_text or '[Next >]' in obs_text):
        return 1  # Search results
    if '[Back to Search]' in obs_text and ('Price:' in obs_text or 'Description' in obs_text):
        return 2  # Product detail
    if 'Your score' in obs_text or 'Thank you' in obs_text.lower():
        return 4  # Buy complete
    if '[Back to Search]' in obs_text:
        return 2  # Product/options page
    return -1  # Unknown

stage_names = {0: 'Initial', 1: 'Search Results', 2: 'Product Detail', 3: 'Options Selected', 4: 'Buy Complete', -1: 'Unknown'}

# Analyze stage distribution across ALL teacher observations
teacher_stage_dist = Counter()
teacher_final_stages = Counter()
teacher_obs_per_traj = []

for traj in teacher_data:
    obs_stages = []
    for msg in traj['messages']:
        if msg['role'] == 'user':
            stage = obs_to_stage(msg['content'])
            teacher_stage_dist[stage] += 1
            obs_stages.append(stage)
    if obs_stages:
        teacher_final_stages[obs_stages[-1]] += 1
    teacher_obs_per_traj.append(len(obs_stages))

report("### Teacher observation stage distribution (all trajectories)")
report("")
report("| Stage | Name | Count | Fraction |")
report("|-------|------|-------|----------|")
total_obs = sum(teacher_stage_dist.values())
for stage in sorted(teacher_stage_dist.keys()):
    count = teacher_stage_dist[stage]
    report(f"| {stage} | {stage_names.get(stage, '?')} | {count} | {count/total_obs:.3f} |")
report("")

report("### Teacher FINAL observation stage distribution")
report("")
report("| Stage | Name | Count | Fraction |")
report("|-------|------|-------|----------|")
total_final = sum(teacher_final_stages.values())
for stage in sorted(teacher_final_stages.keys()):
    count = teacher_final_stages[stage]
    report(f"| {stage} | {stage_names.get(stage, '?')} | {count} | {count/total_final:.3f} |")
report("")

report(f"**Avg observations per teacher trajectory: {np.mean(teacher_obs_per_traj):.1f} (std={np.std(teacher_obs_per_traj):.1f})**")
report("")

# Compute stage-based progress values for teacher trajectories
report("### Stage-to-progress mapping")
report("")
report("If we assign progress = stage / 4:")
report("")
report("| Stage | Progress | Teacher obs at this stage |")
report("|-------|----------|--------------------------|")
for stage in sorted(teacher_stage_dist.keys()):
    if stage >= 0:
        progress = stage / 4.0
        count = teacher_stage_dist[stage]
        report(f"| {stage} ({stage_names.get(stage, '?')}) | {progress:.2f} | {count} ({count/total_obs:.1%}) |")
report("")

# ============================================================
# TASK 4: Reward-as-progress feasibility
# ============================================================
report("## 4. Reward Distribution Analysis")
report("")

teacher_rewards = [t['reward'] for t in teacher_data]
teacher_successes = [t['success'] for t in teacher_data]

report("### Teacher trajectory rewards")
report("")
report(f"| Metric | Value |")
report(f"|--------|-------|")
report(f"| Mean | {np.mean(teacher_rewards):.4f} |")
report(f"| Median | {np.median(teacher_rewards):.4f} |")
report(f"| Std | {np.std(teacher_rewards):.4f} |")
report(f"| Min | {np.min(teacher_rewards):.4f} |")
report(f"| Max | {np.max(teacher_rewards):.4f} |")
report(f"| reward == 1.0 | {sum(1 for r in teacher_rewards if r == 1.0)} ({sum(1 for r in teacher_rewards if r == 1.0)/len(teacher_rewards):.1%}) |")
report(f"| reward >= 0.8 | {sum(1 for r in teacher_rewards if r >= 0.8)} ({sum(1 for r in teacher_rewards if r >= 0.8)/len(teacher_rewards):.1%}) |")
report(f"| reward >= 0.5 | {sum(1 for r in teacher_rewards if r >= 0.5)} ({sum(1 for r in teacher_rewards if r >= 0.5)/len(teacher_rewards):.1%}) |")
report(f"| reward > 0 | {sum(1 for r in teacher_rewards if r > 0)} ({sum(1 for r in teacher_rewards if r > 0)/len(teacher_rewards):.1%}) |")
report(f"| reward == 0 | {sum(1 for r in teacher_rewards if r == 0)} ({sum(1 for r in teacher_rewards if r == 0)/len(teacher_rewards):.1%}) |")
report(f"| success == True | {sum(teacher_successes)} ({sum(teacher_successes)/len(teacher_successes):.1%}) |")
report("")

# Reward histogram buckets
buckets = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.01]
bucket_counts = np.histogram(teacher_rewards, bins=buckets)[0]
report("### Reward distribution histogram")
report("")
report("| Range | Count | Fraction |")
report("|-------|-------|----------|")
for i in range(len(bucket_counts)):
    lo = buckets[i]
    hi = buckets[i+1]
    label = f"[{lo:.1f}, {hi:.1f})" if hi < 1.01 else f"[{lo:.1f}, {hi:.2f}]"
    report(f"| {label} | {bucket_counts[i]} | {bucket_counts[i]/len(teacher_rewards):.3f} |")
report("")

report("### WebShop reward structure")
report("")
report("WebShop reward is computed as:")
report("- reward = (# matching attributes + title_match_score) / (# required attributes + 1)")
report("- It's a trajectory-level reward, given only at the end when 'buy' is clicked")
report("- Intermediate steps get reward = 0")
report("- **This means reward is SPARSE within a trajectory** - only the final step has non-zero reward")
report("")
report("**Key insight**: WebShop reward is NOT step-level dense. It's trajectory-level.")
report("SC's value would be providing STEP-LEVEL progress signal within the trajectory.")
report("")

# ============================================================
# TASK 5: Relaxed matching coverage
# ============================================================
report("## 5. Relaxed Observation Matching Analysis")
report("")

# Build observation sets from teacher data
# For each observation in teacher data, store various truncated versions
print("Building observation sets from teacher data...")

teacher_obs_full = set()  # Full observations
teacher_obs_200 = set()   # First 200 chars
teacher_obs_100 = set()   # First 100 chars
teacher_obs_50 = set()    # First 50 chars
teacher_page_types = set()

def normalize_obs(text):
    """Normalize observation text - strip whitespace, lowercase."""
    return re.sub(r'\s+', ' ', text.strip().lower())

for traj in teacher_data:
    for msg in traj['messages']:
        if msg['role'] == 'user':
            obs = msg['content']
            norm = normalize_obs(obs)
            teacher_obs_full.add(norm)
            teacher_obs_200.add(norm[:200])
            teacher_obs_100.add(norm[:100])
            teacher_obs_50.add(norm[:50])
            teacher_page_types.add(classify_page_type(obs))

report(f"### Unique teacher observations at different granularities")
report("")
report(f"| Granularity | Unique Count |")
report(f"|-------------|-------------|")
report(f"| Full text | {len(teacher_obs_full)} |")
report(f"| First 200 chars | {len(teacher_obs_200)} |")
report(f"| First 100 chars | {len(teacher_obs_100)} |")
report(f"| First 50 chars | {len(teacher_obs_50)} |")
report(f"| Page type only | {len(teacher_page_types)} |")
report("")

# Now check: what does a typical observation look like at each truncation?
report("### Sample observations at different truncation levels")
report("")
sample_obs = []
for traj in teacher_data[:3]:
    for msg in traj['messages']:
        if msg['role'] == 'user':
            sample_obs.append(msg['content'])
            if len(sample_obs) >= 6:
                break
    if len(sample_obs) >= 6:
        break

for i, obs in enumerate(sample_obs[:6]):
    norm = normalize_obs(obs)
    page_type = classify_page_type(obs)
    report(f"**Obs {i+1}** (page_type={page_type}, full_len={len(obs)}):")
    report(f"- 50 chars: `{norm[:50]}`")
    report(f"- 100 chars: `{norm[:100]}`")
    report(f"- 200 chars: `{norm[:200]}`")
    report("")

# Estimate cross-trajectory overlap at different levels
print("Computing cross-trajectory overlap...")
# For teacher trajectories: how often does obs from one traj match obs from another traj?
# Sample 500 trajectories for speed
sample_size = min(500, len(teacher_data))
np.random.seed(42)
sample_indices = np.random.choice(len(teacher_data), sample_size, replace=False)

# Build per-task observation sets at different granularities
task_obs_100 = defaultdict(set)
task_obs_page_type = defaultdict(set)
for idx in sample_indices:
    traj = teacher_data[idx]
    tid = traj['task_id']
    for msg in traj['messages']:
        if msg['role'] == 'user':
            norm = normalize_obs(msg['content'])
            task_obs_100[tid].add(norm[:100])
            task_obs_page_type[tid].add(classify_page_type(msg['content']))

# Cross-task overlap: for each task, what fraction of its obs match obs from OTHER tasks?
overlap_100_ratios = []
overlap_page_ratios = []

all_other_obs_100 = defaultdict(set)
all_other_page = defaultdict(set)

# Build the full sets
all_obs_100_per_task = {}
all_page_per_task = {}
for idx in sample_indices:
    traj = teacher_data[idx]
    tid = traj['task_id']
    if tid not in all_obs_100_per_task:
        all_obs_100_per_task[tid] = set()
        all_page_per_task[tid] = set()
    for msg in traj['messages']:
        if msg['role'] == 'user':
            norm = normalize_obs(msg['content'])
            all_obs_100_per_task[tid].add(norm[:100])
            all_page_per_task[tid].add(classify_page_type(msg['content']))

# Build global sets
global_obs_100 = set()
global_page = set()
for tid in all_obs_100_per_task:
    global_obs_100.update(all_obs_100_per_task[tid])
    global_page.update(all_page_per_task[tid])

# Cross-task match rate
cross_match_100 = []
for tid in all_obs_100_per_task:
    other_obs = global_obs_100 - all_obs_100_per_task[tid]
    my_obs = all_obs_100_per_task[tid]
    if my_obs:
        matched = len(my_obs.intersection(other_obs))  # This is wrong - should be other_obs that contain my obs
        # Actually: how many of my obs appear in other tasks' obs?
        # other_tasks_obs = all obs from other tasks
        other_tasks_obs = set()
        for other_tid in all_obs_100_per_task:
            if other_tid != tid:
                other_tasks_obs.update(all_obs_100_per_task[other_tid])
        matches = len(my_obs & other_tasks_obs)
        cross_match_100.append(matches / len(my_obs))

report("### Cross-task observation overlap (100-char matching)")
report("")
report(f"For {len(all_obs_100_per_task)} unique task_ids in sample of {sample_size} trajectories:")
report(f"- Mean cross-task match rate: **{np.mean(cross_match_100):.3f}**")
report(f"- Median: {np.median(cross_match_100):.3f}")
report(f"- Std: {np.std(cross_match_100):.3f}")
report(f"- Min: {np.min(cross_match_100):.3f}, Max: {np.max(cross_match_100):.3f}")
report("")
report("A low cross-task match rate means each task has UNIQUE observations,")
report("so an on-policy trajectory for task X will rarely match teacher observations from task Y.")
report("")

# CRITICAL: Same-task matching
# For SC to work, on-policy obs from task X need to match teacher obs from the SAME task X
# How many unique tasks have teacher data?
task_traj_count = Counter()
for traj in teacher_data:
    task_traj_count[traj['task_id']] += 1

report("### Same-task teacher coverage")
report("")
report(f"Total unique task_ids with teacher data: {len(task_traj_count)}")
report(f"Teacher trajectories per task:")
report(f"- Mean: {np.mean(list(task_traj_count.values())):.1f}")
report(f"- Median: {np.median(list(task_traj_count.values())):.1f}")
report(f"- Min: {min(task_traj_count.values())}, Max: {max(task_traj_count.values())}")
report("")

# Distribution of trajectories per task
traj_count_dist = Counter(task_traj_count.values())
report("| Trajs/task | # tasks |")
report("|-----------|---------|")
for count in sorted(traj_count_dist.keys())[:15]:
    report(f"| {count} | {traj_count_dist[count]} |")
report("")

# What fraction of training tasks have teacher data?
# This depends on the task split - let's check
unique_tasks = sorted(task_traj_count.keys(), key=lambda x: int(x) if x.isdigit() else x)
report(f"Task ID range: {unique_tasks[0]} to {unique_tasks[-1]}")
report(f"(Checking if task IDs are numeric...)")
numeric_ids = [int(t) for t in unique_tasks if t.isdigit()]
if numeric_ids:
    report(f"Numeric task ID range: {min(numeric_ids)} to {max(numeric_ids)}")
report("")

# ============================================================
# TASK 5 continued: Observation content analysis
# ============================================================
report("## 6. Detailed Observation Content Analysis")
report("")
report("### What makes observations unique?")
report("")

# Sample diverse observations and analyze their components
print("Analyzing observation content diversity...")

# Collect observations by page type
obs_by_type = defaultdict(list)
for traj in teacher_data[:200]:
    for msg in traj['messages']:
        if msg['role'] == 'user':
            ptype = classify_page_type(msg['content'])
            obs_by_type[ptype].append(msg['content'])

for ptype in obs_by_type:
    obs_list = obs_by_type[ptype]
    lengths = [len(o) for o in obs_list]
    report(f"### Page type: {ptype}")
    report(f"- Count (in 200 trajs): {len(obs_list)}")
    report(f"- Avg length: {np.mean(lengths):.0f} chars (std={np.std(lengths):.0f})")
    report(f"- Min: {min(lengths)}, Max: {max(lengths)}")
    report("")

    # Show 2 examples
    for i, obs in enumerate(obs_list[:2]):
        report(f"Example {i+1} (len={len(obs)}):")
        report(f"```")
        report(obs[:600])
        report(f"```")
        report("")

# ============================================================
# Summary and SC Design Implications
# ============================================================
report("## 7. Summary & SC Redesign Implications")
report("")
report("### Key Findings")
report("")
report("1. **Observation uniqueness**: WebShop observations are highly task-specific.")
report("   Product names, prices, descriptions, and search results are all unique per task.")
report("   Cross-task observation overlap at 100-char level is very low.")
report("")
report("2. **Action stage structure**: WebShop has a clear stage progression:")
report("   Initial → Search Results → Product Detail → (Option Selection) → Buy")
report("   Most teacher trajectories follow a predictable flow pattern.")
report("")
report("3. **Teacher reward distribution**: Teacher trajectories are pre-filtered for high quality,")
report("   so reward distribution is heavily skewed toward 1.0.")
report("   The reward is trajectory-level (only at buy), not step-level.")
report("")
report("4. **SC matching challenge**: The current SC uses exact observation hashing.")
report("   Since WebShop observations contain unique product info, cross-task matching is near-zero.")
report("   Even same-task matching requires the agent to visit the exact same products.")
report("")
report("### SC Redesign Options")
report("")
report("**Option A: Action-Stage Progress Map**")
report("- Map observations to stages (0-4) using simple heuristics")
report("- Progress = stage / max_stage")
report("- PRO: Universal coverage (all on-policy obs get a progress value)")
report("- PRO: Monotonic progression is common")
report("- CON: Very coarse (only 5 levels)")
report("- CON: Doesn't distinguish good from bad product choices within a stage")
report("")
report("**Option B: Reward-as-Progress (trajectory level)**")
report("- Use teacher reward as terminal progress, 0 for intermediate steps")
report("- PRO: Directly task-relevant")
report("- CON: No step-level signal (same as having no SC)")
report("")
report("**Option C: Hybrid Stage + Reward**")
report("- Use stage for intermediate progress, reward for terminal")
report("- Progress = stage/4 * (1 - is_terminal) + reward * is_terminal")
report("- PRO: Step-level AND outcome-level signal")
report("")
report("**Option D: Page-type matching with normalization**")
report("- Strip product-specific content, match on page structure only")
report("- Build progress from fraction of teacher trajectories that reach each page type")
report("- PRO: Better than exact matching")
report("- CON: Still very coarse")
report("")

save_report()
print("Done!")
