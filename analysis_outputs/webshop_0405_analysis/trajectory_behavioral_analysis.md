# Trajectory-Level Behavioral Analysis: 0405 WebShop Experiments

**Date**: 2026-04-03
**Validation Step**: 100 (200 tasks each)

## Executive Summary

Hybrid 0405 beats LUFFY on WebShop by +1.28 avg_reward points (0.7656 vs 0.7528), driven
primarily by **better product selection** and **more accurate option clicking** on a
small number of divergent tasks. The improvement is concentrated: 170/200 tasks have
identical rewards, and the top 5 positive contributors account for +2.94 of the +4.39
positive gap. Hybrid's search queries are more faithful to the instruction text (99.5%
vs 88.0% high-coverage queries), which cascades into better product and option matches.

LUFFY+SC 0405 is **fully stable** -- zero language collapse, zero CJK characters -- a
complete fix from the old collapsed LUFFY+SC (0.2211 avg_reward with Chinese output).

## 1. Performance Overview

| Method | avg_reward | perfect(%) | neg | zero | low(<0.3) | mid[0.3,0.7) | high[0.7,1.0) |
|--------|-----------|------------|-----|------|-----------|-------------|--------------|
| **Hybrid 0405** | **0.7656** | **53.0** | 7 | 7 | 5 | 49 | 26 |
| DUET 0405 | 0.7613 | 49.0 | 2 | 8 | 9 | 51 | 32 |
| LUFFY | 0.7528 | 49.5 | 5 | 7 | 8 | 53 | 28 |
| LUFFY+SC 0405 | 0.7087 | 32.5 | 13 | 8 | 5 | 49 | 60 |

**Reward Quantiles**:
| Method | P10 | P25 | P50 | P75 | P90 |
|--------|-----|-----|-----|-----|-----|
| Hybrid 0405 | 0.329 | 0.571 | **1.000** | 1.000 | 1.000 |
| DUET 0405 | 0.333 | 0.571 | 0.909 | 1.000 | 1.000 |
| LUFFY | 0.300 | 0.556 | 0.950 | 1.000 | 1.000 |
| LUFFY+SC 0405 | 0.000 | 0.521 | 0.857 | 1.000 | 1.000 |

Key: Hybrid's median is 1.000 (vs 0.950 for LUFFY), meaning more than half of tasks are
perfectly solved.

## 2. Failure Mode Classification

| Failure Mode | Hybrid | DUET | LUFFY | LUFFY+SC |
|-------------|--------|------|-------|----------|
| Language collapse (CJK) | **0** | **0** | **0** | **0** |
| Repetition loops (3+ same) | 0 | 0 | 1 | 5 |
| Format errors | 0 | 0 | 0 | 0 |
| Wrong product (bought, r<0.3) | 5 | 9 | 8 | 5 |
| Negative rewards | 7 | 2 | 5 | 13 |
| Failed to buy (r=0) | 0 | 0 | 0 | 2 |
| Stuck on options (repeated clicks) | 6 | 8 | 7 | **14** |

Key findings:
- **No language collapse anywhere** -- all 0405 methods are stable
- LUFFY+SC 0405 has the most stuck trajectories (14) and repetition loops (5), suggesting
  SC reward shaping may encourage option exploration that sometimes gets trapped
- Hybrid has slightly more negative rewards (7 vs 5 for LUFFY) -- these are cases where
  it fails to complete purchase

## 3. Cross-Method Comparison on Same Tasks

### Task Agreement
- **170/200 tasks**: Hybrid and LUFFY get the exact same reward
- **17 tasks**: Hybrid wins (diff > 0.05)
- **12 tasks**: LUFFY wins (diff > 0.05)
- Top 5 positive contributors: +2.94 points
- All negative contributors: -1.82 points
- **Net gap: +2.57 points = avg_reward difference of +0.013**

### 4-Way Perfect Score Agreement
| Condition | Count |
|-----------|-------|
| All 4 methods perfect | 59 |
| 3 methods perfect | 37 |
| 2 methods perfect | 9 |
| 1 method perfect | 3 |
| No method perfect | 92 |

### Correlation Matrix (Pearson, per-task rewards)
|          | Hybrid | DUET  | LUFFY | LUFFY+SC |
|----------|--------|-------|-------|----------|
| Hybrid   | 1.000  | 0.928 | 0.936 | 0.822    |
| DUET     | 0.928  | 1.000 | 0.955 | 0.791    |
| LUFFY    | 0.936  | 0.955 | 1.000 | 0.791    |
| LUFFY+SC | 0.822  | 0.791 | 0.791 | 1.000    |

LUFFY+SC is the most decorrelated, consistent with its unique failure modes. Hybrid-LUFFY
correlation (0.936) is slightly lower than DUET-LUFFY (0.955), suggesting Hybrid has
learned some distinct behaviors.

## 4. Root Cause: WHY Hybrid Beats LUFFY

### Search Query Quality
| Metric | Hybrid | DUET | LUFFY | LUFFY+SC |
|--------|--------|------|-------|----------|
| Avg search length (chars) | **154** | 151 | 144 | 150 |
| High instruction coverage (>70%) | **99.5%** | 96.0% | 88.0% | 98.0% |

Critical qualifier term inclusion rates:

| Term | Hybrid | LUFFY | Gap |
|------|--------|-------|-----|
| "officially licensed" | **100%** | 73% | +27pp |
| "slip resistant" | **100%** | 22% | +78pp |
| "non slip" | **100%** | 56% | +44pp |
| "classic fit" | **100%** | 95% | +5pp |
| "slim fit" | **94%** | 78% | +16pp |

**Hybrid preserves instruction qualifiers far better than LUFFY**, especially product
descriptors like "officially licensed" and "slip resistant". This directly determines
which products appear in search results.

### Purchase Completion Rate
| Method | Completion |
|--------|-----------|
| Hybrid 0405 | 98.0% |
| DUET 0405 | **99.0%** |
| LUFFY | 98.0% |
| LUFFY+SC 0405 | 94.0% |

All top-3 methods complete purchases at similar rates. LUFFY+SC lags slightly.

### Root Cause Breakdown (17 tasks where Hybrid > LUFFY)
| Source | Count | % |
|--------|-------|---|
| Different product selected | 8 | 47% |
| Different options (size/color) | 15 | 88% |
| Hybrid bought, LUFFY didn't | 2 | 12% |

### Root Cause Breakdown (12 tasks where LUFFY > Hybrid)
| Source | Count | % |
|--------|-------|---|
| Different product selected | 1 | 8% |
| Different options (size/color) | 12 | 100% |
| LUFFY bought, Hybrid didn't | 2 | 17% |

**Conclusion**: When Hybrid wins, it's often because better search queries led to a
**better product** (47% of wins). When LUFFY wins, the product is usually the same
but LUFFY picks better options (100% involve option differences). This suggests:
- Hybrid's advantage is **search-query-driven product selection**
- LUFFY's residual advantage is **option selection accuracy** on specific tasks

## 5. SC Attribute-Aware Potential (Phi) Analysis

| Method | Phi(final) mean | Phi(max) mean | Phi(avg) mean | Attr match mean |
|--------|----------------|---------------|---------------|-----------------|
| Hybrid 0405 | 0.754 | 0.766 | 0.597 | 0.429 |
| DUET 0405 | 0.763 | 0.767 | 0.604 | 0.428 |
| LUFFY | 0.750 | 0.763 | 0.593 | 0.429 |
| LUFFY+SC 0405 | 0.722 | 0.767 | 0.583 | 0.383 |

Final observation page types:
| Page Type | Hybrid | DUET | LUFFY | LUFFY+SC |
|-----------|--------|------|-------|----------|
| product_detail | 196 | 199 | 195 | 188 |
| search_home (stuck) | 4 | 1 | 5 | 12 |

LUFFY+SC 0405 visits more product pages (745 vs ~600 for others) due to more option
clicking and exploration, but ends up on search_home more often (12 vs 4-5), indicating
occasional inability to recover from bad states.

### Phi on Divergent Tasks
For the biggest Hybrid win (Hybrid=1.0 vs LUFFY=0.2, blazers task):
- Hybrid: Phi progression [0.15 -> 0.85 -> 0.85 -> 0.85] (found right product immediately)
- LUFFY: Phi progression [0.15 -> 0.68 -> 0.68 -> 0.68 -> 0.68 -> 0.00 -> 0.68]
  (stuck clicking wrong options, Phi dropped to 0.00 mid-trajectory)

The SC attribute-aware potential correctly discriminates: better products yield higher
Phi values (0.85 vs 0.68), confirming SC provides useful signal.

## 6. LUFFY+SC 0405 Stability Verification

| Check | Result |
|-------|--------|
| CJK characters in agent output | **0/200** |
| Non-ASCII characters in agent output | **0/200** |
| Repetition loops (3+ same action) | 5/200 |
| Purchase completion | 188/200 (94%) |

**VERDICT: FULLY STABLE. No language collapse.**

The old LUFFY+SC collapsed into Chinese output (avg_reward=0.2211). The 0405 version
with corrected SC implementation eliminates this entirely. However, LUFFY+SC 0405 still
underperforms due to:
- 14/200 trajectories have stuck option clicking (vs 6-8 for other methods)
- 12/200 trajectories end on search_home (vs 1-5 for others)
- 11 tasks where LUFFY+SC << Hybrid/LUFFY by >0.2 reward gap

## 7. Case Studies

### Case Study 1: Hybrid Wins -- Better Product Selection via Search
**Task**: Women's suiting & blazers, button closure, polyester spandex, color: black, size: xx-large

| Method | Reward | Search | Product | Options | Steps |
|--------|--------|--------|---------|---------|-------|
| **Hybrid** | **1.000** | Includes "wash cold" | b08dxl22jn | black, xx-large | 5 |
| LUFFY | 0.200 | Includes "machine wash" | b09m63b87v (wrong) | stuck clicking xnj-tshirt325-black x3 | 8 |
| DUET | 0.250 | Includes "machine wash" | b09m63b87v (wrong) | xnj-tshirt325-black, x-large | 5 |

**Why**: Hybrid's search returned the correct blazer (b08dxl22jn). LUFFY/DUET got a
t-shirt variant (b09m63b87v) due to slightly different search terms, then struggled
with wrong color option names (clicking "xnj-tshirt325-black" instead of proper "black").

### Case Study 2: Hybrid Wins -- Selecting Correct Product ID
**Task**: Officially licensed men's t-shirts, classic fit, color: navy, fit type: men, size: medium

| Method | Reward | Product | Key difference |
|--------|--------|---------|----------------|
| **Hybrid** | **1.000** | b07xpr3r7n | Correct product, selected men+navy+medium |
| LUFFY | 0.429 | b09q67h373 (wrong) | Wrong product, selected d-navy (wrong color name) |
| DUET | 1.000 | b07xpr3r7n | Same as Hybrid |

**Why**: Search query with "officially licensed" surfaced the correct product. LUFFY
dropped this qualifier, got a different product with approximate but wrong color match.

### Case Study 3: LUFFY Wins -- Hybrid Gets Stuck
**Task**: Women's tops, tees & blouses, short sleeve, unique design, relaxed fit, green, x-large

| Method | Reward | Product | Key difference |
|--------|--------|---------|----------------|
| Hybrid | -0.100 | b09qqp3356 | Clicked green twice, never clicked size, never bought |
| **LUFFY** | **0.394** | b09qqp3356 | Clicked green, x-large, then Buy Now |
| DUET | 0.394 | b09qqp3356 | Clicked x-large, then green/black, completed |

**Why**: Hybrid clicked "green" but the option didn't register (possibly already selected),
then clicked green again (stuck), never advanced to size selection. LUFFY progressed
through options correctly. This is a failure of Hybrid's option-clicking robustness.

### Case Study 4: LUFFY Wins -- Better Option Matching
**Task**: Men's loafers & slip-ons, rubber outsole, color: r.brown137, size: 10

| Method | Reward | Product | Key difference |
|--------|--------|---------|----------------|
| Hybrid | 0.571 | b07hp6lvrs | Clicked "brown" (generic, partial match) |
| **LUFFY** | **0.857** | b07s7hdc88 | Clicked "r.brown137" (exact color name match) |

**Why**: LUFFY's search was more concise but found a product with exact color name
matching. Hybrid found a different product where the color option name was generic
"brown" instead of the specific "r.brown137" the instruction asked for.

## 8. Key Takeaways

1. **Hybrid's advantage is search quality**: 99.5% high-coverage search queries (vs 88%
   for LUFFY), preserving critical qualifiers like "officially licensed" and "slip
   resistant" that determine which products appear.

2. **The improvement is sparse but impactful**: Only 17/200 tasks differ by >0.05, but
   Hybrid's wins are larger (+0.255 avg diff) than LUFFY's wins (-0.152 avg diff).

3. **Option selection is the remaining weakness**: When Hybrid loses, it's always due to
   option selection issues (stuck clicking, wrong option names). This affects 6/200
   trajectories.

4. **LUFFY+SC 0405 is completely stable** but underperforms due to excessive option
   exploration (14/200 stuck, 5/200 repetition loops). The SC reward shaping appears
   to encourage exploration that sometimes traps the agent.

5. **SC Phi correctly identifies quality**: Higher Phi values correspond to better
   products (0.85 for perfect matches vs 0.68 for partial matches), validating the
   attribute-aware design.

6. **No method shows language collapse or format errors** in the 0405 generation.
