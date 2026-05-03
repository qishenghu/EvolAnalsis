# Velocity Sprint — cross-server results log

| timestamp | server | name | success_rate | reward_mean | notes |
|---|---|---|---|---|---|
| 2026-05-02 16:24 | 4xA100 | ws_swC_v_pk03_v00 | 39.5% (n=200) | 0.7131 | env=webshop |
| 2026-05-02 17:00 | 4xA100 | INVALIDATED | — | — | pk03_v00 result above (39.5%) ran on PRE-HOTFIX code with whip-saw bug. Restarting all 5 cells with latched code. |
| 2026-05-02 17:23 | 4xA100 | ws_swC_v_pk03_v00 | — | — | val@100 missing (experiments/webshop/ws_swC_v_pk03_v00/validation_log/100.jsonl) |

| 2026-05-02 17:25 | 4xA100 | RE-RESTART | — | — | First restart resumed buggy step_100 ckpt → skipped training. Now also moved checkpoints/ aside. Restarting fresh from scratch. |
| 2026-05-02 21:54 | 4xA100 | ws_swC_v_pk03_v00 | 36.5% (n=200) | 0.7225 | env=webshop |

| 2026-05-02 22:34 | 4xA100 | LATCH_V2 | — | — | v1 latch (threshold 0.3) fired at step 17 on pk03_v00 (single noise dip) → 36.5% (vs buggy 39.5%). Implemented triple-gated v2 latch: min_warmup=30 + plateau_level≥0.85 + persist≥3. Restarting all 4 WS cells. L20X may want to pull. |
| 2026-05-03 03:22 | 4xA100 | ws_swC_v_pk03_v00 | 28.5% (n=200) | 0.6740 | env=webshop |

| 2026-05-03 03:22 | 4xA100 | ws_swC_v_pk03_v00 v2 | 28.5% (n=200) | 0.6740 | latched at step 63. **WORSE than v1 (36.5%) and buggy (39.5%). Hypothesis disproven: more BC warmup → worse.** |
| 2026-05-03 08:02 | 4xA100 | ws_swC_v_pk04_v00 | 26.5% (n=200) | 0.6913 | env=webshop |
| 2026-05-03 08:02 | 4xA100 | ws_swC_v_pk04_v00 v2 | 26.5% (n=200) | 0.6913 | peak=0.4. Trend confirmed: peak↑ → SR↓ (pk03=28.5%, pk04=26.5%). Reward higher (0.69 vs 0.67) means more partial-credit but fewer perfect successes (BC distorts attribute precision). |
