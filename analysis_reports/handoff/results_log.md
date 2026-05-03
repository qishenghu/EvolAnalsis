# Velocity Sprint — cross-server results log

| timestamp | server | name | success_rate | reward_mean | notes |
|---|---|---|---|---|---|
| 2026-05-02 16:24 | 4xA100 | ws_swC_v_pk03_v00 | 39.5% (n=200) | 0.7131 | env=webshop |
| 2026-05-02 17:00 | 4xA100 | INVALIDATED | — | — | pk03_v00 result above (39.5%) ran on PRE-HOTFIX code with whip-saw bug. Restarting all 5 cells with latched code. |
| 2026-05-02 17:23 | 4xA100 | ws_swC_v_pk03_v00 | — | — | val@100 missing (experiments/webshop/ws_swC_v_pk03_v00/validation_log/100.jsonl) |

| 2026-05-02 17:25 | 4xA100 | RE-RESTART | — | — | First restart resumed buggy step_100 ckpt → skipped training. Now also moved checkpoints/ aside. Restarting fresh from scratch. |
| 2026-05-02 21:54 | 4xA100 | ws_swC_v_pk03_v00 | 36.5% (n=200) | 0.7225 | env=webshop |

| 2026-05-02 22:34 | 4xA100 | LATCH_V2 | — | — | v1 latch (threshold 0.3) fired at step 17 on pk03_v00 (single noise dip) → 36.5% (vs buggy 39.5%). Implemented triple-gated v2 latch: min_warmup=30 + plateau_level≥0.85 + persist≥3. Restarting all 4 WS cells. L20X may want to pull. |
| 2026-05-02 12:40 | L20X    | ws_swC_v_pk05_v00       | 27.0% (n=200) | partial_mean=0.378 | latch=**v1**, val@50=30.5%, dropped to 27.0% at step 100. |
| 2026-05-02 14:41 | L20X    | DECISION                | — | — | Pulled v2 code, switching remaining 3 runs to v2 latch. |
| 2026-05-02 14:46 | L20X    | RESTART_v2              | — | — | Launched fresh `run_l20x_velocity_v2.sh` queue: pk03_v00_K15 → pk05_v00 → pk07_v00 → pk05 → pk03_aggr (5 × 3.5h ≈ 17.5h). All v2 latch. ETA ~08:20 next day. |
| 2026-05-03 03:22 | 4xA100 | ws_swC_v_pk03_v00 v2    | 28.5% (n=200) | 0.6740 | latched at step 63. **WORSE than v1 (36.5%) and buggy (39.5%). Hypothesis disproven: more BC warmup → worse.** |
| 2026-05-03 08:02 | 4xA100 | ws_swC_v_pk04_v00 v2    | 26.5% (n=200) | 0.6913 | peak=0.4. Trend confirmed: peak↑ → SR↓ (pk03=28.5%, pk04=26.5%). |
| 2026-05-03 10:17 | 4xA100 | NEW QUEUE: SOTA-hunt    | — | — | Killed v2-latch queue. Launched `run_sota_hunt_2026_05_03.sh`: 4 settings using gap mode + token weighting (3B WS / 1.5B WS / 3B AF / 1.5B AF). See `analysis_reports/handoff/GAP_MODE_BEST_OF_K_PROPOSAL_2026-05-03.md` for code change adding best-of-k variant + experiment proposal asking L20X to test it. |
