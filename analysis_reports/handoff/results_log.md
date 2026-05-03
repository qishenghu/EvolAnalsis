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
| 2026-05-03 02:21 | L20X    | ws_swC_v_pk07_v00 (v2)  | 29.0% (n=200) | env=webshop | latched run, slightly better than pk03/pk05 v00 (~26%) |
| 2026-05-03 02:35 | L20X    | GAP_BOK_TAKEOVER        | — | — | Per A100 GAP_MODE_BEST_OF_K_PROPOSAL: dropping queued `pk03_aggr` (low-info, 5th velocity variant) and inserting `ws_swC_v_gap_bok_pk02` (gap mode + best_of_k=true + token_weighting=true, peak=0.2 valley=0.05). Watchdog active, will fire when current pk05 finishes (~05:50). gap_bok ETA ~09:30. Direct A/B vs 4xA100's `ws_3b_gap_pk02_v05_tw_dr3fast` (mean-gap, ETA ~14:00). |
| 2026-05-03 02:45 | L20X    | FULL_PIVOT_GAP_BOK      | — | — | Aborted pk05 mid-run (step ~10), archived as `_v2partial`. Velocity-mode v2 family confirmed capped at 25-29% across 3 valley=0 variants. Killed both old orchestrator + watchdog. New orchestrator `run_l20x_gap_bok_final.sh` runs 3 best-of-k peak variants on 3B WS: pk02 → pk015 → pk025 (most-likely-winner first). Total 10.5h, ETA ~13:15 today. |
| 2026-05-03 12:43 | L20X    | gap_bok_pk02            | 30.5% (n=200) | partial=0.520 | best-of-k, peak=0.20 valley=0.05 token_weighting=true |
| 2026-05-03 09:30 | L20X    | gap_bok_pk015           | 29.0% (n=200) | partial=0.472 | best-of-k, peak=0.15 |
| 2026-05-03 12:43 | L20X    | gap_bok_pk025           | 36.5% (n=200) | partial=0.536 | best-of-k, peak=0.25 ⭐ best gap-bok |
| 2026-05-03 14:44 | L20X    | LUFFY_RERUN_LAUNCH      | — | — | Sanity-check: rerun pure LUFFY (use_chord=false, use_dr3=false, teacher_policy_shaping=p_div_p_beta_0.1) on L20X. Original 49.5% never reproduced on this infra. If reruns at ~49.5% → DUET* genuinely below LUFFY; if ~30-40% → baseline was inflated and our 36.5% is competitive. ETA ~18:15 today. |
