# Velocity Sprint — cross-server results log

| timestamp | server | name | success_rate | reward_mean | notes |
|---|---|---|---|---|---|
| 2026-05-02 16:24 | 4xA100 | ws_swC_v_pk03_v00 | 39.5% (n=200) | 0.7131 | env=webshop |
| 2026-05-02 17:00 | 4xA100 | INVALIDATED | — | — | pk03_v00 result above (39.5%) ran on PRE-HOTFIX code with whip-saw bug. Restarting all 5 cells with latched code. |
| 2026-05-02 17:23 | 4xA100 | ws_swC_v_pk03_v00 | — | — | val@100 missing (experiments/webshop/ws_swC_v_pk03_v00/validation_log/100.jsonl) |

| 2026-05-02 17:25 | 4xA100 | RE-RESTART | — | — | First restart resumed buggy step_100 ckpt → skipped training. Now also moved checkpoints/ aside. Restarting fresh from scratch. |
| 2026-05-02 21:54 | 4xA100 | ws_swC_v_pk03_v00 | 36.5% (n=200) | 0.7225 | env=webshop |

| 2026-05-02 22:34 | 4xA100 | LATCH_V2 | — | — | v1 latch (threshold 0.3) fired at step 17 on pk03_v00 (single noise dip) → 36.5% (vs buggy 39.5%). Implemented triple-gated v2 latch: min_warmup=30 + plateau_level≥0.85 + persist≥3. Restarting all 4 WS cells. L20X may want to pull. |
| 2026-05-02 12:40 | L20X    | ws_swC_v_pk05_v00       | 27.0% (n=200) | partial_mean=0.378 | latch=**v1**, val@50=30.5%, dropped to 27.0% at step 100 (mild −3.5pp). Latch fired ~step 45 but rs_latched showed sporadic 1→0→1→0 flips (instance reset across FSDP boundaries). Better than buggy whip-saw (1.5%). |
| 2026-05-02 14:41 | L20X    | DECISION                | — | — | Pulled v2 code. pk07_v00 currently running (started 12:41 with v1 in-memory) at step 50. **Keeping pk07_v00 on v1** to gather high-peak v1 signal (per 4×A100 recommendation). **Remaining 3 runs (pk03_v00_K15, pk05, pk03_aggr) will auto-use v2** when their launcher.py spawns fresh from disk. |
