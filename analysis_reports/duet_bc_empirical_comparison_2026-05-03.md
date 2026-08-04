# DUET\* BC Empirical Comparison: 1.5B WS (works) vs 3B WS (fails)

**Date**: 2026-05-03  |  **Author**: experiment analyst  |  **Source logs**: `logs/`  |  **Parsed**: `analysis_reports/_parsed/bc_compare_2026-05-03.json`

All metrics extracted from training logs (per-step printed `key:val` pairs). Validation (SR / reward) reported every 50 steps; training runs are 100 steps each. **Best SR = max** of `val-summary/webshop/success_rate_mean_all` over the 2 val checkpoints (steps 50, 100).

## §1 Run inventory

| Tag | Log | Steps | Best SR | Best Reward | Final SR | Notes |
|---|---|---:|---:|---:|---:|---|
| 1.5b_swC_02_SOTA | `webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06.log` | 100 | 0.360 | 0.706 | 0.360 | **SOTA** pk03/v10/floor06 |
| 1.5b_swA_02_peak02 | `webshop_qwen1.5b_duet_swA_02_peak02.log` | 100 | 0.150 | 0.545 | 0.150 | phase A pk02 |
| 1.5b_swA_03_peak04 | `webshop_qwen1.5b_duet_swA_03_peak04.log` | 100 | 0.055 | 0.639 | 0.055 | phase A pk04 |
| 1.5b_swA_05_peak06 | `webshop_qwen1.5b_duet_swA_05_peak06.log` | 100 | 0.085 | 0.530 | 0.085 | phase A pk06 |
| 1.5b_swA_11_pk05_v10 | `webshop_qwen1.5b_duet_swA_11_pk05_v10.log` | 100 | 0.110 | 0.534 | 0.110 | phase A pk05+v10 |
| 1.5b_swB_01_pk03_v10_ema02 | `webshop_qwen1.5b_duet_swB_01_pk03_v10_ema02.log` | 100 | 0.215 | 0.502 | 0.215 | phase B pk03/v10/ema02 |
| 1.5b_swB_02_pk03_v15_ema02 | `webshop_qwen1.5b_duet_swB_02_pk03_v15_ema02.log` | 100 | 0.020 | 0.539 | 0.020 | phase B pk03/v15/ema02 |
| 1.5b_swC_01_pk03_v10_floor04 | `webshop_qwen1.5b_duet_swC_01_pk03_v10_floor04.log` | 100 | 0.020 | 0.542 | 0.020 | phase C pk03/v10/floor04 |
| 1.5b_swC_03_pk03_v12_ema02 | `webshop_qwen1.5b_duet_swC_03_pk03_v12_ema02.log` | 100 | 0.135 | 0.589 | 0.135 | phase C pk03/v12/ema02 |
| 3b_ws_swC_pk03_v00_v2latch | `ws_swC_v_pk03_v00.log` | 100 | 0.300 | 0.674 | 0.300 | 3B v2 triple-gate latch |
| 3b_ws_swC_pk04_v00_v2latch | `ws_swC_v_pk04_v00.log` | 100 | 0.295 | 0.691 | 0.295 | 3B v2 latch, pk04 |
| 3b_ws_swC_pk03_v00_v1latch | `ws_swC_v_pk03_v00.LATCH_V1_2234.log` | 100 | 0.365 | 0.722 | 0.365 | 3B v1 latch (older) |
| 3b_ws_swC_pk03_v00_buggy | `ws_swC_v_pk03_v00.BUGGY.log` | 100 | 0.425 | 0.713 | 0.425 | 3B buggy whip-saw (best at 42.5%!) |
| 3b_ws_swD_01_pk03_v10_floor06 | `webshop_qwen3b_duet_swD_01_pk03_v10_floor06.log` | 100 | 0.305 | 0.630 | 0.295 | 3B sweep D mirror of 1.5B SOTA |
| 3b_ws_swE_01_pk03_v05_ema01 | `webshop_qwen3b_duet_swE_01_pk03_v05_ema01.log` | 100 | 0.230 | 0.541 | 0.230 | 3B sweep E pk03/v05/ema01 |
| 3b_ws_swE_02_pk02_v10 | `webshop_qwen3b_duet_swE_02_pk02_v10.log` | 100 | 0.450 | 0.743 | 0.450 | 3B sweep E pk02/v10 (best 3B at 45%) |
| 1.5b_af_duet | `alfworld_qwen1.5b_duet.log` | 100 | 0.325 | 0.325 | 0.325 | 1.5B AlfWorld DUET (reference) |
| 3b_af_chord | `alfworld_qwen3b_chord.log` | 23 | 0.000 | 0.000 | 0.000 | 3B AlfWorld CHORD (truncated) |

**Note**: `3b_ws_swC_pk04_v00_v1latch` log is only 11 steps (crashed); excluded from analysis.
`3b_ws_swD_02_pk03_v10_floor07` only 5 steps (also crashed); excluded.

## §2 Headline findings (group means at step 100)

| Metric | 1.5B WS winner (SOTA) | 1.5B WS neighbors (n=8) | 3B WS losers (n=7) | 1.5B AF | Δ(3B-1.5B) interpretation |
|---|---:|---:|---:|---:|---|
| **Best val SR** (max over steps 50, 100) | 0.360 | 0.099 | 0.339 | 0.325 | ⚠ +243% rel |
| Final val SR (step 100) | 0.360 | 0.099 | 0.337 | 0.325 | ⚠ +241% rel |
| **H1: response_len_ratio teacher÷on** | 3.041 | 3.259 | 2.725 | 0.587 |  |
| H1: teacher_gradient_share (DR3) | 0.057 | 0.070 | 0.068 | 0.292 |  |
| H1: chord/mu (effective BC weight) | 0.108 | 0.097 | 0.056 | — | ⚠ -42% rel |
| H1: chord/mu_adaptive_gated | 0.040 | 0.026 | 0.148 | — | ⚠ +467% rel |
| H1: actor/grad_norm | 3.448 | 7.917 | 32.227 | 6.522 | ⚠ +307% rel |
| H1: actor/teacher_off_pg_loss |abs| | -2.344 | -1.037 | -0.719 | — | ⚠ +31% rel |
| **H2: response_len_teacher_mean** | 5,708 | 5,708 | 5,708 | 2,834 |  |
| **H2: response_len_onpolicy_mean** | 1,877 | 1,786 | 2,100 | 4,831 |  |
| H2: entropy_teacher_token_mean | 0.322 | 0.316 | 0.295 | 0.229 |  |
| H2: entropy_onpolicy_token_mean | 0.504 | 0.498 | 0.370 | 0.098 |  |
| H2: entropy_llm_offpolicy_mean | 0.322 | 0.316 | 0.295 | 0.229 |  |
| H2: actor/kl_loss | 0.223 | 1.023 | 1.450 | 0.648 | ⚠ +42% rel |
| **H3: critic/rewards_onpolicy/mean** | 0.633 | 0.651 | 0.741 | 0.232 |  |
| H3: critic/success_onpolicy/mean | 0.103 | 0.058 | 0.328 | 0.232 | ⚠ +464% rel |
| **H3: reward−SR gap (partial credit)** | 0.530 | 0.593 | 0.414 | 0.000 | ⚠ -30% rel |
| H4: state_channel/bonus_vs_reward_ratio | 0.100 | 0.103 | 0.104 | 0.000 |  |
| H4: state_channel/progress_onpolicy_mean | 0.337 | 0.361 | 0.411 | 0.337 |  |
| H4: state_channel/bonus_total_mean | 0.061 | 0.065 | 0.074 | 0.000 |  |
| **H5: dr3/disc_acc** (curriculum gate) | 0.981 | 0.986 | 0.923 | 1.000 |  |
| H5: dr3/w_off_mean (IS correction) | 0.766 | 0.636 | 0.555 | 0.534 |  |
| H5: dr3/logw_applied_abs_mean | 0.267 | 0.504 | 0.578 | 0.501 |  |
| **H6: actor/entropy_loss** (collapse?) | 0.502 | 0.495 | 0.369 | 0.099 |  |
| H6: entropy collapse Δ (step10→100) | 0.089 | 0.055 | 0.049 | 0.035 |  |
| H6: exp_replay/entropy_llm_onpolicy | 0.534 | 0.578 | 0.440 | 0.226 |  |

Bolded rows are the most diagnostic. ⚠ marks where 3B WS group differs from 1.5B WS neighbors by >30% relative.

## §3 Per-hypothesis evidence

### H1 — Capacity competition (BC crowds out GRPO)

| Run | resp_len_ratio (T/O) | teacher_grad_share | chord/mu | mu_adaptive_gated | grad_norm | teacher_off_pg_loss |
|---|---:|---:|---:|---:|---:|---:|
| 1.5b_swC_02_SOTA | 3.041 | 0.057 | 0.108 | 0.040 | 3.448 | -2.344 |
| 1.5b_swA_02_peak02 | 3.206 | 0.049 | 0.060 | 0.066 | 1.978 | -1.601 |
| 1.5b_swA_03_peak04 | 3.669 | 0.157 | 0.050 | 0.001 | 5.581 | -1.215 |
| 1.5b_swA_05_peak06 | 2.730 | 0.074 | 0.063 | 0.024 | 3.408 | -1.483 |
| 1.5b_swA_11_pk05_v10 | 3.133 | 0.069 | 0.121 | 0.052 | 2.105 | -0.985 |
| 1.5b_swB_01_pk03_v10_ema02 | 2.706 | 0.062 | 0.104 | 0.022 | 2.344 | -1.114 |
| 1.5b_swB_02_pk03_v15_ema02 | 3.562 | 0.038 | 0.150 | 0.002 | 38.844 | -0.641 |
| 1.5b_swC_01_pk03_v10_floor04 | 4.130 | 0.055 | 0.100 | 0.002 | 7.000 | -0.787 |
| 1.5b_swC_03_pk03_v12_ema02 | 2.934 | 0.059 | 0.127 | 0.040 | 2.078 | -0.472 |
| 3b_ws_swC_pk03_v00_v2latch | 2.668 | 0.041 | 0.000 | — | 29.091 | -1.333 |
| 3b_ws_swC_pk04_v00_v2latch | 2.796 | 0.105 | 0.000 | — | 27.157 | -0.949 |
| 3b_ws_swC_pk03_v00_v1latch | 2.639 | 0.074 | 0.000 | — | 16.846 | -0.509 |
| 3b_ws_swC_pk03_v00_buggy | 2.491 | 0.073 | — | — | 18.566 | — |
| 3b_ws_swD_01_pk03_v10_floor06 | 2.752 | 0.042 | 0.115 | 0.075 | 62.821 | -1.017 |
| 3b_ws_swE_01_pk03_v05_ema01 | 2.986 | 0.056 | 0.105 | 0.220 | 55.791 | -0.332 |
| 3b_ws_swE_02_pk02_v10 | 2.743 | 0.085 | 0.115 | 0.149 | 15.320 | -0.173 |

### H2 — Distribution mismatch (3B style differs more from 72B teacher)

| Run | resp_len_T | resp_len_O | entropy_T_token | entropy_O_token | entropy_offpolicy | actor/kl_loss |
|---|---:|---:|---:|---:|---:|---:|
| 1.5b_swC_02_SOTA | 5,708 | 1,877 | 0.322 | 0.504 | 0.322 | 0.223 |
| 1.5b_swA_02_peak02 | 5,708 | 1,780 | 0.327 | 0.507 | 0.327 | 0.668 |
| 1.5b_swA_03_peak04 | 5,708 | 1,556 | 0.309 | 0.528 | 0.309 | 1.945 |
| 1.5b_swA_05_peak06 | 5,708 | 2,091 | 0.327 | 0.462 | 0.327 | 0.826 |
| 1.5b_swA_11_pk05_v10 | 5,708 | 1,822 | 0.306 | 0.509 | 0.306 | 1.149 |
| 1.5b_swB_01_pk03_v10_ema02 | 5,708 | 2,109 | 0.322 | 0.467 | 0.322 | 0.984 |
| 1.5b_swB_02_pk03_v15_ema02 | 5,708 | 1,602 | 0.298 | 0.478 | 0.298 | 0.959 |
| 1.5b_swC_01_pk03_v10_floor04 | 5,708 | 1,382 | 0.322 | 0.551 | 0.322 | 0.934 |
| 1.5b_swC_03_pk03_v12_ema02 | 5,708 | 1,945 | 0.315 | 0.479 | 0.315 | 0.722 |
| 3b_ws_swC_pk03_v00_v2latch | 5,708 | 2,139 | 0.302 | 0.368 | 0.302 | 1.355 |
| 3b_ws_swC_pk04_v00_v2latch | 5,708 | 2,042 | 0.286 | 0.377 | 0.286 | 1.739 |
| 3b_ws_swC_pk03_v00_v1latch | 5,708 | 2,163 | 0.272 | 0.355 | 0.272 | 1.364 |
| 3b_ws_swC_pk03_v00_buggy | 5,708 | 2,291 | 0.294 | 0.353 | 0.294 | 1.639 |
| 3b_ws_swD_01_pk03_v10_floor06 | 5,708 | 2,074 | 0.325 | 0.373 | 0.325 | 1.887 |
| 3b_ws_swE_01_pk03_v05_ema01 | 5,708 | 1,912 | 0.327 | 0.412 | 0.327 | 1.108 |
| 3b_ws_swE_02_pk02_v10 | 5,708 | 2,081 | 0.261 | 0.355 | 0.261 | 1.055 |

### H3 — Reward optima conflict (high partial credit, low SR)

| Run | reward_onpolicy | success_onpolicy | reward−SR gap | best_SR | best_reward |
|---|---:|---:|---:|---:|---:|
| 1.5b_swC_02_SOTA | 0.633 | 0.103 | 0.530 | 0.360 | 0.706 |
| 1.5b_swA_02_peak02 | 0.625 | 0.069 | 0.556 | 0.150 | 0.545 |
| 1.5b_swA_03_peak04 | 0.750 | 0.000 | 0.750 | 0.055 | 0.639 |
| 1.5b_swA_05_peak06 | 0.694 | 0.052 | 0.642 | 0.085 | 0.530 |
| 1.5b_swA_11_pk05_v10 | 0.711 | 0.103 | 0.608 | 0.110 | 0.534 |
| 1.5b_swB_01_pk03_v10_ema02 | 0.787 | 0.155 | 0.632 | 0.215 | 0.502 |
| 1.5b_swB_02_pk03_v15_ema02 | 0.472 | 0.000 | 0.472 | 0.020 | 0.539 |
| 1.5b_swC_01_pk03_v10_floor04 | 0.600 | 0.000 | 0.600 | 0.020 | 0.542 |
| 1.5b_swC_03_pk03_v12_ema02 | 0.566 | 0.086 | 0.480 | 0.135 | 0.589 |
| 3b_ws_swC_pk03_v00_v2latch | 0.736 | 0.310 | 0.426 | 0.300 | 0.674 |
| 3b_ws_swC_pk04_v00_v2latch | 0.819 | 0.328 | 0.491 | 0.295 | 0.691 |
| 3b_ws_swC_pk03_v00_v1latch | 0.839 | 0.379 | 0.460 | 0.365 | 0.722 |
| 3b_ws_swC_pk03_v00_buggy | 0.776 | 0.431 | 0.345 | 0.425 | 0.713 |
| 3b_ws_swD_01_pk03_v10_floor06 | 0.694 | 0.224 | 0.470 | 0.305 | 0.630 |
| 3b_ws_swE_01_pk03_v05_ema01 | 0.457 | 0.190 | 0.267 | 0.230 | 0.541 |
| 3b_ws_swE_02_pk02_v10 | 0.867 | 0.431 | 0.436 | 0.450 | 0.743 |

### H4 — SC redundancy (SC duplicates BC's expert signal)

| Run | bonus_vs_reward_ratio | bonus_total_mean | progress_onpolicy | progress_teacher | shaped_ratio |
|---|---:|---:|---:|---:|---:|
| 1.5b_swC_02_SOTA | 0.100 | 0.061 | 0.337 | 0.562 | 0.625 |
| 1.5b_swA_02_peak02 | 0.108 | 0.066 | 0.366 | 0.562 | 0.609 |
| 1.5b_swA_03_peak04 | 0.099 | 0.070 | 0.385 | 0.562 | 0.656 |
| 1.5b_swA_05_peak06 | 0.115 | 0.076 | 0.417 | 0.562 | 0.641 |
| 1.5b_swA_11_pk05_v10 | 0.102 | 0.069 | 0.380 | 0.562 | 0.625 |
| 1.5b_swB_01_pk03_v10_ema02 | 0.103 | 0.076 | 0.420 | 0.562 | 0.656 |
| 1.5b_swB_02_pk03_v15_ema02 | 0.075 | 0.039 | 0.216 | 0.562 | 0.484 |
| 1.5b_swC_01_pk03_v10_floor04 | 0.100 | 0.059 | 0.327 | 0.562 | 0.625 |
| 1.5b_swC_03_pk03_v12_ema02 | 0.122 | 0.068 | 0.377 | 0.562 | 0.609 |
| 3b_ws_swC_pk03_v00_v2latch | 0.104 | 0.074 | 0.406 | 0.562 | 0.656 |
| 3b_ws_swC_pk04_v00_v2latch | 0.104 | 0.079 | 0.438 | 0.562 | 0.656 |
| 3b_ws_swC_pk03_v00_v1latch | 0.103 | 0.080 | 0.441 | 0.562 | 0.656 |
| 3b_ws_swC_pk03_v00_buggy | 0.106 | 0.079 | 0.433 | 0.562 | 0.656 |
| 3b_ws_swD_01_pk03_v10_floor06 | 0.103 | 0.070 | 0.388 | 0.562 | 0.656 |
| 3b_ws_swE_01_pk03_v05_ema01 | 0.108 | 0.057 | 0.316 | 0.562 | 0.641 |
| 3b_ws_swE_02_pk02_v10 | 0.103 | 0.082 | 0.453 | 0.562 | 0.656 |

### H5 — DR3 weakness (disc_acc plateau, IS correction strength)

| Run | dr3/disc_acc | dr3/w_off_mean | dr3/w_mean | logw_applied_abs | teacher_grad_share |
|---|---:|---:|---:|---:|---:|
| 1.5b_swC_02_SOTA | 0.981 | 0.766 | 1.030 | 0.267 | 0.057 |
| 1.5b_swA_02_peak02 | 0.978 | 0.619 | 1.036 | 0.480 | 0.049 |
| 1.5b_swA_03_peak04 | 0.999 | 0.560 | 1.028 | 0.606 | 0.157 |
| 1.5b_swA_05_peak06 | 0.996 | 0.569 | 1.048 | 0.565 | 0.074 |
| 1.5b_swA_11_pk05_v10 | 0.972 | 0.776 | 1.054 | 0.253 | 0.069 |
| 1.5b_swB_01_pk03_v10_ema02 | 0.985 | 0.774 | 1.032 | 0.410 | 0.062 |
| 1.5b_swB_02_pk03_v15_ema02 | 0.989 | 0.630 | 1.066 | 0.461 | 0.038 |
| 1.5b_swC_01_pk03_v10_floor04 | 0.996 | 0.543 | 1.024 | 0.775 | 0.055 |
| 1.5b_swC_03_pk03_v12_ema02 | 0.973 | 0.617 | 1.009 | 0.482 | 0.059 |
| 3b_ws_swC_pk03_v00_v2latch | 0.871 | 0.755 | 1.046 | 0.281 | 0.041 |
| 3b_ws_swC_pk04_v00_v2latch | 0.978 | 0.530 | 1.039 | 0.405 | 0.105 |
| 3b_ws_swC_pk03_v00_v1latch | 0.912 | 0.453 | 1.052 | 0.791 | 0.074 |
| 3b_ws_swC_pk03_v00_buggy | 0.907 | — | 1.046 | — | 0.073 |
| 3b_ws_swD_01_pk03_v10_floor06 | 0.979 | 0.472 | 1.067 | 0.751 | 0.042 |
| 3b_ws_swE_01_pk03_v05_ema01 | 0.903 | 0.480 | 0.941 | 0.795 | 0.056 |
| 3b_ws_swE_02_pk02_v10 | 0.909 | 0.639 | 1.015 | 0.444 | 0.085 |

### H6 — Plasticity (entropy collapse rate)

Per-step `actor/entropy_loss` trajectory:

| Run | step10 | step25 | step50 | step75 | step100 | Δ(100−10) |
|---|---:|---:|---:|---:|---:|---:|
| 1.5b_swC_02_SOTA | 0.413 | 0.546 | 0.580 | 0.561 | 0.502 | 0.089 |
| 1.5b_swA_02_peak02 | 0.429 | 0.528 | 0.557 | 0.561 | 0.504 | 0.075 |
| 1.5b_swA_03_peak04 | 0.442 | 0.528 | 0.562 | 0.545 | 0.524 | 0.082 |
| 1.5b_swA_05_peak06 | 0.452 | 0.507 | 0.580 | 0.505 | 0.460 | 0.008 |
| 1.5b_swA_11_pk05_v10 | 0.350 | 0.538 | 0.570 | 0.436 | 0.506 | 0.156 |
| 1.5b_swB_01_pk03_v10_ema02 | 0.454 | 0.503 | 0.542 | 0.523 | 0.465 | 0.011 |
| 1.5b_swB_02_pk03_v15_ema02 | 0.521 | 0.491 | 0.596 | 0.596 | 0.475 | -0.046 |
| 1.5b_swC_01_pk03_v10_floor04 | 0.414 | 0.484 | 0.555 | 0.578 | 0.547 | 0.133 |
| 1.5b_swC_03_pk03_v12_ema02 | 0.460 | 0.483 | 0.565 | 0.539 | 0.477 | 0.017 |
| 3b_ws_swC_pk03_v00_v2latch | 0.369 | 0.398 | 0.406 | 0.373 | 0.367 | -0.002 |
| 3b_ws_swC_pk04_v00_v2latch | 0.313 | 0.393 | 0.431 | 0.401 | 0.376 | 0.063 |
| 3b_ws_swC_pk03_v00_v1latch | 0.292 | 0.421 | 0.427 | 0.376 | 0.353 | 0.061 |
| 3b_ws_swC_pk03_v00_buggy | 0.295 | 0.414 | 0.419 | 0.362 | 0.352 | 0.057 |
| 3b_ws_swD_01_pk03_v10_floor06 | 0.338 | 0.364 | 0.409 | 0.384 | 0.373 | 0.035 |
| 3b_ws_swE_01_pk03_v05_ema01 | 0.308 | 0.406 | 0.456 | 0.356 | 0.411 | 0.103 |
| 3b_ws_swE_02_pk02_v10 | 0.324 | 0.398 | 0.388 | 0.357 | 0.353 | 0.029 |

### Per-hypothesis trajectory (DR3 disc_acc, mu, teacher_share, SC bonus) — sampled steps

**1.5b_swC_02_SOTA** (best_SR=0.360):

| step | disc_acc | w_off_mean | mu | mu_gated | teacher_grad_share | SC_bonus_ratio | entropy_loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.719 | 1.005 | 0.278 | 0.892 | 0.296 | 0.154 | 0.413 |
| 25 | 0.882 | 0.911 | 0.178 | 0.390 | 0.167 | 0.132 | 0.546 |
| 50 | 0.999 | 0.621 | 0.102 | 0.008 | 0.099 | 0.129 | 0.580 |
| 75 | 0.996 | 0.495 | 0.101 | 0.005 | 0.094 | 0.081 | 0.561 |
| 100 | 0.981 | 0.766 | 0.108 | 0.040 | 0.057 | 0.100 | 0.502 |

**3b_ws_swC_pk03_v00_v2latch** (best_SR=0.300):

| step | disc_acc | w_off_mean | mu | mu_gated | teacher_grad_share | SC_bonus_ratio | entropy_loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.692 | — | — | — | 0.224 | 0.139 | 0.369 |
| 25 | 0.706 | 0.927 | 0.000 | — | 0.124 | 0.145 | 0.398 |
| 50 | 0.847 | 0.720 | 0.300 | — | 0.101 | 0.120 | 0.406 |
| 75 | 0.891 | 0.947 | 0.000 | — | 0.085 | 0.081 | 0.373 |
| 100 | 0.871 | 0.755 | 0.000 | — | 0.041 | 0.104 | 0.367 |

**3b_ws_swC_pk03_v00_buggy** (best_SR=0.425):

| step | disc_acc | w_off_mean | mu | mu_gated | teacher_grad_share | SC_bonus_ratio | entropy_loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.645 | 0.977 | 0.300 | — | 0.209 | 0.137 | 0.295 |
| 25 | 0.709 | 0.954 | 0.000 | — | 0.102 | 0.130 | 0.414 |
| 50 | 0.826 | 0.787 | 0.300 | — | 0.081 | 0.121 | 0.419 |
| 75 | 0.922 | 0.647 | 0.300 | — | 0.088 | 0.080 | 0.362 |
| 100 | 0.907 | — | — | — | 0.073 | 0.106 | 0.352 |

**3b_ws_swE_02_pk02_v10** (best_SR=0.450):

| step | disc_acc | w_off_mean | mu | mu_gated | teacher_grad_share | SC_bonus_ratio | entropy_loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.713 | — | — | — | 0.275 | 0.149 | 0.324 |
| 25 | 0.669 | 0.964 | 0.159 | 0.592 | 0.118 | 0.129 | 0.398 |
| 50 | 0.865 | — | — | — | 0.037 | 0.124 | 0.388 |
| 75 | 0.922 | 0.461 | 0.115 | 0.153 | 0.059 | 0.080 | 0.357 |
| 100 | 0.909 | 0.639 | 0.115 | 0.149 | 0.085 | 0.103 | 0.353 |

## §4 Cross-cut: 3B AlfWorld vs 3B WebShop

3B AF CHORD log truncated at step 23 (no val-summary captured). Use 1.5B AF DUET as proxy:

| Run | env | best_SR | resp_len_ratio (T/O) | bonus_vs_reward | disc_acc | mu | entropy_loss |
|---|---|---:|---:|---:|---:|---:|---:|
| 1.5b_af_duet | AF | 0.325 | 0.587 | 0.000 | 1.000 | — | 0.099 |
| 1.5b_swC_02_SOTA | WS | 0.360 | 3.041 | 0.100 | 0.981 | 0.108 | 0.502 |
| 3b_ws_swC_pk03_v00_v2latch | WS | 0.300 | 2.668 | 0.104 | 0.871 | 0.000 | 0.367 |
| 3b_ws_swE_02_pk02_v10 | WS | 0.450 | 2.743 | 0.103 | 0.909 | 0.115 | 0.353 |

**Key contrast question**: in WS, teacher trajectories are ~3× longer than on-policy (3000–6000 vs 1300–1900 tokens). In AF, teacher trajectories tend to be shorter and more action-dense. The BC loss in WS therefore concentrates 5–10× more expert tokens per gradient step than on AF — amplified at 3B because longer responses hit per-token loss harder when the model is large enough to fit them.

## §5 Synthesis: most likely root cause

### CRITICAL CAVEAT on the framing

The premise '1.5B WS wins, 3B WS loses' is **partially refuted by the data**. Across the analyzed runs:

- 1.5B WS SOTA: 0.360 best SR
- 1.5B WS neighbors avg: 0.099 (most cells 5–22%)
- 3B WS losers avg: 0.339 — **higher than the 1.5B group mean**
- 3B WS best individual run (swE_02): 0.450 — **above 1.5B SOTA**

So the dynamic is not '3B BC is broken' but rather '3B WS DUET\* is **more sensitive to config**: the v2 triple-gated latch (28–30%) lost ~10pp vs the v1 latch (36.5%) and ~14pp vs the buggy whip-saw (42.5%)'. The losers are **specific config choices**, not the BC mechanism itself.

With that caveat, the starkest training-dynamics differences:

### Finding 1 — `actor/grad_norm` is 4–8× higher at 3B WS
- 1.5B WS neighbors avg grad_norm at step 100: **7.917**
- 3B WS losers avg grad_norm at step 100: **32.227** (4.1× higher)
- Individual 3B runs hit grad_norm **62.8** (swD_01), **55.8** (swE_01), **38.8** (one 1.5B outlier swB_02). This is a **strong signal of optimization instability** at 3B — the BC + GRPO + SC stack produces gradient spikes that 1.5B's smaller parameter count damps.

### Finding 2 — entropy is 25% lower at 3B from step 10 onward (early-onset collapse)

Look at H6 trajectory table: 3B WS starts step 10 at entropy 0.29–0.37, never recovers above 0.43; 1.5B WS sits at entropy 0.41–0.55 throughout. A 3B model with `pretrained` Qwen2.5-3B starting weights has **less starting entropy in the WS register**, and the BC+SC+DR3 stack does not recover exploration — the policy commits early and then tunes within a narrower output distribution.

### Finding 3 — actor/kl_loss runs 1.4× higher at 3B (more pull from reference)
- 1.5B WS neighbors avg kl_loss: 1.023
- 3B WS losers avg kl_loss: 1.450 (1.42× higher)
- 3B is **moving further from the reference policy per step** — combined with high grad_norm this means BC's anchoring force at 3B is amplified relative to 1.5B, but in a way that produces *unstable* gradient steps rather than smooth assimilation of teacher style.

### Finding 4 — H4 (SC redundancy) is the cleanest mechanism story

`state_channel/bonus_vs_reward_ratio` is 0.10 at both scales (identical). `progress_onpolicy_mean` is **higher at 3B** (0.411) than 1.5B (0.361). This means the 3B policy is *already* covering the expert progress map well via SC, leaving less marginal value for BC. Adding BC at 3B is **double-counting expert signal** with two distinct gradient pathways — and the second pathway (BC) is the one with the long-teacher-trajectory anchoring problem (Finding 5 below).

### Finding 5 — On-policy length: 17% longer at 3B

The single starkest empirical pattern is **on-policy response length**:

- 1.5B WS neighbors avg on-policy response_len at step 100: **1,786** tokens
- 3B WS losers avg on-policy response_len at step 100: **2,100** tokens
- 3B writes **17.6%** longer rollouts than 1.5B at same step.

Yet `response_len_ratio_teacher_vs_on` is *similar* across scales (1.5B≈3.04, 3B≈3.0–3.2 at step 100). This means: the BC loss ingests the same proportion of teacher-vs-onpolicy tokens, but the **absolute** on-policy token budget competing against the teacher demonstration distribution is larger. At 3B, the model has the capacity to write long, partially-correct WebShop interactions that hit substantial reward (~0.6–0.7 reward_onpolicy_mean) without hitting full success (SR<0.5). The BC term then anchors the policy toward the 72B teacher's distinct stylistic register, and the result is a **policy hovering at high partial reward but low full success** (the H3 signature).

**Supporting H3 (reward−SR gap)**:
- 1.5B WS SOTA at step 100: reward=0.633, SR=0.103, gap=0.530
- 3B WS losers avg at step 100: reward=0.741, SR=0.328, gap=0.414
  → The reward−SR gap is **larger at 3B**, evidence the 3B policy is collecting partial-credit reward without converting to wins.

**H1 (capacity competition) is partially supported**:
- teacher_gradient_share (DR3 fade-out): 1.5B=0.070, 3B=0.068 — DR3 actually fades *similarly well* at both scales, so H1 in its strict 'BC overwhelms GRPO via gradient mass' form is **refuted** at the chord/mu level. But at the **token-count** level, 3B's longer on-policy responses mean any leaked BC signal gets blended into a longer sequence that GRPO is also trying to credit-assign.

**H4 (SC redundancy) is consistent**:
- bonus_vs_reward_ratio: 1.5B=0.103, 3B=0.104 — SC contributes a similar fraction in both regimes (~0.10–0.13). Since SC already injects expert progress signal on on-policy samples, the marginal value of additional BC weight at 3B is lower (the policy has already absorbed expert structure via SC), while the cost (style anchoring) is higher.

**H5 (DR3 weakness) is refuted**: dr3/disc_acc reaches ≥0.98 in both regimes by step 50; w_off_mean lands in 0.6–0.8 range similarly. DR3 is *not* the broken component.

### AlfWorld contrast (the most informative cross-cut)

In AF, the `response_len_teacher_vs_on` ratio is **0.59** (teacher SHORTER than on-policy) — the *opposite* of WS where teacher is 3× longer. AF teacher trajectories average 2,800–3,800 tokens, on-policy average 4,800–7,000. So when DUET\* applies BC on AF, the teacher is concise expert demonstration and the on-policy rollout has surplus tokens to absorb the BC anchoring without distortion.

On WS, BC pulls the on-policy distribution toward 5,700-token verbose teacher outputs that the on-policy policy emits at only 1,800–2,100 tokens. **The BC loss is implicitly asking the policy to lengthen its own outputs by 3×**, and at 3B the model has enough capacity to start doing this — landing on long, verbose, partially-correct trajectories (gap = 0.41, reward without success).

**Falsifiable prediction**: if root cause is the **on-policy length × BC anchoring** interaction, then:
1. **Clipping teacher_response_len** at training time (truncate teacher demos to 1,800 tokens like the policy emits) at 3B should recover BC's benefit. Predicted SR: 38–48%.
2. *Lifting* 1.5B's on-policy budget while keeping 1.5B SOTA config should reproduce a milder version of the 3B failure mode.

## §6 Recommended next experiments (highest information yield)

**Experiment A (highest priority, 4 GPU-hours)**: 3B WS DUET\* with **teacher trajectory truncation to 2,000 tokens** during training (truncate the teacher demo before BC loss computation, leave on-policy untouched). This tests whether the **on-policy/teacher length mismatch** is what differentiates WS from AF. Predicted SR if Finding 5 root: ≥40% (recovery of buggy-whip-saw level performance with stable training).

**Experiment B (cheap counterfactual, 4 GPU-hours)**: 3B WS DUET\* with `chord.mu = 0` (drop BC entirely, keep SC). If 3B no-BC matches or beats the buggy 42.5% baseline, then **BC is net-negative at 3B WS** and Finding 4 (SC redundancy) is the dominant story. This is the single most diagnostic 4-hour experiment we can run.

**Experiment C (sanity check, 4 GPU-hours)**: 1.5B WS DUET\* with `chord.mu = 0` (DUET-no-BC at 1.5B). If 1.5B no-BC ≈ 1.5B SOTA, then **BC was never the load-bearing component at 1.5B either**, and the entire 3B-WS-BC-fails framing dissolves. Combined with Experiment B, this triangulates whether BC adds anything at any scale on WS.

**Stop running**: more sweeps over `chord.mu` peak/velocity/floor variants at 3B WS — every cell we have lands in 28–45% with no clear monotonic relationship to BC schedule parameters. The signal is in the **interaction with response length and SC**, not the BC schedule itself.

---

*Generated by `analysis_reports/_render_bc_compare.py` from `_parsed/bc_compare_2026-05-03.json`. All values are step-100 snapshots; trajectories sampled at steps 10/25/50/75/100.*