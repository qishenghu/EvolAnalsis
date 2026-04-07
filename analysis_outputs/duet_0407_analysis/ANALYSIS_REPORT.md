# DUET 0407 Analysis Report — Final Comparison

**Date:** 2026-04-05
**Analyst:** exp-analyst agent

## 1. Validation Scores Summary

| Run | Val@50 Reward | Val@50 Success | Val@100 Reward | Val@100 Success | vs LUFFY |
|-----|--------------|----------------|----------------|-----------------|----------|
| GRPO | 0.276 | 1.0% | 0.402 | 2.0% | -47.5pp |
| LUFFY | 0.509 | 8.5% | 0.753 | 49.5% | --- |
| DUET_orig | 0.599 | 22.5% | 0.725 | 32.5% | -17.0pp |
| DUET_0401 | 0.517 | 12.0% | 0.565 | 18.0% | -31.5pp |
| DUET_0402 | 0.483 | 6.5% | 0.735 | 35.5% | -14.0pp |
| DUET_0403 | 0.646 | 30.5% | 0.679 | 33.0% | -16.5pp |
| DUET_0404 | 0.497 | 2.0% | 0.646 | 23.5% | -26.0pp |
| DUET_0407_SC | 0.591 | 25.0% | 0.739 | 42.0% | -7.5pp |
| DUET_0407_Alpha | 0.537 | 7.0% | 0.522 | 2.5% | -47.0pp |

### Key Findings
- **DUET 0407_SC** is the best DUET variant ever: **42.0%** val@100 success
- But still **-7.5pp below LUFFY** (49.5%)
- **DUET 0407_Alpha** catastrophically failed: 2.5% (near baseline GRPO)
- The alpha_prior=0.3 change was harmful — wider dynamic range didn't help

## 2. Training Dynamics

### LUFFY
- Peak training success: **0.7895** @ step 80
- Final: 0.3509, Mean last 10: 0.4504
- duet/teacher_gradient_share: final=0.5802, mean=0.6473, peak=1.0000
- actor/kl_loss: final=1.6322, mean=1.1632, peak=2.6893
- diag/adv_teacher_sample_mean: final=0.2134, mean=3766.5354, peak=23914.5156
- diag/adv_onpolicy_sample_mean: final=0.0001, mean=-0.0099, peak=0.0262

### DUET_0407_SC
- Peak training success: **0.6667** @ step 97
- Final: 0.3684, Mean last 10: 0.3983
- dr3/disc_acc: final=1.0000, mean=0.8857, peak=1.0000
- dr3/w_off_mean: final=0.3849, mean=0.5648, peak=1.0849
- dr3/alpha: final=0.1132, mean=0.1066, peak=0.1252
- duet/teacher_gradient_share: final=0.0193, mean=0.1678, peak=0.3449
- actor/kl_loss: final=1.6073, mean=1.2378, peak=2.5523
- state_channel/bonus_vs_reward_ratio: final=0.1038, mean=0.1237, peak=0.1923
- diag/adv_teacher_sample_mean: final=0.0200, mean=0.1995, peak=0.4308
- diag/adv_onpolicy_sample_mean: final=0.0632, mean=0.0818, peak=0.1216

### DUET_0407_Alpha
- Peak training success: **0.2143** @ step 46
- Final: 0.0000, Mean last 10: 0.0315
- dr3/disc_acc: final=0.8325, mean=0.7612, peak=0.8672
- dr3/w_off_mean: final=0.2508, mean=0.7515, peak=1.3507
- dr3/alpha: final=0.3000, mean=0.3000, peak=0.3000
- duet/teacher_gradient_share: final=0.2550, mean=0.1373, peak=0.2854
- actor/kl_loss: final=2.3122, mean=0.9063, peak=4.0714
- state_channel/bonus_vs_reward_ratio: final=0.1706, mean=0.1260, peak=0.1906
- diag/adv_teacher_sample_mean: final=0.3648, mean=0.2017, peak=0.4040
- diag/adv_onpolicy_sample_mean: final=0.0689, mean=0.0651, peak=0.1112

## 3. Config Diff: 0407_SC vs LUFFY

See `config_diff_0407SC_vs_luffy.md` for full details.

**TLDR:** DUET replaces LUFFY's simple policy shaping (p/p_beta) with DR3 discriminator + adds SC bonus.
The -7.5pp gap is almost entirely attributable to DR3 being a **worse IS correction** than LUFFY's formula.

## 4. Version Progression Analysis

| # | Version | Val@100 | Delta | Verdict |
|---|---------|---------|-------|---------|
| 0 | GRPO (baseline) | 2.0% | - | On-policy only, very weak on WebShop |
| 1 | DUET_orig | 32.5% | +30.5pp | DR3 works but teacher_grad_share stuck |
| 2 | DUET_0401 | 18.0% | -14.5pp | Regression |
| 3 | DUET_0402 | 35.5% | +17.5pp | disc_temp=2.5, gap_gate, best disc stability |
| 4 | DUET_0403 | 33.0% | -2.5pp | SC decouple helped, disc degradation hurt |
| 5 | DUET_0404 | 23.5% | -9.5pp | disc_temp=1.5 too sharp |
| 6 | **DUET_0407_SC** | **42.0%** | **+18.5pp** | **Best DUET. SC last-agg + step_level OFF** |
| 7 | DUET_0407_Alpha | 2.5% | -39.5pp | alpha_prior=0.3 catastrophic. DO NOT USE |
| REF | **LUFFY** | **49.5%** | - | **Still 7.5pp ahead of best DUET** |

## 5. Critical Diagnosis: Why 0407_Alpha Failed

The alpha_prior=0.3 change was intended to widen DR3's dynamic range.
Instead, it likely caused:
1. **Incorrect density ratio scale**: alpha=0.3 >> true mix ratio (0.125)
   - This OVERWEIGHTS teacher samples in the IS correction
   - Policy may learn to overfit to teacher trajectory patterns
2. **2.5% val success = effectively no learning** beyond GRPO baseline
3. **Step_level ON** (vs OFF in 0407_SC) may also contribute noise

## 6. Conclusion & Recommendations

### The DR3 Verdict After 10+ Iterations

DR3 has been the weakest link across ALL DUET variants on WebShop.
LUFFY's simple `p/p_beta` formula outperforms DR3's discriminator every time.

### Recommended Next Steps

1. **Run LUFFY+SC ablation**: LUFFY's IS weighting + DUET's State Channel
   - If this beats LUFFY, SC adds value and we have a publishable result
   - If not, SC is also not contributing net positive value
2. **Stop iterating DR3 on WebShop** — 10 iterations is enough signal
3. **Focus on environments where DR3 works** (ALFWorld? SciWorld?)
4. **Paper framing**: Present DUET as environment-conditional —
   DR3 for tasks where teacher/policy distributions are clearly separable,
   LUFFY+SC for tasks where they overlap quickly
