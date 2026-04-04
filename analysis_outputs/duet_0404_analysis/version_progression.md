# DUET Version Progression Analysis

## Validation Scores

| Version | Val@50 Reward | Val@50 Success | Val@100 Reward | Val@100 Success |
|---------|--------------|----------------|----------------|-----------------|
| GRPO | 0.276 | 1.0% | 0.402 | 2.0% |
| LUFFY **BEST** | 0.509 | 8.5% | 0.753 | 49.5% |
| DUET_orig | 0.599 | 22.5% | 0.725 | 32.5% |
| DUET_0401 | 0.517 | 12.0% | 0.565 | 18.0% |
| DUET_0402 | 0.483 | 6.5% | 0.735 | 35.5% |
| DUET_0403 | 0.646 | 30.5% | 0.679 | 33.0% |
| DUET_0404 | 0.497 | 2.0% | 0.646 | 23.5% |

## Version Changes and Effects

| Version | Key Changes | Val@100 Success | Outcome |
|---------|-------------|-----------------|---------|
| DUET_orig | Baseline DUET | 32.5% | Below LUFFY 49.5% |
| DUET_0401 | Unknown changes | 18.0% | REGRESSION |
| DUET_0402 | disc_temp=2.5, gap_gate ON | 35.5% | Best DUET so far |
| DUET_0403 | SC decouple, adv clip, gap_gate OFF, disc_temp=2.5 | 33.0% | Slight regression from 0402 |
| DUET_0404 | disc_temp=1.5, gap_gate OFF, SC decouple, adv clip | 23.5% | WORST since 0401 |

## Training Dynamics Comparison

### LUFFY
- Peak success: 0.7895 @ step 80
- Final success: 0.3509
- Final 5-step avg: 0.4502
- Total steps: 99

### DUET_orig
- Peak success: 0.5965 @ step 90
- Final success: 0.2321
- Final 5-step avg: 0.3506
- Total steps: 98

### DUET_0402
- Peak success: 0.5965 @ step 90
- Final success: 0.3036
- Final 5-step avg: 0.4277
- Total steps: 98

### DUET_0403
- Peak success: 0.8070 @ step 80
- Final success: 0.1429
- Final 5-step avg: 0.2383
- Total steps: 98

### DUET_0404
- Peak success: 0.8070 @ step 80
- Final success: 0.1429
- Final 5-step avg: 0.2383
- Total steps: 98
