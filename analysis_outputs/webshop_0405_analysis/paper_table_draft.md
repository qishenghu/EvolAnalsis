# Paper Table Draft: WebShop Qwen2.5-3B Results

## Main Results Table

| Method | Action Ch. | State Ch. | val@100 Mean | Perfect Rate | Neg Rate | Steps to 0.7 |
|--------|-----------|-----------|-------------|-------------|---------|--------------|
| On-policy GRPO | -- | -- | 0.402 | 2.0% | 17.5% | never |
| CHORD | SFT | -- | -0.100 | 0.0% | 100% | never |
| LUFFY | pi/pi_beta | -- | 0.753 | 49.5% | 2.5% | 63 |
| LUFFY+SC | pi/pi_beta | attr-aware | 0.709 | 32.5% | 6.5% | 35 |
| DUET (AC only) | DR3 | -- | 0.725* | 32.5%* | 1.0%* | 44* |
| **DUET** | **DR3** | **attr-aware** | **0.761** | **49.0%** | **1.0%** | **24** |
| Hybrid | DR3+pi/pi_beta | attr-aware | 0.766 | 53.0% | 3.5% | 24 |

*DUET (AC only) uses old SC (stage-based); needs rerun with 0405 codebase for fair comparison.

## Ablation: SC Match Mode

| SC Match Mode | val@100 Mean | Late-train trend |
|---------------|-------------|-----------------|
| None (LUFFY) | 0.753 | stable |
| stage (old) | 0.221 (collapsed) | DEGRADING |
| attribute_aware (0405) | 0.709 | improving |

## Ablation: DR3 disc_temperature

| disc_temperature | val@100 Mean | disc_acc@100 | Late-train |
|-----------------|-------------|-------------|------------|
| 1.5 (old Hybrid) | 0.512 | 0.886 | DEGRADING |
| 1.0 (0405 Hybrid) | 0.766 | 0.996 | improving |

Note: disc_temperature was changed simultaneously with SC fixes,
so the individual contribution cannot be cleanly isolated.
