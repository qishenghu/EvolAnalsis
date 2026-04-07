# Config Diff: DUET 0407_SC vs LUFFY

These are the EXACT parameters that differ. Everything else is identical.

| Parameter Path | LUFFY | DUET 0407_SC | Impact |
|---------------|-------|-------------|--------|
| `actor.use_dr3` | **false** | **true** | Enables DR3 discriminator-based IS correction |
| `actor.teacher_policy_shaping_enable` | **true** | **false** | LUFFY uses pi/pi_beta shaping |
| `actor.teacher_policy_shaping_mode` | `p_div_p_beta` | (absent) | LUFFY's IS weight formula |
| `actor.teacher_policy_shaping_beta` | `0.1` | (absent) | LUFFY's temperature parameter |
| `actor.dr3` (entire block) | (absent) | Full DR3 config | 30+ DR3 hyperparams |
| `exp_manager.teacher_experience.policy_shaping.enable` | **true** | **false** | LUFFY applies shaping at data level |
| `exp_manager.teacher_experience.policy_shaping.mode` | `p_div_p_beta` | (absent) | |
| `exp_manager.teacher_experience.policy_shaping.beta` | `0.1` | (absent) | |
| `exp_manager.state_channel.enable` | (absent/false) | **true** | SC only in DUET |
| `exp_manager.state_channel.exclude_teacher` | (absent) | **true** | SC bonus on-policy only |
| `exp_manager.state_channel.beta` | (absent) | **0.15** | SC bonus scale |
| `exp_manager.state_channel.match_mode` | (absent) | `attribute_aware` | SC matching strategy |
| `exp_manager.state_channel.progress_agg` | (absent) | `last` | Use last observation for progress |
| `exp_manager.state_channel.grpo_decouple` | (absent) | **true** | SC decoupled from GRPO advantage |
| `exp_manager.state_channel.step_level.enable` | (absent) | **false** | Step-level deltas disabled |

## Summary of DUET Components ON TOP of LUFFY's Approach

LUFFY = Teacher mixing + policy shaping (p/p_beta, beta=0.1)
DUET 0407_SC = LUFFY base - LUFFY policy shaping + DR3 + SC

### What DUET REPLACES from LUFFY:
1. **LUFFY's policy shaping** (teacher_policy_shaping + exp_manager.policy_shaping)
   -> Replaced by DR3's discriminator-based density ratio correction

### What DUET ADDS on top:
1. **DR3 Action Channel** (~30 hyperparams): discriminator, dual ESS clipping, alpha sync, etc.
2. **State Channel**: dense progress bonus for on-policy samples (beta=0.15, attribute_aware)

### The Gap Diagnosis

Since 0407_SC is the closest to LUFFY (42.0% vs 49.5%, gap=-7.5pp),
the remaining gap must come from either:
1. DR3 IS correction being WORSE than LUFFY's policy shaping (likely)
2. SC bonus interfering with learning (possible but SC is decoupled)
3. DR3+SC interaction effects (less likely given SC is decoupled)

To isolate: run LUFFY + SC (no DR3). If it matches LUFFY, DR3 is the culprit.
To confirm: run DUET with LUFFY-style policy shaping instead of DR3.
