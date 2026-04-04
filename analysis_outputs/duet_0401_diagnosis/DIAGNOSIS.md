# DUET 0401 Diagnosis: Why It Underperforms

## Summary
DUET 0401 (std floor fix + stage SC) achieves **18% success** on WebShop validation, vs **32.5% for original DUET** and **49.5% for LUFFY**. Despite fixing two real problems (advantage explosion + SC coverage), the fix created a new one: **premature teacher fade-out**.

## Validation Scores

| Method | Step 50 | Step 100 |
|--------|---------|----------|
| DUET 0401 (std floor + stage SC) | reward=0.517, success=**12%** | reward=0.565, success=**18%** |
| DUET orig (hash SC, no std floor) | reward=0.599, success=**22.5%** | reward=0.725, success=**32.5%** |
| LUFFY | reward=0.509, success=**8.5%** | reward=0.753, success=**49.5%** |

## Config Difference
Only ONE config change between 0401 and orig: `state_channel.match_mode: "stage"` (0401) vs `"hash"` (orig).
The std floor fix is a CODE change that affects both runs (but orig ran on old code without it).

## What the Std Floor Fix Fixed
Original DUET had **catastrophic advantage explosion**:
- Teacher advantages: 0.24 → **4,840** (step 99)
- PG loss: -0.16 → **-7,154** (step 99)
- Teacher gradient share oscillates wildly: 0.21 → 1.00 → 0.12 → 1.00

DUET 0401 has stable, well-behaved training:
- Teacher advantages: 0.40 → 0.08 (reasonable)
- PG loss: -0.75 → -0.11 (normal)
- Teacher gradient share smoothly decays: 0.35 → 0.06

## What Stage SC Fixed
Original DUET hash-based SC was **completely broken** on WebShop:
- Coverage: 10.9-12.5% (tiny)
- SC bonus: **exactly 0.0** throughout
- Progress (on-policy): **0.0** throughout

Stage SC in 0401 works:
- Coverage: **87.5%**
- SC bonus: 0.036-0.067 (3-7% of reward)
- Progress (on-policy): 0.20-0.38

## ROOT CAUSE: Premature Teacher Fade-Out

The std floor fix made DR3 work correctly. But "correctly" means the discriminator learns to separate teacher/policy distributions quickly (disc_acc → 87%), which drives teacher gradient share down fast:

| Step | DUET 0401 | DUET orig | LUFFY |
|------|-----------|-----------|-------|
| 1    | 35.4%     | 21.1%     | 37.2% |
| 25   | **16.8%** | 100%*     | 100%* |
| 50   | **7.6%**  | 11.7%     | 100%* |
| 99   | **6.1%**  | 100%*     | **58%** |

*\*Oscillating wildly in orig; sustained in LUFFY*

**At only 16% success rate, DUET 0401 has already reduced teacher influence to 6%.** Teachers are fading before the policy has learned from them.

**LUFFY maintains 58% teacher gradient share at step 99** and reaches 35% success. The persistent teacher signal is critical for WebShop.

## Why DUET Orig Still Beats 0401 Despite Instability
DUET orig's wild oscillations (teacher_gradient_share bouncing between 0 and 1.0) inadvertently provided **periodic bursts of full teacher signal**. Despite the training being unstable (advantages exploding to 4800+), these bursts pushed the policy to learn something before collapsing. Its training success peaked at **41% at step 50** before crashing.

## SC Bonus Is Not Compensating
SC provides ~5-9% bonus relative to task reward. This gives useful dense reward shaping but **cannot replace the direct gradient contribution from teacher samples**. The fundamental issue is DR3 fading teachers too fast, and SC doesn't have enough magnitude to compensate.

## Hypotheses for Why DR3 Fades Too Fast
1. **Discriminator overfits to surface features**: Teacher demos (72B model) have different formatting/length from 3B policy outputs. The discriminator can distinguish them without seeing quality differences.
2. **Data ratio asymmetry**: 12.5% teacher vs 87.5% on-policy. DR3 density ratios are inherently high, pushing teacher weight down.
3. **WebShop has harder exploration**: Unlike ALFWorld (discrete tasks), WebShop requires precise product matching. The policy needs more teacher guidance for longer.

## Recommended Next Steps
1. **Slow down DR3 fade-out**: Add a minimum teacher gradient share floor (e.g., `min_teacher_share: 0.15`) or reduce discriminator learning rate
2. **Test DR3-free variant**: DUET with just SC (no DR3) to isolate SC's contribution
3. **Increase teacher ratio**: n_teacher=2 per task group (25% instead of 12.5%) to give DR3 more teacher data
4. **Warmup DR3**: Delay DR3 activation until step 20-30, letting teachers contribute fully during early training
