# v25 Trajectory Collapse: Behavioral Characterization

**Variant.** WebShop Qwen2.5-1.5B DUET v25 — widens PPO teacher clip `off_cliprange_high: 0.6 → 2.0`, drops the BC anchor that v24 carried. Aggregate metrics: val@50 = 0.523, val@100 = **-0.041**; `response_length/mean` 2081 → 959; `progress_onpolicy_mean` 0.445 → 0.036.

**Data.** `rollout_log/{49,99}.jsonl` (64 trajs each), `validation_log/{50,100}.jsonl` (200 trajs each), contrast against `webshop_qwen1.5b_duet_v12/rollout_log/99.jsonl`.

## 1. Aggregate degradation

| metric | v25 @ step 49 | v25 @ step 99 | v25 val @ 100 | v12 @ step 99 |
| --- | --- | --- | --- | --- |
| mean score | 0.609 | 0.117 | -0.041 | 0.487 |
| frac score ≤ 0 | 6% | 73% | 91.5% | 25% |
| assistant turns ≤ 2 | 3% | 72% | 91% | 19% |
| doubled `<action>` open | 0% | 47% | 69.5% | 9% |
| action body contains fake tags (`<search>`, `<click>`, `<when>`, `<story>`) | 0% | 69% | 84% | 8% |

The collapse is *structural*, not quality: 91% of val@100 trajectories die within 2 turns because the very first assistant emission cannot be parsed by WebShop's action regex. Progress-map drops because the agent never gets past the search home state.

## 2. Step 49 vs step 99 — raw excerpts

**Step 49, traj 0 (score 0.70, clean).** Action body is syntactically valid; thought/act discipline is intact.
```
<think>
The goal is to find compatible apple wearable technology ...
</think>
<action>
search[Apple Watch rose gold 40mm]</action>
```

**Step 99, traj 1 (score -0.10, classic v25 failure).** Doubled `<action>` open, then a CoT-style `<think>` block, then a second nested `<action>` containing the actual query — the outer regex grabs the first `<action>...</action>` whose body is `<action>\n<think>\n...`, which the env rejects.
```
<action>
<action>
<think>
The search query should include keywords that describe the specific details ...
</think>
<action>
search[clothing tops - loose fit - short sleeve - long sleeve - c5-army green - size small - less than 50.00 dollars]
```

**Step 99, traj 2 (score -0.10, invented tags).** Policy invents `<click>` and `<result>` tags that never existed in the system prompt:
```
<result>success</result>
<action>
<click>
search[clothing - loose fit women's tops, tees & blouses ...]
</click></action>
```

**Step 99, traj 8/9/10 (score -0.10, `<search>` tag invented).** Action body is replaced entirely by a fake `<search>...</search>` element:
```
<action>
<search>slim fit, hand wash, machine wash men's tuxedo shirts ... price: less than 50.00 dollars</search></action>
```

**Step 99, traj 11 (score -0.10, `<when>` XML hallucination).** The action body is an invented XSLT-like construct — pure format hallucination:
```
<action>
<when result="search[your query]">
<when result="click[search results]">
<when result="click[Find]">
<when result="click[Accept]">
...
```

**Step 99, traj 14 (recursive doubling).** Same `<action>\n<think>...\n<action>` pattern nested **five** levels deep in a single turn.

**v12 step 99 (same task, `c5-army green small` query).** Clean format:
```
<think>The query should include keywords like 'loose fit women's tops' ...</think>
<action>search[loose fit women's tops tees & blouses c5-army green small less than 50 dollars]</action>
```

## 3. Failure-mode distribution (20 hand-classified v25 step-99 trajs)

Using the user's taxonomy:

| mode | description | count / 20 |
| --- | --- | --- |
| **a** | Malformed action tags (doubled opens, nested fake XML inside `<action>`) | **15** |
| b | Teacher-verbatim reproduction | 0 |
| c | Single-action loop across turns | 2 (both also mode a) |
| d | Empty / whitespace assistant output | 0 (1 fully-empty traj also has no action) |
| e | Hallucinated click target absent from clickables | 0 |
| f | Other | 1 |

Modes b and e are absent. The policy still *intends* semantically reasonable actions (the inner `search[...]` args usually describe the correct product attributes); what broke is the *tag grammar wrapping* those actions.

## 4. Teacher-verbatim check

For the first 20 v25@99 trajs, longest substring match between `assistant[t]` and `user[t-1]` observation was computed over 40-char windows. Only 10.5% of assistant turns reused any 40-char span from the prior observation, and all hits were generic product-description phrases (not teacher hallmark text). **v25 did not overfit to teacher-token reproduction.** The teacher trajectories are clean `<think>…</think>\n<action>search[…]</action>` format — if the student were imitating them, we would see *fewer* invented tags, not more.

## 5. Diagnosis

This is **not** widened-clip → teacher overfitting. It is widened-clip → **PPO step instability on on-policy tokens** that corrupted the format distribution:

- Raising `off_cliprange_high` from 0.6 to 2.0 tripled the allowed negative surprise bound on teacher samples, meaning tokens where the teacher was much more confident than the current policy can now contribute gradient up to 3× larger than under v24's clip. Teacher samples in this run have small `w_hat` under DR3, so their raw weight is already small — the clip rarely bound them before.
- But the widened clip also widened the asymmetric update for on-policy samples where the behaviour policy (old actor) had very low probability on a token that the current actor is now raising. Without BC as a regularizer (v24 had BC; v25 does not), the only force keeping the format tokens (`<action>`, `</action>`) near the SFT prior was the frozen reference KL — which by step ~60–70 had drifted enough that `<action>` token log-probs could be updated by large PPO steps.
- The symptom profile — doubled opening tags, recursive nesting, replacement of `<action>` by `<search>/<click>/<when>` — is the signature of the **`<action>` open-tag token drifting off the prior while `<think>` and `<action>` boilerplate gets re-sampled twice per turn**. Turn length halved (2081 → 959) precisely because the env now truncates after the first malformed turn (usually the second turn) instead of running to completion at 5–8 turns. Entropy proxy: at step 49 the 64 trajs had mean 3.48 action tags, at step 99 only 2.70, and of those "actions" 69% are syntactically invalid — so valid action throughput is ~0.8 per traj, a 4× drop.

So the mechanism is **policy-format drift driven by unbounded PPO updates on low-probability tokens**, not teacher imitation. The actual WebShop *semantic* reasoning (which product, which attribute, which price) is still largely correct inside the garbage wrapper — the model still writes `search[loose fit women's tops ... c5-army green small]` inside an invalid tag.

## 6. Verdict for the paper

**BC's role in v24 is stability, not support lift.** v25 proves that the widened teacher clip can admit teacher gradient magnitude in principle — but without BC as a format anchor, the on-policy PPO step itself destroys the action-tag grammar within 50 updates. The support-lift hypothesis (BC expands reachable action distribution) is refuted by v25: the teacher was reached through the wider clip, and still the policy collapsed. What BC actually provides is a per-token lower bound on format-token log-probabilities that prevents PPO from moving `<action>` boundary tokens off their SFT values. Drop that anchor and the off-policy budget becomes self-destructive.

One-liner: *BC in DUET is not a path to expert support; it is the scaffolding that keeps PPO from eating the action grammar.*
