# Qwen3.5 Agentic RL Context Management

Status: exact-snapshot and the configured 32K/10K contract are implemented; the WebShop h15 serial context A/B passes its trace-identity gate and exercises natural compaction plus exact 10K completions; a repaired WebShop v7 canary completed one real update, one zero-advantage skip, paired 32-task validation, and final checkpoint, while sustained GRPO stability, ALFWorld, and the exact-32K GPU update boundary remain pending  
Last updated: 2026-08-01  
Scope: Qwen3.5-4B/2B on ALFWorld and WebShop, first with pure GRPO, then DUET with DeepSeek-v4-flash teacher data

## Executive decision

The first context mechanism should not be an LLM summary. It should be a deterministic, token-budgeted renderer over an immutable event log:

1. always retain the task goal/instructions and current observation losslessly;
2. retain recent whole turns;
3. represent older turns as action plus a bounded observation record;
4. evict oldest whole turns when the prompt budget is reached;
5. freeze the exact prompt IDs, sampled completion IDs, and rollout log-probabilities at every decision;
6. train only a decision under the exact condition that produced it.

The key choice is item 6. The earlier implementation compressed the final transcript after sampling, then trained old actions under newly rewritten prefixes. Tuning the reasoning budget `B` cannot correct that conditional-distribution error.

The initial implementation uses one token-weighted decision snapshot per trajectory. GRPO grouping and reward remain at full-trajectory level. This is a low-memory bridge to later segmented/all-turn training and is strictly more faithful than scoring every historical action in one post-hoc transcript.

The production length contract is now fixed:

```text
configured max_model_len   32,768 tokens
maximum rollout prompt    22,528 tokens
maximum one-turn response 10,240 tokens
                         -------
total                     32,768 tokens
```

The 10,240-token number is the reserve for the **current sampled completion**, including its sampled reasoning and action. It is a per-environment-decision maximum, not a total-trajectory or historical-reasoning budget. The native generation template's forced `<think>\n` opening belongs to the prompt and is counted inside the 22,528-token prompt budget; the 10,240 limit applies to token IDs actually returned by the server. The native Qwen3.5 chat template removes `<think>...</think>` from prior assistant turns before rendering the next prompt, and exact-snapshot training gives loss only to the selected current completion. Consequently the baseline uses `reasoning_history_tokens: 0`: historical assistant state is canonical action text, while the selected response retains all sampled reasoning tokens.

This is an implementation and data-contract decision, not yet a GRPO-stability conclusion. The historical WebShop v6 path exercised two rollout/scoring/update/reload cycles, but its three-decision horizon could not activate the four-recent-turn compactor. A matched 200-task initial-checksum baseline is now available: mean reward moved from `0.00644444` to `-0.00180556`, with paired delta `-0.00825`, bootstrap 95% CI `[-0.02025, -0.001]`, and improved/equal/degraded counts `10/166/24`. This establishes degradation of the **old v6 update path**; it does not test the subsequently repaired path. The repaired v7 canary below completed safely and had a small paired 32-task delta whose interval crosses zero, but it contains only one actual optimizer step. The h15 v3 serial validation activates context management and passes a causal trace-identity check, but is validation-only. Neither can establish sustained training stability or benchmark-level non-degradation.

## Repository diagnosis

### Root cause: rollout and training conditions differed

The failed context implementation mutated earlier `assistant` and `user` messages after later environment steps:

- old assistant reasoning was replaced with action-only text;
- old observations were stripped or character-truncated;
- the corresponding `token_arr` was regenerated;
- all historical assistant tokens still received training loss in the final monolithic sample.

For an action sampled as

\[
a_t \sim \pi_{\theta_b}(\cdot \mid C_t),
\]

the learner therefore evaluated it under a different, retrospectively produced prefix \(\tilde C_t\). This is neither ordinary on-policy GRPO nor a controlled off-policy correction.

A production-tokenizer replay with `/data/shared_models/Qwen3.5-4B-thinkraw`, six turns, sliding window 3, and reasoning budget 1024 found:

- exact rollout prompt lengths: `[33, 208, 383, 558, 643, 728]`;
- reconstructed training-prefix lengths: `[33, 118, 203, 288, 463, 638]`;
- exact token identity: **0/6 turns matched**.

Turn 0 had the same length but different IDs because the forced Qwen thinking header and sampled reasoning were later rendered as a plain history assistant turn. Thus length equality was not sufficient evidence.

### Independent contaminants in the old runs

The previous ALFWorld/WebShop curves cannot be used to estimate a context effect because they combine several independent failures:

| Failure | Consequence |
|---|---|
| rollout allowed 10,240 response tokens while training allowed 8,192 | terminal action could be removed while terminal reward was retained |
| response overflow was silently prefix-truncated twice | corrupted reward-to-token attribution |
| empty completion was rewritten to literal `im_end` | fabricated assistant turns entered trajectories |
| outer rollout resubmission was unbounded | dead vLLM services could cause effectively infinite retries |
| vLLM log-probabilities were not placed in `DataProto` | behavior/actor consistency code never ran |
| context safety check ignored its input and counted raw history | the checked token view differed from the served token view |
| ALFWorld stopped at step 23 and WebShop at step 4 after rollout-server failures | neither run reached validation |

The `B=0` collapse, `B=-1` OOM, and `B=1024` partial run are therefore confounded endpoints, not a valid compression ablation. They also optimized a different object: the legacy monolithic sample assigned loss to historical actions, whereas snapshot training masks the prompt and trains only the selected turn's exact sampled completion. Those results cannot justify setting the new 10,240-token response reserve as a historical-`think` budget.

## What the literature implies

The most relevant recent work converges on two principles: acting must use a bounded reconstructed state, and learning must explicitly understand segmentation/reconstruction.

- [ReSum](https://arxiv.org/abs/2509.13313) reports that an ordinary agent is not automatically aligned to summary-conditioned reasoning. Its GRPO variant broadcasts trajectory advantage across explicitly segmented trajectories. This argues against adding an external DeepSeek summary to inference while training a reconstructed final transcript.
- [CompactionRL](https://arxiv.org/abs/2607.05378) jointly trains execution and summary generation with token-level normalization and cross-trajectory advantage estimation. It supports learned compaction as a later algorithmic contribution, not as an untrained preprocessing trick.
- [ECHO](https://arxiv.org/abs/2606.31650) keeps source-indexed memory records so reconstructed context retains provenance for credit assignment. This motivates keeping the raw State Channel view and recording which turns a renderer selected.
- [ZipRL](https://arxiv.org/abs/2605.28069) uses non-uniform compression and modifies GRPO credit through hindsight replay. It is evidence that adaptive compression is coupled to the learning objective rather than a safe drop-in string transform.
- [ACON](https://arxiv.org/abs/2510.00615) optimizes benchmark-specific compression guidelines from full-success/compressed-failure pairs and reports 26--54% peak-token reduction. Its thresholds and guidelines are task-specific; they should not be copied directly to ALFWorld/WebShop.
- [Context-lite / Verlog](https://openreview.net/forum?id=6CE5PLsZdW) demonstrates customizable bounded history in multi-turn RL, but its published formulation uses GAE and explicitly does not establish compatibility with value-free GRPO. Its memory idea is useful; its objective is not a direct implementation recipe here.
- [From History to State](https://arxiv.org/abs/2605.05413) uses deterministic state tracking on ALFWorld, WebShop, and SciWorld and reports 2--7x prompt-token reduction. This supports a benchmark-specific state ledger as the next step after the exact-snapshot baseline.
- [On Effectiveness and Efficiency of Agentic Tool-calling and RL Training](https://arxiv.org/abs/2606.00135) finds multi-turn results sensitive to templates and how prior reasoning/history is carried forward. This directly supports hashing and persisting the template and rendered prompt.

### Designs rejected for the first gate

- **Post-hoc monolithic transcript:** cannot represent the different context used at each action.
- **Character slicing:** does not respect token budgets or semantic/turn boundaries.
- **Global left/right truncation:** can remove goals, current action sets, or terminal actions.
- **Online DeepSeek summaries:** add cost, nondeterminism, latency, and an untrained summary-conditioned distribution shift.
- **Token-importance pruning:** harder to audit and can split structured observations or action markup.
- **Full-context teacher, compressed-context student:** teacher actions are conditioned on information unavailable to the student.
- **A universal ALFWorld/WebShop compression threshold:** the two environments place critical state in different structures.

## Correctness contract

The raw trajectory is append-only. Context management produces a per-decision view and never edits the event log.

For every accepted decision `t`:

```text
hash(ids actually sent to rollout server)
    == stored DecisionSnapshot.prompt_hash
    == hash(training prompt_ids)

server sampled token IDs
    == DecisionSnapshot.completion_token_ids
    == training response_ids

len(server token logprobs)
    == len(training response_ids)

vLLM processed logprob(sampled token | exact prompt, sampling processors)
    == stored rollout_log_probs
    == PPO old-policy denominator on valid rollout tokens

actor full-sequence recomputation with the same temperature/processors
    is an independent numerical-drift audit, not a replacement denominator
```

The token and conditioning identities above are exact. The last comparison is deliberately not written as numerical equality: Qwen3.5's BF16 Gated DeltaNet takes a recurrent cached-decode path in vLLM and a full teacher-forced path in the FSDP actor, and those two execution paths have a measurable numerical floor even with identical weights and token IDs. Define

\[
\delta_i = \log \pi_{\text{actor, full}}(y_i) -
            \log \pi_{\text{vLLM, cached}}(y_i).
\]

The hard safety gate checks the distribution and tail of `δ`; PPO still uses the actual vLLM value that sampled each token.

Additional hard invariants:

- no response truncation after reward is known;
- no current-observation truncation in the correctness baseline;
- goal/instructions and current legal actions/clickables are retained;
- prompt length is checked on the exact rendered IDs sent to the model;
- empty completions and missing token IDs/log-probabilities fail the trajectory;
- trajectory/server retries are finite;
- every rollout server must pass health, weight reload, and prefix-cache reset;
- all external rollout replicas must report a non-empty, identical parameter-checksum fingerprint after each reload;
- parameter checksums are queried only while the servers are awake inside the ordered reload transaction; an out-of-band checksum probe against a level-1 sleeping server is prohibited;
- exact rollout-logprob presence and token alignment are hard requirements, and calibrated rollout/actor numerical drift is a hard safety gate rather than a warning that may be ignored;
- actor and reference use the same padding, temperature-order, and precision contract for KL computation;
- raw observations remain available to DUET's State Channel;
- context/template hashes are saved with each selected sample.

## Current 32K baseline design

### Rendering

`StructuredContextPolicy` constructs a prompt from raw `ExtendedMessage.content`:

```text
protected initialization (system + task goal/current initial state)
        |
older whole turns: action-only + compact observation
        |
recent whole turns: action-only + full observation
        |
current observation: full, never silently cropped
        |
Qwen3.5 generation prompt: forced native <think> prefix
```

Old turns are removed oldest-first only when the exact chat-template rendering exceeds the prompt budget. Observations are clipped by tokenizer IDs, not characters. The result records raw/managed token counts, compressed/dropped turns, and observation clipping counts.

The current baseline removes historical hidden reasoning. This is not merely a compression preference. In the native Qwen3.5 template, every prior assistant message is parsed at `</think>` and only the post-reasoning content is emitted before the latest user query. A tokenizer check with 500 repeated private-reasoning fragments produced exactly the same next-turn prompt IDs as action-only history. Thus `reasoning_history_tokens: 10240` would be a misleading no-op for model-facing IDs while retaining large strings in snapshots and logs. The fixed setting is `reasoning_history_tokens: 0`.

This is also consistent with the snapshot objective. The selected current response contains the full sampled reasoning plus action and receives the response loss. Historical assistant turns are prompt tokens with zero loss. The legacy concern that an old action was trained after deleting the reasoning that produced it does not apply to this sample construction.

### Immutable decision training

At each model call, `DecisionSnapshot` freezes:

- exact prompt messages and prompt token IDs;
- SHA-256 prompt and template hashes;
- raw sampled completion token IDs;
- one vLLM log-probability per sampled token;
- unmodified sampled content and stop/finish reasons;
- context statistics for that decision.

The final trajectory reward is copied to the selected decision sample. `data_id` remains the GRPO group ID, so every rollout for one task still shares the intended trajectory-level relative-advantage group regardless of the smoke run's configured group size.

### Why one token-weighted turn

Let trajectory `τ` contain decision completions with lengths \(L_t\), total \(L=\sum_tL_t\), and token losses \(\ell_{t,i}\). The desired trajectory token mean is

\[
\bar\ell_\tau = \frac{1}{L}\sum_t\sum_{i=1}^{L_t}\ell_{t,i}.
\]

If turn `t` is selected with probability \(L_t/L\), then

\[
\mathbb{E}_t\left[\frac{1}{L_t}\sum_i\ell_{t,i}\right]
= \bar\ell_\tau.
\]

The implementation derives a stable pseudo-random target from the run seed, task ID, data ID, and rollout ID. This makes A/B runs reproducible and keeps selection independent of completion content. The estimator uses less memory than all-turn segmented training, at the cost of higher variance. Once the snapshot baseline is stable, all decision snapshots can be trained with explicit per-trajectory normalization.

### Behavior-policy log-probabilities

[vLLM 0.19.1's serving contract](https://docs.vllm.ai/en/v0.19.1/cli/serve/#--logprobs-mode) defines two relevant modes:

- `raw_logprobs` returns probabilities before sampling processors. It is vLLM's default and therefore excludes temperature, top-k, and top-p transformations.
- `processed_logprobs` returns the distribution after those processors, which is the distribution that actually sampled the token.

The FSDP actor recomputation must apply the same temperature transform as the sampling server. The earlier external servers were launched without an explicit mode, so vLLM returned its default raw values while the actor returned temperature-processed values. This produced systematic absolute log-probability drift: ALFWorld mean/max `0.047/0.862` at temperature 0.9, and WebShop `0.084/2.858` at temperature 0.6 in the first snapshot gate. The larger WebShop error is consistent with the stronger temperature transform; it is not evidence that prompt snapshotting failed.

Changing the servers to `--logprobs-mode processed_logprobs` removed that semantic mismatch, but did **not** make recurrent decode and full teacher forcing numerically identical. The first two residual 32K gates recorded:

| Lane | Temperature | Mean absolute logprob drift | Maximum absolute drift | Gate result |
|---|---:|---:|---:|---|
| ALFWorld | 0.9 | 0.022094 | 0.375049 | failed 0.01 / 0.25 limits |
| WebShop | 0.6 | 0.018737 | 0.395515 | failed 0.01 / 0.25 limits |

The corresponding logs are `logs/alfworld_qwen35_4b_grpo_snapshot_32k_gate.log` and `logs/webshop_qwen35_4b_grpo_snapshot_32k_gate.log`. These runs correctly failed their then-configured `0.01/0.25` mean/max limits. Inspection first exposed and fixed one real actor-side numerical mismatch:

```text
vLLM processed_logprobs:
    bf16 logits -> promote to fp32 -> divide by temperature -> log_softmax

upstream verl actor:
    bf16 logits -> in-place bf16 divide by temperature -> log_softmax
```

The second form rounds every scaled logit in bf16 before normalization. With Qwen3.5's 248,320-token vocabulary, that rounding is visible through the vocabulary-wide normalizer. The FP32-temperature scorer below fixes this ordering error. It does not, however, remove the separate numerical difference between Qwen3.5 GDN cached decode and full-sequence teacher forcing.

#### Fixed-token replay evidence

The v5 failure artifact allowed the remaining hypotheses to be separated on identical token IDs:

| Comparison | Valid tokens | Mean absolute drift | p99 | Maximum | Interpretation |
|---|---:|---:|---:|---:|---|
| FSDP actor scorer vs corrected HF full-forward replay, processed at T=0.6 | 742 | `0.99--1.06e-6` per sample | -- | `<=3.6e-6` | actor scorer/formula matches HF full forward |
| vLLM teacher-forced raw prompt logprobs vs HF full-forward raw logprobs | 742 | `0.009875` | `0.10879` | `0.24608` | cross-stack full-forward numerical drift |
| same compiled vLLM, cached decode vs teacher-forced prefill, T=1 identity processors | 512 | `0.009214` | `0.107996` | `0.121375` | same-stack recurrent cached/full drift |
| same HF stack, tokenwise cache vs full forward, T=1 | 487 | `0.00694` | `0.08579` | `0.11296` | cached/full drift exists without vLLM |

The first offline HF replay had accidentally converted FP32 RoPE buffers to BF16; the figures above are from the corrected replay, which casts parameters to BF16 but preserves buffer dtypes. Batch balancing was also ruled out: every tensor was reordered together, the aligned-row error was about `0.012--0.014`, wrong-row comparisons were `0.457--0.610`, and one-token offsets were `0.377--0.475`. Replaying the same sample on all four rollout ports produced bitwise-identical vLLM prompt logprobs, ruling out replica, weight, and cache variance for that request.

vLLM 0.19.1 prompt logprobs are raw even when decode returns processed logprobs, so they cannot be compared directly with a T=0.6 processed rollout scalar. The cached/prefill diagnostic therefore used T=1, `top_p=1`, `top_k=0`, disabled penalties, `min_tokens=0`, and `ignore_eos=true`. Setting `min_tokens=128` would have masked EOS and invalidated the comparison.

Backend variants did not reduce the floor: disabling packed recurrent decode gave mean/p99/max `0.009726/0.10999/0.2032`; eager packed decode gave `0.009646/0.11103/0.1283` and was slower. FlashInfer GDN prefill is unsupported on A100 in this vLLM build and falls back to Triton. The production choice is therefore compiled mode, Triton GDN prefill, and packed recurrent decode enabled. These measurements justify a bounded drift gate; they do not prove equivalence on arbitrary 32K inputs.

The server contract still requires `--logprobs-mode processed_logprobs`. Rollout sampling keeps `top_k=-1` and `top_p=1`, so actor and vLLM apply the same effective processor: temperature. If either backend later adds a penalty, top-k/top-p filter, or other processor, the actor recomputation must add the same transform before the gate can pass.

### Exact FP32-temperature scorer and Qwen3.5 padding

The Qwen3.5 FSDP scorer now follows the vLLM numerical order without materializing full-sequence fp32 vocabulary logits. On CUDA, sampled-token logprobs use the FlashAttention Triton cross-entropy kernel with `logit_scale=1 / temperature`; the kernel loads logits into fp32 before applying that scale. Entropy, when requested, is computed in small fp32 chunks. The model is asked for only the final `valid_response_length + 1` logits: the last prompt position plus the valid response positions needed for the causal next-token labels.

Qwen3.5 adds a separate Gated DeltaNet padding constraint. Its hybrid recurrent state must see the same unpadded token stream that vLLM served. The upstream Qwen3.5 padding helper can skip masking for micro-batch size `B=1`, while the training batch still contains left-padded prompts and right-padded responses. Nonzero pad embeddings before the real prompt can therefore alter the recurrent state even though those positions have zero loss.

For the exact scorer, the Qwen3.5 gate consequently uses `model.use_remove_padding: false` and scorer micro-batch size 1 rather than verl's rmpad/Ulysses path. Each micro-forward:

1. requires a contiguous valid response-token prefix;
2. removes response-side right padding;
3. crops prompt-side left padding from `input_ids`, `attention_mask`, and `position_ids`;
4. asserts that no padding remains before the model forward;
5. scores the valid response, then right-pads logprobs and entropy back to the batch tensor shape.

This crop is an exact representation change for one sample, not context truncation: no valid prompt or response token is removed. It also avoids evaluating a short sample at another sample's padded 32K length. Current unit tests cover the fp32 formula, chunked entropy, left-crop preservation, response right-padding restoration, causal label alignment, and rejection of unsupported `B>1` or rmpad configurations. They do not substitute for the pending GPU exact-32K boundary test or an ALFWorld natural gate.

### Actor/reference symmetry

The numerical scoring contract applies to both sides of KL, not only to the trainable actor. For the current FSDP gates:

- `Role.RefPolicy` maps to the same HET FSDP worker family as `Role.ActorRollout`;
- actor and standalone reference both instantiate `HETDataParallelPPOActor`;
- `ref.behavior_logprob_fp32_temperature` is interpolated from the actor setting;
- static validation requires actor/ref scorer flags to be equal;
- both sides use the same `B=1` padding crop and fp32-temperature operator.

Actor and reference logprobs need not remain numerically equal after the actor updates—the models then have different parameters—but they must be computed under the same token, padding, temperature, and precision semantics. This symmetric contract is currently scoped to the FSDP Qwen3.5 gate; it should not be generalized to an untested Megatron or DR3-hidden path.

On valid sampled tokens, PPO uses the stored rollout value as `old_log_probs` (`use_rollout_log_probs_as_old: true`). Actor recomputation remains an independent drift audit. The calibrated WebShop v6/ALFWorld v3 safety gate is:

| Metric over valid sampled tokens | Limit |
|---|---:|
| mean `abs(δ)` | `0.02` |
| p99 `abs(δ)` | `0.25` |
| fraction of `exp(δ)` outside the configured PPO interval `[0.8, 1.28]` | `0.01` |
| catastrophic maximum `abs(δ)` | `0.75` |

All four checks are enabled in both production YAMLs; equality with a limit passes, and a negative threshold would disable that check. The trainer reports every threshold violation and atomically writes the aligned tensors and threshold metadata before raising, so no optimizer update occurs on a failed batch. Missing, misaligned, or empty rollout log-probabilities are unconditional errors. The finite-value gate is also fail-closed: any NaN/Inf in valid rollout/current log-probabilities, token deltas, ratios, or aggregate drift metrics is persisted as a failure artifact and raises before threshold comparison or optimization. These limits bound observed backend numerical drift; they are neither token-identity tolerances nor training-quality targets.

### External-server state contract

With `external_sleep_between_steps: true`, servers must be launched with `SLEEP_MODE=1`. The order for a dirty actor is:

```text
wake level-1 sleeping servers
    -> verify health
    -> export current FSDP bf16 safetensors
    -> reload every vLLM replica
    -> require positive exported/loaded tensor counts
    -> probe configured parameter checksums on every replica
    -> require one identical, non-empty checksum fingerprint
    -> reset every prefix cache
    -> verify health
    -> generate
    -> sleep
```

Startup performs the same sync before the first rollout, so a resumed FSDP checkpoint cannot silently sample from the base model. A cross-trainer edge case made the first managed h15 attempt fail: the prior trainer left vLLM in level-1 sleep, `/health` still returned 200, and a new manager incorrectly assumed the service was awake before reload. The manager now starts with unknown sleep state and performs one idempotent `/wake_up` before its first health/reload transaction. Fresh managed and control serial runs both completed real startup synchronization at checksum `6812228a4a79493c`, without the earlier CUDA invalid-argument failure.

The server manifest records the model/tokenizer/template files, launcher and worker extension, resolved experiment YAML, `MAX_MODEL_LEN`, `LOGPROBS_MODE`, sleep mode, eager/compiled mode, GDN prefill backend, and packed-recurrent setting with hashes. The current checksum probe proves cross-server agreement after reload; for a nonzero actor update, an unchanged fingerprint across steps is an additional anomaly to investigate, and a future end-to-end check should compare the reloaded values with trainer-side exported tensors rather than overstate replica agreement as source equality.

## Benchmark policies

Both benchmarks use the same hard length contract:

- `max_model_len`: 32,768;
- `prompt_length`, `data.max_prompt_length`, and `context_management.max_prompt_tokens`: 22,528;
- `response_length` and `data.max_response_length`: 10,240;
- `reasoning_history_tokens`: 0;
- `min_recent_turns`: 1;
- `recent_observation_max_tokens`: -1;
- `allow_current_observation_truncation`: false.

These are priority budgets, not targets to fill. The renderer keeps protected initialization/goal and the latest action/current observation first, then complete recent interaction units, then action-only plus compact old observations. It drops oldest old units before any recent unit. Unused prompt capacity is desirable when additional text is stale rather than state-bearing.

Both current Qwen3.5 FSDP gate configurations additionally set `model.use_remove_padding: false`, actor/ref `behavior_logprob_fp32_temperature: true`, actor/ref logprob micro-batch size 1, and Ulysses sequence parallel size 1. These are exact-scorer constraints rather than changes to the 22,528/10,240 context allocation.

### ALFWorld gate

- model/prompt/response budgets: 32,768 / 22,528 / 10,240 tokens;
- `max_env_len: 4096` on the legacy `ExtendedMessage` future view; the structured snapshot renderer reads immutable raw content and does **not** silently apply this crop;
- recent full observations: 2 whole turns;
- old observation cap: 160 tokens after repetitive action-hint removal;
- current goal, room state, inventory, and `AVAILABLE ACTIONS`: protected;
- current natural-smoke horizon: 3 decisions; a context-effect run must exceed this window, with 12 decisions retained as the planned full gate.

ALFWorld repeats large legal-action lists. Removing those lists only from old observations is a high-confidence deterministic saving because the current list is authoritative. Across 9,056 archived Qwen3.5 trajectory samples, the largest observed ALFWorld user message was about 717 tokens; after removing old `AVAILABLE ACTIONS`, the largest result among the 1,000 longest unique observations was 132 tokens. Thus 160 retains the observed state-bearing result while two lossless recent turns avoid injecting several stale legal-action lists.

### WebShop gate

- model/prompt/response budgets: 32,768 / 22,528 / 10,240 tokens;
- `max_env_len: 4096` on the legacy `ExtendedMessage` future view; the structured snapshot renderer reads immutable raw content and does **not** silently apply this crop;
- recent full pages: 4 whole turns;
- old page/result cap: 512 tokens;
- shopping constraints, current page, options, and clickable elements: protected;
- historical snapshot-training v6 horizon: 3 decisions, which cannot activate this compactor;
- current causal-canary horizon: 15 decisions, with the K=4/512 treatment first eligible at decision index 5.

WebShop state is distributed across product titles, constraints, options, and navigation. Four recent turns cover the common search -> results -> product -> option/buy chain without clipping current clickables. Across 9,984 archived Qwen3.5 samples, the largest user page was 3,241 tokens and the ordinary long tail was about 1,000 tokens, so the observed corpus happened to fit below 4,096. That observation is not a model-facing hard cap in snapshot mode: an oversized protected current page either fits inside the total 22,528-token prompt budget unchanged or raises `ContextBudgetError`. Old product prose can still be very long; the 512-token record is deliberately bounded. A deterministic structured shopping ledger is preferred over more aggressive generic clipping in the next version.

## Experiment protocol under 8x A100 80GB

Only two four-GPU lanes are available. Candidate selection must therefore be staged.

### Gate 0: static and synthetic contract

Before natural rollouts, both resolved YAMLs must pass the length validator and assert:

- `22528 + 10240 == 32768` at rollout, data, context-manager, actor, and reference boundaries;
- external vLLM is launched with `MAX_MODEL_LEN=32768`, `LOGPROBS_MODE=processed_logprobs`, and `SLEEP_MODE=1`;
- the server manifest hashes the exact experiment YAML, model/tokenizer/template, launcher, and reload extension;
- a 22,528-token prompt is accepted without left/right cropping;
- a 10,240-token completion is stored byte/token-identically, while 10,241 tokens fail rather than truncate;
- an exact 32,768-token synthetic snapshot passes serialization, actor old-log-probability computation, reference computation, and one update without OOM;
- Qwen3.5 actor and reference both use the fp32-temperature scorer, `use_remove_padding` is false, and their scorer flags cannot diverge silently;
- a micro-batch-1 sample with prompt-side left padding and response-side right padding scores identically to its unpadded valid-token form, while non-contiguous masks and unsupported larger scorer micro-batches fail explicitly;
- a protected goal/current observation that cannot fit raises `ContextBudgetError` before generation;
- repeated construction of one event log yields identical prompt IDs and hashes;
- historical reasoning and action-only history produce identical native-template prompt IDs, and the configured canonical renderer stores action-only history.

The static validator currently passes invariants I1--I9 for both production YAMLs, including exact `22528 + 10240 == 32768` allocation and every actor/rollout/ref/critic token cap at 32,768. A real-tokenizer CPU canary now exercises the renderer through `Linear_CMT`:

| Profile | Synthetic raw prompt | Managed prompt | Old observations clipped | Oldest whole turns evicted |
|---|---:|---:|---:|---:|
| ALFWorld K=2/160 | 121,893 | 22,469 | 188 | 71 |
| WebShop K=4/512 | 45,361 | 22,025 | 68 | 30 |

For both profiles the canary verifies deterministic IDs/hash, action-only historical assistants, visible token-bound clipping, oldest-first complete-turn eviction, exact initialization and recent/current observations, and equality between the model-facing messages/IDs and the captured decision snapshot. Its four focused tests pass.

The CPU canary is renderer/snapshot evidence, not a model execution test. The h15 serial WebShop run subsequently generated and preserved exact 10,240-token completions on GPU, but its natural maximum prompt plus completion was only 13,197. It still did not send a 22,528-token prompt or run a 32,768-token actor/reference forward/backward update. That GPU exact-boundary case remains mandatory because Qwen3.5's 248,320-token output vocabulary makes logits memory proportional to actual sequence length. Setting a 32K serving ceiling, passing the CPU canary, or observing a natural 10K completion does not exercise the full training-memory boundary.

### Gate 1: concurrent natural smoke, integration only

Run the two environments concurrently:

- lane A: `alfworld_qwen35_4b_grpo_snapshot_gate.yaml` on GPUs 0--3;
- lane B: `webshop_qwen35_4b_grpo_snapshot_gate.yaml` on GPUs 4--7.

A short natural smoke can expose data-contract, server-state, and obvious stability failures. It proves only the sequence lengths that happened to be sampled. Even if every request had a 10,240-token allowance, short responses do not prove that the 10K response or 32K train boundary works. The current three-decision smoke is also structurally incapable of exercising either compactor: before the final model call there are at most two completed historical turns, versus `recent_turns=2` for ALFWorld and `recent_turns=4` for WebShop.

The natural smoke passes only if:

- prompt/response identity violations: 0;
- silent truncations: 0;
- fabricated/empty completions accepted: 0;
- exact `length` completions preserve all returned tokens/log-probabilities, do not fabricate EOS or call `env.step` with a partial action, immediately end with the configured failure reward, and remain below the configured 10% decision-level circuit breaker;
- missing rollout log-probabilities: 0, and valid-logprob token count equals sampled response-token count;
- rollout/actor mean absolute drift <= 0.02, p99 <= 0.25, outside-PPO-clip fraction <= 0.01, and catastrophic maximum <= 0.75;
- `training/behavior_logprobs_from_rollout == 1` on every update;
- current-observation clips: 0;
- NaN/Inf loss, KL, entropy, or gradients: 0;
- server retry budget exhaustion: 0;
- wake, positive reload counts, identical non-empty replica checksum, cache reset, health check, and sleep all succeed at startup and every dirty update;
- context statistics are emitted every update;
- managed prompts never exceed 22,528 and sampled prompt plus response never exceeds 32,768;
- goal/current legal actions or clickables are present losslessly in decoded snapshots;
- historical `<think>` text does not leak into a next-turn native prompt;
- no OOM or server death occurs.

`response_length/clip_ratio` and `prompt_length/clip_ratio` are not valid truncation gates in the current metric helper: they compare each sequence with that batch's dynamically padded maximum, so at least one ordinary sample is marked “clipped.” Use the stored completion length against 10,240 and the actual `finish_reason` instead.

The callback raises on an empty completion or missing token IDs/logprobs. A genuine `length` completion follows the explicit terminal contract above and is counted against the optimizer-side decision-fraction circuit breaker; it is not silently accepted as an ordinary action. Other unexpected non-`stop` reasons are still recorded rather than governed by their own hard gate and remain an implementation gap.

Success rate from this small smoke is descriptive only. Four or fewer task groups can move from nonzero success to zero by task mix alone; it is neither degradation evidence nor a reason to select one context policy.

#### WebShop v2--v6 execution record

| Run | What reached the trainer | Numerical evidence / failure | Optimizer update |
|---|---|---|---|
| v2 | startup weight sync | an out-of-band checksum diagnostic was issued while level-1 sleep was active; the next `/wake_up` returned CUDA illegal-memory-access | no |
| v3 | startup | the shell/executor that owned the rollout services exited; port 8211 was then connection-refused during the health check | no |
| v4 | one complete rollout batch | 506 tokens; mean/p99/max `0.017251/0.185691/0.241299`, outside-clip `0.005929`; failed the then-strict mean `0.01` gate | no |
| v5 | one complete rollout batch and saved failure artifact | 742 tokens; mean/p99/max `0.013345/0.186318/0.453400`, outside-clip `0.004043`; failed the then-strict mean/max gate | no |
| v6 step 1 | rollout, actor/ref scoring, update, reload | 506 tokens; mean/p99/max `0.017251/0.185691/0.241299`, outside-clip `0.005929`; all four calibrated gates passed; `behavior_logprobs_from_rollout=1` | yes |
| v6 step 2 | rollout, actor/ref scoring, update, reload | 459 tokens; mean/p99/max `0.013467/0.187151/0.313326`, outside-clip `0.002179`; all four calibrated gates passed; actual rollout logprobs were installed before update | yes |

v5's failure artifact also proves tensor propagation and selection: raw/managed prompt lengths were `[1401, 1401, 1095, 1095]`, selected decisions were `[2, 2, 1, 1]`, and compressed/dropped/clipped counts were all zero. It does **not** prove compaction. WebShop v6 likewise had zero reduction: step 1 raw and managed means were both 1,179.25 (maximum 1,401), and step 2 selected raw/managed lengths were `[1354, 328, 328, 324]`. This is the expected consequence of K=4 with only three decisions, not evidence that the K=4/512 policy improves context.

The v6 server checksum was `6812228a...` initially, `e80333b1...` after step 1, and `7d154caa...` after step 2, identical across all four replicas each time. Together with the update timing, this proves that both optimizer/reload paths changed sampled weights. Step 1 nevertheless had zero reward variance and zero GRPO advantage across its four trajectories. The old path still ran an optimizer step, allowing KL numerical residue and AdamW decay to move weights despite zero GRPO signal. The current gate enables `skip_zero_advantage_grpo_update`, which skips that pure-GRPO actor update entirely; this repair still requires a new training canary.

After step 2, v6 completed deterministic post-update validation on 200 tasks. Its mean was `-0.00180556`, versus `0.00644444` for the matched 200-task initial-checksum baseline. The paired delta was `-0.00825`, bootstrap 95% CI `[-0.02025, -0.001]`, with improved/equal/degraded counts `10/166/24`. This is evidence that the historical v6 updates degraded the paired evaluation, not merely a descriptive post-update curve. One validation completion re-tokenizes to exactly 10,240 tokens, proving the serving allowance but not the actor/ref/update boundary.

The historical process then exited nonzero during final checkpoint archival because the base model lacks `generation_config.json`; although model, optimizer, and extra-state shards existed in the Ray working directory, that run remains a failed checkpoint transaction. The checkpoint manager now constructs a local generation-config fallback only for this missing-file compatibility case, but the old archive is not retroactively complete. Together with zero-advantage update skipping, this removes two concrete failure sources without proving that the repaired GRPO path is stable.

The supported claims at this point are limited to: the two production profiles declare one internally consistent 22,528/10,240/32,768 regime; the CPU canary activates deterministic compression and whole-turn eviction with the real tokenizer; the h15 WebShop validation activates natural K=4/512 compaction and preserves exact 10,240-token completions; the old WebShop path used actual rollout logprobs but its paired post-update result degraded; and the repaired path now completes a minimal guarded update/skip/checkpoint cycle. It is **not** yet supported to claim that a 10K response survives the actor/ref/update boundary, that an exact 32K GPU update is memory-safe, that repaired GRPO is stable over a meaningful horizon, that benchmark-level reward is non-degrading, or that the result transfers to ALFWorld or Qwen3.5-2B.

#### WebShop repaired-path v7 canary

`webshop_qwen35_4b_grpo_snapshot_32k_gate_v7_repaired` ran with four synchronized rollout replicas but one outstanding environment request, a fixed 32-task pre/post validation order, two consumed GRPO batches, and a three-decision horizon. It exited successfully.

- Startup wake/reload passed on all four replicas at checksum `6812228a4a79493c`.
- Batch 1 had nonzero group-relative advantage and applied one actor update. All behavior-drift gates passed over 868 valid tokens: mean/p99/max `0.012583/0.199467/0.376572`, outside-clip fraction `0.004608`. Before batch 2, all replicas reloaded checksum `5b1e9567941929c`.
- Batch 2 had exactly zero effective advantage. The trainer reported `actor_update_skipped_zero_advantage=1`, `actor_update_applied=0`, and `external_rollout_weights_marked_stale=0`. Its 1,493-token drift metrics also passed: mean/p99/max `0.009703/0.176208/0.262153`, outside-clip `0.001340`.
- Both batches used actual rollout log-probabilities as the PPO denominator, had zero length-truncated decisions, and passed the nonfinite and group-integrity gates.
- Final checkpointing completed instead of failing on the missing base `generation_config.json`. The actor directory contains four model shards, four optimizer shards, four extra-state shards, `config.json`, tokenizer/template files, and the generated fallback `generation_config.json`.

The 32 unique validation tasks and their order match exactly before and after the single applied update. Mean reward changed from `-0.0203125` to `-0.0187500`, paired delta `+0.0015625`, with improved/equal/degraded counts `3/27/2`; a 100,000-resample paired bootstrap interval was `[-0.0046875, 0.0078125]`. Both evaluations had zero successes and no length events. This is no immediate degradation signal, not evidence of improvement: the interval crosses zero, the reward range is nearly degenerate, and only one optimizer step occurred.

This canary deliberately does not validate long-context training. Its horizon of three is below K=4 treatment onset, its largest validation prompt plus completion was 2,235 tokens, and no 10K completion entered actor/ref/update. The rollout-server supervisor manifest was created for the preceding h15 run; the runtime parameters matched v7 and the frozen v7 launcher record is complete, but a formal v7 reproduction should restart the servers with the v7 experiment YAML so the server manifest hashes that exact config. The 50GB checkpoint remains in the recorded Ray working directory rather than being copied into the repository workspace.

### Gate 2: paired causal comparison

The first WebShop h15 A/B exposed a validation-protocol bug before it measured context quality. In v2, both arms used the same 16 tasks, template, weights, and checksum, and managed context first activated at step 5. However, four external replicas and concurrent dynamic batching produced treatment-before divergence even at temperature zero: among 71 pre-treatment decisions with the same prompt hash, only 61 had the same completion hash. The raw score gap (`0.11875` managed versus `0.18750` control) is therefore invalid as a context effect. All of its net score difference came from a task whose completion had already diverged before treatment.

The v3 serial rerun fixed the causal execution contract to one server, one validation task per batch, one outstanding request, greedy decoding, seed 2025, and no trajectory resubmission. Its frozen launcher records are `webshop_qwen35_4b_context_h15_managed_eval_v3_serial` and `webshop_qwen35_4b_context_h15_control_eval_v3_serial`. Both startup syncs reported checksum `6812228a4a79493c`, and their frozen `agentevolver` source trees are identical.

The trace-identity gate passed:

| Check | Result |
|---|---:|
| same unique tasks and order | 16/16 |
| decision 0--4 prompt/completion/token-count/finish/template identity | 79/79 |
| trajectories reaching decision 5 in both arms | 13 |
| managed raw tokens equal control prompt tokens at decision 5 | 13/13 |
| managed first activation exactly at decision 5 | 13/13 |
| control context activations | 0 |

The frozen v3 JSONL stores raw prompt token counts but predates a separate raw-prompt hash. Its step 0--4 full identity plus equal step-5 raw lengths is strong evidence of the same pre-treatment state. The current implementation now persists `raw_prompt_hash` in every decision snapshot and audit record, so future runs can require direct raw token identity at the branch point.

Managed context was active on 114 decisions and reduced cumulative prompt tokens from `660,900` raw to `570,941` rendered: `89,959` tokens saved, 13.61% over all decisions and 16.38% over active decisions. It clipped 266 old-observation instances and dropped zero whole turns; the largest natural raw prompt was only 8,023, so this run tested observation compaction but not the 22,528-token eviction boundary. Maximum rendered prompt was 6,627, maximum completion was exactly 10,240, and maximum prompt plus completion was 13,197.

Performance was nearly tied: managed/control mean rewards were `0.118750/0.121875`, paired delta `-0.003125`, with `3/9/4` improved/equal/degraded tasks. A 100,000-resample paired bootstrap interval was `[-0.01875, 0.01250]`; both arms had 3/16 positive-reward tasks. Managed had two correct length-terminal episodes and control one; the shared pre-treatment case was identical, while the extra managed case occurred after context changed the path. Thus v3 is a valid structural and small causal canary, but 16 tasks, one seed, one checkpoint, and validation-only execution cannot establish non-degradation or GRPO stability.

The reusable h15 configs now make `max_env_worker: 1` explicit and use v4 experiment names so they cannot append to the frozen v3 artifacts. The next formal comparison must use more fixed paired tasks and multiple seeds. ALFWorld still requires its own full-history versus K=2/160 comparison. The strongest future protocol is to generate the common pre-treatment prefix once, snapshot the environment, and fork both policies at the treatment step; absent environment forking, every run must retain the 100% pre-treatment prompt/completion/finish identity gate.

### Gate 3: sustained concurrent training

After each benchmark has a passing context candidate, run the chosen ALFWorld and WebShop policies concurrently for the intended horizon, with fixed-task validation at meaningful milestones. Repeat final configurations across seeds before making paper claims.

Suggested acceptance thresholds:

- at least 25% p95 prompt-token reduction after the policy becomes active;
- no OOM or reward-bearing truncation;
- no more than 5 percentage points paired strict-SR degradation at a milestone;
- no increase in invalid/malformed action rate beyond the paired confidence interval;
- rollout-vs-actor drift remains within the four hard limits: mean 0.02, p99 0.25, outside-clip fraction 0.01, and maximum 0.75;
- all rollout replicas continue to pass reload-count, checksum, cache-reset, and health gates;
- KL, entropy, and gradient norms remain finite without a new monotonic collapse signature.

## Teacher and DUET follow-up

The first gates are pure GRPO. Existing teacher data must not be mixed into them.

For DeepSeek-v4-flash collection, “same context” means the same deterministic context plan and message content, not the same tokenizer IDs (DeepSeek and Qwen tokenize differently). Each teacher decision must persist:

- canonical context-plan/message JSON hash;
- provider/model name and request ID;
- exact messages sent to DeepSeek;
- returned action/content and finish reason;
- the student tokenizer/template hash used later for BC/DR3 conversion.

The current minimal teacher collector uses full history and a different tokenizer contract, so it does not satisfy this requirement. The parked 4,000-trajectory DeepSeek-v4-pro ALFWorld dataset must not be presented as compressed-context-aligned teacher data.

DUET integration should begin only after the collector acts under the selected benchmark policy. Teacher actions remain explicitly off-policy; Baseline Separation, DR3, and adaptive BC handle that status. They must never be relabeled as behavior-policy samples.

## 2B extension

After the 4B context/data contract passes, use the same renderer and snapshot implementation for Qwen3.5-2B. Re-run the length/memory gate rather than assuming a larger token budget: the output-vocabulary logits and rollout KV/cache implementation, not parameter count alone, determine the safe sequence regime.

## Reproducibility artifacts

Every formal run should archive:

- resolved YAML and git/worktree diff;
- model path plus tokenizer/template hashes;
- rollout server versions, addresses, `MAX_MODEL_LEN`, `LOGPROBS_MODE`, sleep mode, eager/compiled mode, GDN backend, and packed-recurrent setting;
- actual active request concurrency, per-request seed, and task-to-server/request routing identity;
- the launcher manifest with experiment-YAML, server-extension, and model-contract hashes;
- per-sync exported/loaded counts, checksum probe count/fingerprint, cache-reset result, and timing;
- per-step raw/managed prompt-token distributions;
- selected decision indices, rendered prompt hashes, completion hashes, finish reasons, and raw-prompt hashes at causal branch points;
- processed rollout behavior log-probabilities, their valid-token mask, and rollout-vs-actor mean/quantile/ratio/position drift metrics;
- resolved actor/reference scorer flags, padding mode, scorer micro-batch size, and actor/reference KL diagnostics under the symmetric numerical contract;
- abnormal completion, truncation, retry, invalid-action, OOM, and server-death counts;
- a separate exact-32K boundary-test result, not an inference from natural rollout maxima;
- archived CPU long-context canary reports for both production profiles;
- raw trajectory view for State Channel and model-facing snapshot view for training;
- fixed-task validation outputs, treatment-before trace-identity results, and paired intervals, not only aggregate WandB curves;
- final checkpoint success/failure and the exact archived model/optimizer files; partial shard writes are not a completed checkpoint.

No context mechanism should be called “non-degrading” until both the structural gates and the benchmark-specific paired performance gates pass.
