#!/usr/bin/env python
"""Standalone length/memory-regime validator for experiment configs (F1).

Checks the invariants that keep the rollout engine, the context manager
and the FSDP micro-batching in one consistent length regime:

  I1. context-manager budget: max_seq_length (= rollout.max_model_len -
      data.max_response_length) == data.max_prompt_length
      (mirrors the assert in cmt_linear_think.py; cmt_linear/cmt_memory
      derive max_seq_length the same way)
  I2. data.max_prompt_length + data.max_response_length <= rollout.max_model_len
  I3. every *_max_token_len_per_gpu >= the longest sequence a batch can contain
      (= data.max_prompt_length + data.max_response_length). verl's
      rearrange_micro_batches asserts max_token_len >= max_seq_len of the batch,
      NOT >= max_model_len: the latter is only the vLLM serving ceiling and is
      cheap (GDN KV ~32KB/token), while *_max_token_len_per_gpu sets how many
      tokens land in one micro-batch and therefore drives the logits memory
      (tokens x 248320 vocab x 2B; 12288 fits on 4xA100, 16384 OOMs). Tying the
      two together forced max_model_len down every time memory ran out.
  I4. rollout.prompt_length == rollout.max_model_len - rollout.response_length
  I5. data.train_batch_size >= actor.ppo_mini_batch_size, matching the trainer
      startup assertion so an invalid gate fails in this cheap preflight.
  I6. rollout.prompt_length == data.max_prompt_length ==
      context_management.max_prompt_tokens.
  I7. rollout.response_length == data.max_response_length.
  I8. Qwen3.5's exact fp32-temperature scorer uses the supported padded GDN
      path, never verl remove-padding/Ulysses packing.
  I9. actor and reference scorers use the same fp32-temperature contract.

Usage:
    python config/validate_length_config.py <experiment.yaml> [more.yaml ...]

Config resolution mirrors the experiment yamls' hydra defaults chain
[ppo_trainer, agentevolver, _self_] with a pragmatic direct OmegaConf merge:
external/config_fallback/ppo_trainer.yaml <- config/agentevolver.yaml <- the
experiment yaml (hydra/defaults keys stripped). Exit code 0 iff every file
passes every invariant.
"""

import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
FALLBACK_YAML = REPO_ROOT / "external" / "config_fallback" / "ppo_trainer.yaml"
DEFAULTS_YAML = REPO_ROOT / "config" / "agentevolver.yaml"

TOKEN_LEN_KEYS = [
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu",
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu",
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu",
    "critic.ppo_max_token_len_per_gpu",
    "critic.forward_max_token_len_per_gpu",
]


def load_merged(exp_yaml: Path):
    """Merge ppo_trainer <- agentevolver <- experiment yaml (defaults chain)."""
    base = OmegaConf.load(FALLBACK_YAML)
    defaults = OmegaConf.load(DEFAULTS_YAML)
    exp_dict = OmegaConf.to_container(OmegaConf.load(exp_yaml), resolve=False)
    # Strip hydra bookkeeping keys the compose step would consume.
    for key in ("hydra", "defaults"):
        exp_dict.pop(key, None)
    return OmegaConf.merge(base, defaults, OmegaConf.create(exp_dict))


def sel_int(cfg, key):
    val = OmegaConf.select(cfg, key)
    if val is None:
        raise KeyError(f"missing config key: {key}")
    return int(val)


def validate(exp_yaml: Path) -> bool:
    cfg = load_merged(exp_yaml)

    max_model_len = sel_int(cfg, "actor_rollout_ref.rollout.max_model_len")
    prompt_length = sel_int(cfg, "actor_rollout_ref.rollout.prompt_length")
    response_length = sel_int(cfg, "actor_rollout_ref.rollout.response_length")
    data_max_prompt = sel_int(cfg, "data.max_prompt_length")
    data_max_response = sel_int(cfg, "data.max_response_length")
    train_batch_size = sel_int(cfg, "data.train_batch_size")
    ppo_mini_batch_size = sel_int(
        cfg, "actor_rollout_ref.actor.ppo_mini_batch_size"
    )
    context_max_prompt = sel_int(
        cfg,
        "actor_rollout_ref.rollout.context_management.max_prompt_tokens",
    )
    max_seq_length = max_model_len - data_max_response  # cmt_* derivation
    use_remove_padding = bool(
        OmegaConf.select(cfg, "actor_rollout_ref.model.use_remove_padding")
    )
    actor_fp32_temperature = bool(
        OmegaConf.select(
            cfg,
            "actor_rollout_ref.actor.behavior_logprob_fp32_temperature",
        )
    )
    ref_fp32_temperature = bool(
        OmegaConf.select(
            cfg,
            "actor_rollout_ref.ref.behavior_logprob_fp32_temperature",
        )
    )

    checks = [
        (
            "I1 context-manager budget leaves room for the init prompt",
            max_seq_length >= data_max_prompt,
            f"max_model_len({max_model_len}) - data.max_response_length({data_max_response}) "
            f"= {max_seq_length} vs data.max_prompt_length({data_max_prompt})",
        ),
        (
            "I2 data prompt+response <= max_model_len",
            data_max_prompt + data_max_response <= max_model_len,
            f"{data_max_prompt} + {data_max_response} = {data_max_prompt + data_max_response} "
            f"vs max_model_len({max_model_len})",
        ),
        (
            "I4 rollout.prompt_length == max_model_len - response_length",
            prompt_length == max_model_len - response_length,
            f"prompt_length({prompt_length}) vs {max_model_len} - {response_length} "
            f"= {max_model_len - response_length}",
        ),
        (
            "I5 train_batch_size >= ppo_mini_batch_size",
            train_batch_size >= ppo_mini_batch_size,
            f"train_batch_size({train_batch_size}) vs "
            f"ppo_mini_batch_size({ppo_mini_batch_size})",
        ),
        (
            "I6 rollout/data/context prompt budgets are identical",
            prompt_length == data_max_prompt == context_max_prompt,
            f"rollout.prompt_length({prompt_length}), "
            f"data.max_prompt_length({data_max_prompt}), "
            f"context.max_prompt_tokens({context_max_prompt})",
        ),
        (
            "I7 rollout.response_length == data.max_response_length",
            response_length == data_max_response,
            f"rollout.response_length({response_length}) vs "
            f"data.max_response_length({data_max_response})",
        ),
        (
            "I8 exact Qwen3.5 scorer disables remove padding",
            not actor_fp32_temperature or not use_remove_padding,
            f"behavior_logprob_fp32_temperature({actor_fp32_temperature}), "
            f"model.use_remove_padding({use_remove_padding})",
        ),
        (
            "I9 actor/ref temperature contracts are identical",
            actor_fp32_temperature == ref_fp32_temperature,
            f"actor({actor_fp32_temperature}) vs ref({ref_fp32_temperature})",
        ),
    ]
    longest_seq = data_max_prompt + data_max_response
    for key in TOKEN_LEN_KEYS:
        val = sel_int(cfg, key)
        checks.append(
            (
                f"I3 {key} >= longest possible sequence",
                val >= longest_seq,
                f"{val} vs data.max_prompt_length + data.max_response_length "
                f"= {longest_seq} (max_model_len {max_model_len} is the serving "
                f"ceiling and does not bound this)",
            )
        )

    ok = True
    print(f"== {exp_yaml} ==")
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            ok = False
        print(f"  [{status}] {name}: {detail}")
    return ok


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    all_ok = True
    for arg in argv[1:]:
        path = Path(arg)
        if not path.exists():
            print(f"== {path} ==\n  [FAIL] file not found")
            all_ok = False
            continue
        try:
            if not validate(path):
                all_ok = False
        except Exception as exc:  # missing keys, malformed yaml, ...
            print(f"  [FAIL] {type(exc).__name__}: {exc}")
            all_ok = False
    print("ALL PASS" if all_ok else "FAILURES DETECTED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
