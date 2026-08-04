#!/usr/bin/env python3
"""Fail-closed preflight for the queued Qwen3.5-4B batch-16 runs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


EXPECTED = {
    "webshop": {
        "experiment": "webshop_qwen35_4b_grpo_32k_baseline_b16_gpu47_seed2025_s100",
        "env_url": "http://127.0.0.1:8083",
        "recent_turns": 4,
        "old_observation_tokens": 512,
        "temperature": 0.6,
    },
    "alfworld": {
        "experiment": "alfworld_qwen35_4b_grpo_32k_baseline_b16_gpu47_seed2025_s100",
        "env_url": "http://127.0.0.1:8081",
        "recent_turns": 2,
        "old_observation_tokens": 160,
        "temperature": 0.9,
    },
}

BASELINE = {
    "webshop": "webshop_qwen35_4b_grpo_32k_baseline_gpu47_v1_s200.yaml",
    "alfworld": "alfworld_qwen35_4b_grpo_32k_baseline_gpu47_v4_serial_decode_safe.yaml",
}

ALLOWED_BASELINE_DIFFS = {
    "webshop": {
        "trainer.experiment_name",
        "trainer.default_local_dir",
        "trainer.validation_data_dir",
        "trainer.rollout_data_dir",
        "trainer.total_training_steps",
        "actor_rollout_ref.actor.ppo_mini_batch_size",
        "critic.ppo_mini_batch_size",
        "data.train_batch_size",
        "task_manager.bs",
    },
    "alfworld": {
        "trainer.experiment_name",
        "trainer.default_local_dir",
        "trainer.validation_data_dir",
        "trainer.rollout_data_dir",
        "trainer.test_freq",
        "actor_rollout_ref.actor.ppo_mini_batch_size",
        "actor_rollout_ref.rollout.rollout_logprob_drift_max_threshold",
        "critic.ppo_mini_batch_size",
        "data.train_batch_size",
        "data.max_train_tasks",
        "exp_manager.reme.workspace_id",
        "task_manager.bs",
    },
}


def _compose(path: Path):
    with initialize_config_dir(config_dir=str(path.parent.resolve()), version_base=None):
        return compose(config_name=path.stem)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _diff_paths(left, right, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths: set[str] = set()
        for key in left.keys() | right.keys():
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.add(child)
            else:
                paths.update(_diff_paths(left[key], right[key], child))
        return paths
    if left != right:
        return {prefix}
    return set()


def _check(path: Path, benchmark: str) -> list[str]:
    cfg = _compose(path)
    expected = EXPECTED[benchmark]
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    trainer = cfg.trainer
    data = cfg.data
    actor = cfg.actor_rollout_ref.actor
    rollout = cfg.actor_rollout_ref.rollout
    ref = cfg.actor_rollout_ref.ref
    context = rollout.context_management

    require(trainer.experiment_name == expected["experiment"], "experiment name")
    require(trainer.project_name == "agentevolver", "W&B project")
    require(set(trainer.logger) == {"console", "wandb"}, "online W&B logger")
    require(trainer.n_gpus_per_node == 4 and trainer.nnodes == 1, "4-GPU topology")
    require(trainer.total_epochs == 1, "one epoch")
    require(trainer.total_training_steps == 100, "100 outer steps")
    require(trainer.save_freq == 50 and trainer.test_freq == 25, "save/eval cadence")
    require(trainer.val_before_train is True, "fixed step-0 validation")
    require(trainer.resume_mode == "disable", "fresh run")
    require(OmegaConf.select(cfg, "trainer.resume_from_path") is None, "no resume path")

    require(data.train_batch_size == 16, "task batch 16")
    require(data.max_train_tasks == 1600, "1,600-task curriculum")
    require(
        data.train_batch_size * trainer.total_training_steps == data.max_train_tasks,
        "batch x steps must equal one curriculum pass",
    )
    require(data.max_val_tasks == 200 and data.validation_shuffle is False, "fixed validation-200")
    require(data.seed == 2025 and data.task_seed == 2026, "paired run/task seeds")
    require(data.max_prompt_length == 22528, "22,528-token prompt budget")
    require(data.max_response_length == 10240, "10,240-token response budget")

    require(cfg.actor_rollout_ref.model.path == "/data/shared_models/Qwen3.5-4B-think", "model")
    require(cfg.actor_rollout_ref.model.use_remove_padding is False, "padded Qwen3.5 scorer")
    require(actor.ppo_mini_batch_size == 16, "actor mini-batch 16")
    require(actor.ppo_micro_batch_size_per_gpu == 1, "actor micro-batch one")
    require(actor.ppo_max_token_len_per_gpu == 32768, "actor 32K token cap")
    require(actor.skip_zero_advantage_grpo_update is True, "zero-advantage guard")

    require(rollout.thinking_mode == "native_qwen35", "Qwen3.5 thinking mode")
    require(rollout.prompt_length == 22528, "rollout prompt budget")
    require(rollout.response_length == 10240, "rollout response budget")
    require(rollout.max_model_len == 32768, "32K model context")
    require(rollout.prompt_length + rollout.response_length == rollout.max_model_len, "exact 32K split")
    require(rollout.n == 8, "eight GRPO samples per task")
    require(data.train_batch_size * rollout.n == 128, "128 trajectories per outer step")
    require(rollout.max_env_worker == 32 and rollout.max_num_seqs == 1, "bounded rollout concurrency")
    require(rollout.log_prob_micro_batch_size_per_gpu == 1, "rollout scorer micro-batch one")
    require(rollout.log_prob_max_token_len_per_gpu == 32768, "rollout scorer 32K cap")
    require(rollout.use_rollout_log_probs_as_old is True, "behavior logprobs as PPO old policy")
    require(rollout.rollout_logprob_drift_mean_threshold == 0.02, "mean drift gate")
    require(rollout.rollout_logprob_drift_p99_threshold == 0.25, "p99 drift gate")
    require(rollout.rollout_importance_ratio_outside_clip_threshold == 0.01, "ratio tail gate")
    require(rollout.rollout_logprob_drift_max_threshold == -1, "raw max diagnostic-only")
    require(rollout.max_trajectory_resubmits == 1, "bounded trajectory retry")
    require(rollout.multi_turn.max_steps == 30, "30-decision horizon")
    require(float(rollout.temperature) == expected["temperature"], "benchmark temperature")
    require(list(rollout.external_server_addresses) == [
        "127.0.0.1:8211",
        "127.0.0.1:8212",
        "127.0.0.1:8213",
        "127.0.0.1:8214",
    ], "GPU4-7 external rollout lane")
    require(rollout.val_kwargs.n == 1 and rollout.val_kwargs.temperature == 0, "greedy validation")
    require(rollout.val_kwargs.seed == 2025, "fixed validation seed")

    require(context.enabled is True and context.snapshot_training is True, "v5 snapshot training")
    require(context.max_prompt_tokens == 22528, "context prompt cap")
    require(context.recent_turns == expected["recent_turns"], "benchmark recent-turn policy")
    require(
        context.history_observation_max_tokens == expected["old_observation_tokens"],
        "benchmark old-observation cap",
    )
    require(context.allow_current_observation_truncation is False, "protected current observation")
    require(context.reasoning_history_tokens == 0, "no replayed historical think")
    require(context.snapshot_selection == "token_weighted", "snapshot selection")
    require(context.snapshot_selection_seed == 2025, "snapshot seed")

    require(ref.log_prob_micro_batch_size_per_gpu == 1, "reference micro-batch one")
    require(ref.log_prob_max_token_len_per_gpu == 32768, "reference 32K cap")
    require(cfg.critic.ppo_max_token_len_per_gpu == 32768, "critic PPO 32K cap")
    require(cfg.critic.forward_max_token_len_per_gpu == 32768, "critic forward 32K cap")
    require(cfg.env_service.env_type == benchmark, "benchmark environment type")
    require(cfg.env_service.env_url == expected["env_url"], "benchmark environment URL")

    baseline = _compose(path.parent / BASELINE[benchmark])
    resolved = OmegaConf.to_container(cfg, resolve=True)
    baseline_resolved = OmegaConf.to_container(baseline, resolve=True)
    observed_diffs = _diff_paths(baseline_resolved, resolved)
    require(
        observed_diffs == ALLOWED_BASELINE_DIFFS[benchmark],
        "resolved config changed outside the pre-registered batch-16 diff: "
        f"observed={sorted(observed_diffs)}",
    )

    if errors:
        print(f"FAIL {path}")
        for error in errors:
            print(f"  - {error}")
    else:
        print(
            f"PASS {benchmark}: {trainer.experiment_name}; "
            f"tasks={data.max_train_tasks}, batch={data.train_batch_size}, "
            f"steps={trainer.total_training_steps}, trajectories={data.max_train_tasks * rollout.n}, "
            f"sha256={_sha256(path)}"
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--webshop", type=Path, required=True)
    parser.add_argument("--alfworld", type=Path, required=True)
    args = parser.parse_args()

    errors = []
    for benchmark, path in (("webshop", args.webshop), ("alfworld", args.alfworld)):
        if not path.is_file():
            print(f"FAIL missing {benchmark} config: {path}")
            errors.append(f"missing {path}")
            continue
        errors.extend(_check(path, benchmark))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
