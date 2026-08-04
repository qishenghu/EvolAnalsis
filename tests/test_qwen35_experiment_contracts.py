from copy import deepcopy
from pathlib import Path
import subprocess

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]


def _compose_from(config_dir: Path, config_name: str):
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name=config_name)


def _assert_32k_reasoning_contract(cfg):
    rollout = cfg.actor_rollout_ref.rollout
    assert rollout.prompt_length == 22528
    assert rollout.response_length == 10240
    assert rollout.max_model_len == 32768
    assert cfg.data.max_prompt_length == 22528
    assert cfg.data.max_response_length == 10240
    assert rollout.prompt_length + rollout.response_length == rollout.max_model_len
    assert cfg.actor_rollout_ref.actor.ppo_max_token_len_per_gpu == 32768
    assert rollout.log_prob_max_token_len_per_gpu == 32768
    assert cfg.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu == 32768
    assert cfg.actor_rollout_ref.actor.skip_zero_advantage_grpo_update is True
    assert rollout.use_rollout_log_probs_as_old is True


def _normalized_ab_contract(cfg):
    """Remove only the intentional treatment and output-location differences."""
    contract = deepcopy(OmegaConf.to_container(cfg, resolve=True))
    trainer = contract["trainer"]
    for key in (
        "experiment_name",
        "default_local_dir",
        "validation_data_dir",
        "rollout_data_dir",
    ):
        trainer.pop(key)
    context = contract["actor_rollout_ref"]["rollout"]["context_management"]
    for key in (
        "recent_turns",
        "history_observation_max_tokens",
    ):
        context.pop(key)
    return contract


def _normalized_v4_v5_resume_contract(cfg):
    """Remove exactly the intentionally changed v5 resume/diagnostic fields."""
    contract = deepcopy(OmegaConf.to_container(cfg, resolve=True))
    trainer = contract["trainer"]
    for key in (
        "experiment_name",
        "default_local_dir",
        "validation_data_dir",
        "rollout_data_dir",
        "val_before_train",
        "resume_mode",
        "resume_from_path",
    ):
        trainer.pop(key)
    contract["actor_rollout_ref"]["rollout"].pop(
        "rollout_logprob_drift_max_threshold"
    )
    return contract


def test_webshop_h15_managed_and_control_are_paired_except_policy_knobs():
    config_dir = (
        REPO_ROOT
        / "config/duet_paper_experiments_configs/iclr2027/webshop"
    )
    managed = _compose_from(
        config_dir, "webshop_qwen35_4b_context_h15_managed_eval"
    )
    control = _compose_from(
        config_dir, "webshop_qwen35_4b_context_h15_control_eval"
    )

    for cfg in (managed, control):
        _assert_32k_reasoning_contract(cfg)
        assert cfg.trainer.val_before_train is True
        assert cfg.trainer.val_only is True
        assert cfg.trainer.resume_mode == "disable"
        assert cfg.data.max_val_tasks == 16
        assert cfg.data.validation_shuffle is False
        assert cfg.data.val_batch_size == 1
        assert cfg.data.dataloader_num_workers == 0
        assert cfg.thread_pool.max_workers == 1
        assert cfg.actor_rollout_ref.rollout.max_env_worker == 1
        assert cfg.actor_rollout_ref.rollout.multi_turn.max_steps == 15
        assert cfg.actor_rollout_ref.rollout.max_trajectory_resubmits == 0
        assert cfg.actor_rollout_ref.rollout.external_server_addresses == [
            "127.0.0.1:8211"
        ]
        val_kwargs = cfg.actor_rollout_ref.rollout.val_kwargs
        assert val_kwargs.n == 1
        assert val_kwargs.temperature == 0
        assert val_kwargs.top_p == 1.0
        assert val_kwargs.top_k == -1
        assert val_kwargs.do_sample is False
        assert val_kwargs.seed == 2025

    assert _normalized_ab_contract(managed) == _normalized_ab_contract(control)

    assert managed.actor_rollout_ref.model.path == control.actor_rollout_ref.model.path
    assert managed.actor_rollout_ref.rollout.context_management.recent_turns == 4
    assert (
        managed.actor_rollout_ref.rollout.context_management.history_observation_max_tokens
        == 512
    )
    assert control.actor_rollout_ref.rollout.context_management.recent_turns == 15
    assert (
        control.actor_rollout_ref.rollout.context_management.history_observation_max_tokens
        == -1
    )


def test_validation_seed_reaches_exact_request_sampling_params():
    from agentevolver.module.env_manager.env_manager import ParallelEnvManager

    config_dir = (
        REPO_ROOT
        / "config/duet_paper_experiments_configs/iclr2027/webshop"
    )
    cfg = _compose_from(
        config_dir, "webshop_qwen35_4b_context_h15_managed_eval"
    )
    manager = ParallelEnvManager.__new__(ParallelEnvManager)
    manager.rollout_config = cfg.actor_rollout_ref.rollout

    params = manager._sampling_params_for_mode("validate")

    assert params["n"] == 1
    assert params["max_completion_tokens"] == 10240
    assert params["temperature"] == 0
    assert params["top_p"] == 1.0
    assert params["top_k"] == -1
    assert params["seed"] == 2025


def test_qwen35_2b_smoke_uses_same_10k_32k_contract():
    config_dir = (
        REPO_ROOT
        / "config/duet_paper_experiments_configs/iclr2027/alfworld"
    )
    cfg = _compose_from(config_dir, "alfworld_qwen35_2b_grpo_smoke")

    _assert_32k_reasoning_contract(cfg)
    assert cfg.actor_rollout_ref.model.path.endswith("Qwen3.5-2B-think")
    assert cfg.actor_rollout_ref.model.use_remove_padding is False
    assert cfg.actor_rollout_ref.rollout.thinking_mode == "native_qwen35"
    assert cfg.actor_rollout_ref.rollout.context_management.enabled is True


def test_alfworld_4b_full_baseline_contract_is_production_scale_on_gpu47():
    config_dir = (
        REPO_ROOT
        / "config/duet_paper_experiments_configs/iclr2027/alfworld"
    )
    cfg = _compose_from(
        config_dir, "alfworld_qwen35_4b_grpo_32k_baseline_gpu47_v3_decode_safe"
    )

    _assert_32k_reasoning_contract(cfg)
    assert (
        cfg.trainer.experiment_name
        == "alfworld_qwen35_4b_grpo_32k_baseline_gpu47_v3_decode_safe"
    )
    assert cfg.actor_rollout_ref.model.path.endswith("Qwen3.5-4B-think")
    assert cfg.actor_rollout_ref.model.use_remove_padding is False
    assert cfg.actor_rollout_ref.rollout.thinking_mode == "native_qwen35"
    assert cfg.actor_rollout_ref.rollout.external_server_addresses == [
        "127.0.0.1:8211",
        "127.0.0.1:8212",
        "127.0.0.1:8213",
        "127.0.0.1:8214",
    ]
    assert cfg.trainer.n_gpus_per_node == 4
    assert cfg.trainer.total_training_steps == 100
    assert cfg.trainer.save_freq == 50
    assert cfg.trainer.test_freq == 50
    assert cfg.trainer.val_before_train is True
    assert cfg.trainer.resume_mode == "disable"
    assert cfg.data.train_batch_size == 8
    assert cfg.data.max_train_tasks == 800
    assert cfg.actor_rollout_ref.rollout.n == 8
    assert cfg.actor_rollout_ref.rollout.max_env_worker == 32
    assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 8
    assert cfg.actor_rollout_ref.rollout.multi_turn.max_steps == 30
    assert cfg.task_manager.strategy_args.max_explore_step == 30
    assert cfg.data.max_val_tasks == 200
    assert cfg.data.validation_shuffle is False
    assert cfg.actor_rollout_ref.rollout.context_management.enabled is True
    assert cfg.actor_rollout_ref.rollout.context_management.recent_turns == 2
    assert (
        cfg.actor_rollout_ref.rollout.context_management.history_observation_max_tokens
        == 160
    )
    assert cfg.actor_rollout_ref.rollout.use_rollout_log_probs_as_old is True
    assert cfg.actor_rollout_ref.actor.skip_zero_advantage_grpo_update is True
    assert cfg.actor_rollout_ref.rollout.rollout_logprob_drift_mean_threshold == 0.02
    assert cfg.actor_rollout_ref.rollout.rollout_logprob_drift_p99_threshold == 0.25
    assert (
        cfg.actor_rollout_ref.rollout.rollout_importance_ratio_outside_clip_threshold
        == 0.01
    )
    assert cfg.actor_rollout_ref.rollout.rollout_logprob_drift_max_threshold == 0.75


def test_alfworld_4b_v4_serial_run_inherits_strict_production_contract():
    config_dir = (
        REPO_ROOT
        / "config/duet_paper_experiments_configs/iclr2027/alfworld"
    )
    cfg = _compose_from(
        config_dir,
        "alfworld_qwen35_4b_grpo_32k_baseline_gpu47_v4_serial_decode_safe",
    )

    _assert_32k_reasoning_contract(cfg)
    assert cfg.trainer.experiment_name.endswith("v4_serial_decode_safe")
    assert cfg.trainer.total_training_steps == 100
    assert cfg.trainer.save_freq == 50
    assert cfg.trainer.test_freq == 50
    assert cfg.trainer.val_before_train is True
    assert cfg.trainer.save_checkpoint_on_identity_failure is True
    assert cfg.trainer.export_actor_weights_on_identity_failure is True
    assert cfg.actor_rollout_ref.model.use_remove_padding is False
    assert cfg.actor_rollout_ref.rollout.max_num_seqs == 1
    assert cfg.actor_rollout_ref.rollout.enforce_eager is False
    assert cfg.actor_rollout_ref.rollout.n == 8
    assert cfg.actor_rollout_ref.rollout.rollout_logprob_drift_mean_threshold == 0.02
    assert cfg.actor_rollout_ref.rollout.rollout_logprob_drift_p99_threshold == 0.25
    assert (
        cfg.actor_rollout_ref.rollout.rollout_importance_ratio_outside_clip_threshold
        == 0.01
    )
    assert cfg.actor_rollout_ref.rollout.rollout_logprob_drift_max_threshold == 0.75


def test_alfworld_4b_v5_resume_changes_only_resume_outputs_and_raw_max_gate():
    config_dir = (
        REPO_ROOT
        / "config/duet_paper_experiments_configs/iclr2027/alfworld"
    )
    v4 = _compose_from(
        config_dir,
        "alfworld_qwen35_4b_grpo_32k_baseline_gpu47_v4_serial_decode_safe",
    )
    v5 = _compose_from(
        config_dir,
        "alfworld_qwen35_4b_grpo_32k_baseline_gpu47_v5_resume_diagnostic_max",
    )

    _assert_32k_reasoning_contract(v5)
    assert (
        v5.trainer.experiment_name
        == "alfworld_qwen35_4b_grpo_32k_baseline_gpu47_v5_resume_diagnostic_max"
    )
    assert v5.trainer.default_local_dir.endswith(
        "/alfworld_qwen35_4b_grpo_32k_baseline_gpu47_v5_resume_diagnostic_max"
    )
    assert v5.trainer.validation_data_dir.endswith(
        "/alfworld_qwen35_4b_grpo_32k_baseline_gpu47_v5_resume_diagnostic_max/validation_log"
    )
    assert v5.trainer.rollout_data_dir.endswith(
        "/alfworld_qwen35_4b_grpo_32k_baseline_gpu47_v5_resume_diagnostic_max/rollout_log"
    )
    assert v5.trainer.total_training_steps == 100
    assert v5.trainer.val_before_train is False
    assert v5.trainer.resume_mode == "resume_path"
    assert v5.trainer.resume_from_path == (
        "/data/home/qisheng/EvolAnalsis/checkpoints/agentevolver/"
        "alfworld_qwen35_4b_grpo_32k_baseline_gpu47_v5_resume_diagnostic_max/"
        "recovery/global_step_1"
    )
    assert v5.actor_rollout_ref.rollout.rollout_logprob_drift_max_threshold == -1
    assert v5.actor_rollout_ref.rollout.rollout_logprob_drift_mean_threshold == 0.02
    assert v5.actor_rollout_ref.rollout.rollout_logprob_drift_p99_threshold == 0.25
    assert (
        v5.actor_rollout_ref.rollout.rollout_importance_ratio_outside_clip_threshold
        == 0.01
    )
    assert v5.actor_rollout_ref.rollout.use_rollout_log_probs_as_old is True
    assert v5.actor_rollout_ref.rollout.context_management.recent_turns == 2
    assert (
        v5.actor_rollout_ref.rollout.context_management.history_observation_max_tokens
        == 160
    )

    assert _normalized_v4_v5_resume_contract(v4) == (
        _normalized_v4_v5_resume_contract(v5)
    )


def test_webshop_4b_s200_baseline_is_paper_scale_and_v5_managed():
    config_dir = (
        REPO_ROOT
        / "config/duet_paper_experiments_configs/iclr2027/webshop"
    )
    cfg = _compose_from(
        config_dir,
        "webshop_qwen35_4b_grpo_32k_baseline_gpu47_v1_s200",
    )

    _assert_32k_reasoning_contract(cfg)
    assert (
        cfg.trainer.experiment_name
        == "webshop_qwen35_4b_grpo_32k_baseline_gpu47_v1_s200"
    )
    assert cfg.trainer.project_name == "agentevolver"
    assert set(cfg.trainer.logger) == {"console", "wandb"}
    assert cfg.trainer.n_gpus_per_node == 4
    assert cfg.trainer.total_epochs == 1
    assert cfg.trainer.total_training_steps == 200
    assert cfg.trainer.save_freq == 50
    assert cfg.trainer.test_freq == 25
    assert cfg.trainer.val_before_train is True
    assert cfg.trainer.resume_mode == "disable"
    assert cfg.trainer.save_checkpoint_on_identity_failure is True
    assert cfg.trainer.export_actor_weights_on_identity_failure is True

    assert cfg.actor_rollout_ref.model.path.endswith("Qwen3.5-4B-think")
    assert cfg.actor_rollout_ref.model.use_remove_padding is False
    assert cfg.actor_rollout_ref.rollout.thinking_mode == "native_qwen35"
    assert cfg.actor_rollout_ref.actor.entropy_coeff == 0
    assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 8
    assert cfg.actor_rollout_ref.rollout.n == 8
    assert cfg.actor_rollout_ref.rollout.max_num_seqs == 1
    assert cfg.actor_rollout_ref.rollout.max_env_worker == 32
    assert cfg.actor_rollout_ref.rollout.max_trajectory_resubmits == 1
    assert cfg.actor_rollout_ref.rollout.multi_turn.max_steps == 30
    assert cfg.task_manager.strategy_args.max_explore_step == 30
    assert cfg.actor_rollout_ref.rollout.temperature == 0.6
    assert cfg.actor_rollout_ref.rollout.rollout_logprob_drift_max_threshold == -1
    assert cfg.actor_rollout_ref.rollout.rollout_logprob_drift_mean_threshold == 0.02
    assert cfg.actor_rollout_ref.rollout.rollout_logprob_drift_p99_threshold == 0.25
    assert (
        cfg.actor_rollout_ref.rollout.rollout_importance_ratio_outside_clip_threshold
        == 0.01
    )

    context = cfg.actor_rollout_ref.rollout.context_management
    assert context.enabled is True
    assert context.recent_turns == 4
    assert context.history_observation_max_tokens == 512
    assert context.recent_observation_max_tokens == -1
    assert context.allow_current_observation_truncation is False
    assert context.reasoning_history_tokens == 0
    assert context.snapshot_training is True
    assert context.snapshot_selection == "token_weighted"

    assert cfg.data.train_batch_size == 8
    assert cfg.data.max_train_tasks == 1600
    assert cfg.data.max_train_tasks // cfg.data.train_batch_size == 200
    assert cfg.data.max_val_tasks == 200
    assert cfg.data.validation_shuffle is False
    assert cfg.data.dataloader_num_workers == 0
    assert cfg.env_service.env_type == "webshop"
    assert cfg.env_service.env_url == "http://127.0.0.1:8083"
    assert cfg.env_service.env_params.action_format == "react_tags"
    assert cfg.exp_manager.teacher_experience.enable is False
    assert cfg.exp_manager.state_channel.enable is False
    assert cfg.exp_manager.experience_replay.enable is False


def test_rollout_launcher_defaults_to_decode_safe_qwen35_gdn_path():
    launcher = (REPO_ROOT / "start_rollout_servers.sh").read_text()

    assert (
        'VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE="'
        '${VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE:-0}"'
    ) in launcher
    assert "vllm_enable_fla_packed_recurrent_decode=" in launcher


def test_webshop_gpu47_queue_is_contract_checked_and_handoff_gated():
    queue_path = REPO_ROOT / "run_iclr_gpu47_queue.sh"
    subprocess.run(["bash", "-n", str(queue_path)], check=True)
    queue = queue_path.read_text(encoding="utf-8")

    assert "webshop_qwen35_4b_grpo_32k_baseline_gpu47_v1_s200" in queue
    assert 'unset WANDB_API_KEY WANDB_IDENTITY_TOKEN_FILE WANDB_CREDENTIALS_FILE' in queue
    assert "preflight_wandb_online" in queue
    assert 'GPU47_HANDOFF_VERIFIED:-0' in queue
    assert 'GPU47_LANE_LOCK_FD:-' in queue
    assert "GPU47_ROLLOUT_GPU_PIDS" in queue
    assert "--query-compute-apps=pid" in queue
    assert 'LANE_RAY_TMPDIR="${RAY_TMPDIR}/ws35s200"' in queue
    ray_socket_example = (
        "/data/ray/ws35s200/"
        "session_2026-08-02_16-16-27_595984_1439351/sockets/plasma_store"
    )
    assert len(ray_socket_example.encode()) <= 107
    for required_field in (
        "model_dir=/data/shared_models/Qwen3.5-4B-think",
        "gpus=4,5,6,7",
        "max_model_len=32768",
        "max_num_seqs=1",
        "logprobs_mode=processed_logprobs",
        "vllm_enable_fla_packed_recurrent_decode=0",
        "health_verified_ports=8211 8212 8213 8214",
    ):
        assert required_field in queue
    queue_commands = "\n".join(
        line for line in queue.splitlines() if not line.lstrip().startswith("#")
    )
    for forbidden in ("launcher.py --kill", "pkill", "ray stop", "kill -9"):
        assert forbidden not in queue_commands


def test_alfworld_to_webshop_handoff_is_exact_and_non_destructive():
    handoff_path = (
        REPO_ROOT
        / "run_scripts/20_main_paper/handoff_alfworld_to_webshop_qwen35_gpu47.sh"
    )
    subprocess.run(["bash", "-n", str(handoff_path)], check=True)
    handoff = handoff_path.read_text(encoding="utf-8")

    assert 'ALFWORLD_LAUNCHER_PID="${ALFWORLD_LAUNCHER_PID:?' in handoff
    assert 'ALFWORLD_TASKRUNNER_PID="${ALFWORLD_TASKRUNNER_PID:?' in handoff
    assert 'ALFWORLD_GPU_WORKER_PIDS:?' in handoff
    assert 'GPU47_ROLLOUT_GPU_PIDS:?' in handoff
    assert "validation_log/100.jsonl" in handoff
    assert "latest_checkpointed_iteration.txt" in handoff
    assert 'grep -aFq "step:100 -"' in handoff
    assert "model_world_size_4_rank_*.pt" in handoff
    assert "optim_world_size_4_rank_*.pt" in handoff
    assert "GPU47_HANDOFF_VERIFIED=1" in handoff
    assert "GPU47_LANE_LOCK_FD=7" in handoff
    assert 'check_gpu_lane_exact "at arm time" 1' in handoff
    assert 'check_gpu_lane_exact "after ALFWorld teardown" 0' in handoff
    assert 'len(set(ids))==200' in handoff
    handoff_commands = "\n".join(
        line for line in handoff.splitlines() if not line.lstrip().startswith("#")
    )
    for forbidden in ("launcher.py --kill", "pkill", "ray stop", "kill -9"):
        assert forbidden not in handoff_commands


def test_webshop_preray_recovery_preserves_failure_and_gpu_gate():
    recovery_path = (
        REPO_ROOT
        / "run_scripts/20_main_paper/recover_webshop_after_preray_afunix_gpu47.sh"
    )
    subprocess.run(["bash", "-n", str(recovery_path)], check=True)
    recovery = recovery_path.read_text(encoding="utf-8")

    assert "AF_UNIX path length cannot exceed 107 bytes" in recovery
    assert "failed_preray_afunix_20260802_161627" in recovery
    assert 'grep -aFq "TaskRunner pid="' in recovery
    assert "--query-compute-apps=pid" in recovery
    assert "GPU47_LANE_LOCK_FD=7" in recovery
    assert "GPU47_HANDOFF_VERIFIED=1" in recovery
    assert 'mv "${FAILED_LOG}" "${FAILED_LOG_ARCHIVE}"' in recovery
    assert 'mv "${FAILED_RECORD}" "${FAILED_RECORD_ARCHIVE}"' in recovery
    recovery_commands = "\n".join(
        line for line in recovery.splitlines() if not line.lstrip().startswith("#")
    )
    for forbidden in ("launcher.py --kill", "pkill", "ray stop", "kill -9", "rm -"):
        assert forbidden not in recovery_commands
