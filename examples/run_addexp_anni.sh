ulimit -n 1048576
ulimit -s 16384

# completion_callback=none
env_url=http://localhost:8000
current_time=$(date "+%Y%m%d_%H%M%S")
log_file="dlc_log_${current_time}.log"
export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export ES_HOSTS=http://11.160.132.46:8200
export HF_ENDPOINT=https://hf-mirror.com


# Environment activate
. /mnt/data/zouanni.zan/miniconda3/etc/profile.d/conda.sh;
conda activate appworld;
cd /mnt/data/zouanni.zan/service_0815/EnvService
bash env_sandbox/appworld.sh &


# Set ExperienceMaker service
conda activate em;
cd /mnt/data/zouanni.zan/service_0815/ExperienceMaker
experiencemaker \
  http_service.host="127.0.0.1" \
  http_service.port=8001 \
  llm.default.model_name=qwen-max-2025-01-25 \
  embedding_model.default.model_name=text-embedding-v4 \
  vector_store.default.backend=local_file \
  op.rerank_experience_op.params.enable_llm_rerank=false \
  op.experience_validation_op.params.validation_threshold=0.3 \
  op.experience_deduplication_op.params.similarity_threshold=0.3 \
  http_service.limit_concurrency=256 \
  thread_pool.max_workers=256 &
sleep 30
em_url=http://localhost:8001
val_rollout_expmode="mixed"
train_rollout_expmode="mixed"
rollout_expratio=0.5
train_sample_expmode="discard"
train_sample_keepratio=0.0
clip_ratio_high=0.2
off_cliprange_high=0.2

conda activate verl
cd /mnt/data/zouanni.zan/BeyondAgent;

set -xeu

export HYDRA_FULL_ERROR=1
# export RAY_DEBUG_POST_MORTEM=1


PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/config"

export PYTHONFAULTHANDLER=1

python3 -m beyondagent.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='beyond_agent_dataflow' \
    env_service.env_url=$env_url \
    experience_maker.base_url=$em_url \
    experience_maker.enable_context_generator=True \
    experience_maker.enable_summarizer=False \
    experience_maker.workspace_id="qwen2.5-14b_appworld_syndata" \
    experience_maker.updated_freq=0 \
    actor_rollout_ref.actor.off_cliprange_high=${off_cliprange_high} \
    hybrid_experience_training.val_rollout_expmode=${val_rollout_expmode} \
    hybrid_experience_training.train_rollout_expmode=${train_rollout_expmode} \
    hybrid_experience_training.train_sample_expmode=${train_sample_expmode} \
    hybrid_experience_training.rollout_expratio=${rollout_expratio} \
    hybrid_experience_training.train_sample_keepratio=${train_sample_keepratio} \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.max_prompt_length=6000 \
    data.max_response_length=17480 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.use_qwen3=False \
    actor_rollout_ref.rollout.enable_request_id=False \
    actor_rollout_ref.rollout.prompt_length=23480 \
    actor_rollout_ref.rollout.response_length=2048 \
    actor_rollout_ref.rollout.max_model_len=23480 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.model.path=/mnt/data/zouanni.zan/models/Qwen2.5-14B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.n_gpus_per_node=8 \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.project_name='ba-techreport' \
    trainer.experiment_name="batech-qwen25_14b-appworld_baseline" \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=30 \
    trainer.val_before_train=False \
    trainer.validation_data_dir="experiments/exp_${current_time}/validation_log" \
    trainer.rollout_data_dir="experiments/exp_${current_time}/rollout_log" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=23480 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=23480 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=23480 \
    critic.ppo_max_token_len_per_gpu=23480 \
    critic.forward_max_token_len_per_gpu=23480 \
    data.train_files=null \
    data.val_files=null \
    env_service.env_type=appworld \
    task_manager.n=0 \
    task_manager.train_data_path=/mnt/data/zouanni.zan/synthetic_data/tasks_explored.train.appworld0909_1.json \
    task_manager.mixture.synthetic_data_ratio=99999.0 \
    task_manager.mixture.use_original_tasks=True \
    task_manager.grader.synthetic_grader=llm-binary-gt-no_constraint \
    task_manager.llm_client=qwen3-235b-a22b-instruct-2507 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    2>&1 | tee "$log_file" \