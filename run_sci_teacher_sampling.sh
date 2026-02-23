conda run -n agentevolver --no-capture-output \
  python scripts/synthesize_sciworld_teacher_from_gold.py \
  --model_path /data/code/exp/models/Qwen/Qwen2.5-72B-Instruct \
  --tensor_parallel_size 4 \
  --inputs data/teacher_trajectories/sciworld_gold_augmented_new.jsonl \
  --output data/teacher_trajectories/sciworld_gold_qwen72b_synth.jsonl \
  --max_steps_per_task 50 \
  --resume \
  --export_base data/teacher_trajectories/sciworld_gold_qwen72b_filtered \
  --export_threshold 1.0
