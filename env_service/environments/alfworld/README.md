# AlfWorld Environment Integration

This document describes how to integrate AgentGym's AlfWorld environment into AgentEvolver for Experience Pool and GRPO training.

## Overview

The AlfWorld environment has been integrated into AgentEvolver by:
1. Creating an `AlfworldEnv` class that implements the `BaseEnv` interface
2. Registering it with the environment registry
3. Adding launch script support
4. Configuring launcher to support AlfWorld

## Prerequisites

1. **AgentGym AlfWorld Environment**: Ensure `AgentGym/agentenv-alfworld/` is properly set up
   ```bash
   cd AgentGym/agentenv-alfworld
   bash setup.sh
   ```

2. **AlfWorld Data**: Make sure `ALFWORLD_DATA` is set (default: `~/.cache/alfworld`)
   ```bash
   export ALFWORLD_DATA=~/.cache/alfworld
   alfworld-download
   ```

## Configuration

### 1. Environment Variables

Add to your `.env` file (or `example.env`):

```bash
# AlfWorld Configuration
ALFWORLD_PATH="./env_service/launch_script"
ALFWORLD_SCRIPT="bash alfworld.sh"
ALFWORLD_DATA=~/.cache/alfworld  # Optional, defaults to ~/.cache/alfworld
```

### 2. Training Configuration

In your training YAML config (`config/agentevolver.yaml` or similar), set:

```yaml
env_service:
  env_type: "alfworld"
  env_url: "http://127.0.0.1:8080"

data:
  # If using environment-provided tasks
  train_files: null  # Will load from env_service
  val_files: null
```

## Usage

### 1. Launch Environment Service

```bash
# Option 1: Using launcher
python launcher.py --conf config/your_config.yaml --with-alfworld

# Option 2: Direct launch
cd env_service/launch_script
bash alfworld.sh
```

### 2. Start Training

```bash
python launcher.py \
  --conf config/your_config.yaml \
  --with-alfworld \
  --with-reme  # If using Experience Pool
```

## How It Works

### Task ID Format

- **Seed Tasks**: Task IDs are game indices (0, 1, 2, ...) as strings
- **Task Loading**: `get_query_list(split)` loads from `mappings_train.json` or `mappings_test.json`
- **Game Selection**: The `task_id` is used as an index into the games list

### Environment Interface

The `AlfworldEnv` class:
- Wraps AgentGym's `ALFWorld_Wrapper`
- Implements `BaseEnv` interface:
  - `get_init_state()`: Initializes environment with a specific game
  - `step()`: Executes actions and returns observations
  - `evaluate()`: Returns reward (0.0 or 1.0 for success)
  - `get_query_list()`: Returns list of available task IDs
  - `close()`: Cleans up resources

### Integration Points

1. **TaskManager**: Loads seed tasks via `load_tasks_from_environment()`
   - Calls `env_service.get_env_profile(env_type="alfworld", split="train")`
   - Gets list of task IDs (game indices)

2. **EnvWorker**: Creates environment instances during rollout
   - Calls `env.create_instance(env_type="alfworld", task_id=...)`
   - Uses task_id as game index

3. **Experience Pool**: Works seamlessly with AlfWorld
   - Trajectories are summarized and stored
   - Historical experience can be retrieved for context

4. **GRPO Training**: Standard GRPO/PPO training works with AlfWorld
   - Advantages computed from rewards
   - Policy updates based on trajectories

## Troubleshooting

### Issue: "Environment 'alfworld' not found"

**Solution**: Make sure the environment is registered:
- Check that `alfworld_env.py` exists in `env_service/environments/alfworld/`
- Verify `@Registry.register("alfworld")` decorator is present
- Ensure `env_service/env_service.py` can import the module

### Issue: "Failed to create AlfWorld environment"

**Solution**: 
- Check `ALFWORLD_DATA` is set correctly
- Verify `configs/base_config.yaml` exists in AgentGym directory
- Ensure AlfWorld dependencies are installed

### Issue: "Could not find game index for task_id"

**Solution**:
- Task IDs should be integers (as strings) representing game indices
- Check that `mappings_train.json` and `mappings_test.json` exist
- Verify task_id is within valid range [0, len(games))

## File Structure

```
env_service/
├── environments/
│   └── alfworld/
│       ├── __init__.py
│       └── alfworld_env.py      # Main environment implementation
├── launch_script/
│   └── alfworld.sh              # Launch script
└── ...

AgentGym/
└── agentenv-alfworld/
    ├── agentenv_alfworld/
    │   ├── env_wrapper.py       # AgentGym wrapper
    │   └── ...
    └── configs/
        ├── mappings_train.json
        └── mappings_test.json
```

## Next Steps

1. **Test Integration**: Run a simple training loop to verify everything works
2. **Configure Experience Pool**: Set up ReMe if you want to use Experience Pool
3. **Tune Hyperparameters**: Adjust GRPO/PPO parameters for AlfWorld tasks
4. **Monitor Training**: Use logview to track training progress

## Notes

- AlfWorld uses discrete actions (text commands)
- Rewards are sparse (0.0 for failure, 1.0 for success)
- Task IDs are game indices, not semantic task descriptions
- The environment supports "Text", "Embody", and "Hybrid" world types (default: "Text")

