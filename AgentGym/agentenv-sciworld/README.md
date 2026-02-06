# Agent Environments - SciWorld

## Setup
Before running: You will have to have Java 1.8+ installed on your system (shipped with most linux distributions).
``` sh
conda create --name agentenv-sciworld python=3.8
conda activate agentenv-sciworld
pip install -e .
```

## Launch

``` sh
sciworld --host 0.0.0.0 --port 36001
```

## Gold path / Ground-truth trajectories

ScienceWorld can (optionally) generate a **gold action sequence** ("gold path") for a task variation.
This is useful to construct ground-truth trajectories by replaying those actions and recording
observations/scores.

- **Generate gold path**: call `/reset` with `generate_gold_path=true`
- **Fetch gold actions**: call `GET /gold_action_sequence?id=...`

Important notes:
- The gold action sequence is **not guaranteed to be optimal**.
- Gold path generation may fail for some tasks/variations.
