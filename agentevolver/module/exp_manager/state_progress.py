"""
DUET State Channel: Expert Progress-Based Reward Shaping

This module implements the State Channel of the DUET framework. It constructs
a progress function Φ(s) from expert trajectory state sequences, computes
trajectory-level progress P(τ) = mean(Φ(s_t)), and provides shaped reward
R'(τ) = R(τ) + β·P(τ) to break reward sparsity in GRPO.

Key classes:
    ExpertProgressMap: Builds and caches per-task hash maps from teacher
        trajectory observations to progress values in [0, 1].

Key functions:
    normalize_observation: Strip dynamic suffixes (AVAILABLE ACTIONS, etc.)
    extract_observations_from_steps: Extract obs from Trajectory.steps
    extract_observations_from_batch_messages: Extract obs from batch data
"""

import re
from typing import Dict, List, Optional, Tuple

from loguru import logger


# ---------------------------------------------------------------------------
# Observation normalization
# ---------------------------------------------------------------------------

def normalize_observation(obs_text: str, env_type: str = "alfworld") -> str:
    """
    Strip dynamic suffixes from environment observations so the same
    underlying state always hashes to the same key.

    ALFWorld appends "\\nAVAILABLE ACTIONS: ..." after every observation.
    WebShop appends "\\n\\nYou can use: ..." and "\\nClickable elements: ...".
    SciWorld appends "\\nPossible actions: ...".
    """
    if not obs_text:
        return ""

    text = obs_text.strip()

    if env_type == "alfworld":
        idx = text.find("\nAVAILABLE ACTIONS:")
        if idx != -1:
            text = text[:idx]
    elif env_type == "webshop":
        patterns = [
            r'\n\nYou can use:.*$',
            r'\n\nClickable elements:.*$',
            r'\nClickable elements:.*$',
            r'\nAvailable actions:.*$',
        ]
        for pat in patterns:
            text = re.sub(pat, '', text, flags=re.DOTALL)
    elif env_type == "sciworld":
        patterns = [
            r'\nPossible actions:.*$',
            r'\nAvailable actions:.*$',
        ]
        for pat in patterns:
            text = re.sub(pat, '', text, flags=re.DOTALL)

    return text.strip()


# ---------------------------------------------------------------------------
# Observation extraction helpers
# ---------------------------------------------------------------------------

def extract_observations_from_steps(
    steps: List[dict],
    env_type: str = "alfworld",
    skip_initial: int = 3,
) -> List[str]:
    """
    Extract normalized environment observations from a Trajectory's .steps list.

    Teacher trajectories loaded via exp_manager have .steps as List[dict]
    (originally from the "messages" field in JSONL). The first *skip_initial*
    messages are typically: system prompt, assistant ack, initial user query.
    Subsequent role=="user" messages are environment observations.
    """
    observations = []
    for i, msg in enumerate(steps):
        if i < skip_initial:
            continue
        if msg.get("role") == "user":
            normalized = normalize_observation(msg.get("content", ""), env_type)
            if normalized:
                observations.append(normalized)
    return observations


def extract_observations_from_batch_messages(
    msg_data,
    env_type: str = "alfworld",
    skip_initial: int = 3,
) -> List[str]:
    """
    Extract normalized observations from the batch non_tensor_batch["messages"]
    format.  Each element is {"messages": [{"role": ..., "content": ...}, ...]}.
    """
    if msg_data is None:
        return []

    # Unwrap the {"messages": [...]} wrapper
    if isinstance(msg_data, dict) and "messages" in msg_data:
        msg_list = msg_data["messages"]
    elif isinstance(msg_data, (list, tuple)):
        msg_list = msg_data
    else:
        return []

    return extract_observations_from_steps(msg_list, env_type, skip_initial)


# ---------------------------------------------------------------------------
# ExpertProgressMap
# ---------------------------------------------------------------------------

class ExpertProgressMap:
    """
    Builds a per-task hash map from normalized environment observations to
    progress values in [0, 1].  Constructed once from teacher trajectories
    and queried during training to compute trajectory progress P(τ).

    For each task, the map records the *maximum* progress index at which
    each observation appears across all expert trajectories for that task.
    """

    def __init__(
        self,
        teacher_task2trajectories: Dict[str, list],
        env_type: str = "alfworld",
        match_mode: str = "hash",
    ):
        self.env_type = env_type
        self.match_mode = match_mode
        # task_id -> {normalized_obs_string -> progress_float}
        self.progress_maps: Dict[str, Dict[str, float]] = {}

        total_states = 0
        total_tasks = 0

        for task_id, trajectories in teacher_task2trajectories.items():
            progress_map: Dict[str, float] = {}
            for traj in trajectories:
                steps = traj.steps if hasattr(traj, 'steps') else []
                if not steps:
                    continue
                obs_list = extract_observations_from_steps(steps, env_type)
                T = len(obs_list)
                if T == 0:
                    continue
                for j, obs in enumerate(obs_list):
                    # j/(T-1) so the last observation = 1.0
                    progress = j / max(T - 1, 1)
                    progress_map[obs] = max(progress_map.get(obs, 0.0), progress)
                total_states += T

            if progress_map:
                self.progress_maps[task_id] = progress_map
                total_tasks += 1

        total_keys = sum(len(m) for m in self.progress_maps.values())
        logger.info(
            f"[State Channel] Built ExpertProgressMap: "
            f"{total_tasks} tasks, {total_states} total expert observations, "
            f"{total_keys} unique state keys"
        )

    # ------------------------------------------------------------------
    # Core lookups
    # ------------------------------------------------------------------

    def has_task(self, task_id: str) -> bool:
        return task_id in self.progress_maps

    def get_potential(self, task_id: str, observation: str) -> float:
        """Φ(s): return the state progress value in [0, 1], or 0.0 if unmatched."""
        pmap = self.progress_maps.get(task_id)
        if pmap is None:
            return 0.0
        return pmap.get(observation, 0.0)

    def compute_trajectory_progress(
        self, task_id: str, observations: List[str]
    ) -> float:
        """P(τ) = (1/T) Σ Φ(s_t)  — average progress over the trajectory."""
        if not observations:
            return 0.0
        total = sum(self.get_potential(task_id, obs) for obs in observations)
        return total / len(observations)

    def compute_step_deltas(
        self, task_id: str, observations: List[str]
    ) -> Tuple[List[float], List[float]]:
        """
        Compute per-step potentials and deltas.
        Returns:
            potentials: [Φ(s_0), Φ(s_1), ...]
            deltas:     [Φ(s_1)-Φ(s_0), Φ(s_2)-Φ(s_1), ...]  (len = len(obs)-1)
        """
        potentials = [self.get_potential(task_id, obs) for obs in observations]
        deltas = [potentials[t + 1] - potentials[t] for t in range(len(potentials) - 1)]
        return potentials, deltas

    def get_coverage_stats(
        self, task_id: str, observations: List[str]
    ) -> Dict[str, float]:
        """Coverage diagnostics for a single trajectory."""
        if not observations:
            return {"coverage": 0.0, "matched": 0, "total": 0,
                    "mean_potential": 0.0, "max_potential": 0.0}
        pmap = self.progress_maps.get(task_id, {})
        potentials = [self.get_potential(task_id, obs) for obs in observations]
        # Check key existence (not > 0) since the first state has progress=0.0
        matched = sum(1 for obs in observations if obs in pmap)
        return {
            "coverage": matched / len(observations),
            "matched": matched,
            "total": len(observations),
            "mean_potential": sum(potentials) / len(potentials),
            "max_potential": max(potentials),
        }

    def get_global_stats(self) -> Dict[str, int]:
        """Summary statistics for the whole map."""
        return {
            "num_tasks": len(self.progress_maps),
            "total_keys": sum(len(m) for m in self.progress_maps.values()),
        }
