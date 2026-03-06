from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agentevolver.schema.trajectory import Reward


@dataclass
class CellContinuation:
    continuation_id: str
    task_id: str
    cell_id: str
    source: str
    suffix_steps: List[dict]
    success_label: float
    final_reward: float
    progress_score: float = 0.0
    old_log_probs: Optional[List[float]] = None
    response_mask: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CellStats:
    visit_count: int = 0
    success_count: int = 0
    best_reward: float = 0.0
    progress_score_mean: float = 0.0
    teacher_count: int = 0
    onpolicy_count: int = 0
    replay_count: int = 0
    source_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if self.visit_count <= 0:
            return 0.0
        return float(self.success_count) / float(self.visit_count)

    @property
    def utility(self) -> float:
        p = self.success_rate
        return p * (1.0 - p)


@dataclass
class FrontierCell:
    cell_id: str
    task_id: str
    frontier_hash: str
    frontier_depth: int
    prefix_steps: List[dict]
    suffix_pool: List[CellContinuation] = field(default_factory=list)
    stats: CellStats = field(default_factory=CellStats)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_continuation(self, continuation: CellContinuation) -> None:
        self.suffix_pool.append(continuation)
        self.stats.visit_count += 1
        if continuation.success_label > 0:
            self.stats.success_count += 1
        self.stats.best_reward = max(self.stats.best_reward, float(continuation.final_reward))

        prev_n = max(self.stats.visit_count - 1, 0)
        if self.stats.visit_count == 1:
            self.stats.progress_score_mean = float(continuation.progress_score)
        else:
            self.stats.progress_score_mean = (
                self.stats.progress_score_mean * prev_n + float(continuation.progress_score)
            ) / float(self.stats.visit_count)

        src = continuation.source or "unknown"
        self.stats.source_counts[src] = self.stats.source_counts.get(src, 0) + 1
        if src == "teacher":
            self.stats.teacher_count += 1
        elif src.startswith("onpolicy"):
            self.stats.onpolicy_count += 1
        else:
            self.stats.replay_count += 1


@dataclass
class FrontierReplaySample:
    task_id: str
    cell_id: str
    group_id: int
    prefix_steps: List[dict]
    suffix_steps: List[dict]
    reward: Reward
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
