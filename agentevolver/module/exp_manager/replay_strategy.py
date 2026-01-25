"""
Replay Strategy Module for RePO (Replay-Enhanced Policy Optimization).

This module implements the 4 replay strategies from the RePO paper:
1. FullScopeStrategy: Use all historical samples for a prompt
2. RecencyStrategy: Use the most recent K samples (default, recommended)
3. RewardOrientedStrategy: Use samples with highest rewards
4. VarianceDrivenStrategy: Select samples that maximize reward variance

Reference: RePO Paper Section 3.3 - Diverse Replay Strategies
"""

import random
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agentevolver.schema.trajectory import Trajectory


class ReplayStrategy(ABC):
    """
    Abstract base class for replay strategies.
    
    All strategies implement the `select` method to choose trajectories
    from the replay buffer for a given task.
    """
    
    @abstractmethod
    def select(
        self,
        trajectories: List["Trajectory"],
        k: int,
    ) -> List["Trajectory"]:
        """
        Select k trajectories from the available trajectories.
        
        Args:
            trajectories: List of available trajectories for a task
            k: Number of trajectories to select (G_off in the paper)
            
        Returns:
            List of selected trajectories
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class FullScopeStrategy(ReplayStrategy):
    """
    Full-scope Strategy: Use all historical samples for a prompt.
    
    From RePO paper:
    - Advantage: Most thorough data utilization
    - Disadvantage: May include too many samples from old policies, 
      leading to large distribution shift
    
    Note: This strategy ignores the `k` parameter and returns all trajectories.
    The importance ratio clipping will handle distribution shift.
    """
    
    def select(
        self,
        trajectories: List["Trajectory"],
        k: int,
    ) -> List["Trajectory"]:
        """Return all available trajectories (ignores k)."""
        return trajectories.copy()


class RecencyStrategy(ReplayStrategy):
    """
    Recency-based Strategy: Use the most recent K samples.
    
    From RePO paper:
    - Advantage: Closer to current policy, smaller distribution shift
    - Disadvantage: May lose valuable early samples
    
    This is the DEFAULT and RECOMMENDED strategy in RePO.
    
    Trajectories are sorted by `stored_step` (global training step when stored),
    with most recent first.
    """
    
    def select(
        self,
        trajectories: List["Trajectory"],
        k: int,
    ) -> List["Trajectory"]:
        """Select the k most recently stored trajectories."""
        if len(trajectories) <= k:
            return trajectories.copy()
        
        # Sort by stored_step (descending - most recent first)
        # Fallback to stored_timestamp if stored_step not available
        def get_recency_key(traj: "Trajectory") -> float:
            if hasattr(traj, 'metadata') and traj.metadata:
                # Prefer stored_step (integer step count)
                if 'stored_step' in traj.metadata:
                    return float(traj.metadata['stored_step'])
                # Fallback to stored_timestamp
                if 'stored_timestamp' in traj.metadata:
                    return float(traj.metadata['stored_timestamp'])
            return 0.0
        
        sorted_trajs = sorted(
            trajectories,
            key=get_recency_key,
            reverse=True  # Most recent first
        )
        return sorted_trajs[:k]


class RewardOrientedStrategy(ReplayStrategy):
    """
    Reward-oriented Strategy: Use samples with highest rewards.
    
    From RePO paper:
    - Advantage: Reinforces known good behaviors
    - Disadvantage: May overfit to specific patterns
    
    Trajectories are sorted by reward (descending - highest first).
    """
    
    def select(
        self,
        trajectories: List["Trajectory"],
        k: int,
    ) -> List["Trajectory"]:
        """Select the k trajectories with highest rewards."""
        if len(trajectories) <= k:
            return trajectories.copy()
        
        def get_reward(traj: "Trajectory") -> float:
            # Try metadata first (may have stored reward)
            if hasattr(traj, 'metadata') and traj.metadata:
                if 'reward' in traj.metadata:
                    return float(traj.metadata['reward'])
            # Fallback to reward.outcome
            if hasattr(traj, 'reward') and traj.reward:
                if hasattr(traj.reward, 'outcome'):
                    return float(traj.reward.outcome)
            return 0.0
        
        sorted_trajs = sorted(
            trajectories,
            key=get_reward,
            reverse=True  # Highest reward first
        )
        return sorted_trajs[:k]


class VarianceDrivenStrategy(ReplayStrategy):
    """
    Variance-driven Strategy: Select samples that maximize reward variance.
    
    From RePO paper:
    - Purpose: Mitigate vanishing gradient signal when reward discrimination is low
    - Implementation: Greedily select samples to maximize within-group reward variance
    
    This strategy is useful when:
    - Most samples have similar rewards (low discrimination)
    - Need to amplify learning signal from diverse outcomes
    
    Algorithm:
    1. Start with samples having min and max rewards
    2. Greedily add samples that maximize variance of selected set
    """
    
    def select(
        self,
        trajectories: List["Trajectory"],
        k: int,
    ) -> List["Trajectory"]:
        """Select k trajectories that maximize reward variance."""
        if len(trajectories) <= k:
            return trajectories.copy()
        
        # Extract rewards
        def get_reward(traj: "Trajectory") -> float:
            if hasattr(traj, 'metadata') and traj.metadata:
                if 'reward' in traj.metadata:
                    return float(traj.metadata['reward'])
            if hasattr(traj, 'reward') and traj.reward:
                if hasattr(traj.reward, 'outcome'):
                    return float(traj.reward.outcome)
            return 0.0
        
        rewards = [get_reward(t) for t in trajectories]
        n = len(trajectories)
        
        # Edge case: k <= 2, just return min and max reward samples
        if k <= 2:
            sorted_idx = sorted(range(n), key=lambda i: rewards[i])
            if k == 1:
                # Return the one with max reward
                return [trajectories[sorted_idx[-1]]]
            else:
                # Return min and max
                return [trajectories[sorted_idx[0]], trajectories[sorted_idx[-1]]]
        
        # Start with min and max reward samples
        sorted_idx = sorted(range(n), key=lambda i: rewards[i])
        selected_idx = [sorted_idx[0], sorted_idx[-1]]  # min and max
        
        # Greedily add samples that maximize variance
        remaining = set(range(n)) - set(selected_idx)
        
        while len(selected_idx) < k and remaining:
            best_idx = None
            best_var = -1.0
            
            for idx in remaining:
                # Compute variance if we add this sample
                test_rewards = [rewards[i] for i in selected_idx + [idx]]
                var = float(np.var(test_rewards))
                
                if var > best_var:
                    best_var = var
                    best_idx = idx
            
            if best_idx is not None:
                selected_idx.append(best_idx)
                remaining.remove(best_idx)
            else:
                # No improvement possible, break
                break
        
        return [trajectories[i] for i in selected_idx]


class RandomStrategy(ReplayStrategy):
    """
    Random Strategy: Randomly select K samples.
    
    This is a simple baseline strategy not in the original RePO paper,
    but useful for comparison and debugging.
    """
    
    def select(
        self,
        trajectories: List["Trajectory"],
        k: int,
    ) -> List["Trajectory"]:
        """Randomly select k trajectories."""
        if len(trajectories) <= k:
            return trajectories.copy()
        
        return random.sample(trajectories, k)


class ReplayStrategyFactory:
    """
    Factory class for creating ReplayStrategy instances.
    
    Usage:
        strategy = ReplayStrategyFactory.create("recency")
        selected = strategy.select(trajectories, k=8)
    """
    
    _strategies = {
        "full_scope": FullScopeStrategy,
        "fullscope": FullScopeStrategy,
        "recency": RecencyStrategy,
        "latest": RecencyStrategy,  # Alias
        "reward": RewardOrientedStrategy,
        "reward_oriented": RewardOrientedStrategy,
        "variance": VarianceDrivenStrategy,
        "variance_driven": VarianceDrivenStrategy,
        "random": RandomStrategy,
    }
    
    @classmethod
    def create(cls, strategy_name: str) -> ReplayStrategy:
        """
        Create a ReplayStrategy instance by name.
        
        Args:
            strategy_name: Name of the strategy (case-insensitive)
                - "full_scope" or "fullscope": FullScopeStrategy
                - "recency" or "latest": RecencyStrategy (default)
                - "reward" or "reward_oriented": RewardOrientedStrategy
                - "variance" or "variance_driven": VarianceDrivenStrategy
                - "random": RandomStrategy
                
        Returns:
            ReplayStrategy instance
            
        Raises:
            ValueError: If strategy_name is not recognized
        """
        name = strategy_name.lower().strip()
        
        if name not in cls._strategies:
            available = list(cls._strategies.keys())
            raise ValueError(
                f"Unknown replay strategy: '{strategy_name}'. "
                f"Available strategies: {available}"
            )
        
        return cls._strategies[name]()
    
    @classmethod
    def available_strategies(cls) -> List[str]:
        """Return list of available strategy names."""
        return list(cls._strategies.keys())
