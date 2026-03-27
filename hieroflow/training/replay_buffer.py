"""
Prioritised replay buffer for off-policy GFlowNet training.

Stores completed trajectories (sketch-level and tactic-level) and samples
them with priority proportional to their training signal (|TD-error| + ε).

Implementation uses a ``SumTree`` for O(log n) priority sampling and
separate capacity partitions for sketch/tactic trajectories.

Reference: Schaul et al. (2015) "Prioritized Experience Replay."
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Trajectory dataclass
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    """
    A single completed trajectory stored in the replay buffer.

    Covers both sketch-level (outer GFlowNet) and tactic-level (inner
    GFlowNet) trajectories.  The ``level`` field controls which sub-buffer
    the trajectory is routed to.
    """

    __slots__ = (
        "trajectory_id",
        "level",
        "states",
        "actions",
        "log_pf",
        "log_pb",
        "log_reward",
        "timestamp",
        "priority",
    )

    trajectory_id: str
    """Unique UUID4 string identifying this trajectory."""

    level: str
    """Either 'sketch' (outer) or 'tactic' (inner)."""

    states: list[Any]
    """Sequence of state snapshots visited during the trajectory."""

    actions: list[Any]
    """Sequence of actions taken."""

    log_pf: float
    """Sum of forward log-probabilities Σ log P_F(a_t|s_t)."""

    log_pb: float
    """Sum of backward log-probabilities Σ log P_B(s_t|s_{t+1})."""

    log_reward: float
    """Log-space terminal reward log R(τ)."""

    timestamp: float
    """Wall-clock time when this trajectory was added (for staleness checks)."""

    priority: float
    """Current sampling priority (updated by ``update_priorities``)."""


# ---------------------------------------------------------------------------
# SumTree
# ---------------------------------------------------------------------------

class _SumTree:
    """
    Binary sum-tree for O(log n) priority sampling.

    Stores ``capacity`` leaf priorities.  Supports:
    - ``update(idx, priority)``: update the priority of leaf *idx*.
    - ``sample()``: sample a leaf proportional to priority.
    - ``total``: total sum of all priorities.

    This is a private implementation detail of ``PrioritisedReplayBuffer``.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        # Size = 2 * capacity for a full binary tree
        self._tree = np.zeros(2 * capacity, dtype=np.float64)
        self._size = 0  # Number of valid leaves

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return self._size

    @property
    def total(self) -> float:
        return float(self._tree[1])  # Root holds the total sum

    def update(self, idx: int, priority: float) -> None:
        """Set the priority of leaf *idx* and propagate the change upward."""
        leaf_idx = idx + self._capacity  # Leaves start at capacity
        delta = priority - self._tree[leaf_idx]
        self._tree[leaf_idx] = priority
        # Propagate
        parent = leaf_idx >> 1
        while parent >= 1:
            self._tree[parent] += delta
            parent >>= 1

    def sample(self) -> tuple[int, float]:
        """
        Sample a leaf index proportional to priority.

        Returns:
            ``(leaf_idx, priority)`` where leaf_idx ∈ [0, capacity).
        """
        if self.total <= 0:
            # Uniform fallback
            idx = int(np.random.randint(0, max(self._size, 1)))
            return idx, max(self._tree[idx + self._capacity], 1e-8)

        value = np.random.uniform(0, self.total)
        node = 1  # Start at root
        while node < self._capacity:
            left = node << 1
            right = left + 1
            if value <= self._tree[left]:
                node = left
            else:
                value -= self._tree[left]
                node = right

        leaf_idx = node - self._capacity
        return leaf_idx, float(self._tree[node])

    def add(self, priority: float) -> int:
        """
        Add a new leaf with *priority* using a circular write pointer.

        Returns the leaf index used.
        """
        idx = self._size % self._capacity
        self.update(idx, priority)
        if self._size < self._capacity:
            self._size += 1
        return idx


# ---------------------------------------------------------------------------
# PrioritisedReplayBuffer
# ---------------------------------------------------------------------------

class PrioritisedReplayBuffer:
    """
    Prioritised replay buffer for both sketch and tactic trajectories.

    Trajectories are stored in two separate sub-buffers (sketch / tactic)
    and sampled with ratio 1:3 (tactic over-sampled because they are cheaper
    to generate).

    Priority is set to ``|TD_error| + ε`` where:
        TD_error = log_Z + log_pf - log_reward - log_pb

    Importance sampling weights (for bias correction) are annealed from
    ``beta`` toward 1.0 over ``beta_anneal_steps`` steps.
    """

    _EPSILON: float = 1e-6
    """Small constant added to priorities to ensure non-zero sampling."""

    def __init__(
        self,
        max_size: int = 50_000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_anneal_steps: int = 100_000,
    ) -> None:
        """
        Initialise the buffer.

        Args:
            max_size:           Maximum total trajectories across both levels.
            alpha:              Priority exponent (0 = uniform, 1 = full priority).
            beta:               Initial importance-sampling exponent.
            beta_anneal_steps:  Steps over which beta is annealed to 1.0.
        """
        self.max_size = max_size
        self.alpha = alpha
        self.beta_start = beta
        self.beta_anneal_steps = beta_anneal_steps
        self._step = 0

        # Split capacity 1:3 between sketch and tactic
        sketch_cap = max_size // 4
        tactic_cap = max_size - sketch_cap

        self._sketch_tree = _SumTree(sketch_cap)
        self._tactic_tree = _SumTree(tactic_cap)

        # Actual trajectory storage (circular buffers)
        self._sketch_store: list[Trajectory | None] = [None] * sketch_cap
        self._tactic_store: list[Trajectory | None] = [None] * tactic_cap

        # ID → (level, idx) for priority updates
        self._id_to_loc: dict[str, tuple[str, int]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def beta(self) -> float:
        """Current beta value (annealed toward 1.0)."""
        fraction = min(1.0, self._step / max(self.beta_anneal_steps, 1))
        return self.beta_start + fraction * (1.0 - self.beta_start)

    def add(self, trajectory: Trajectory) -> None:
        """
        Add a trajectory to the appropriate sub-buffer.

        Priority is computed as:
            |log_Z + log_pf - log_reward - log_pb| + ε

        Note: ``log_Z`` is not stored in ``Trajectory``; we approximate it
        as 0 here.  The trainer calls ``update_priorities`` once log_Z is
        available.

        Args:
            trajectory: The completed trajectory to store.

        Raises:
            ValueError: If ``trajectory.level`` is not 'sketch' or 'tactic'.
        """
        if trajectory.level not in ("sketch", "tactic"):
            raise ValueError(
                f"Unknown trajectory level '{trajectory.level}'. "
                "Expected 'sketch' or 'tactic'."
            )

        td_error = abs(
            trajectory.log_pf - trajectory.log_reward - trajectory.log_pb
        )
        priority = (td_error + self._EPSILON) ** self.alpha

        if trajectory.level == "sketch":
            idx = self._sketch_tree.add(priority)
            self._sketch_store[idx] = trajectory
        else:
            idx = self._tactic_tree.add(priority)
            self._tactic_store[idx] = trajectory

        # Track the location for future priority updates
        self._id_to_loc[trajectory.trajectory_id] = (trajectory.level, idx)
        trajectory.priority = priority

    def sample(
        self, batch_size: int
    ) -> tuple[list[Trajectory], torch.Tensor]:
        """
        Sample a batch of trajectories with importance-sampling weights.

        Samples are drawn at a 1:3 sketch:tactic ratio.  If one sub-buffer
        is empty, all samples are drawn from the other.

        Args:
            batch_size: Total number of trajectories to return.

        Returns:
            ``(trajectories, importance_weights)`` where
            importance_weights is a ``torch.Tensor`` of shape ``[batch_size]``
            normalised so that the maximum weight is 1.
        """
        self._step += 1

        n_sketch = max(1, batch_size // 4)
        n_tactic = batch_size - n_sketch

        # Clamp to available items
        n_sketch = min(n_sketch, self._sketch_tree.size)
        n_tactic = min(n_tactic, self._tactic_tree.size)
        if n_sketch + n_tactic == 0:
            return [], torch.zeros(0)

        sketch_samples = self._sample_from(
            self._sketch_tree, self._sketch_store, n_sketch
        )
        tactic_samples = self._sample_from(
            self._tactic_tree, self._tactic_store, n_tactic
        )

        all_samples = sketch_samples + tactic_samples
        trajectories = [t for t, _ in all_samples]
        raw_weights = [w for _, w in all_samples]

        # Importance-sampling correction: w_i = (1/N * 1/P(i))^beta
        N = self._sketch_tree.size + self._tactic_tree.size
        total_priority = self._sketch_tree.total + self._tactic_tree.total
        weights = torch.tensor(
            [
                ((N * p / total_priority) ** (-self.beta))
                for p in raw_weights
            ],
            dtype=torch.float32,
        )
        # Normalise by max weight for stability
        weights = weights / weights.max().clamp(min=1e-8)

        return trajectories, weights

    def update_priorities(
        self,
        trajectory_ids: list[str],
        new_td_errors: torch.Tensor,
    ) -> None:
        """
        Update priorities for a batch of trajectories after a gradient step.

        Args:
            trajectory_ids: IDs of trajectories to update.
            new_td_errors:  ``torch.Tensor`` of shape ``[len(trajectory_ids)]``
                            with new |TD_error| values.
        """
        for tid, td_err in zip(trajectory_ids, new_td_errors.detach().cpu()):
            if tid not in self._id_to_loc:
                continue
            level, idx = self._id_to_loc[tid]
            priority = (float(abs(td_err.item())) + self._EPSILON) ** self.alpha
            if level == "sketch":
                self._sketch_tree.update(idx, priority)
                if self._sketch_store[idx] is not None:
                    self._sketch_store[idx].priority = priority  # type: ignore[union-attr]
            else:
                self._tactic_tree.update(idx, priority)
                if self._tactic_store[idx] is not None:
                    self._tactic_store[idx].priority = priority  # type: ignore[union-attr]

    def get_stats(self) -> dict[str, float | int]:
        """
        Return buffer statistics for logging.

        Returns:
            Dict with keys: ``buffer_size``, ``sketch_size``, ``tactic_size``,
            ``mean_priority``, ``mean_log_reward``, ``sketch_tactic_ratio``.
        """
        sketch_trajs = [
            t for t in self._sketch_store if t is not None
        ]
        tactic_trajs = [
            t for t in self._tactic_store if t is not None
        ]
        all_trajs = sketch_trajs + tactic_trajs

        mean_priority = (
            float(np.mean([t.priority for t in all_trajs]))
            if all_trajs
            else 0.0
        )
        mean_log_reward = (
            float(np.mean([t.log_reward for t in all_trajs]))
            if all_trajs
            else 0.0
        )
        ratio = (
            len(sketch_trajs) / max(len(tactic_trajs), 1)
        )
        return {
            "buffer_size": len(all_trajs),
            "sketch_size": len(sketch_trajs),
            "tactic_size": len(tactic_trajs),
            "mean_priority": mean_priority,
            "mean_log_reward": mean_log_reward,
            "sketch_tactic_ratio": ratio,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_from(
        self,
        tree: _SumTree,
        store: list[Trajectory | None],
        n: int,
    ) -> list[tuple[Trajectory, float]]:
        """Sample *n* trajectories from *tree* / *store*."""
        results: list[tuple[Trajectory, float]] = []
        for _ in range(n):
            idx, priority = tree.sample()
            traj = store[idx]
            if traj is not None:
                results.append((traj, priority))
        return results
