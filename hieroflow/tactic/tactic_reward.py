"""
Tactic reward: Lean verification result (binary + partial).

The inner reward is computed from the ``TacticResult`` returned by
``LeanEnv.step()``.  All values are stored in log-space as required by the
cross-cutting design rules.

Reward schedule:
    - 1.0  (log 0.0)   if tactic closes the current goal completely
    - 0.5  (log -0.69) if tactic reduces the goal count (partial progress)
    - 0.0  (clamped)   if tactic causes a Lean error
    - -0.1 (clamped)   if tactic times out
"""

from __future__ import annotations

import math

import torch

from hieroflow.environment.proof_state import TacticResult

# Stable floor for log-space rewards
_LOG_ZERO: float = -100.0


def compute_tactic_log_reward(
    result: TacticResult,
    original_goal_count: int,
    timeout: bool = False,
    partial_reward_scale: float = 0.5,
) -> torch.Tensor:
    """
    Compute the inner GFlowNet reward for a single tactic application.

    Args:
        result:              ``TacticResult`` from ``LeanEnv.step()``.
        original_goal_count: Number of open goals *before* the tactic.
        timeout:             Whether the tactic exceeded the time budget.
        partial_reward_scale: Scale for partial (goal-reducing) reward.

    Returns:
        Scalar ``torch.Tensor`` in log-space, clamped to ``[_LOG_ZERO, 0]``.
    """
    if timeout:
        # Timeout penalty: log(max(0, -0.1)) is undefined; use floor
        return torch.tensor(_LOG_ZERO, dtype=torch.float32)

    if result.new_state is None or not result.success:
        return torch.tensor(_LOG_ZERO, dtype=torch.float32)

    if result.new_state.is_terminal:
        # Complete proof → reward = 1.0
        return torch.tensor(0.0, dtype=torch.float32)

    if result.goals_closed > 0 and original_goal_count > 0:
        # Partial progress
        fraction = result.goals_closed / original_goal_count
        raw = partial_reward_scale * fraction
        log_r = math.log(max(raw, 1e-45))
        return torch.tensor(max(log_r, _LOG_ZERO), dtype=torch.float32)

    # No progress
    return torch.tensor(_LOG_ZERO, dtype=torch.float32)
