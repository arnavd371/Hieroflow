"""
Trajectory Balance (TB) loss for GFlowNets.

Reference: Malkin et al. (2022) "Trajectory Balance: Improved Credit Assignment
in GFlowNets".  https://arxiv.org/abs/2201.13259

Also implements:
- Detailed Balance (DB) loss (alternative for ablations).
- SubTB(λ) loss from Madan et al. (2023) — interpolates between TB (λ=1)
  and DB (λ→0) via geometric weighting over sub-trajectories.

All computations operate entirely in log-space to avoid numerical underflow.
"""

from __future__ import annotations

import logging
import warnings

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Reward underflow threshold
_REWARD_UNDERFLOW_THRESHOLD: float = -100.0


# ---------------------------------------------------------------------------
# trajectory_balance_loss
# ---------------------------------------------------------------------------

def trajectory_balance_loss(
    log_pf_trajectory: torch.Tensor,
    log_pb_trajectory: torch.Tensor,
    log_reward: torch.Tensor,
    log_z: nn.Parameter,
) -> torch.Tensor:
    """
    Compute the Trajectory Balance loss for a single trajectory.

    Implements:
        L = (log Z + Σ log P_F(a_t|s_t) - log R(τ) - Σ log P_B(s_t|s_{t+1}))^2

    All inputs must be in log-space.  No exponentiation is performed.

    Args:
        log_pf_trajectory: ``[T]`` tensor of forward log-probs at each step.
        log_pb_trajectory: ``[T]`` tensor of backward log-probs at each step.
        log_reward:        Scalar — log R(τ) for the complete trajectory.
        log_z:             Learnable ``nn.Parameter`` — log of the partition
                           function Z.

    Returns:
        Scalar TB loss (non-negative).
    """
    if log_reward.item() < _REWARD_UNDERFLOW_THRESHOLD:
        warnings.warn(
            f"log_reward = {log_reward.item():.2f} is below the underflow "
            f"threshold {_REWARD_UNDERFLOW_THRESHOLD}.  Check reward computation.",
            stacklevel=2,
        )

    log_pf_sum = log_pf_trajectory.sum()
    log_pb_sum = log_pb_trajectory.sum()
    balance = log_z.squeeze() + log_pf_sum - log_reward - log_pb_sum
    return balance.pow(2)


# ---------------------------------------------------------------------------
# detailed_balance_loss
# ---------------------------------------------------------------------------

def detailed_balance_loss(
    log_pf: torch.Tensor,
    log_pb: torch.Tensor,
    log_flow_in: torch.Tensor,
    log_flow_out: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the Detailed Balance loss for a single transition.

    Implements:
        L_DB = (log F(s) + log P_F(s→s') - log F(s') - log P_B(s'→s))^2

    All inputs must be in log-space.

    Args:
        log_pf:       Log forward probability of the transition s → s'.
        log_pb:       Log backward probability of the transition s' → s.
        log_flow_in:  Log flow into state s (log F(s)).
        log_flow_out: Log flow out of state s' (log F(s')).

    Returns:
        Scalar DB loss (non-negative).
    """
    balance = log_flow_in + log_pf - log_flow_out - log_pb
    return balance.pow(2)


# ---------------------------------------------------------------------------
# SubTBLoss
# ---------------------------------------------------------------------------

class SubTBLoss(nn.Module):
    """
    SubTB(λ) loss from Madan et al. (2023).

    Interpolates between Trajectory Balance (λ=1) and Detailed Balance
    (λ→0) via a geometric weighting scheme over sub-trajectories.

    Reference: Madan et al. (2023) "Learning GFlowNets from partial
    episodes for improved convergence and stability."
    https://arxiv.org/abs/2209.12782

    The loss sums over all sub-trajectories τ_{i:j} within the full
    trajectory, each weighted by λ^(j-i).

    All computations remain in log-space.
    """

    def __init__(self, lambda_param: float = 0.9) -> None:
        """
        Initialise SubTBLoss.

        Args:
            lambda_param: Geometric weighting factor in (0, 1].
                          λ=1 recovers TB; λ→0 recovers DB.
        """
        super().__init__()
        if not (0.0 < lambda_param <= 1.0):
            raise ValueError(
                f"lambda_param must be in (0, 1], got {lambda_param}"
            )
        self.lambda_param = lambda_param

    def forward(
        self,
        log_pfs: torch.Tensor,
        log_pbs: torch.Tensor,
        log_flows: torch.Tensor,
        log_reward: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the SubTB(λ) loss.

        Args:
            log_pfs:    ``[T]`` tensor of forward log-probs.
            log_pbs:    ``[T]`` tensor of backward log-probs.
            log_flows:  ``[T+1]`` tensor of log-flows at each state
                        (including the terminal state).
            log_reward: Scalar — log terminal reward.

        Returns:
            Scalar SubTB loss (non-negative).
        """
        T = log_pfs.size(0)
        if T == 0:
            return torch.tensor(0.0, requires_grad=True)

        if log_reward.item() < _REWARD_UNDERFLOW_THRESHOLD:
            warnings.warn(
                f"log_reward = {log_reward.item():.2f} below underflow threshold.",
                stacklevel=2,
            )

        # Override the terminal flow with the log-reward
        # (enforces the boundary condition F(s_terminal) = R(s_terminal))
        log_flows_with_reward = log_flows.clone()
        if log_flows_with_reward.size(0) > T:
            log_flows_with_reward[T] = log_reward

        total_loss = torch.tensor(0.0)
        weight_sum = torch.tensor(0.0)

        for i in range(T):
            for j in range(i + 1, T + 1):
                length = j - i
                weight = self.lambda_param ** length

                # Sub-trajectory balance:
                # log F(s_i) + Σ_{t=i}^{j-1} log P_F(t) - log F(s_j) - Σ_{t=i}^{j-1} log P_B(t)
                log_pf_sub = log_pfs[i:j].sum()
                log_pb_sub = log_pbs[i:j].sum()

                log_fi = log_flows_with_reward[i] if i < log_flows_with_reward.size(0) else torch.tensor(0.0)
                log_fj = log_flows_with_reward[j] if j < log_flows_with_reward.size(0) else log_reward

                balance = log_fi + log_pf_sub - log_fj - log_pb_sub
                total_loss = total_loss + weight * balance.pow(2)
                weight_sum = weight_sum + weight

        if weight_sum > 0:
            return total_loss / weight_sum
        return total_loss


# ---------------------------------------------------------------------------
# Unit test (__main__)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Trajectory Balance loss unit test ===")

    # Perfect GFlowNet: log Z + sum(log P_F) == log R + sum(log P_B) exactly
    T = 4
    log_z = nn.Parameter(torch.tensor(2.0))

    # Construct a balanced trajectory
    log_pf = torch.tensor([0.5, 0.3, -0.2, -0.1])
    log_pb = torch.tensor([-0.3, 0.1, 0.2, -0.4])
    log_r = torch.tensor(
        log_z.item() + log_pf.sum().item() - log_pb.sum().item()
    )

    loss = trajectory_balance_loss(log_pf, log_pb, log_r, log_z)
    assert abs(loss.item()) < 1e-6, f"Expected ~0 loss, got {loss.item()}"
    print(f"  TB loss for perfect GFlowNet: {loss.item():.6e}  ✓")

    # Non-zero loss for imbalanced trajectory
    log_pf_bad = torch.tensor([0.5, 0.3, -0.2, -0.1])
    log_pb_bad = torch.tensor([-0.3, 0.1, 0.2, -0.4])
    log_r_bad = torch.tensor(-1.0)  # Mismatch

    loss_bad = trajectory_balance_loss(log_pf_bad, log_pb_bad, log_r_bad, log_z)
    assert loss_bad.item() > 0, "Expected positive loss for imbalanced trajectory"
    print(f"  TB loss for imbalanced trajectory: {loss_bad.item():.4f}  ✓")

    # SubTB test
    print("\n=== SubTB(0.9) unit test ===")
    sub_tb = SubTBLoss(lambda_param=0.9)
    log_flows = torch.zeros(T + 1)
    sub_loss = sub_tb(log_pf, log_pb, log_flows, log_r)
    print(f"  SubTB loss: {sub_loss.item():.4f}")

    print("\nAll tests passed.")
