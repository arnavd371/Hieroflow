"""
Sketch reward: P(some tactic completion closes goal | sketch node).

The outer reward is the fraction of sketch nodes for which TacticFlow
successfully found a valid Lean tactic.  This is a noisy but tractable
proxy for proof success.

All computations are in log-space as required by the cross-cutting rules.
"""

from __future__ import annotations

import math

import torch

from hieroflow.sketch.sketch_dag import ProofSketch, SketchNodeState


def compute_sketch_log_reward(
    sketch: ProofSketch,
    tactic_successes: dict[str, bool],
) -> torch.Tensor:
    """
    Compute the outer GFlowNet reward in log-space.

    The reward is defined as:
        R_outer = fraction of ASSIGNED/CLOSED nodes where TacticFlow succeeded.

    If the sketch is empty or all attempts failed, returns ``log(ε)`` clamped
    to -100 for numerical stability.

    Args:
        sketch:           The completed (or partially completed) sketch.
        tactic_successes: Mapping from node_id → True/False indicating whether
                          TacticFlow found a valid tactic for that node.

    Returns:
        Scalar ``torch.Tensor`` with the log-space reward.
    """
    if not tactic_successes:
        return torch.tensor(-100.0)

    num_successes = sum(1 for v in tactic_successes.values() if v)
    fraction = num_successes / len(tactic_successes)
    log_r = max(math.log(max(fraction, 1e-45)), -100.0)
    return torch.tensor(log_r, dtype=torch.float32)


def compute_node_difficulty_weights(sketch: ProofSketch) -> dict[str, float]:
    """
    Return a weight per node proportional to its estimated difficulty.

    These weights can be used to down-sample easy nodes in the inner rollout
    and focus compute on harder ones.

    Args:
        sketch: A ``ProofSketch`` with populated obligation estimated_depths.

    Returns:
        Dict mapping node_id → normalised weight in (0, 1].
    """
    total_depth = sum(
        n.obligation.estimated_depth for n in sketch.nodes.values()
    )
    if total_depth == 0:
        n_nodes = len(sketch.nodes)
        return {nid: 1.0 / n_nodes for nid in sketch.nodes}

    return {
        nid: node.obligation.estimated_depth / total_depth
        for nid, node in sketch.nodes.items()
    }
