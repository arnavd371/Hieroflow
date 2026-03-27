"""
SketchFlow: outer GFlowNet over proof sketch DAGs.

SketchFlow operates over the space of ``ProofSketch`` DAGs.  At each step it
selects an OPEN node and assigns it an abstract strategy
(induction / contradiction / rewrite / case_split / direct).

The GFlowNet objective is Trajectory Balance (Malkin et al., 2022):
    L = (log Z + Σ log P_F(a|s) - log R - Σ log P_B(s'|s))^2

The backward policy P_B is uniform over parent sketches, which is the
standard choice for DAG-structured GFlowNets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from hieroflow.sketch.sketch_dag import ProofSketch, SketchNodeState
from hieroflow.sketch.sketch_encoder import SketchEncoder


# ---------------------------------------------------------------------------
# Strategy vocabulary
# ---------------------------------------------------------------------------

STRATEGIES: list[str] = [
    "induction",
    "contradiction",
    "rewrite",
    "case_split",
    "direct",
]

STRATEGY_TO_IDX: dict[str, int] = {s: i for i, s in enumerate(STRATEGIES)}


# ---------------------------------------------------------------------------
# SketchAction
# ---------------------------------------------------------------------------

@dataclass
class SketchAction:
    """
    A single action taken by SketchFlow.

    An action selects one OPEN node and assigns it a strategy.  The
    ``log_prob`` is the log probability of this action under the current
    forward policy, recorded for TB-loss computation.
    """

    __slots__ = ("node_id", "strategy", "log_prob")

    node_id: str
    """ID of the OPEN node being assigned a strategy."""

    strategy: str
    """Abstract strategy string (one of ``STRATEGIES``)."""

    log_prob: float
    """Log probability log P_F(action | sketch) under the forward policy."""


# ---------------------------------------------------------------------------
# SketchTrajectory
# ---------------------------------------------------------------------------

@dataclass
class SketchTrajectory:
    """
    A complete trajectory produced by SketchFlow during one rollout.

    ``sketches[i]`` is the sketch *before* ``actions[i]`` was applied.
    ``terminal_reward`` is set once TacticFlow has evaluated the sketch.
    """

    sketches: list[ProofSketch] = field(default_factory=list)
    """Sequence of sketch states visited during the rollout."""

    actions: list[SketchAction] = field(default_factory=list)
    """Sequence of actions taken (one per sketch transition)."""

    terminal_reward: float | None = None
    """Log-space terminal reward, set after inner rollout completes."""


# ---------------------------------------------------------------------------
# SketchFlow
# ---------------------------------------------------------------------------

class SketchFlow(nn.Module):
    """
    Outer GFlowNet that samples proof sketches as abstract DAGs.

    Architecture:
    1. Encode the current sketch with ``SketchEncoder`` to get global and
       per-node embeddings.
    2. For each OPEN node, compute action logits:
       ``score(node, strategy) = MLP(node_emb ‖ strategy_emb)``
    3. Normalise across (node, strategy) pairs to get P_F distribution.
    4. Sample or argmax to pick the next action.

    The backward policy P_B is uniform over parent sketches (implemented as
    a constant 1/|open_actions| for each step).
    """

    def __init__(
        self,
        encoder: SketchEncoder,
        hidden_dim: int = 512,
        num_strategies: int = len(STRATEGIES),
    ) -> None:
        """
        Initialise SketchFlow.

        Args:
            encoder:       Pre-built ``SketchEncoder`` for graph encoding.
            hidden_dim:    Hidden dimension for action MLP.
            num_strategies: Number of abstract strategies (default 5).
        """
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.num_strategies = num_strategies

        # Learned strategy embeddings (one per strategy)
        self.strategy_embeddings = nn.Embedding(num_strategies, hidden_dim)

        # Action scoring MLP: [node_emb ‖ strategy_emb] → scalar
        node_emb_dim = encoder.hidden_dim
        self.action_mlp = nn.Sequential(
            nn.Linear(node_emb_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Learnable log partition function (separate from TacticFlow's)
        self.log_Z = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    # Core GFlowNet methods
    # ------------------------------------------------------------------

    def forward_policy(
        self, sketch: ProofSketch
    ) -> dict[str, torch.Tensor]:
        """
        Compute forward-policy logits for all OPEN nodes.

        Args:
            sketch: Current ``ProofSketch`` state.

        Returns:
            Dict mapping each OPEN node_id → ``torch.Tensor`` of shape
            ``[num_strategies]`` (raw logits, not probabilities).

        Returns an empty dict if there are no OPEN nodes.
        """
        open_nodes = sketch.get_open_nodes()
        if not open_nodes:
            return {}

        # Per-node embeddings for all nodes in the sketch
        node_embs = self.encoder.encode_nodes(sketch)  # [N, hidden_dim]

        # Map node_id → index in node_embs
        node_id_to_idx = {
            nid: idx for idx, nid in enumerate(sketch.nodes.keys())
        }

        strategy_idxs = torch.arange(self.num_strategies, dtype=torch.long)
        strategy_embs = self.strategy_embeddings(strategy_idxs)  # [S, hidden_dim]

        result: dict[str, torch.Tensor] = {}
        for node in open_nodes:
            node_emb = node_embs[node_id_to_idx[node.node_id]]  # [node_emb_dim]
            # Expand node_emb to pair with each strategy
            node_emb_exp = node_emb.unsqueeze(0).expand(
                self.num_strategies, -1
            )  # [S, node_emb_dim]
            pairs = torch.cat([node_emb_exp, strategy_embs], dim=-1)  # [S, node_emb_dim+hidden_dim]
            logits = self.action_mlp(pairs).squeeze(-1)  # [S]
            result[node.node_id] = logits

        return result

    def backward_policy(self, sketch: ProofSketch) -> torch.Tensor:
        """
        Compute the (log) backward policy for the current sketch state.

        For DAG GFlowNets the standard choice is uniform over parent sketches.
        Here we approximate this as ``-log(num_open_nodes * num_strategies)``.

        Args:
            sketch: Current ``ProofSketch`` state.

        Returns:
            Scalar ``torch.Tensor`` with the log backward probability.
        """
        num_open = len(sketch.get_open_nodes())
        if num_open == 0:
            return torch.tensor(0.0)
        log_pb = -math.log(num_open * self.num_strategies)
        return torch.tensor(log_pb)

    def sample_action(
        self, sketch: ProofSketch, temperature: float = 1.0
    ) -> SketchAction:
        """
        Sample an action from the forward policy at the current sketch state.

        Actions are sampled by (a) flattening all (node, strategy) pairs into
        one multinomial, then (b) recovering the node/strategy indices.

        Args:
            sketch:      Current sketch state.
            temperature: Softmax temperature (lower → more greedy).

        Returns:
            A ``SketchAction`` with the selected node_id, strategy, and log_prob.

        Raises:
            ValueError: If there are no OPEN nodes.
        """
        open_nodes = sketch.get_open_nodes()
        if not open_nodes:
            raise ValueError("Cannot sample action: no OPEN nodes in sketch")

        logits_per_node = self.forward_policy(sketch)

        # Flatten: build a list of (node_id, strategy_idx, logit)
        all_node_ids: list[str] = []
        all_strat_idxs: list[int] = []
        all_logits: list[torch.Tensor] = []

        for node in open_nodes:
            logits = logits_per_node[node.node_id]  # [S]
            for s_idx in range(self.num_strategies):
                all_node_ids.append(node.node_id)
                all_strat_idxs.append(s_idx)
                all_logits.append(logits[s_idx])

        flat_logits = torch.stack(all_logits) / temperature  # [N*S]
        log_probs = F.log_softmax(flat_logits, dim=0)
        probs = log_probs.exp()

        action_idx = torch.multinomial(probs, num_samples=1).item()
        assert isinstance(action_idx, int)

        selected_node_id = all_node_ids[action_idx]
        selected_strategy = STRATEGIES[all_strat_idxs[action_idx]]
        log_prob = log_probs[action_idx].item()

        return SketchAction(
            node_id=selected_node_id,
            strategy=selected_strategy,
            log_prob=log_prob,
        )

    # ------------------------------------------------------------------
    # Trajectory Balance loss
    # ------------------------------------------------------------------

    def trajectory_balance_loss(
        self,
        trajectory: SketchTrajectory,
        reward: float,
    ) -> torch.Tensor:
        """
        Compute the TB loss for a completed sketch trajectory.

        Implements:
            L = (log Z + Σ log P_F(a|s) - log R - Σ log P_B(s'|s))^2

        Args:
            trajectory: A ``SketchTrajectory`` with recorded actions and
                        sketches.
            reward:     The terminal reward in *linear* space (R ≥ 0).
                        Internally converted to log-space.

        Returns:
            Scalar loss ``torch.Tensor``.
        """
        log_r = torch.tensor(
            max(math.log(max(reward, 1e-45)), -100.0), dtype=torch.float32
        )

        log_pf_sum = sum(a.log_prob for a in trajectory.actions)
        log_pf_tensor = torch.tensor(log_pf_sum, dtype=torch.float32, requires_grad=True)

        # Re-compute P_F using current parameters for the recorded sketches
        # (we re-score each (state, action) pair so gradients flow)
        pf_terms: list[torch.Tensor] = []
        for sketch, action in zip(trajectory.sketches, trajectory.actions):
            logits_per_node = self.forward_policy(sketch)
            if action.node_id not in logits_per_node:
                continue
            logits = logits_per_node[action.node_id]  # [S]
            log_probs_node = F.log_softmax(logits, dim=0)
            strat_idx = STRATEGY_TO_IDX.get(action.strategy, 0)
            pf_terms.append(log_probs_node[strat_idx])

        if pf_terms:
            log_pf_tensor = torch.stack(pf_terms).sum()
        else:
            log_pf_tensor = torch.zeros(1)

        # Uniform backward policy
        log_pb_sum = sum(
            self.backward_policy(sketch).item() for sketch in trajectory.sketches
        )
        log_pb_tensor = torch.tensor(log_pb_sum, dtype=torch.float32)

        balance = self.log_Z.squeeze() + log_pf_tensor - log_r - log_pb_tensor
        return balance.pow(2)
