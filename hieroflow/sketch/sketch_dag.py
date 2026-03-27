"""
ProofSketch: a partial DAG of ProofObligations.

The ProofSketch is the *state space* of the outer GFlowNet (SketchFlow).
At each step, SketchFlow selects an OPEN node and assigns it an abstract
strategy (e.g. "induction", "contradiction").  Once all nodes are CLOSED
the sketch is complete and TacticFlow can fill in the concrete tactics.

Design note: ``to_feature_matrix`` must stay in sync with the feature
dimension constant ``NODE_FEAT_DIM = 12`` used by ``SketchEncoder``.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import torch

from hieroflow.environment.obligation import ObligationType, ProofObligation

if TYPE_CHECKING:
    pass

# Feature dimension: 8 (ObligationType one-hot) + 3 (SketchNodeState one-hot) + 1 (depth)
NODE_FEAT_DIM: int = 12


# ---------------------------------------------------------------------------
# SketchNodeState
# ---------------------------------------------------------------------------

class SketchNodeState(Enum):
    """
    Lifecycle state of a single node in a ProofSketch.

    OPEN     → the node has been created but no strategy assigned yet.
    ASSIGNED → a strategy has been assigned; TacticFlow will fill it in.
    CLOSED   → TacticFlow successfully closed the goal for this node.
    FAILED   → TacticFlow exhausted all options without closing the goal.
    """

    OPEN = auto()
    ASSIGNED = auto()
    CLOSED = auto()
    FAILED = auto()


# ---------------------------------------------------------------------------
# SketchNode
# ---------------------------------------------------------------------------

@dataclass
class SketchNode:
    """
    Single node in a ``ProofSketch`` DAG.

    Each node represents one ``ProofObligation`` that arises during the proof.
    The outer GFlowNet (SketchFlow) operates at this granularity.
    """

    __slots__ = (
        "node_id",
        "obligation",
        "state",
        "assigned_strategy",
        "children",
        "parent_id",
    )

    node_id: str
    """Unique identifier (UUID4 string)."""

    obligation: ProofObligation
    """The abstract proof obligation this node represents."""

    state: SketchNodeState
    """Current lifecycle state."""

    assigned_strategy: str | None
    """Abstract strategy assigned by SketchFlow (e.g. 'induction')."""

    children: list[str]
    """Node IDs of child nodes (subgoals created by the assigned strategy)."""

    parent_id: str | None
    """Parent node ID, or None for the root."""


# ---------------------------------------------------------------------------
# ProofSketch
# ---------------------------------------------------------------------------

class ProofSketch:
    """
    Partial DAG of ``ProofObligation`` nodes being built by SketchFlow.

    The root node corresponds to the original theorem.  Child nodes arise
    when a tactic strategy (e.g. induction) splits a goal into subgoals.

    The class exposes both a mutable interface (used during rollout) and a
    read-only interface for the GNN encoder.

    Feature matrix convention (``to_feature_matrix``):
        Each node is represented as a ``NODE_FEAT_DIM``-dimensional vector:
        - dims 0–7:   one-hot over ``ObligationType`` (8 values)
        - dims 8–10:  one-hot over ``SketchNodeState``
                      (OPEN=8, ASSIGNED=9, CLOSED/FAILED=10)
        - dim  11:    estimated_depth / 10.0 (normalised to ≈[0,1])
    """

    def __init__(self, root_obligation: ProofObligation) -> None:
        """
        Create a new ProofSketch with a single OPEN root node.

        Args:
            root_obligation: The ``ProofObligation`` for the top-level theorem.
        """
        root_id = str(uuid.uuid4())
        self.nodes: dict[str, SketchNode] = {
            root_id: SketchNode(
                node_id=root_id,
                obligation=root_obligation,
                state=SketchNodeState.OPEN,
                assigned_strategy=None,
                children=[],
                parent_id=None,
            )
        }
        self.root_id: str = root_id

    # ------------------------------------------------------------------
    # Mutating methods
    # ------------------------------------------------------------------

    def add_node(
        self,
        obligation: ProofObligation,
        parent_id: str,
        strategy: str | None = None,
    ) -> str:
        """
        Add a new OPEN child node to *parent_id*.

        Args:
            obligation: The ``ProofObligation`` for the new node.
            parent_id:  ID of the existing parent node.
            strategy:   Optional strategy string (if already known).

        Returns:
            The new node's ID.

        Raises:
            KeyError: If *parent_id* does not exist in this sketch.
        """
        if parent_id not in self.nodes:
            raise KeyError(f"Parent node '{parent_id}' not found in ProofSketch")

        node_id = str(uuid.uuid4())
        state = SketchNodeState.ASSIGNED if strategy else SketchNodeState.OPEN
        node = SketchNode(
            node_id=node_id,
            obligation=obligation,
            state=state,
            assigned_strategy=strategy,
            children=[],
            parent_id=parent_id,
        )
        self.nodes[node_id] = node
        self.nodes[parent_id].children.append(node_id)
        return node_id

    def assign_strategy(self, node_id: str, strategy: str) -> None:
        """
        Assign an abstract strategy to an OPEN node, moving it to ASSIGNED.

        Args:
            node_id:  ID of the node to update.
            strategy: Strategy string (e.g. 'induction', 'contradiction').

        Raises:
            KeyError:    If *node_id* does not exist.
            ValueError:  If the node is not in the OPEN state.
        """
        node = self._get_node(node_id)
        if node.state != SketchNodeState.OPEN:
            raise ValueError(
                f"Node '{node_id}' is in state {node.state.name}, expected OPEN"
            )
        node.state = SketchNodeState.ASSIGNED
        node.assigned_strategy = strategy

    def close_node(self, node_id: str) -> None:
        """
        Mark *node_id* as CLOSED (TacticFlow successfully filled it in).

        Raises:
            KeyError: If *node_id* does not exist.
        """
        self._get_node(node_id).state = SketchNodeState.CLOSED

    def fail_node(self, node_id: str) -> None:
        """
        Mark *node_id* as FAILED (TacticFlow could not close it).

        Raises:
            KeyError: If *node_id* does not exist.
        """
        self._get_node(node_id).state = SketchNodeState.FAILED

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_open_nodes(self) -> list[SketchNode]:
        """Return all nodes currently in the OPEN state."""
        return [n for n in self.nodes.values() if n.state == SketchNodeState.OPEN]

    def is_complete(self) -> bool:
        """
        Return True iff every leaf node is CLOSED.

        A leaf is a node with no children.
        """
        leaves = [n for n in self.nodes.values() if not n.children]
        return bool(leaves) and all(
            n.state == SketchNodeState.CLOSED for n in leaves
        )

    def is_failed(self) -> bool:
        """
        Return True iff any node is in the FAILED state.

        (The outer GFlowNet treats this as a terminal-failure condition.)
        """
        return any(n.state == SketchNodeState.FAILED for n in self.nodes.values())

    # ------------------------------------------------------------------
    # GNN feature matrix
    # ------------------------------------------------------------------

    def to_feature_matrix(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (node_features, edge_index) tensors for GNN encoding.

        Node ordering is deterministic: sorted by insertion order (dict order
        is preserved in Python 3.7+).

        Returns:
            node_features: ``torch.Tensor`` of shape ``[N, NODE_FEAT_DIM]``
            edge_index:    ``torch.Tensor`` of shape ``[2, E]`` (COO format),
                           where each column is a (parent → child) edge.
        """
        node_ids = list(self.nodes.keys())
        id_to_idx: dict[str, int] = {nid: i for i, nid in enumerate(node_ids)}

        num_ob_types = len(ObligationType)  # 8
        features: list[list[float]] = []
        for nid in node_ids:
            node = self.nodes[nid]
            # One-hot obligation type
            ob_onehot = [0.0] * num_ob_types
            ob_idx = list(ObligationType).index(node.obligation.obligation_type)
            ob_onehot[ob_idx] = 1.0

            # One-hot state (OPEN, ASSIGNED, CLOSED/FAILED combined)
            state_onehot = [0.0, 0.0, 0.0]
            if node.state == SketchNodeState.OPEN:
                state_onehot[0] = 1.0
            elif node.state == SketchNodeState.ASSIGNED:
                state_onehot[1] = 1.0
            else:
                state_onehot[2] = 1.0

            # Normalised depth
            norm_depth = min(node.obligation.estimated_depth / 10.0, 1.0)

            features.append(ob_onehot + state_onehot + [norm_depth])

        node_features = torch.tensor(features, dtype=torch.float32)

        # Build edge list (parent → child)
        src_list: list[int] = []
        dst_list: list[int] = []
        for nid, node in self.nodes.items():
            src_idx = id_to_idx[nid]
            for child_id in node.children:
                dst_idx = id_to_idx[child_id]
                src_list.append(src_idx)
                dst_list.append(dst_idx)

        if src_list:
            edge_index = torch.tensor(
                [src_list, dst_list], dtype=torch.long
            )
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return node_features, edge_index

    # ------------------------------------------------------------------
    # Serialisation / copy
    # ------------------------------------------------------------------

    def clone(self) -> "ProofSketch":
        """
        Return a deep copy of this ProofSketch.

        Used by beam search to branch the sketch state.
        """
        return copy.deepcopy(self)

    def serialise(self) -> dict:
        """
        Serialise the sketch to a JSON-compatible dict.

        The format is ``{"root_id": str, "nodes": {id: {…}, …}}``.
        """
        return {
            "root_id": self.root_id,
            "nodes": {
                nid: {
                    "node_id": node.node_id,
                    "state": node.state.name,
                    "assigned_strategy": node.assigned_strategy,
                    "children": node.children,
                    "parent_id": node.parent_id,
                    "obligation": {
                        "obligation_type": node.obligation.obligation_type.name,
                        "abstracted_goal": node.obligation.abstracted_goal,
                        "hypothesis_types": node.obligation.hypothesis_types,
                        "target_type": node.obligation.target_type,
                        "estimated_depth": node.obligation.estimated_depth,
                    },
                }
                for nid, node in self.nodes.items()
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_node(self, node_id: str) -> SketchNode:
        try:
            return self.nodes[node_id]
        except KeyError:
            raise KeyError(f"Node '{node_id}' not found in ProofSketch") from None

    def __len__(self) -> int:
        """Number of nodes in the sketch."""
        return len(self.nodes)

    def __repr__(self) -> str:
        open_n = len(self.get_open_nodes())
        return (
            f"ProofSketch(nodes={len(self.nodes)}, open={open_n}, "
            f"complete={self.is_complete()}, failed={self.is_failed()})"
        )
