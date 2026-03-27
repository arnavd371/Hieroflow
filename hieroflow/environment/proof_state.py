"""
Lean 4 proof state dataclasses for the HieroFlow GFlowNet training loop.

These dataclasses form the shared vocabulary between the environment layer
and both GFlowNet levels.  They are deliberately free of torch/numpy so
that they can be pickled and sent across process boundaries (e.g. to a
remote LeanDojo worker).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# LeanProofState
# ---------------------------------------------------------------------------

@dataclass
class LeanProofState:
    """
    Snapshot of a Lean 4 proof at a given point in time.

    Each field corresponds to information that LeanDojo can extract from the
    proof server after executing a tactic.  The dataclass is intentionally
    flat so that it serialises cleanly to JSON / msgpack for the replay buffer.
    """

    __slots__ = (
        "theorem_name",
        "goals",
        "tactic_history",
        "depth",
        "is_terminal",
        "is_error",
    )

    theorem_name: str
    """Name of the top-level theorem being proved (e.g. 'Nat.add_comm')."""

    goals: list[str]
    """Raw Lean goal strings, one per open goal.  Empty ↔ proof is complete."""

    tactic_history: list[str]
    """Ordered list of tactics applied so far in this proof attempt."""

    depth: int
    """Number of tactics applied so far (len(tactic_history))."""

    is_terminal: bool
    """True iff the proof is complete (no remaining goals)."""

    is_error: bool
    """True iff the last tactic produced a Lean error."""


# ---------------------------------------------------------------------------
# TacticResult
# ---------------------------------------------------------------------------

@dataclass
class TacticResult:
    """
    Outcome of applying a single tactic to a LeanProofState.

    Produced by ``LeanEnv.step()`` and consumed by TacticFlow's reward signal.
    ``new_state`` is ``None`` only when the tactic caused a catastrophic error
    (e.g. Lean process timeout).
    """

    __slots__ = (
        "tactic",
        "new_state",
        "lean_feedback",
        "success",
        "goals_closed",
    )

    tactic: str
    """The tactic string that was applied."""

    new_state: LeanProofState | None
    """Resulting proof state, or None on fatal error."""

    lean_feedback: str
    """Raw text feedback from Lean (error messages, warnings, etc.)."""

    success: bool
    """True iff the tactic was accepted by Lean without error."""

    goals_closed: int
    """Number of goals that were fully closed by this tactic (≥0)."""


# ---------------------------------------------------------------------------
# GoalTreeNode  (internal to GoalTree)
# ---------------------------------------------------------------------------

@dataclass
class GoalTreeNode:
    """
    Single node in a GoalTree.

    A node represents a goal string at a particular point in the proof search.
    Its children are the subgoals created when a tactic was applied to it.
    ``tactic`` is the tactic that split *this* node into its children.
    """

    __slots__ = ("node_id", "goal", "tactic", "parent_id", "children")

    node_id: str
    """Unique identifier for this node (UUID4 string)."""

    goal: str
    """The Lean goal string at this node (may be empty if the goal was closed)."""

    tactic: str | None
    """The tactic applied at this node to produce the children, or None if leaf."""

    parent_id: str | None
    """Parent node id, or None for the root."""

    children: list[str]
    """List of child node ids."""


# ---------------------------------------------------------------------------
# GoalTree
# ---------------------------------------------------------------------------

class GoalTree:
    """
    Tree recording how goals split over the course of a proof attempt.

    The root is the initial theorem goal.  Each internal node stores the
    tactic that was applied, and the children are the resulting subgoals.
    A leaf node is *closed* iff its goal string is empty (Lean returned no
    remaining goals for it).

    This structure is consumed by ``diversity_metrics.unique_subgoal_rate``
    and by the inner-reward computation in ``HieroFlowTrainer``.
    """

    def __init__(self, root_goal: str) -> None:
        """Initialise the tree with a single root node for *root_goal*."""
        root_id = str(uuid.uuid4())
        self._nodes: dict[str, GoalTreeNode] = {
            root_id: GoalTreeNode(
                node_id=root_id,
                goal=root_goal,
                tactic=None,
                parent_id=None,
                children=[],
            )
        }
        self._root_id: str = root_id

    # ------------------------------------------------------------------
    # Mutating methods
    # ------------------------------------------------------------------

    def add_child(self, parent_id: str, goal: str, tactic: str) -> str:
        """
        Attach a new child node to *parent_id*.

        Records *tactic* as the tactic that produced the child subgoal *goal*.
        Returns the new child node's id.

        Raises ``KeyError`` if *parent_id* does not exist in the tree.
        """
        if parent_id not in self._nodes:
            raise KeyError(f"Parent node '{parent_id}' not found in GoalTree")

        child_id = str(uuid.uuid4())
        child = GoalTreeNode(
            node_id=child_id,
            goal=goal,
            tactic=None,
            parent_id=parent_id,
            children=[],
        )
        self._nodes[child_id] = child
        # Record the tactic on the parent (it produced this child)
        self._nodes[parent_id].tactic = tactic
        self._nodes[parent_id].children.append(child_id)
        return child_id

    # ------------------------------------------------------------------
    # Read-only methods
    # ------------------------------------------------------------------

    @property
    def root_id(self) -> str:
        """Id of the root node."""
        return self._root_id

    def get_leaves(self) -> list[GoalTreeNode]:
        """
        Return all leaf nodes (nodes with no children).

        A leaf may be open (non-empty goal) or closed (empty goal string).
        """
        return [n for n in self._nodes.values() if not n.children]

    def is_complete(self) -> bool:
        """
        Return True iff every leaf node represents a closed goal.

        A goal is considered closed when its goal string is empty, mirroring
        LeanDojo's convention that an empty goal list means the proof is done.
        """
        return all(leaf.goal == "" for leaf in self.get_leaves())

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise the entire tree to a JSON-compatible dict.

        The schema is ``{"root_id": str, "nodes": {node_id: {…}, …}}``.
        """
        return {
            "root_id": self._root_id,
            "nodes": {
                nid: {
                    "node_id": node.node_id,
                    "goal": node.goal,
                    "tactic": node.tactic,
                    "parent_id": node.parent_id,
                    "children": node.children,
                }
                for nid, node in self._nodes.items()
            },
        }

    def __len__(self) -> int:
        """Number of nodes in the tree."""
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def __getitem__(self, node_id: str) -> GoalTreeNode:
        return self._nodes[node_id]
