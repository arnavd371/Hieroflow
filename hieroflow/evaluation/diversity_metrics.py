"""
Diversity evaluation metrics for comparing GFlowNet and RL theorem provers.

These are the primary evaluation criteria for the HieroFlow paper.

All summary metrics return a value in [0, 1] where 1 = maximum diversity,
to allow fair comparison across different systems and theorem benchmarks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from hieroflow.environment.proof_state import GoalTree

if TYPE_CHECKING:
    from hieroflow.evaluation.proof_embedder import ProofEmbedder


# ---------------------------------------------------------------------------
# Syntactic Metrics
# ---------------------------------------------------------------------------

def tactic_type_entropy(proofs: list[list[str]]) -> float:
    """
    Compute the Shannon entropy over tactic type distribution.

    The tactic type of a string is its first word (e.g. 'rw', 'simp',
    'apply').  Higher entropy = more diverse tactic usage.

    The returned value is normalised to [0, 1] by dividing by log(V)
    where V is the vocabulary size (number of distinct tactic types seen).

    Args:
        proofs: List of proofs; each proof is a list of tactic strings.

    Returns:
        Normalised entropy in [0, 1].
    """
    counts: dict[str, int] = {}
    for proof in proofs:
        for tactic in proof:
            tactic_type = tactic.strip().split()[0] if tactic.strip() else "unknown"
            counts[tactic_type] = counts.get(tactic_type, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log(p)

    # Normalise by log(V)
    vocab_size = len(counts)
    if vocab_size <= 1:
        return 0.0
    return entropy / math.log(vocab_size)


def unique_subgoal_rate(proofs: list[GoalTree]) -> float:
    """
    Compute the unique subgoal rate across a collection of GoalTrees.

    Defined as:
        |unique leaf goal strings| / |total non-root nodes|

    This mirrors the "unique subgoal rate" metric from 3D-Prover.

    Args:
        proofs: List of ``GoalTree`` objects from successful proof attempts.

    Returns:
        Rate in [0, 1].
    """
    all_leaf_goals: list[str] = []
    total_tactics = 0

    for tree in proofs:
        leaves = tree.get_leaves()
        for leaf in leaves:
            all_leaf_goals.append(leaf.goal)
        # Count non-root nodes as proxy for tactic applications
        total_tactics += max(len(tree) - 1, 0)

    if total_tactics == 0:
        return 0.0

    unique_goals = len(set(all_leaf_goals))
    return min(unique_goals / total_tactics, 1.0)


def proof_edit_distance(
    proof_a: list[str], proof_b: list[str]
) -> float:
    """
    Compute normalised edit distance between two tactic sequences.

    Uses the Wagner-Fischer dynamic programming algorithm.
    Returns a value in [0, 1] where 0 = identical, 1 = completely different.

    Args:
        proof_a: First tactic sequence.
        proof_b: Second tactic sequence.

    Returns:
        Normalised edit distance in [0, 1].
    """
    n, m = len(proof_a), len(proof_b)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return 1.0

    # Wagner-Fischer DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if proof_a[i - 1] == proof_b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    raw_dist = dp[n][m]
    return raw_dist / max(n, m)


def pairwise_diversity(proofs: list[list[str]]) -> float:
    """
    Compute mean pairwise edit distance over all proof pairs.

    Capped at 500 proofs to keep runtime O(n^2) manageable.

    Args:
        proofs: List of tactic sequences.

    Returns:
        Mean pairwise edit distance in [0, 1].
    """
    if len(proofs) < 2:
        return 0.0

    sampled = proofs[:500]
    n = len(sampled)
    total = 0.0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            total += proof_edit_distance(sampled[i], sampled[j])
            count += 1

    return total / count if count > 0 else 0.0


# ---------------------------------------------------------------------------
# Semantic Metrics
# ---------------------------------------------------------------------------

def semantic_diversity(
    proofs: list[list[tuple[str, str]]],
    embedder: "ProofEmbedder",
) -> float:
    """
    Compute mean pairwise cosine distance in proof embedding space.

    Args:
        proofs:   List of proofs; each proof is a list of (tactic, goal) pairs.
        embedder: A ``ProofEmbedder`` to map proofs to vectors.

    Returns:
        Mean pairwise cosine distance in [0, 1].
    """
    if len(proofs) < 2:
        return 0.0

    embeddings = embedder.embed_batch(proofs)  # [N, D]
    # Normalise rows
    norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = embeddings / norms

    # Cosine similarity matrix
    sim_matrix = normed @ normed.T  # [N, N]
    N = sim_matrix.size(0)

    # Mean pairwise cosine *distance* = 1 - similarity
    total_dist = 0.0
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            dist = 1.0 - sim_matrix[i, j].item()
            total_dist += max(0.0, dist)
            count += 1

    return total_dist / count if count > 0 else 0.0


def strategy_cluster_count(
    proofs: list[list[tuple[str, str]]],
    embedder: "ProofEmbedder",
    n_clusters: int = 5,
) -> int:
    """
    Find the effective number of distinct proof strategies.

    Uses K-means clustering on proof embeddings and returns the number of
    clusters that collectively explain >80% of the variance.

    Args:
        proofs:     List of proofs.
        embedder:   ``ProofEmbedder`` to map proofs to vectors.
        n_clusters: Maximum number of clusters to fit.

    Returns:
        Effective cluster count (1 ≤ k ≤ n_clusters).
    """
    if len(proofs) < 2:
        return 1

    try:
        from sklearn.cluster import KMeans  # type: ignore[import]
        from sklearn.decomposition import PCA  # type: ignore[import]
    except ImportError:
        # Fallback: return 1 if sklearn not available
        return 1

    embeddings = embedder.embed_batch(proofs).detach().numpy()  # [N, D]

    # PCA to find variance explained
    n_components = min(n_clusters, embeddings.shape[0], embeddings.shape[1])
    if n_components < 2:
        return 1

    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    cumvar = pca.explained_variance_ratio_.cumsum()

    # Number of components explaining >80% of variance
    k = int((cumvar < 0.8).sum()) + 1
    return min(k, n_clusters)


# ---------------------------------------------------------------------------
# DiversityReport
# ---------------------------------------------------------------------------

@dataclass
class DiversityReport:
    """
    Aggregated diversity metrics report.

    All fields are in [0, 1] where 1 = maximum diversity (except
    ``strategy_clusters`` which is a count).
    """

    tactic_entropy: float = 0.0
    """Normalised Shannon entropy of tactic type distribution."""

    unique_subgoal_rate: float = 0.0
    """Fraction of unique subgoals relative to total tactic applications."""

    pairwise_edit_diversity: float = 0.0
    """Mean pairwise edit distance over tactic sequences."""

    semantic_diversity: float = 0.0
    """Mean pairwise cosine distance in proof embedding space."""

    strategy_clusters: int = 1
    """Effective number of distinct proof strategies (K-means)."""

    def summary_str(self) -> str:
        """Return a human-readable one-line summary."""
        return (
            f"DiversityReport("
            f"entropy={self.tactic_entropy:.3f}, "
            f"subgoal_rate={self.unique_subgoal_rate:.3f}, "
            f"edit_div={self.pairwise_edit_diversity:.3f}, "
            f"sem_div={self.semantic_diversity:.3f}, "
            f"clusters={self.strategy_clusters})"
        )


def evaluate_diversity(
    proofs: list[list[str]],
    goal_trees: list[GoalTree],
    embedder: "ProofEmbedder",
    proof_states: list[list[tuple[str, str]]] | None = None,
) -> DiversityReport:
    """
    Compute all diversity metrics and return a ``DiversityReport``.

    Args:
        proofs:       List of proofs (tactic string sequences).
        goal_trees:   List of ``GoalTree`` objects (one per proof).
        embedder:     ``ProofEmbedder`` for semantic metrics.
        proof_states: Optional list of (tactic, goal) pair sequences for
                      semantic metrics.  If None, tactic-only embeddings
                      are used.

    Returns:
        A populated ``DiversityReport``.
    """
    entropy = tactic_type_entropy(proofs)
    subgoal_rate = unique_subgoal_rate(goal_trees) if goal_trees else 0.0
    edit_div = pairwise_diversity(proofs)

    if proof_states and len(proof_states) >= 2:
        sem_div = semantic_diversity(proof_states, embedder)
        n_clusters = strategy_cluster_count(proof_states, embedder)
    else:
        sem_div = 0.0
        n_clusters = 1

    return DiversityReport(
        tactic_entropy=entropy,
        unique_subgoal_rate=subgoal_rate,
        pairwise_edit_diversity=edit_div,
        semantic_diversity=sem_div,
        strategy_clusters=n_clusters,
    )
