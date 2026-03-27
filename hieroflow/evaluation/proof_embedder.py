"""
Contrastive proof embedding for semantic diversity measurement.

``ProofEmbedder`` maps a proof (sequence of tactic + goal-state pairs) into
a fixed-size vector.  The embedding captures the *semantic* structure of the
proof strategy, enabling cosine-distance comparisons between proofs that use
different variable names but the same logical structure.

Architecture:
    (tactic_i, goal_i) pairs
    → sentence transformer encode each pair
    → mean-pool over tactic steps
    → linear projection to embedding_dim
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ProofEmbedder(nn.Module):
    """
    Embeds a proof (list of (tactic, goal) pairs) into R^embedding_dim.

    Uses a pretrained sentence transformer to encode each (tactic → goal)
    pair independently, then mean-pools over tactic steps to produce a
    single proof-level embedding.

    The sentence transformer is loaded lazily on first use to avoid
    mandatory dependency on ``sentence_transformers`` at import time.

    The interface follows the ``evaluation/diversity_metrics.py`` contract:
    - ``embed(proof_states)``       → ``[embedding_dim]``
    - ``embed_batch(proofs)``       → ``[N, embedding_dim]``
    """

    _SENTENCE_MODEL_NAME: str = "all-MiniLM-L6-v2"
    """Default sentence-transformer model (small, fast, good quality)."""

    def __init__(self, embedding_dim: int = 256) -> None:
        """
        Initialise the embedder.

        Args:
            embedding_dim: Output embedding dimension.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self._sentence_model: Any | None = None

        # Projection from sentence-transformer dim (384 for MiniLM) to embedding_dim
        self.projection = nn.Linear(384, embedding_dim)

    def _get_sentence_model(self) -> Any:
        """Lazily load the sentence transformer (avoids import at module level)."""
        if self._sentence_model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import]
                self._sentence_model = SentenceTransformer(self._SENTENCE_MODEL_NAME)
            except ImportError:
                logger.warning(
                    "sentence_transformers not installed.  "
                    "ProofEmbedder will use random projections as fallback."
                )
                self._sentence_model = _RandomFallbackEncoder(384)
        return self._sentence_model

    def embed(
        self, proof_states: list[tuple[str, str]]
    ) -> torch.Tensor:
        """
        Embed a single proof given as a list of (tactic, goal) pairs.

        Each pair is formatted as ``"tactic → goal"`` before encoding.

        Args:
            proof_states: List of (tactic_applied, resulting_goal_string) pairs.

        Returns:
            ``torch.Tensor`` of shape ``[embedding_dim]``.
        """
        if not proof_states:
            return torch.zeros(self.embedding_dim)

        model = self._get_sentence_model()
        texts = [f"{tac} → {goal}" for tac, goal in proof_states]

        # Encode with sentence transformer
        raw = model.encode(texts, convert_to_tensor=True)  # [T, 384]
        if not isinstance(raw, torch.Tensor):
            raw = torch.tensor(raw, dtype=torch.float32)
        raw = raw.float()

        # Mean pool over tactic steps
        pooled = raw.mean(dim=0)  # [384]
        return self.projection(pooled)  # [embedding_dim]

    def embed_batch(
        self, proofs: list[list[tuple[str, str]]]
    ) -> torch.Tensor:
        """
        Embed a batch of proofs.

        Args:
            proofs: List of N proofs; each is a list of (tactic, goal) pairs.

        Returns:
            ``torch.Tensor`` of shape ``[N, embedding_dim]``.
        """
        embeddings = torch.stack([self.embed(p) for p in proofs], dim=0)
        return embeddings  # [N, embedding_dim]

    def forward(
        self, proof_states: list[tuple[str, str]]
    ) -> torch.Tensor:
        """Alias for ``embed`` (for use as an ``nn.Module``)."""
        return self.embed(proof_states)


# ---------------------------------------------------------------------------
# Fallback encoder (no sentence_transformers installed)
# ---------------------------------------------------------------------------

class _RandomFallbackEncoder:
    """
    Deterministic character-hash–based encoder used when
    sentence_transformers is not available.

    # TEST ONLY — this is not a meaningful semantic encoder.
    """

    def __init__(self, dim: int) -> None:
        self._dim = dim

    def encode(
        self, texts: list[str], convert_to_tensor: bool = False
    ) -> torch.Tensor:
        results = []
        for text in texts:
            # Deterministic hash-based pseudo-embedding
            seed = sum(ord(c) for c in text) % (2 ** 31)
            gen = torch.Generator()
            gen.manual_seed(seed)
            vec = torch.randn(self._dim, generator=gen)
            results.append(vec)
        return torch.stack(results, dim=0)
