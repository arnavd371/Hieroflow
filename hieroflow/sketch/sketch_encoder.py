"""
GNN encoder for ProofSketch DAGs.

``SketchEncoder`` converts a ``ProofSketch`` into a fixed-size graph embedding
used by SketchFlow to decide the next action.  ``ObligationEmbedder`` produces
per-node initial features from a ``ProofObligation`` independently of graph
structure.

Architecture:
    ProofSketch → ObligationEmbedder (per-node) → GATConv × num_layers
    → LayerNorm + residual → mean-pool → global embedding [1, output_dim]

The node feature dimension is fixed at ``NODE_FEAT_DIM = 12`` as defined in
``sketch_dag.py``.  When ``ObligationEmbedder`` is used, its output is
concatenated with the one-hot features before being projected to
``hidden_dim``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hieroflow.sketch.sketch_dag import NODE_FEAT_DIM, ProofSketch

try:
    from torch_geometric.nn import GATConv  # type: ignore[import]
    _HAS_TORCH_GEOMETRIC = True
except ImportError:  # pragma: no cover
    _HAS_TORCH_GEOMETRIC = False
    # Provide a minimal fallback so the module can be imported without
    # torch_geometric in test environments.
    class GATConv(nn.Module):  # type: ignore[no-redef]
        """Stub GATConv for environments without torch_geometric."""

        def __init__(self, in_channels: int, out_channels: int, heads: int = 1, **_: object) -> None:
            super().__init__()
            self.lin = nn.Linear(in_channels, out_channels * heads)
            self.out_channels = out_channels
            self.heads = heads

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            return self.lin(x)


from hieroflow.environment.obligation import ObligationType, ProofObligation


# ---------------------------------------------------------------------------
# ObligationEmbedder
# ---------------------------------------------------------------------------

class ObligationEmbedder(nn.Module):
    """
    Embeds a ``ProofObligation`` into a dense vector, independent of graph
    structure.

    This module provides richer initial node features than the raw one-hot
    encoding in ``ProofSketch.to_feature_matrix``.  Its output is used as
    the initial node representation before the GNN layers in ``SketchEncoder``.

    Architecture:
        obligation_type → Embedding(vocab_size, embed_dim)
        estimated_depth → Linear(1, embed_dim)
        concat → Linear(2 * embed_dim, hidden_dim) → ReLU
    """

    def __init__(
        self,
        vocab_size: int = 8,
        embed_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        """
        Initialise the embedder.

        Args:
            vocab_size:  Number of ``ObligationType`` values (default 8).
            embed_dim:   Embedding dimension for the type lookup table.
            hidden_dim:  Output dimension of the embedder.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.type_embedding = nn.Embedding(vocab_size, embed_dim)
        self.depth_proj = nn.Linear(1, embed_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, obligation: ProofObligation) -> torch.Tensor:
        """
        Embed a single ``ProofObligation``.

        Args:
            obligation: The ``ProofObligation`` to embed.

        Returns:
            Dense embedding of shape ``[hidden_dim]``.
        """
        ob_types = list(ObligationType)
        type_idx = torch.tensor(
            ob_types.index(obligation.obligation_type), dtype=torch.long
        )
        type_emb = self.type_embedding(type_idx)  # [embed_dim]

        depth_input = torch.tensor(
            [[obligation.estimated_depth / 10.0]], dtype=torch.float32
        )
        depth_emb = self.depth_proj(depth_input).squeeze(0)  # [embed_dim]

        combined = torch.cat([type_emb, depth_emb], dim=0)  # [2 * embed_dim]
        return self.output_proj(combined)  # [hidden_dim]

    def embed_batch(self, obligations: list[ProofObligation]) -> torch.Tensor:
        """
        Embed a list of ``ProofObligation`` objects in batch.

        Args:
            obligations: List of N obligations.

        Returns:
            ``torch.Tensor`` of shape ``[N, hidden_dim]``.
        """
        return torch.stack([self.forward(ob) for ob in obligations], dim=0)


# ---------------------------------------------------------------------------
# SketchEncoder
# ---------------------------------------------------------------------------

class SketchEncoder(nn.Module):
    """
    Graph neural network encoder for ``ProofSketch`` DAGs.

    Converts a variable-size DAG into a fixed-size global embedding used by
    ``SketchFlow`` to compute action logits.

    Architecture per layer:
        x_new = GATConv(x, edge_index) + Linear(x_input)  [residual]
        x_new = LayerNorm(x_new)
        x_new = ReLU(x_new)

    The input projection maps ``node_feat_dim`` → ``hidden_dim`` before the
    first GNN layer.  A final linear layer maps ``hidden_dim`` → ``output_dim``
    after mean-pool aggregation.
    """

    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        hidden_dim: int = 256,
        num_layers: int = 4,
        output_dim: int = 512,
        gat_heads: int = 4,
    ) -> None:
        """
        Initialise the encoder.

        Args:
            node_feat_dim: Input feature dimension per node (default 12).
            hidden_dim:    Hidden dimension used throughout the GNN.
            num_layers:    Number of GATConv message-passing layers.
            output_dim:    Dimension of the global graph embedding.
            gat_heads:     Number of attention heads in each GATConv.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Project raw node features to hidden_dim
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        # GAT layers (output_channels = hidden_dim // gat_heads per head,
        # concat=True → hidden_dim total)
        assert hidden_dim % gat_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by gat_heads ({gat_heads})"
        )
        per_head_dim = hidden_dim // gat_heads

        self.gat_layers = nn.ModuleList(
            [
                GATConv(
                    hidden_dim,
                    per_head_dim,
                    heads=gat_heads,
                    concat=True,
                    add_self_loops=True,
                )
                for _ in range(num_layers)
            ]
        )

        # Residual projections (input → hidden_dim for each layer's skip connection)
        self.residual_projs = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        # Global readout
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def encode_nodes(self, sketch: ProofSketch) -> torch.Tensor:
        """
        Return per-node embeddings for all nodes in *sketch*.

        Args:
            sketch: The ``ProofSketch`` to encode.

        Returns:
            ``torch.Tensor`` of shape ``[N, hidden_dim]``.
        """
        node_features, edge_index = sketch.to_feature_matrix()
        # node_features: [N, node_feat_dim]
        x = self.input_proj(node_features)  # [N, hidden_dim]

        for layer_idx in range(self.num_layers):
            x_res = self.residual_projs[layer_idx](x)
            x = self.gat_layers[layer_idx](x, edge_index)  # [N, hidden_dim]
            x = self.layer_norms[layer_idx](x + x_res)
            x = torch.relu(x)

        return x  # [N, hidden_dim]

    def forward(self, sketch: ProofSketch) -> torch.Tensor:
        """
        Compute a global graph embedding for *sketch*.

        Args:
            sketch: The ``ProofSketch`` to encode.

        Returns:
            ``torch.Tensor`` of shape ``[1, output_dim]``.
        """
        node_embs = self.encode_nodes(sketch)  # [N, hidden_dim]
        global_emb = node_embs.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        return self.output_proj(global_emb)  # [1, output_dim]
