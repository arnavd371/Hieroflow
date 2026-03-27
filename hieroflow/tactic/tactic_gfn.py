"""
TacticFlow: inner GFlowNet conditioned on a ProofObligation.

TacticFlow generates Lean 4 tactic token sequences autoregressively.  It
conditions on a ``ProofObligation`` (from SketchFlow) via a cross-attention
adapter inserted after the LLM backbone's final hidden layer.

The LLM backbone is passed in as a HuggingFace ``PreTrainedModel`` — this
module only implements the conditioning adapter and the GFlowNet objective.

GFlowNet objective: Trajectory Balance (token-level).
    L = (log Z + Σ_t log P_F(a_t|s_t) - log R - Σ_t log P_B(s_t|s_{t+1}))^2

The backward policy is uniform over token positions (standard choice for
autoregressive GFlowNets).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from hieroflow.environment.obligation import ProofObligation
from hieroflow.environment.proof_state import LeanProofState
from hieroflow.sketch.sketch_encoder import ObligationEmbedder

if TYPE_CHECKING:
    from transformers import PreTrainedModel  # type: ignore[import]


# ---------------------------------------------------------------------------
# ObligationCrossAttention
# ---------------------------------------------------------------------------

class ObligationCrossAttention(nn.Module):
    """
    Cross-attention adapter that conditions LM hidden states on a
    ``ProofObligation`` embedding.

    Keys and values come from the obligation embedding (projected to
    ``lm_hidden_dim``); queries come from the LM's final hidden states.

    This adapter is inserted *after* the LLM backbone's last transformer
    layer.  Its output is added to the LM hidden states (residual) before
    the LM head computes logits.
    """

    def __init__(
        self,
        lm_hidden_dim: int,
        obligation_embed_dim: int = 128,
        num_heads: int = 8,
    ) -> None:
        """
        Initialise the cross-attention adapter.

        Args:
            lm_hidden_dim:        Hidden dimension of the LLM backbone.
            obligation_embed_dim: Output dimension of ``ObligationEmbedder``.
            num_heads:            Number of attention heads.
        """
        super().__init__()
        self.lm_hidden_dim = lm_hidden_dim
        self.num_heads = num_heads

        # Project obligation embedding to lm_hidden_dim for K/V
        self.kv_proj = nn.Linear(obligation_embed_dim, lm_hidden_dim)

        # Multi-head cross-attention: Q from LM hidden, K/V from obligation
        self.attn = nn.MultiheadAttention(
            embed_dim=lm_hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(lm_hidden_dim)

    def forward(
        self,
        lm_hidden_states: torch.Tensor,
        obligation_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply cross-attention conditioning.

        Args:
            lm_hidden_states: ``[batch, seq_len, lm_hidden_dim]`` from LM.
            obligation_emb:   ``[obligation_embed_dim]`` from ObligationEmbedder.

        Returns:
            Conditioned hidden states ``[batch, seq_len, lm_hidden_dim]``.
        """
        # Project obligation to key/value shape: [1, 1, lm_hidden_dim]
        kv = self.kv_proj(obligation_emb).unsqueeze(0).unsqueeze(0)

        batch = lm_hidden_states.size(0)
        kv_expanded = kv.expand(batch, -1, -1)  # [batch, 1, lm_hidden_dim]

        attn_out, _ = self.attn(
            query=lm_hidden_states,
            key=kv_expanded,
            value=kv_expanded,
        )
        # Residual connection + LayerNorm
        return self.layer_norm(lm_hidden_states + attn_out)


# ---------------------------------------------------------------------------
# TacticFlow
# ---------------------------------------------------------------------------

class TacticFlow(nn.Module):
    """
    Inner GFlowNet that generates Lean 4 tactic token sequences.

    TacticFlow wraps a pretrained LLM backbone with a cross-attention
    conditioning adapter.  It is trained with the Trajectory Balance
    objective at the token level.

    The LLM backbone is *not* modified in-place; its parameters may be
    frozen during the early stages of training (controlled by the trainer).

    Interface contract:
    - Receives a ``ProofObligation`` from SketchFlow.
    - Returns a tactic string (and its log-probability) to the environment.
    - Nothing else crosses the SketchFlow ↔ TacticFlow boundary.
    """

    def __init__(
        self,
        lm_backbone: "PreTrainedModel",
        obligation_embedder: ObligationEmbedder,
        hidden_dim: int = 512,
        max_tactic_length: int = 128,
        cross_attention_heads: int = 8,
    ) -> None:
        """
        Initialise TacticFlow.

        Args:
            lm_backbone:          HuggingFace PreTrainedModel used as token
                                  generator (e.g. deepseek-coder-1.3b).
            obligation_embedder:  Pre-built ``ObligationEmbedder`` shared with
                                  ``SketchEncoder``.
            hidden_dim:           Hidden dimension for the conditioning adapter.
            max_tactic_length:    Maximum number of tokens to generate.
            cross_attention_heads: Number of attention heads in the cross-attn
                                   adapter.
        """
        super().__init__()
        self.lm_backbone = lm_backbone
        self.obligation_embedder = obligation_embedder
        self.max_tactic_length = max_tactic_length

        # Infer the LM hidden dimension
        lm_hidden_dim: int = getattr(
            lm_backbone.config, "hidden_size",
            getattr(lm_backbone.config, "n_embd", hidden_dim),
        )
        self.lm_hidden_dim = lm_hidden_dim

        # Cross-attention adapter
        self.cross_attention = ObligationCrossAttention(
            lm_hidden_dim=lm_hidden_dim,
            obligation_embed_dim=obligation_embedder.hidden_dim,
            num_heads=cross_attention_heads,
        )

        # Learnable log partition function (separate from SketchFlow's)
        self.log_z = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        proof_state: LeanProofState,
        obligation: ProofObligation,
        tactic_prefix: list[int],
    ) -> torch.Tensor:
        """
        Compute logits over the next tactic token.

        Args:
            proof_state:    Current ``LeanProofState`` (used to construct the
                            LM prompt: goal strings + tactic history).
            obligation:     ``ProofObligation`` from SketchFlow for conditioning.
            tactic_prefix:  Token ids of the tactic generated so far.

        Returns:
            ``torch.Tensor`` of shape ``[vocab_size]`` — raw logits over the
            next token.
        """
        # Build the input prompt: "<goals>\n<tactic_history>\n<prefix>"
        prompt_ids = self._build_prompt_ids(proof_state, tactic_prefix)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)

        # Run LM backbone (no gradient for backbone parameters by default)
        lm_outputs = self.lm_backbone(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        # Final hidden states: [1, seq_len, lm_hidden_dim]
        hidden_states = lm_outputs.hidden_states[-1]

        # Condition on the obligation via cross-attention
        obligation_emb = self.obligation_embedder(obligation)  # [obligation_embed_dim]
        conditioned = self.cross_attention(hidden_states, obligation_emb)

        # Use the LM's language model head on the conditioned hidden state
        # of the *last* token position to predict the next token.
        last_hidden = conditioned[:, -1, :]  # [1, lm_hidden_dim]
        logits = self.lm_backbone.lm_head(last_hidden).squeeze(0)  # [vocab_size]
        return logits

    def sample_tactic(
        self,
        proof_state: LeanProofState,
        obligation: ProofObligation,
        temperature: float = 1.0,
        max_length: int = 128,
    ) -> tuple[str, float]:
        """
        Autoregressively sample a complete tactic string.

        Stops when the EOS token is generated or ``max_length`` is reached.

        Args:
            proof_state:  Current Lean proof state.
            obligation:   ``ProofObligation`` for conditioning.
            temperature:  Sampling temperature (lower → more greedy).
            max_length:   Hard cap on tactic token length.

        Returns:
            A tuple ``(tactic_string, log_prob_of_full_tactic)``.
        """
        tokenizer = getattr(self.lm_backbone, "tokenizer", None)
        eos_id: int = (
            self.lm_backbone.config.eos_token_id
            if hasattr(self.lm_backbone.config, "eos_token_id")
            else 2
        )

        tactic_tokens: list[int] = []
        cumulative_log_prob: float = 0.0

        for _ in range(min(max_length, self.max_tactic_length)):
            logits = self.forward(proof_state, obligation, tactic_tokens)
            log_probs = F.log_softmax(logits / temperature, dim=-1)
            probs = log_probs.exp()

            next_token = torch.multinomial(probs, num_samples=1).item()
            assert isinstance(next_token, int)

            cumulative_log_prob += log_probs[next_token].item()
            tactic_tokens.append(next_token)

            if next_token == eos_id:
                break

        # Decode tokens to string
        if tokenizer is not None:
            tactic_string = tokenizer.decode(tactic_tokens, skip_special_tokens=True)
        else:
            tactic_string = " ".join(str(t) for t in tactic_tokens)

        return tactic_string, cumulative_log_prob

    def trajectory_balance_loss(
        self,
        tactic_tokens: list[int],
        log_pf: float,
        lean_reward: float,
        log_z: nn.Parameter | None = None,
    ) -> torch.Tensor:
        """
        Compute the TB loss for a single tactic trajectory.

        Implements (token-level):
            L = (log Z + Σ log P_F(t) - log R - Σ log P_B(t))^2

        The backward policy is uniform: log P_B(t) = -log(|tactic_tokens|).

        Args:
            tactic_tokens: Token id sequence for the generated tactic.
            log_pf:        Sum of token-level log forward probabilities.
            lean_reward:   Terminal reward in *linear* space from Lean.
            log_z:         Explicit log Z parameter; defaults to ``self.log_z``.

        Returns:
            Scalar loss ``torch.Tensor``.
        """
        if log_z is None:
            log_z = self.log_z

        log_r = torch.tensor(
            max(math.log(max(lean_reward, 1e-45)), -100.0), dtype=torch.float32
        )
        T = max(len(tactic_tokens), 1)
        log_pb = torch.tensor(-math.log(T) * T, dtype=torch.float32)
        log_pf_tensor = torch.tensor(log_pf, dtype=torch.float32, requires_grad=True)

        balance = log_z.squeeze() + log_pf_tensor - log_r - log_pb
        return balance.pow(2)

    def forward_pass_log_probs(
        self,
        proof_state: LeanProofState,
        obligation: ProofObligation,
        tactic_tokens: list[int],
    ) -> torch.Tensor:
        """
        Compute token-level log probabilities for a *given* tactic token sequence.

        This method is used during off-policy training to re-score stored
        trajectories from the replay buffer.

        Args:
            proof_state:   Current Lean proof state.
            obligation:    ``ProofObligation`` for conditioning.
            tactic_tokens: Complete token sequence to score.

        Returns:
            ``torch.Tensor`` of shape ``[T]`` — log P_F(token_t | prefix_t)
            for each position t.
        """
        log_prob_list: list[torch.Tensor] = []
        for t in range(len(tactic_tokens)):
            prefix = tactic_tokens[:t]
            logits = self.forward(proof_state, obligation, prefix)
            log_probs = F.log_softmax(logits, dim=-1)
            log_prob_list.append(log_probs[tactic_tokens[t]])

        return torch.stack(log_prob_list)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt_ids(
        self,
        proof_state: LeanProofState,
        tactic_prefix: list[int],
    ) -> list[int]:
        """
        Construct the integer token id prompt from a ``LeanProofState``.

        The prompt concatenates:
        1. Encoded goal strings (one per open goal).
        2. Encoded tactic history.
        3. The tactic prefix generated so far.

        When no tokenizer is attached to the backbone, falls back to a
        minimal mock that treats each character as one token id (for tests).
        """
        tokenizer = getattr(self.lm_backbone, "tokenizer", None)
        if tokenizer is None:
            # Minimal fallback: encode the goal strings as ASCII ordinals
            text = "\n".join(proof_state.goals + proof_state.tactic_history)
            base_ids = [ord(c) % 256 for c in text[:64]]
            return base_ids + tactic_prefix

        sep = tokenizer.eos_token_id or 2
        goal_text = "\n".join(proof_state.goals)
        history_text = "\n".join(proof_state.tactic_history)
        prompt_text = f"-- Goals:\n{goal_text}\n-- History:\n{history_text}\n-- Tactic: "

        prompt_ids: list[int] = tokenizer.encode(
            prompt_text, add_special_tokens=False
        )
        return prompt_ids + tactic_prefix
