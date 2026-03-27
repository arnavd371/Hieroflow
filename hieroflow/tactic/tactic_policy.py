"""
LLM-based tactic policy (wraps a pretrained code model).

``TacticPolicy`` provides a high-level interface for generating Lean 4
tactics using a pretrained code LLM.  It is a thin wrapper around
``TacticFlow`` that handles tokenisation, beam search, and candidate
de-duplication.

In training, ``TacticFlow.sample_tactic`` is called directly.  This class
is intended for inference and evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from hieroflow.environment.obligation import ProofObligation
from hieroflow.environment.proof_state import LeanProofState
from hieroflow.tactic.tactic_gfn import TacticFlow

logger = logging.getLogger(__name__)


@dataclass
class TacticCandidate:
    """A single candidate tactic with its log-probability."""

    __slots__ = ("tactic", "log_prob", "rank")

    tactic: str
    """The Lean 4 tactic string."""

    log_prob: float
    """Log probability of this tactic under TacticFlow's forward policy."""

    rank: int
    """Rank among all candidates (0 = highest log_prob)."""


class TacticPolicy:
    """
    High-level inference wrapper around ``TacticFlow``.

    Generates multiple tactic candidates for a given (proof_state,
    obligation) pair via repeated sampling.  Candidates are de-duplicated
    and sorted by log-probability.

    This class exists at the interface between TacticFlow and the
    ``LeanEnv``: it produces candidate strings that are then executed by
    Lean.  It must NOT be used to compute training losses (use
    ``TacticFlow`` directly for that).
    """

    def __init__(
        self,
        tactic_flow: TacticFlow,
        num_candidates: int = 16,
        temperature: float = 1.0,
        max_length: int = 128,
    ) -> None:
        """
        Initialise the policy.

        Args:
            tactic_flow:    The trained ``TacticFlow`` model.
            num_candidates: Number of tactic candidates to generate.
            temperature:    Sampling temperature for ``TacticFlow``.
            max_length:     Maximum tactic token length.
        """
        self.tactic_flow = tactic_flow
        self.num_candidates = num_candidates
        self.temperature = temperature
        self.max_length = max_length

    def generate_candidates(
        self,
        proof_state: LeanProofState,
        obligation: ProofObligation,
    ) -> list[TacticCandidate]:
        """
        Generate ranked tactic candidates.

        Args:
            proof_state: Current ``LeanProofState``.
            obligation:  ``ProofObligation`` from SketchFlow.

        Returns:
            List of ``TacticCandidate`` objects sorted by descending
            log-probability, de-duplicated.
        """
        seen: dict[str, float] = {}

        for _ in range(self.num_candidates):
            try:
                tactic, log_prob = self.tactic_flow.sample_tactic(
                    proof_state=proof_state,
                    obligation=obligation,
                    temperature=self.temperature,
                    max_length=self.max_length,
                )
                tactic = tactic.strip()
                if not tactic:
                    continue
                # Keep the highest log_prob for duplicate tactics
                if tactic not in seen or log_prob > seen[tactic]:
                    seen[tactic] = log_prob
            except Exception as exc:  # noqa: BLE001
                logger.warning("TacticPolicy sample error: %s", exc)

        sorted_tactics = sorted(seen.items(), key=lambda kv: kv[1], reverse=True)
        return [
            TacticCandidate(tactic=t, log_prob=lp, rank=i)
            for i, (t, lp) in enumerate(sorted_tactics)
        ]

    def best_tactic(
        self,
        proof_state: LeanProofState,
        obligation: ProofObligation,
    ) -> TacticCandidate | None:
        """
        Return the single highest-probability tactic candidate.

        Returns:
            The best ``TacticCandidate``, or None if generation failed.
        """
        candidates = self.generate_candidates(proof_state, obligation)
        return candidates[0] if candidates else None
