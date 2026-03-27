"""
Curriculum sampler for theorem difficulty scheduling.

Implements a difficulty-aware theorem sampler that starts with easy theorems
and gradually introduces harder ones as training progresses.  This is
analogous to curriculum learning in supervised settings.

The difficulty of a theorem is estimated heuristically from its
``ProofObligation.estimated_depth`` value.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TheoremDifficulty:
    """Theorem with an estimated difficulty score."""

    theorem_name: str
    """Fully qualified Lean 4 theorem name."""

    difficulty: float
    """Estimated difficulty in [0, ∞).  Higher → harder."""

    success_rate: float = 0.0
    """Empirical success rate accumulated during training."""

    attempts: int = 0
    """Number of training attempts so far."""


class CurriculumSampler:
    """
    Adaptive curriculum sampler for theorems.

    Starts by sampling preferentially from easier theorems.  As training
    progresses (measured by a step counter), the difficulty threshold is
    raised using an exponential schedule.

    Additionally implements *self-paced* curriculum: theorems that the model
    solves consistently (success_rate > high_threshold) are temporarily
    down-weighted, focusing compute on medium-difficulty problems.
    """

    def __init__(
        self,
        theorems: list[str],
        difficulties: list[float] | None = None,
        ramp_steps: int = 10_000,
        easy_fraction: float = 0.7,
    ) -> None:
        """
        Initialise the sampler.

        Args:
            theorems:      List of theorem names.
            difficulties:  Optional per-theorem difficulty scores.  If None,
                           difficulty is assigned uniformly at random in [0, 1).
            ramp_steps:    Number of steps to ramp from easy-only to full mix.
            easy_fraction: Fraction of easy theorems in the initial curriculum.
        """
        if difficulties is None:
            difficulties = [random.random() for _ in theorems]

        if len(difficulties) != len(theorems):
            raise ValueError(
                f"difficulties length {len(difficulties)} != "
                f"theorems length {len(theorems)}"
            )

        self._items: list[TheoremDifficulty] = [
            TheoremDifficulty(name, diff)
            for name, diff in zip(theorems, difficulties)
        ]
        self._items.sort(key=lambda x: x.difficulty)
        self._ramp_steps = ramp_steps
        self._easy_fraction = easy_fraction
        self._step = 0

    def sample(self) -> str:
        """
        Sample a theorem name according to the current curriculum.

        Returns:
            A theorem name string.
        """
        self._step += 1

        # Ramp factor: 0 at start → 1 after ramp_steps
        ramp = min(1.0, self._step / max(self._ramp_steps, 1))

        # Number of theorems accessible at the current stage
        n_total = len(self._items)
        n_accessible = max(
            1,
            int(n_total * (self._easy_fraction + (1 - self._easy_fraction) * ramp)),
        )
        accessible = self._items[:n_accessible]

        # Weight inversely by success rate to focus on hard-but-solvable
        weights = [
            max(1.0 - item.success_rate, 0.05) for item in accessible
        ]
        chosen = random.choices(accessible, weights=weights, k=1)[0]
        return chosen.theorem_name

    def update(
        self, theorem_name: str, success: bool
    ) -> None:
        """
        Update the success rate of *theorem_name* after a training attempt.

        Uses an exponential moving average with α = 0.1.

        Args:
            theorem_name: The theorem that was attempted.
            success:      Whether the proof succeeded.
        """
        for item in self._items:
            if item.theorem_name == theorem_name:
                item.attempts += 1
                alpha = 0.1
                item.success_rate = (
                    (1 - alpha) * item.success_rate + alpha * float(success)
                )
                break

    def get_stats(self) -> dict[str, Any]:
        """Return curriculum statistics for logging."""
        n_total = len(self._items)
        ramp = min(1.0, self._step / max(self._ramp_steps, 1))
        n_accessible = max(
            1,
            int(n_total * (self._easy_fraction + (1 - self._easy_fraction) * ramp)),
        )
        mean_success = (
            sum(i.success_rate for i in self._items[:n_accessible]) / n_accessible
        )
        return {
            "step": self._step,
            "n_accessible": n_accessible,
            "n_total": n_total,
            "ramp_fraction": ramp,
            "mean_success_rate": mean_success,
        }
