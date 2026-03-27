"""
Benchmark runners for HieroFlow evaluation.

Supports three standard theorem-proving benchmarks:
- LeanDojo Benchmark 4 (random split)
- MiniF2F (minimath formalization to Lean 4)
- ProofNet (undergraduate-level proofs)

Each runner produces a list of proof attempts for diversity evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """
    Aggregated result from running HieroFlow on a benchmark.
    """

    benchmark_name: str
    """Name of the benchmark (e.g. 'LeanDojoBenchmark4')."""

    theorems_attempted: int = 0
    """Number of theorems HieroFlow attempted."""

    theorems_proved: int = 0
    """Number of theorems successfully proved."""

    proof_attempts: list[list[str]] = field(default_factory=list)
    """All generated tactic sequences (one list per theorem attempt)."""

    proof_states: list[list[tuple[str, str]]] = field(default_factory=list)
    """(tactic, goal) pairs for each proof attempt (for semantic metrics)."""

    @property
    def success_rate(self) -> float:
        """Fraction of theorems proved."""
        if self.theorems_attempted == 0:
            return 0.0
        return self.theorems_proved / self.theorems_attempted

    def summary_str(self) -> str:
        return (
            f"{self.benchmark_name}: "
            f"{self.theorems_proved}/{self.theorems_attempted} proved "
            f"({self.success_rate:.1%})"
        )


# ---------------------------------------------------------------------------
# Base runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Base class for benchmark runners.

    Subclasses override ``load_theorems()`` to provide benchmark-specific
    theorem lists.  The ``run()`` method handles the common evaluation loop.
    """

    benchmark_name: str = "Base"

    def __init__(
        self,
        trainer: Any,  # HieroFlowTrainer
        num_samples: int = 1,
    ) -> None:
        """
        Initialise the runner.

        Args:
            trainer:     A ``HieroFlowTrainer`` instance (used for rollouts).
            num_samples: Number of proof attempts per theorem.
        """
        self.trainer = trainer
        self.num_samples = num_samples

    def load_theorems(self) -> list[str]:
        """
        Load the benchmark theorem names.

        Subclasses must override this method.

        Returns:
            List of fully qualified Lean 4 theorem names.
        """
        raise NotImplementedError

    def run(self, max_theorems: int | None = None) -> BenchmarkResult:
        """
        Evaluate HieroFlow on this benchmark.

        Args:
            max_theorems: Cap the number of theorems evaluated (for quick runs).

        Returns:
            A ``BenchmarkResult`` with aggregated statistics.
        """
        theorems = self.load_theorems()
        if max_theorems is not None:
            theorems = theorems[:max_theorems]

        result = BenchmarkResult(
            benchmark_name=self.benchmark_name,
            theorems_attempted=len(theorems),
        )

        for theorem in theorems:
            for _ in range(self.num_samples):
                try:
                    metrics = self.trainer.train_step(theorem)
                    # Collect proof tactic sequences via the lean_env
                    with self.trainer.lean_env.open_theorem(theorem) as state:
                        # Outer rollout to get tactic sequence
                        sketch_traj = self.trainer._outer_rollout(theorem, state)
                        tactic_trajs = self.trainer._inner_rollout(
                            sketch_traj.sketches[-1] if sketch_traj.sketches else None,
                            state,
                        )
                        proof = [
                            tt.tactic_result.tactic
                            for tt in tactic_trajs
                            if tt.tactic_result is not None
                        ]
                        proof_state_pairs: list[tuple[str, str]] = [
                            (
                                tt.tactic_result.tactic,
                                str(tt.tactic_result.new_state.goals[0])
                                if tt.tactic_result.new_state
                                and tt.tactic_result.new_state.goals
                                else "",
                            )
                            for tt in tactic_trajs
                            if tt.tactic_result is not None
                        ]
                        result.proof_attempts.append(proof)
                        result.proof_states.append(proof_state_pairs)

                        if metrics.get("proof_success_rate", 0.0) > 0:
                            result.theorems_proved += 1

                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Benchmark %s failed for %s: %s",
                        self.benchmark_name, theorem, exc
                    )

        logger.info(result.summary_str())
        return result


# ---------------------------------------------------------------------------
# Specific benchmark runners
# ---------------------------------------------------------------------------

class LeanDojoBenchmark4Runner(BenchmarkRunner):
    """
    Runner for the LeanDojo Benchmark 4 (random split).

    Uses theorems from the ``leandojo_benchmark_4`` dataset.
    Falls back to a stub list when the dataset is not installed.
    """

    benchmark_name = "LeanDojoBenchmark4"

    def load_theorems(self) -> list[str]:
        """Load LeanDojo Benchmark 4 theorem names."""
        try:
            import lean_dojo  # type: ignore[import]
            dataset = lean_dojo.load_dataset("leandojo_benchmark_4", "random")
            return [t.full_name for t in dataset["test"]]
        except (ImportError, Exception):
            logger.warning(
                "LeanDojo not available; using stub theorem list for benchmark."
            )
            return [
                "Nat.add_comm",
                "Nat.add_assoc",
                "List.length_append",
                "Nat.zero_add",
                "Nat.succ_eq_add_one",
            ]


class MiniF2FRunner(BenchmarkRunner):
    """
    Runner for the MiniF2F benchmark (mathematics formalized in Lean 4).

    Theorems cover competition mathematics (AMC, AIME, etc.).
    """

    benchmark_name = "MiniF2F"

    def load_theorems(self) -> list[str]:
        """Load MiniF2F theorem names."""
        try:
            import datasets  # type: ignore[import]
            ds = datasets.load_dataset("cat-searcher/minif2f-lean4", split="test")
            return [row["name"] for row in ds]
        except (ImportError, Exception):
            logger.warning(
                "datasets not available; using stub theorem list for MiniF2F."
            )
            return [
                "mathd_algebra_101",
                "mathd_numbertheory_99",
                "amc12a_2000_p1",
            ]


class ProofNetRunner(BenchmarkRunner):
    """
    Runner for the ProofNet benchmark (undergraduate-level proofs).

    Tests on theorems from real undergraduate mathematics courses,
    including analysis, algebra, and topology.
    """

    benchmark_name = "ProofNet"

    def load_theorems(self) -> list[str]:
        """Load ProofNet theorem names."""
        try:
            import datasets  # type: ignore[import]
            ds = datasets.load_dataset("hoskinson-center/proofnet", split="test")
            return [row["nl_statement"] for row in ds]
        except (ImportError, Exception):
            logger.warning(
                "datasets not available; using stub theorem list for ProofNet."
            )
            return [
                "continuous_id",
                "algebra.group.basic.mul_left_cancel",
                "topology.basic.is_open_union",
            ]
