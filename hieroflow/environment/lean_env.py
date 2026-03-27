"""
LeanDojo wrapper — proof state management, tactic execution, and reward.

``LeanEnv`` is the single point of contact with the Lean 4 proof assistant.
It wraps LeanDojo's ``Dojo`` context manager, translating between Lean's raw
API and the HieroFlow dataclasses (``LeanProofState``, ``TacticResult``).

Design contract:
- Only this module may import lean_dojo.
- All other modules receive ``LeanProofState`` / ``TacticResult`` objects.
- The reward for TacticFlow is computed here and lives in log-space.
"""

from __future__ import annotations

import logging
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

from hieroflow.environment.proof_state import LeanProofState, TacticResult

logger = logging.getLogger(__name__)

# Sentinel for reward values stored in log-space
_LOG_ZERO: float = -100.0


@dataclass
class LeanEnvConfig:
    """Configuration for a ``LeanEnv`` instance."""

    __slots__ = (
        "timeout_seconds",
        "max_depth",
        "partial_reward_scale",
        "repo_path",
        "project_name",
    )

    timeout_seconds: float
    """Hard wall-clock timeout per tactic call (seconds)."""

    max_depth: int
    """Maximum number of tactics allowed before the episode is truncated."""

    partial_reward_scale: float
    """
    Scale for the partial (goal-reducing) reward.

    When a tactic reduces the goal count but doesn't close the proof,
    the reward is ``partial_reward_scale * (goals_closed / total_goals)``.
    """

    repo_path: str
    """Absolute path to the Lean 4 project repository on disk."""

    project_name: str
    """Lean project name (used by LeanDojo for finding declaration files)."""


class LeanEnv:
    """
    LeanDojo-backed theorem proving environment.

    Wraps LeanDojo's low-level ``Dojo`` API to provide a Gym-like interface:
    - ``reset(theorem_name)`` → initial ``LeanProofState``
    - ``step(state, tactic)``  → ``TacticResult``

    The class is *not* thread-safe.  Use one instance per worker process.

    Example::

        env = LeanEnv(config)
        with env.open_theorem("Nat.add_comm") as state:
            result = env.step(state, "induction n")
            ...
    """

    def __init__(self, config: LeanEnvConfig) -> None:
        """
        Initialise the environment.

        The LeanDojo Dojo context is NOT opened here; it is opened lazily
        inside ``open_theorem`` to avoid holding Lean server resources between
        episodes.
        """
        self.config = config
        self._dojo: object | None = None  # lean_dojo.Dojo instance
        self._current_theorem: str | None = None
        self._episode_depth: int = 0

    @contextmanager
    def open_theorem(
        self, theorem_name: str
    ) -> Generator[LeanProofState, None, None]:
        """
        Context manager that opens a Lean proof session for *theorem_name*.

        Yields the initial ``LeanProofState``.  The session is closed when
        the context exits.  Calls ``_close_session`` on exit to release Lean
        server resources.

        Args:
            theorem_name: Fully qualified Lean 4 theorem name.

        Yields:
            The initial proof state for the theorem.
        """
        self._current_theorem = theorem_name
        self._episode_depth = 0

        try:
            # Attempt to import lean_dojo; fall back gracefully in unit tests.
            try:
                import lean_dojo  # type: ignore[import]

                theorem = lean_dojo.LeanTheorem(theorem_name, self.config.repo_path)
                with lean_dojo.Dojo(theorem) as (dojo, state):
                    self._dojo = dojo
                    initial_state = self._convert_state(theorem_name, state, [])
                    yield initial_state
            except ImportError:
                logger.warning(
                    "lean_dojo not installed — yielding stub initial state.  "
                    "This is acceptable in unit-test contexts only."
                )
                yield self._stub_state(theorem_name)
        finally:
            self._dojo = None
            self._current_theorem = None

    def step(self, state: LeanProofState, tactic: str) -> TacticResult:
        """
        Apply *tactic* to *state* and return the result.

        Enforces the per-tactic timeout and the maximum episode depth.
        Reward computation (log-space) is handled separately in
        ``compute_log_reward``.

        Args:
            state:  Current ``LeanProofState``.
            tactic: Lean 4 tactic string to apply.

        Returns:
            ``TacticResult`` with the new state (or None on fatal error).
        """
        if self._dojo is None:
            return self._stub_step(state, tactic)

        if self._episode_depth >= self.config.max_depth:
            return TacticResult(
                tactic=tactic,
                new_state=None,
                lean_feedback="Max depth exceeded",
                success=False,
                goals_closed=0,
            )

        start = time.monotonic()
        try:
            import lean_dojo  # type: ignore[import]

            result = self._dojo.run_tac(lean_dojo.TacticState, tactic)  # type: ignore[attr-defined]
            elapsed = time.monotonic() - start

            if elapsed > self.config.timeout_seconds:
                return TacticResult(
                    tactic=tactic,
                    new_state=None,
                    lean_feedback=f"Timeout after {elapsed:.1f}s",
                    success=False,
                    goals_closed=0,
                )

            if isinstance(result, lean_dojo.ProofFinished):  # type: ignore[attr-defined]
                new_state = LeanProofState(
                    theorem_name=state.theorem_name,
                    goals=[],
                    tactic_history=state.tactic_history + [tactic],
                    depth=state.depth + 1,
                    is_terminal=True,
                    is_error=False,
                )
                self._episode_depth += 1
                return TacticResult(
                    tactic=tactic,
                    new_state=new_state,
                    lean_feedback="Proof complete",
                    success=True,
                    goals_closed=len(state.goals),
                )
            elif isinstance(result, lean_dojo.TacticState):  # type: ignore[attr-defined]
                new_goals = list(result.goals)
                goals_closed = max(0, len(state.goals) - len(new_goals))
                new_state = self._convert_state(
                    state.theorem_name,
                    result,
                    state.tactic_history + [tactic],
                )
                self._episode_depth += 1
                return TacticResult(
                    tactic=tactic,
                    new_state=new_state,
                    lean_feedback=str(result),
                    success=True,
                    goals_closed=goals_closed,
                )
            else:
                # LeanDojo error state
                self._episode_depth += 1
                return TacticResult(
                    tactic=tactic,
                    new_state=None,
                    lean_feedback=str(result),
                    success=False,
                    goals_closed=0,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("LeanEnv.step exception: %s", exc)
            return TacticResult(
                tactic=tactic,
                new_state=None,
                lean_feedback=str(exc),
                success=False,
                goals_closed=0,
            )

    def compute_log_reward(
        self,
        result: TacticResult,
        original_goal_count: int,
        timeout: bool = False,
    ) -> float:
        """
        Compute the log-space reward for a ``TacticResult``.

        Reward schedule (matching ``HieroFlowTrainer._compute_inner_reward``):
        - Complete proof:    log(1.0) = 0.0
        - Partial progress:  log(partial_reward_scale * fraction_closed)
        - Error:             log(0.0) = ``_LOG_ZERO``
        - Timeout:           log(max(0.0, -0.1)) → clamped to ``_LOG_ZERO``

        Args:
            result:              The ``TacticResult`` from ``step()``.
            original_goal_count: Number of goals before the tactic was applied.
            timeout:             Whether the tactic timed out.

        Returns:
            Log-reward as a float, clamped to [``_LOG_ZERO``, 0].
        """
        if timeout or result.new_state is None:
            return _LOG_ZERO

        if result.success and result.new_state.is_terminal:
            return 0.0  # log(1.0)

        if result.success and result.goals_closed > 0 and original_goal_count > 0:
            fraction = result.goals_closed / original_goal_count
            raw = self.config.partial_reward_scale * fraction
            return max(_LOG_ZERO, math.log(max(raw, 1e-45)))

        return _LOG_ZERO

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _convert_state(
        self,
        theorem_name: str,
        lean_state: object,
        history: list[str],
    ) -> LeanProofState:
        """Convert a raw LeanDojo TacticState into a ``LeanProofState``."""
        goals = [str(g) for g in getattr(lean_state, "goals", [])]
        return LeanProofState(
            theorem_name=theorem_name,
            goals=goals,
            tactic_history=list(history),
            depth=len(history),
            is_terminal=len(goals) == 0,
            is_error=False,
        )

    def _stub_state(self, theorem_name: str) -> LeanProofState:
        """Return a stub initial state for use in unit tests (no Lean server)."""
        return LeanProofState(
            theorem_name=theorem_name,
            goals=[f"⊢ {theorem_name}"],
            tactic_history=[],
            depth=0,
            is_terminal=False,
            is_error=False,
        )

    def _stub_step(self, state: LeanProofState, tactic: str) -> TacticResult:
        """
        Return a stub result when no Lean server is available.

        # TEST ONLY — do not call in production training loops.
        """
        new_state = LeanProofState(
            theorem_name=state.theorem_name,
            goals=[],
            tactic_history=state.tactic_history + [tactic],
            depth=state.depth + 1,
            is_terminal=True,
            is_error=False,
        )
        return TacticResult(
            tactic=tactic,
            new_state=new_state,
            lean_feedback="stub",
            success=True,
            goals_closed=len(state.goals),
        )
