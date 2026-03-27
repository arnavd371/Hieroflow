"""
Joint training loop for HieroFlow.

Training alternates between SketchFlow (outer) and TacticFlow (inner):

1. Sample a theorem from the curriculum.
2. Run SketchFlow to generate a proof sketch (outer rollout).
3. For each node in the sketch, run TacticFlow to fill in tactics (inner rollout).
4. Execute the complete proof attempt in Lean via LeanEnv.
5. Compute rewards and update both GFlowNets.

The alternation ratio (default 1 outer : 3 inner) is controlled by config.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from hieroflow.environment.lean_env import LeanEnv
from hieroflow.environment.obligation import ObligationExtractor
from hieroflow.environment.proof_state import LeanProofState, TacticResult
from hieroflow.sketch.sketch_dag import ProofSketch, SketchNodeState
from hieroflow.sketch.sketch_gfn import SketchFlow, SketchTrajectory
from hieroflow.sketch.sketch_reward import compute_sketch_log_reward
from hieroflow.tactic.tactic_gfn import TacticFlow
from hieroflow.tactic.tactic_reward import compute_tactic_log_reward
from hieroflow.training.replay_buffer import PrioritisedReplayBuffer, Trajectory
from hieroflow.training.trajectory_balance import trajectory_balance_loss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TacticTrajectory
# ---------------------------------------------------------------------------

@dataclass
class TacticTrajectory:
    """
    A completed tactic-level trajectory for one sketch node.

    Produced by ``HieroFlowTrainer._inner_rollout`` and stored in the
    replay buffer as a 'tactic'-level ``Trajectory``.
    """

    node_id: str
    """Sketch node this trajectory corresponds to."""

    tactic_tokens: list[int] = field(default_factory=list)
    """Token ids of the generated tactic."""

    log_pf: float = 0.0
    """Sum of forward log-probs over the tactic token sequence."""

    log_pb: float = 0.0
    """Sum of backward log-probs (uniform: -log T per step)."""

    tactic_result: TacticResult | None = None
    """Lean execution result for this tactic."""

    log_reward: float = -100.0
    """Log-space inner reward."""


# ---------------------------------------------------------------------------
# TrainingMetrics
# ---------------------------------------------------------------------------

@dataclass
class TrainingMetrics:
    """Metrics snapshot for one training step."""

    step: int = 0
    outer_loss: float = 0.0
    inner_loss: float = 0.0
    outer_reward: float = 0.0
    inner_reward: float = 0.0
    proof_success_rate: float = 0.0
    mean_sketch_depth: float = 0.0
    buffer_size: int = 0


# ---------------------------------------------------------------------------
# HieroFlowTrainer
# ---------------------------------------------------------------------------

class HieroFlowTrainer:
    """
    Alternating trainer for SketchFlow (outer) and TacticFlow (inner).

    Manages the full training loop including rollouts, reward computation,
    gradient updates, replay-buffer population, and logging.

    See ``train_step`` for the per-theorem update cycle.
    """

    def __init__(
        self,
        sketch_gfn: SketchFlow,
        tactic_gfn: TacticFlow,
        lean_env: LeanEnv,
        replay_buffer: PrioritisedReplayBuffer,
        config: dict[str, Any],
    ) -> None:
        """
        Initialise the trainer.

        Args:
            sketch_gfn:    Outer GFlowNet (SketchFlow).
            tactic_gfn:    Inner GFlowNet (TacticFlow).
            lean_env:      LeanDojo-backed environment.
            replay_buffer: Shared prioritised replay buffer.
            config:        Hyperparameter dict (see ``configs/base.yaml``).
        """
        self.sketch_gfn = sketch_gfn
        self.tactic_gfn = tactic_gfn
        self.lean_env = lean_env
        self.replay_buffer = replay_buffer
        self.config = config

        self._obligation_extractor = ObligationExtractor()
        self._step = 0

        sketch_lr: float = config.get("sketch_lr", 1e-4)
        tactic_lr: float = config.get("tactic_lr", 1e-4)

        self._sketch_opt = torch.optim.Adam(
            sketch_gfn.parameters(), lr=sketch_lr
        )
        self._tactic_opt = torch.optim.Adam(
            tactic_gfn.parameters(), lr=tactic_lr
        )

        self._grad_clip: float = config.get("grad_clip", 1.0)
        self._outer_inner_ratio: int = config.get("outer_inner_ratio", 3)

        # Optional wandb logging
        self._use_wandb: bool = config.get("use_wandb", False)
        if self._use_wandb:
            try:
                import wandb  # type: ignore[import]
                self._wandb = wandb
            except ImportError:
                logger.warning("wandb not installed; disabling wandb logging")
                self._use_wandb = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_step(self, theorem: str) -> dict[str, float]:
        """
        Execute one full training round for *theorem*.

        Steps:
        1. Reset the Lean environment.
        2. Run outer rollout (SketchFlow).
        3. Run inner rollout (TacticFlow for each sketch node).
        4. Compute rewards.
        5. Update sketch GFlowNet (every step).
        6. Update tactic GFlowNet (every ``outer_inner_ratio`` inner steps
           per outer step).

        Args:
            theorem: Lean 4 theorem name.

        Returns:
            Dict of scalar metrics for this step.
        """
        self._step += 1

        with self.lean_env.open_theorem(theorem) as proof_state:
            # 1. Outer rollout
            sketch_traj = self._outer_rollout(theorem, proof_state)

            # 2. Inner rollout
            tactic_trajs = self._inner_rollout(sketch_traj.sketches[-1] if sketch_traj.sketches else None, proof_state)

            # 3. Compute rewards
            outer_reward = self._compute_outer_reward(
                sketch_traj.sketches[-1] if sketch_traj.sketches else None,
                tactic_trajs,
            )
            inner_rewards = [
                self._compute_inner_reward(tt.tactic_result)
                for tt in tactic_trajs
            ]

            # 4. Store trajectories in replay buffer
            self._store_sketch_trajectory(sketch_traj, outer_reward)
            for tt, ir in zip(tactic_trajs, inner_rewards):
                self._store_tactic_trajectory(tt, ir)

            # 5. Update sketch GFlowNet
            outer_loss = self._update_sketch_gfn(sketch_traj, outer_reward)

            # 6. Update tactic GFlowNet (over-sampled)
            inner_loss = 0.0
            for _ in range(self._outer_inner_ratio):
                if tactic_trajs:
                    inner_loss = self._update_tactic_gfn(tactic_trajs, inner_rewards)

            # 7. Compute metrics
            proof_success = float(
                any(
                    tt.tactic_result is not None
                    and tt.tactic_result.success
                    and tt.tactic_result.new_state is not None
                    and tt.tactic_result.new_state.is_terminal
                    for tt in tactic_trajs
                )
            )
            mean_sketch_depth = float(
                len(sketch_traj.sketches[-1].nodes)
                if sketch_traj.sketches
                else 0
            )
            buf_stats = self.replay_buffer.get_stats()

            metrics = {
                "step": self._step,
                "outer_loss": outer_loss,
                "inner_loss": inner_loss,
                "outer_reward": outer_reward,
                "inner_reward": float(
                    sum(inner_rewards) / max(len(inner_rewards), 1)
                ),
                "proof_success_rate": proof_success,
                "mean_sketch_depth": mean_sketch_depth,
                "buffer_size": buf_stats["buffer_size"],
            }

            if self._use_wandb:
                self._wandb.log(metrics, step=self._step)

            return metrics

    def train(
        self,
        num_steps: int,
        theorems: list[str],
        eval_fn: Any = None,
    ) -> None:
        """
        Main training loop.

        Args:
            num_steps:  Total number of training steps.
            theorems:   Pool of theorems to sample from.
            eval_fn:    Optional callable(trainer, step) → dict for evaluation.
        """
        try:
            from tqdm import tqdm  # type: ignore[import]
            pbar = tqdm(range(num_steps), desc="HieroFlow training")
        except ImportError:
            pbar = range(num_steps)  # type: ignore[assignment]

        eval_every: int = self.config.get("eval_every", 500)
        save_every: int = self.config.get("save_every", 1000)

        for global_step in pbar:
            theorem = theorems[global_step % len(theorems)]
            try:
                metrics = self.train_step(theorem)
            except Exception as exc:  # noqa: BLE001
                logger.error("train_step failed for %s: %s", theorem, exc)
                continue

            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(  # type: ignore[union-attr]
                    outer_loss=f"{metrics['outer_loss']:.3f}",
                    inner_loss=f"{metrics['inner_loss']:.3f}",
                )

            if global_step % eval_every == 0 and eval_fn is not None:
                eval_fn(self, global_step)

            if global_step % save_every == 0:
                self._save_checkpoint(global_step)

    # ------------------------------------------------------------------
    # Rollout methods
    # ------------------------------------------------------------------

    def _outer_rollout(
        self, theorem: str, proof_state: LeanProofState
    ) -> SketchTrajectory:
        """
        Run SketchFlow to generate a proof sketch.

        Args:
            theorem:     Theorem name (used to build the root obligation).
            proof_state: Initial Lean proof state.

        Returns:
            A ``SketchTrajectory`` with the visited sketches and actions.
        """
        max_sketch_depth: int = self.config.get("max_sketch_depth", 8)
        temperature: float = self.config.get("sketch_temperature", 1.0)

        # Build initial sketch from the proof state's first goal
        root_goal = proof_state.goals[0] if proof_state.goals else theorem
        root_obligation = self._obligation_extractor.extract(root_goal)
        sketch = ProofSketch(root_obligation)

        traj = SketchTrajectory()
        traj.sketches.append(sketch.clone())

        for _ in range(max_sketch_depth):
            if not sketch.get_open_nodes():
                break
            if sketch.is_complete() or sketch.is_failed():
                break

            action = self.sketch_gfn.sample_action(sketch, temperature=temperature)
            sketch.assign_strategy(action.node_id, action.strategy)
            traj.actions.append(action)
            traj.sketches.append(sketch.clone())

        return traj

    def _inner_rollout(
        self,
        sketch: ProofSketch | None,
        proof_state: LeanProofState,
    ) -> list[TacticTrajectory]:
        """
        Run TacticFlow for each ASSIGNED node in *sketch*.

        Args:
            sketch:      The proof sketch from the outer rollout.
            proof_state: Current Lean proof state.

        Returns:
            List of ``TacticTrajectory`` objects, one per ASSIGNED node.
        """
        if sketch is None:
            return []

        temperature: float = self.config.get("tactic_temperature", 1.0)
        results: list[TacticTrajectory] = []

        assigned_nodes = [
            n for n in sketch.nodes.values()
            if n.state == SketchNodeState.ASSIGNED
        ]

        for node in assigned_nodes:
            obligation = node.obligation
            tactic_str, log_pf = self.tactic_gfn.sample_tactic(
                proof_state=proof_state,
                obligation=obligation,
                temperature=temperature,
            )
            T = max(len(tactic_str.split()), 1)
            log_pb = -math.log(T) * T

            tactic_result = self.lean_env.step(proof_state, tactic_str)

            traj = TacticTrajectory(
                node_id=node.node_id,
                tactic_tokens=[],  # Not tracked at string level
                log_pf=log_pf,
                log_pb=log_pb,
                tactic_result=tactic_result,
                log_reward=-100.0,  # Filled below
            )
            results.append(traj)

            # Update proof state for subsequent tactics
            if tactic_result.new_state is not None:
                proof_state = tactic_result.new_state

        return results

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_outer_reward(
        self,
        sketch: ProofSketch | None,
        tactic_trajs: list[TacticTrajectory],
    ) -> float:
        """
        Compute outer GFlowNet reward in linear space.

        R_outer = fraction of sketch nodes where TacticFlow found a valid tactic.

        Args:
            sketch:       The final proof sketch.
            tactic_trajs: Inner rollout trajectories.

        Returns:
            Float reward in [0, 1].
        """
        if not tactic_trajs:
            return 0.0

        successes = {
            tt.node_id: (
                tt.tactic_result is not None and tt.tactic_result.success
            )
            for tt in tactic_trajs
        }
        log_r = compute_sketch_log_reward(
            sketch or ProofSketch(
                self._obligation_extractor.extract("⊢ True")
            ),
            successes,
        )
        # Convert from log-space to linear for the TB loss interface
        return float(math.exp(max(log_r.item(), -100.0)))

    def _compute_inner_reward(
        self, tactic_result: TacticResult | None
    ) -> float:
        """
        Compute inner GFlowNet reward in linear space.

        Reward schedule:
        - 1.0  if tactic closes the proof
        - 0.5  if tactic reduces open goals
        - 0.0  if error / None
        - -0.1 if timeout

        Args:
            tactic_result: The ``TacticResult`` from ``LeanEnv.step()``.

        Returns:
            Float reward (may be slightly negative for timeout penalty).
        """
        if tactic_result is None:
            return 0.0

        if not tactic_result.success or tactic_result.new_state is None:
            return 0.0

        if tactic_result.new_state.is_terminal:
            return 1.0

        if tactic_result.goals_closed > 0:
            return 0.5

        return 0.0

    # ------------------------------------------------------------------
    # GFlowNet update methods
    # ------------------------------------------------------------------

    def _update_sketch_gfn(
        self,
        trajectory: SketchTrajectory,
        reward: float,
    ) -> float:
        """
        Compute TB loss for SketchFlow and take a gradient step.

        Args:
            trajectory: The completed sketch trajectory.
            reward:     Linear-space outer reward.

        Returns:
            Scalar loss value (float).
        """
        if not trajectory.actions:
            return 0.0

        self._sketch_opt.zero_grad()
        loss = self.sketch_gfn.trajectory_balance_loss(trajectory, reward)
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.sketch_gfn.parameters(), self._grad_clip
        )
        self._sketch_opt.step()
        return float(loss.item())

    def _update_tactic_gfn(
        self,
        trajectories: list[TacticTrajectory],
        rewards: list[float],
    ) -> float:
        """
        Compute TB loss for TacticFlow and take a gradient step.

        Averages the loss over all tactic trajectories in the batch.

        Args:
            trajectories: List of tactic trajectories.
            rewards:      Corresponding linear-space inner rewards.

        Returns:
            Mean scalar loss value (float).
        """
        if not trajectories:
            return 0.0

        self._tactic_opt.zero_grad()
        losses: list[torch.Tensor] = []
        for tt, reward in zip(trajectories, rewards):
            loss = self.tactic_gfn.trajectory_balance_loss(
                tactic_tokens=tt.tactic_tokens,
                log_pf=tt.log_pf,
                lean_reward=reward,
            )
            losses.append(loss)

        mean_loss = torch.stack(losses).mean()
        mean_loss.backward()
        nn.utils.clip_grad_norm_(
            self.tactic_gfn.parameters(), self._grad_clip
        )
        self._tactic_opt.step()
        return float(mean_loss.item())

    # ------------------------------------------------------------------
    # Storage and checkpointing
    # ------------------------------------------------------------------

    def _store_sketch_trajectory(
        self, traj: SketchTrajectory, reward: float
    ) -> None:
        """Store sketch trajectory in the replay buffer."""
        log_r = max(math.log(max(reward, 1e-45)), -100.0)
        t = Trajectory(
            trajectory_id=str(uuid.uuid4()),
            level="sketch",
            states=traj.sketches,
            actions=traj.actions,
            log_pf=sum(a.log_prob for a in traj.actions),
            log_pb=0.0,
            log_reward=log_r,
            timestamp=time.time(),
            priority=1.0,
        )
        self.replay_buffer.add(t)

    def _store_tactic_trajectory(
        self, tt: TacticTrajectory, reward: float
    ) -> None:
        """Store tactic trajectory in the replay buffer."""
        log_r = max(math.log(max(reward, 1e-45)), -100.0)
        t = Trajectory(
            trajectory_id=str(uuid.uuid4()),
            level="tactic",
            states=[],
            actions=tt.tactic_tokens,
            log_pf=tt.log_pf,
            log_pb=tt.log_pb,
            log_reward=log_r,
            timestamp=time.time(),
            priority=1.0,
        )
        self.replay_buffer.add(t)

    def _save_checkpoint(self, step: int) -> None:
        """Save model checkpoints to disk."""
        save_dir = self.config.get("checkpoint_dir", "checkpoints")
        import os
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            self.sketch_gfn.state_dict(),
            os.path.join(save_dir, f"sketch_gfn_step{step}.pt"),
        )
        torch.save(
            self.tactic_gfn.state_dict(),
            os.path.join(save_dir, f"tactic_gfn_step{step}.pt"),
        )
        logger.info("Saved checkpoint at step %d to %s", step, save_dir)
