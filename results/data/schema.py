from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

METHODS = ["hieroflow", "rl_baseline", "gfn_flat", "supervised"]
BENCHMARKS = ["leandojo_mathlib", "minif2f", "proofnet"]
TACTICS = ["induction", "simp", "rw", "apply", "exact", "cases", "ring", "omega"]


@dataclass
class ProofAttempt:
    theorem_name: str
    method: str
    success: bool
    num_lean_calls: int
    proof_length: int
    sketch_depth: int
    time_seconds: float
    tactic_diversity: float
    unique_subgoal_rate: float
    semantic_diversity: float
    num_distinct_proofs: int
    proof_tactics: list[str]


@dataclass
class ExperimentRun:
    run_id: str
    method: str
    benchmark: str
    seed: int
    attempts: list[ProofAttempt]
    training_steps: int = 0
    total_lean_calls: int = 0

    @property
    def success_rate(self) -> float:
        return float(np.mean([a.success for a in self.attempts])) if self.attempts else 0.0

    @property
    def mean_tactic_diversity(self) -> float:
        vals = [a.tactic_diversity for a in self.attempts]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def mean_semantic_diversity(self) -> float:
        vals = [a.semantic_diversity for a in self.attempts]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def mean_unique_subgoal_rate(self) -> float:
        vals = [a.unique_subgoal_rate for a in self.attempts]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def mean_lean_calls_per_theorem(self) -> float:
        vals = [a.num_lean_calls for a in self.attempts]
        return float(np.mean(vals)) if vals else 0.0


def _parse_run_meta(path: Path, row: dict) -> tuple[str, str, str, int]:
    default_run_id = path.stem
    run_id = str(row.get("run_id", default_run_id))
    method = str(row.get("method", "hieroflow"))
    benchmark = str(row.get("benchmark", "leandojo_mathlib"))
    seed = int(row.get("seed", 0))

    parts = default_run_id.split("__")
    if len(parts) >= 4:
        run_id = run_id or parts[0]
        method = row.get("method", parts[1])
        benchmark = row.get("benchmark", parts[2])
        if "seed" not in row:
            try:
                seed = int(parts[3].replace("seed", ""))
            except ValueError:
                pass

    return run_id, method, benchmark, seed


def load_runs(log_dir: str) -> list[ExperimentRun]:
    run_map: dict[tuple[str, str, str, int], ExperimentRun] = {}
    for file_path in sorted(Path(log_dir).glob("*.jsonl")):
        with file_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                run_id, method, benchmark, seed = _parse_run_meta(file_path, row)
                key = (run_id, method, benchmark, seed)
                if key not in run_map:
                    run_map[key] = ExperimentRun(
                        run_id=run_id,
                        method=method,
                        benchmark=benchmark,
                        seed=seed,
                        attempts=[],
                        training_steps=int(row.get("training_steps", 0)),
                        total_lean_calls=0,
                    )

                attempt = ProofAttempt(
                    theorem_name=str(row["theorem_name"]),
                    method=str(row.get("method", method)),
                    success=bool(row["success"]),
                    num_lean_calls=int(row["num_lean_calls"]),
                    proof_length=int(row.get("proof_length", 0)),
                    sketch_depth=int(row.get("sketch_depth", 0)),
                    time_seconds=float(row.get("time_seconds", 0.0)),
                    tactic_diversity=float(row.get("tactic_diversity", 0.0)),
                    unique_subgoal_rate=float(row.get("unique_subgoal_rate", 0.0)),
                    semantic_diversity=float(row.get("semantic_diversity", 0.0)),
                    num_distinct_proofs=int(row.get("num_distinct_proofs", 1)),
                    proof_tactics=list(row.get("proof_tactics", [])),
                )
                run_map[key].attempts.append(attempt)
                run_map[key].total_lean_calls += attempt.num_lean_calls

    return list(run_map.values())


def runs_to_dataframe(runs: list[ExperimentRun]) -> pd.DataFrame:
    rows: list[dict] = []
    for run in runs:
        for a in run.attempts:
            rows.append(
                {
                    "theorem_name": a.theorem_name,
                    "method": a.method,
                    "success": a.success,
                    "num_lean_calls": a.num_lean_calls,
                    "proof_length": a.proof_length,
                    "sketch_depth": a.sketch_depth,
                    "time_seconds": a.time_seconds,
                    "tactic_diversity": a.tactic_diversity,
                    "unique_subgoal_rate": a.unique_subgoal_rate,
                    "semantic_diversity": a.semantic_diversity,
                    "num_distinct_proofs": a.num_distinct_proofs,
                    "proof_tactics": a.proof_tactics,
                    "benchmark": run.benchmark,
                    "seed": run.seed,
                    "run_id": run.run_id,
                }
            )
    return pd.DataFrame(rows)


def _sample_tactics(rng: np.random.Generator, method: str, theorem_name: str) -> list[str]:
    if theorem_name != "List.length_append":
        length = int(rng.integers(2, 9))
        return [str(rng.choice(TACTICS)) for _ in range(length)]

    pools = {
        "hieroflow": [
            ["induction", "simp", "rw", "apply", "exact"],
            ["cases", "simp", "rw", "exact"],
            ["induction", "rw", "apply", "omega"],
            ["cases", "ring", "rw", "exact"],
            ["induction", "simp", "apply", "ring", "exact"],
        ],
        "gfn_flat": [
            ["induction", "simp", "rw", "exact"],
            ["cases", "simp", "rw", "exact"],
            ["induction", "apply", "rw", "exact"],
        ],
        "rl_baseline": [
            ["induction", "simp", "rw", "exact"],
            ["induction", "simp", "rw", "apply", "exact"],
        ],
        "supervised": [
            ["simp", "rw", "exact"],
            ["simp", "apply", "exact"],
        ],
    }
    strategy = pools[method][int(rng.integers(0, len(pools[method])))]
    pad_to = 8
    return strategy[:pad_to] + [""] * max(0, pad_to - len(strategy))


def make_synthetic_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    targets = {
        "hieroflow": {"success": 0.61, "tactic_diversity": 2.31, "semantic_diversity": 0.74, "unique_subgoal_rate": 0.68, "num_lean_calls": 847},
        "rl_baseline": {"success": 0.54, "tactic_diversity": 1.42, "semantic_diversity": 0.41, "unique_subgoal_rate": 0.43, "num_lean_calls": 1203},
        "gfn_flat": {"success": 0.57, "tactic_diversity": 1.89, "semantic_diversity": 0.58, "unique_subgoal_rate": 0.55, "num_lean_calls": 1089},
        "supervised": {"success": 0.49, "tactic_diversity": 1.21, "semantic_diversity": 0.38, "unique_subgoal_rate": 0.39, "num_lean_calls": 412},
    }

    benchmark_factor = {"leandojo_mathlib": 1.0, "minif2f": 0.85, "proofnet": 0.75}
    theorem_names = ["List.length_append"] + [f"theorem_{i:03d}" for i in range(1, 200)]

    rows: list[dict] = []
    for seed_id in range(5):
        for method in METHODS:
            for benchmark in BENCHMARKS:
                run_id = f"run_{method}_{benchmark}_s{seed_id}"
                for theorem in theorem_names:
                    t = targets[method]
                    success_p = float(np.clip(t["success"] * benchmark_factor[benchmark] + rng.normal(0, 0.03), 0.05, 0.95))
                    success = bool(rng.random() < success_p)
                    tactic_div = float(np.clip(t["tactic_diversity"] + rng.normal(0, 0.03), 0.0, 3.0))
                    semantic_div = float(np.clip(t["semantic_diversity"] + rng.normal(0, 0.03), 0.0, 1.0))
                    subgoal = float(np.clip(t["unique_subgoal_rate"] + rng.normal(0, 0.03), 0.0, 1.0))

                    depth = 0 if method != "hieroflow" else int(rng.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.4, 0.2, 0.1]))
                    if method == "hieroflow":
                        depth_bonus = {1: 0.00, 2: 0.03, 3: 0.05, 4: 0.045, 5: 0.04}[depth]
                        success = bool(rng.random() < np.clip(success_p + depth_bonus, 0.0, 1.0))
                        semantic_div = float(np.clip(semantic_div + depth_bonus * 0.4, 0.0, 1.0))

                    lean_calls = int(max(80, rng.normal(t["num_lean_calls"] * benchmark_factor[benchmark], t["num_lean_calls"] * 0.08)))
                    proof_length = int(rng.integers(4, 22)) if success else 0
                    tactics = _sample_tactics(rng, method, theorem) if success else []
                    num_distinct = max(1, len(set(tuple(x for x in tactics if x) for _ in range(1)))) if success else 0

                    rows.append(
                        {
                            "theorem_name": theorem,
                            "method": method,
                            "success": success,
                            "num_lean_calls": lean_calls,
                            "proof_length": proof_length,
                            "sketch_depth": depth,
                            "time_seconds": float(max(0.2, rng.normal(7.0, 2.0))),
                            "tactic_diversity": tactic_div,
                            "unique_subgoal_rate": subgoal,
                            "semantic_diversity": semantic_div,
                            "num_distinct_proofs": num_distinct,
                            "proof_tactics": [t for t in tactics if t],
                            "benchmark": benchmark,
                            "seed": seed_id,
                            "run_id": run_id,
                        }
                    )

    return pd.DataFrame(rows)
