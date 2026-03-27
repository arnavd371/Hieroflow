"""
ProofObligation and ObligationExtractor for Lean 4 goals.

A ``ProofObligation`` is an abstracted representation of a Lean 4 goal that
strips away concrete variable names, retaining only the logical structure
type.  This forms the *interface* between SketchFlow (outer GFlowNet) and
TacticFlow (inner GFlowNet): SketchFlow produces ProofObligations; TacticFlow
consumes them.

Design note: the abstraction must be stable across different variable names
but still carry enough structure for the outer GFlowNet to reason about
*strategy* (e.g. "this is an induction goal") without knowing the specific
theorem.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Final


# ---------------------------------------------------------------------------
# ObligationType
# ---------------------------------------------------------------------------

class ObligationType(Enum):
    """
    Abstract classification of a Lean 4 proof obligation.

    Each value corresponds to a high-level proof strategy.  The outer
    GFlowNet (SketchFlow) chooses strategies at this granularity; it never
    sees concrete variable names.
    """

    INDUCTION = auto()
    CONTRADICTION = auto()
    REWRITE = auto()
    CASE_SPLIT = auto()
    DIRECT = auto()
    EXISTENTIAL = auto()
    UNIVERSAL = auto()
    UNKNOWN = auto()


# ---------------------------------------------------------------------------
# ProofObligation
# ---------------------------------------------------------------------------

@dataclass
class ProofObligation:
    """
    Abstract representation of a single Lean 4 proof goal.

    Variable names in the original goal are replaced with typed placeholders
    (e.g. ``?n:Nat``, ``?h:Prop``) so that the outer GFlowNet can reason
    about *structure* rather than *identity*.

    This dataclass is the sole object passed across the SketchFlow ↔
    TacticFlow boundary — nothing else should leak between the two levels.
    """

    __slots__ = (
        "obligation_type",
        "abstracted_goal",
        "hypothesis_types",
        "target_type",
        "estimated_depth",
    )

    obligation_type: ObligationType
    """High-level strategy classification for this goal."""

    abstracted_goal: str
    """Goal string with variable names replaced by typed placeholders."""

    hypothesis_types: list[str]
    """Types of hypotheses in the local context (names stripped)."""

    target_type: str
    """The Lean type being proved (right-hand side of the goal turnstile)."""

    estimated_depth: int
    """Heuristic estimate: how many tactic steps this goal likely needs."""


# ---------------------------------------------------------------------------
# _TYPE_CONSTRUCTORS – names kept verbatim during abstraction
# ---------------------------------------------------------------------------

_TYPE_CONSTRUCTORS: Final[frozenset[str]] = frozenset(
    {
        "Nat", "Int", "Float", "Bool", "String", "Char",
        "List", "Array", "Option", "Result", "Except",
        "Prop", "Type", "Sort", "True", "False",
        "And", "Or", "Not", "Iff", "Eq", "Ne", "HEq",
        "Fin", "UInt8", "UInt16", "UInt32", "UInt64",
        "Int8", "Int16", "Int32", "Int64",
        "Prod", "Sigma", "Subtype", "Sum",
        "Set", "Multiset", "Finset",
        "Nat.succ", "Nat.zero",
    }
)

# Lean 4 keywords / tactics that should not be treated as variable names
_LEAN_KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "fun", "let", "have", "show", "from", "match", "with",
        "if", "then", "else", "do", "return", "pure",
        "theorem", "lemma", "def", "abbrev", "instance",
        "induction", "cases", "rcases", "obtain", "intro",
        "apply", "exact", "rw", "simp", "ring", "omega",
        "norm_num", "decide", "trivial", "assumption",
        "contradiction", "absurd", "exfalso",
        "constructor", "left", "right", "exists", "use",
        "specialize", "have", "suffices", "calc",
    }
)

# Lowercase identifiers that look like bound variables
_BOUND_VAR_RE: Final[re.Pattern[str]] = re.compile(r"\b([a-z][a-zA-Z0-9_']*)\b")

# Hypothesis line: "h : <type>" or "h₁ : <type>"
_HYP_RE: Final[re.Pattern[str]] = re.compile(
    r"^([a-zA-Z_][a-zA-Z0-9_'₀-₉]*)\s*:\s*(.+)$"
)


# ---------------------------------------------------------------------------
# ObligationExtractor
# ---------------------------------------------------------------------------

class ObligationExtractor:
    """
    Heuristic extractor that converts a raw Lean 4 goal string into a
    ``ProofObligation``.

    The classifier uses regex patterns and keyword matching rather than a
    full Lean parser.  It is intentionally approximate: the goal is to give
    the outer GFlowNet a useful structural signal, not a formal proof-type
    annotation.

    Usage::

        extractor = ObligationExtractor()
        obligation = extractor.extract("n m : Nat ⊢ n + m = m + n")
    """

    def extract(self, goal_string: str) -> ProofObligation:
        """
        Parse *goal_string* and return the corresponding ``ProofObligation``.

        The goal string may contain hypothesis lines separated by newlines
        followed by a line starting with ``⊢`` (or ``|-``).

        Args:
            goal_string: Raw Lean 4 goal string as returned by LeanDojo.

        Returns:
            A fully populated ``ProofObligation``.
        """
        lines = [ln.strip() for ln in goal_string.strip().splitlines() if ln.strip()]

        # Split into hypotheses and target
        hyp_lines: list[str] = []
        target_line: str = ""
        for line in lines:
            if line.startswith("⊢") or line.startswith("|-"):
                target_line = line.lstrip("⊢").lstrip("|-").strip()
            else:
                hyp_lines.append(line)

        if not target_line and lines:
            target_line = lines[-1]

        hypothesis_types = self._extract_hypothesis_types(hyp_lines)
        target_type = target_line
        obligation_type = self._classify(hyp_lines, target_line)
        abstracted_goal = self._abstract_goal(goal_string)
        estimated_depth = self._estimate_depth(obligation_type, hyp_lines, target_line)

        return ProofObligation(
            obligation_type=obligation_type,
            abstracted_goal=abstracted_goal,
            hypothesis_types=hypothesis_types,
            target_type=target_type,
            estimated_depth=estimated_depth,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_hypothesis_types(self, hyp_lines: list[str]) -> list[str]:
        """
        Return the *types* of hypotheses, stripping names.

        E.g. "h : n + m = m + n" → "n + m = m + n".
        """
        types: list[str] = []
        for line in hyp_lines:
            m = _HYP_RE.match(line)
            if m:
                types.append(m.group(2).strip())
            else:
                # Could be a variable declaration "n m : Nat"
                colon_idx = line.find(":")
                if colon_idx != -1:
                    types.append(line[colon_idx + 1:].strip())
        return types

    def _classify(self, hyp_lines: list[str], target: str) -> ObligationType:
        """
        Classify the obligation type using keyword/pattern heuristics.

        Priority order (highest wins):
        1. CONTRADICTION  — False / absurd / contradiction in hypotheses or target
        2. INDUCTION      — induction/Nat.rec patterns
        3. CASE_SPLIT     — Or / cases / match patterns
        4. EXISTENTIAL    — ∃ / Exists
        5. UNIVERSAL      — ∀ / forall
        6. REWRITE        — Eq (=) in target with simp/ring suitability
        7. DIRECT         — simple closed-form targets
        8. UNKNOWN        — fallback
        """
        combined_hyps = " ".join(hyp_lines).lower()
        target_lower = target.lower()
        target_raw = target

        # 1. Contradiction
        if (
            "false" in target_lower
            or "absurd" in combined_hyps
            or "contradiction" in combined_hyps
            or re.search(r"\bfalse\b", combined_hyps)
        ):
            return ObligationType.CONTRADICTION

        # 2. Induction
        if (
            re.search(r"\binduction\b", combined_hyps + " " + target_lower)
            or "nat.rec" in target_lower
            or re.search(r"\bn\.succ\b|\bn\.zero\b", target_lower)
        ):
            return ObligationType.INDUCTION

        # 3. Case split (Or / if-then-else / cases)
        if (
            re.search(r"\bor\b|\bor\.inl\b|\bor\.inr\b", target_lower)
            or re.search(r"\bcases\b|\bmatch\b", combined_hyps + " " + target_lower)
            or "∨" in target_raw
        ):
            return ObligationType.CASE_SPLIT

        # 4. Existential
        if "∃" in target_raw or re.search(r"\bexists\b|\bexist\b", target_lower):
            return ObligationType.EXISTENTIAL

        # 5. Universal
        if "∀" in target_raw or re.search(r"\bforall\b", target_lower):
            return ObligationType.UNIVERSAL

        # 6. Rewrite (equality target)
        if (
            "=" in target_raw
            and "≠" not in target_raw
            and not re.search(r"\biff\b|↔", target_lower + target_raw)
        ):
            return ObligationType.REWRITE

        # 7. Direct (propositions that look provable in one step)
        if re.search(r"\btrue\b|\btrivial\b|\brefl\b", target_lower):
            return ObligationType.DIRECT

        return ObligationType.UNKNOWN

    def _abstract_goal(self, goal_string: str) -> str:
        """
        Replace bound variable names with typed placeholders.

        Type constructors (Nat, List, Prop, etc.) are kept verbatim.
        Lean keywords and tactic names are also kept verbatim.
        All other lowercase identifiers are replaced with ``?name:Type``
        if a type annotation is available, or ``?var`` otherwise.
        """
        # Build a mapping from variable name → inferred type
        var_type: dict[str, str] = {}
        lines = goal_string.strip().splitlines()
        for line in lines:
            line = line.strip()
            # "x y : T" or "h : P"
            colon_idx = line.find(":")
            if colon_idx != -1 and not line.startswith("⊢") and not line.startswith("|-"):
                names_part = line[:colon_idx].strip()
                type_part = line[colon_idx + 1:].strip()
                for name in names_part.split():
                    name = name.strip()
                    if name and re.match(r"^[a-zA-Z_][a-zA-Z0-9_'₀-₉]*$", name):
                        var_type[name] = type_part

        def replace_var(match: re.Match[str]) -> str:
            name = match.group(1)
            if name in _TYPE_CONSTRUCTORS or name in _LEAN_KEYWORDS:
                return name
            if name[0].isupper():
                return name  # Keep type-level identifiers
            if name in var_type:
                return f"?{name}:{var_type[name]}"
            return f"?{name}"

        return _BOUND_VAR_RE.sub(replace_var, goal_string)

    def _estimate_depth(
        self,
        obligation_type: ObligationType,
        hyp_lines: list[str],
        target: str,
    ) -> int:
        """
        Heuristic estimate of how many tactics this goal needs.

        These rough numbers guide the outer GFlowNet's sketch depth budget.
        They are *not* used for reward computation.
        """
        base: dict[ObligationType, int] = {
            ObligationType.DIRECT: 1,
            ObligationType.REWRITE: 2,
            ObligationType.UNIVERSAL: 2,
            ObligationType.EXISTENTIAL: 2,
            ObligationType.CASE_SPLIT: 3,
            ObligationType.CONTRADICTION: 2,
            ObligationType.INDUCTION: 4,
            ObligationType.UNKNOWN: 3,
        }
        depth = base[obligation_type]

        # Bonus for large number of hypotheses
        depth += min(len(hyp_lines) // 3, 2)

        # Bonus for nested quantifiers
        depth += target.count("∀") + target.count("∃")

        # Bonus for compound targets
        depth += target.count("∧") + target.count("∨")

        return depth


# ---------------------------------------------------------------------------
# Unit tests (run with: python -m hieroflow.environment.obligation)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    extractor = ObligationExtractor()

    EXAMPLES: list[tuple[str, str]] = [
        (
            "Induction on natural numbers",
            "n : Nat\n⊢ n + 0 = n",
        ),
        (
            "Contradiction from False hypothesis",
            "h : False\n⊢ 1 = 2",
        ),
        (
            "Universal quantifier goal",
            "⊢ ∀ (n : Nat), n + 0 = n",
        ),
        (
            "Existential goal",
            "⊢ ∃ (n : Nat), n > 0",
        ),
        (
            "Rewrite / ring goal",
            "a b : Nat\n⊢ a + b = b + a",
        ),
    ]

    for desc, goal in EXAMPLES:
        obl = extractor.extract(goal)
        print(f"\n=== {desc} ===")
        print(f"  Goal:            {goal!r}")
        print(f"  Type:            {obl.obligation_type.name}")
        print(f"  Abstracted:      {obl.abstracted_goal!r}")
        print(f"  Hyp types:       {obl.hypothesis_types}")
        print(f"  Target:          {obl.target_type!r}")
        print(f"  Estimated depth: {obl.estimated_depth}")

    print("\nAll 5 examples processed successfully.")
