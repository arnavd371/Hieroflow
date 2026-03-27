"""
HieroFlow: Hierarchical GFlowNet for Lean 4 Theorem Proving.

Two-level GFlowNet architecture:
- Outer GFlowNet (SketchFlow): samples proof sketches as abstract DAGs of proof obligations
- Inner GFlowNet (TacticFlow): fills in concrete Lean 4 tactics conditioned on sketch nodes
"""
