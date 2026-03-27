# Hieroflow

HieroFlow is a hierarchical GFlowNet approach for Lean 4 theorem proving.

## Research results and visualization pipeline

This repository includes a publication-oriented results pipeline under
`results/` for generating NeurIPS-ready
figures, tables, and significance reports.

### What is included

- Global plotting style (`results/plot_style.py`) with:
  - 300 DPI defaults
  - serif fonts and publication-safe sizing
  - color-blind safe method palette and hatch patterns
  - convenience helpers for figure saving and significance brackets
- Data schema and loading (`results/data/schema.py`, `results/data/loader.py`)
  - dataclasses for proof attempts and experiment runs
  - JSONL run loading to tidy pandas DataFrames
  - realistic synthetic data generation for pre-experiment iteration
- Figure scripts (`results/figures/fig1_main_result.py` … `fig6_appendix.py`)
  - standalone runnable modules
  - each script uses synthetic data if run directly
  - outputs PDF + PNG to `results/figures/output/`
- Table generators (`results/tables/table1_main.py`, `table2_ablation.py`, `table3_diversity.py`)
  - LaTeX strings for direct insertion into paper drafts
- Statistical testing (`results/stats/significance.py`, `results/stats/effect_size.py`)
  - bootstrap confidence intervals
  - paired Wilcoxon signed-rank tests
  - Cohen’s d effect sizes and relative-improvement helpers
- Master driver (`results/generate_all.py`)
  - generates all figures/tables/statistics in one run
  - supports real logs via `--log-dir`

### Quick start

Run all artifacts using synthetic data:

```bash
PYTHONPATH=. python -m results.generate_all
```

Or with real experiment logs:

```bash
PYTHONPATH=. python -m results.generate_all \
  --log-dir /path/to/jsonl/logs \
  --output-dir results/figures/output
```

### Current generated results (synthetic validation)

Running `results.generate_all` currently produces:

- `fig1_main_result.pdf/png`
- `fig2_diversity.pdf/png`
- `fig3_sketch_ablation.pdf/png`
- `fig4_efficiency.pdf/png`
- `fig5_qualitative.pdf/png`
- `fig6_appendix.pdf/png`

with table outputs printed to stdout and a significance summary dataframe report.
