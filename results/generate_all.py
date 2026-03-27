from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from results.data.schema import make_synthetic_data
from results.data.loader import load_dataframe
from results.plot_style import set_output_dir
from results.figures.fig1_main_result import make_figure as make_fig1
from results.figures.fig2_diversity import make_figure as make_fig2
from results.figures.fig3_sketch_ablation import make_figure as make_fig3
from results.figures.fig4_efficiency import make_figure as make_fig4
from results.figures.fig5_qualitative import make_figure as make_fig5
from results.figures.fig6_appendix import make_figure as make_fig6
from results.figures.fig7_training_curves import make_figure as make_fig7
from results.figures.fig8_per_benchmark import make_figure as make_fig8
from results.figures.fig9_diversity_scatter import make_figure as make_fig9
from results.tables.table1_main import generate_table1
from results.tables.table2_ablation import generate_table2
from results.tables.table3_diversity import generate_table3
from results.stats.significance import run_all_significance_tests


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all HieroFlow paper figures/tables")
    parser.add_argument("--log-dir", type=str, default=None, help="Directory containing JSONL run logs")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/figures/output",
        help="Directory for figure outputs",
    )
    args = parser.parse_args()

    t0 = time.time()
    out_dir = Path(args.output_dir)
    set_output_dir(out_dir)

    if args.log_dir:
        df = load_dataframe(args.log_dir)
    else:
        print("WARNING: Using synthetic data. Pass --log-dir to use real experiment logs.")
        df = make_synthetic_data()

    saved_files = []
    for fig_fn in [make_fig1, make_fig2, make_fig3, make_fig4, make_fig5, make_fig6, make_fig7, make_fig8, make_fig9]:
        saved_files.extend(fig_fn(df, output_dir=out_dir))

    generate_table1(df)
    generate_table2(df)
    generate_table3(df)
    run_all_significance_tests(df)

    elapsed = time.time() - t0
    print("\nSaved outputs:")
    for path in saved_files:
        print(f" - {path}")
    print(f"Total time elapsed: {elapsed:.2f}s")

    checklist = [
        "fig1_main_result.pdf",
        "fig2_diversity.pdf",
        "fig3_sketch_ablation.pdf",
        "fig4_efficiency.pdf",
        "fig5_qualitative.pdf",
        "fig6_appendix.pdf",
        "fig7_training_curves.pdf",
        "fig8_per_benchmark.pdf",
        "fig9_diversity_scatter.pdf",
    ]
    print()
    for item in checklist:
        mark = "✓" if (out_dir / item).exists() else "✗"
        print(f"[{mark}] {item}")
    print("All figures ready for NeurIPS submission.")


if __name__ == "__main__":
    main()
