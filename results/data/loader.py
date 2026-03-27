from __future__ import annotations

import pandas as pd

from .schema import load_runs, runs_to_dataframe


def load_dataframe(log_dir: str) -> pd.DataFrame:
    return runs_to_dataframe(load_runs(log_dir))
