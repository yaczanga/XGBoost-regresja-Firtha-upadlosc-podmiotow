from pathlib import Path

import pandas as pd


def data_load(path: str | Path) -> pd.DataFrame:
    df = pd.read_stata(path)
    return df
