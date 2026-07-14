"""Shared data loading and feature preparation for the swap-hedging models.

Single source of truth for how the Russian-bank panel data is cleaned, so the
exploratory comparison script, the training CLI, and batch scoring all see
exactly the same features.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Mevliutov_Data.csv"

TARGET = "Hedge_indicator"
_JUNK_COLUMNS = ["Bank", "Reg", "Unnamed: 15"]


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature cleaning shared by training and inference.

    - drops identifier/junk columns
    - converts percent-formatted strings ("32.4%") to floats
    - one-hot encodes remaining categoricals
    - KNN-imputes missing numeric values
    """
    X = df.drop(columns=[c for c in _JUNK_COLUMNS + [TARGET] if c in df.columns]).copy()

    for c in X.select_dtypes(include="object").columns:
        s = X[c].dropna().astype(str)
        if s.str.contains("%").any():
            X[c] = (X[c].astype(str).str.replace("%", "", regex=False)
                    .replace({"nan": np.nan, "None": np.nan}).astype(float))

    X = pd.get_dummies(X, columns=list(X.select_dtypes(include="object").columns),
                       drop_first=True)
    X = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(X),
                     columns=X.columns, index=X.index)
    return X


def load_and_prepare(path: Path = DATA_PATH):
    """Load the panel dataset and return (features, target)."""
    df = pd.read_csv(path)
    y = df[TARGET].astype(int)
    X = clean_features(df)
    return X, y
