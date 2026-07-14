"""Batch scoring CLI for the persisted swap-hedging model.

Reads a CSV with the same raw schema as the training data (the target column
is optional and ignored), applies the identical feature preparation, and
writes hedge-adoption probabilities plus thresholded predictions.

Usage:
    python src/predict.py data/Mevliutov_Data.csv --out predictions.csv
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from data_prep import clean_features

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "models" / "swap_hedging_xgb.joblib"


def load_bundle(path: Path = DEFAULT_BUNDLE) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python src/train.py` first to train "
            "and persist the model.")
    return joblib.load(path)


def score(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Return the input frame with hedge_probability / hedge_prediction added."""
    X = clean_features(df)
    # Align one-hot columns with what the model was trained on
    X = X.reindex(columns=bundle["columns"], fill_value=0.0)
    X_scaled = bundle["scaler"].transform(X)

    proba = bundle["model"].predict_proba(X_scaled)[:, 1]
    out = df.copy()
    out["hedge_probability"] = proba.round(4)
    out["hedge_prediction"] = (proba >= bundle["threshold"]).astype(int)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--out", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()

    bundle = load_bundle(args.bundle)
    scored = score(pd.read_csv(args.input_csv), bundle)
    scored.to_csv(args.out, index=False)
    positives = int(scored["hedge_prediction"].sum())
    print(f"Scored {len(scored)} rows -> {args.out} "
          f"({positives} predicted hedgers, threshold={bundle['threshold']:.2f})")


if __name__ == "__main__":
    main()
