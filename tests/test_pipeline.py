"""Tests for the swap-hedging data prep, training, and scoring pipeline.

Uses a reduced XGBoost (20 trees) so the whole suite runs in seconds while
still asserting real signal (AUC well above chance) on the held-out split.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_prep import DATA_PATH, TARGET, clean_features, load_and_prepare  # noqa: E402
from predict import score  # noqa: E402
from train import tune_threshold  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return load_and_prepare()


# ------------------------------------------------------------- data prep ---

def test_dataset_exists():
    assert DATA_PATH.exists(), "panel dataset missing from data/"


def test_features_are_numeric_and_complete(data):
    X, y = data
    assert X.isna().sum().sum() == 0, "imputation left missing values"
    assert all(np.issubdtype(dt, np.number) for dt in X.dtypes), \
        "non-numeric feature survived preparation"
    assert len(X) == len(y) > 1000


def test_target_is_binary_and_imbalanced(data):
    _, y = data
    assert set(y.unique()) == {0, 1}
    positive_rate = y.mean()
    assert 0.02 < positive_rate < 0.30, \
        f"unexpected positive rate {positive_rate:.3f} - data drift?"


def test_junk_columns_dropped(data):
    X, _ = data
    for junk in ("Bank", "Reg", "Unnamed: 15", TARGET):
        assert junk not in X.columns


# ------------------------------------------------------ training behavior ---

@pytest.fixture(scope="module")
def quick_model(data):
    X, y = data
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=47)
    scaler = StandardScaler().fit(X_tr)
    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    model = xgb.XGBClassifier(n_estimators=20, max_depth=4,
                              scale_pos_weight=pos_weight,
                              eval_metric="logloss", random_state=47)
    model.fit(scaler.transform(X_tr), y_tr)
    return model, scaler, X, X_te, y_te


def test_model_beats_chance_comfortably(quick_model):
    model, scaler, _, X_te, y_te = quick_model
    auc = roc_auc_score(y_te, model.predict_proba(scaler.transform(X_te))[:, 1])
    assert auc > 0.85, f"AUC {auc:.3f} - pipeline degraded"


def test_tune_threshold_prefers_high_f1():
    y = np.array([0] * 90 + [1] * 10)
    proba = np.where(y == 1, 0.4, 0.1).astype(float)  # perfect separation at 0.4
    t = tune_threshold(y, proba)
    assert 0.1 < t <= 0.4


# --------------------------------------------------------------- scoring ---

def test_score_roundtrip(quick_model, data):
    model, scaler, X, _, _ = quick_model
    _, y = data
    raw = pd.read_csv(DATA_PATH)
    bundle = {"model": model, "scaler": scaler,
              "columns": list(X.columns), "threshold": 0.5}
    scored = score(raw.head(50), bundle)
    assert {"hedge_probability", "hedge_prediction"} <= set(scored.columns)
    assert scored["hedge_probability"].between(0, 1).all()
    assert set(scored["hedge_prediction"].unique()) <= {0, 1}
    assert len(scored) == 50


def test_score_handles_missing_onehot_columns(quick_model):
    """New data missing a categorical level must still score (reindex path)."""
    model, scaler, X, _, _ = quick_model
    raw = pd.read_csv(DATA_PATH).head(10)
    if "Status" in raw.columns:
        raw = raw.drop(columns=["Status"])
    bundle = {"model": model, "scaler": scaler,
              "columns": list(X.columns), "threshold": 0.5}
    scored = score(raw, bundle)
    assert len(scored) == 10
