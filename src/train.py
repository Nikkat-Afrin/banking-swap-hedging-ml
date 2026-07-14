"""Train and persist the production swap-hedging model.

Pipeline:
  1. Stratified k-fold cross-validation of the XGBoost classifier (the winner
     of the model comparison) to get an honest generalization estimate.
  2. Decision-threshold tuning on out-of-fold predictions (max F1) — with an
     ~8% positive class, the default 0.5 cutoff is rarely optimal.
  3. Final fit on the full training split, evaluation on a held-out test set.
  4. Persist a self-contained inference bundle (model + scaler + feature
     columns + tuned threshold) and write reports/metrics.json.

Usage:
    python src/train.py                # defaults
    python src/train.py --folds 10 --test-size 0.25
"""

import argparse
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from data_prep import load_and_prepare

RNG = 47
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORTS = ROOT / "reports"


def build_model(pos_weight: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=pos_weight, eval_metric="logloss",
        random_state=RNG,
    )


def tune_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Pick the probability cutoff that maximizes F1 on out-of-fold preds."""
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, (proba >= t).astype(int), zero_division=0)
              for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    X, y = load_and_prepare()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=RNG)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = pd.DataFrame(scaler.transform(X_tr), columns=X.columns, index=X_tr.index)
    X_te_s = pd.DataFrame(scaler.transform(X_te), columns=X.columns, index=X_te.index)

    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

    # ---- 1-2. cross-validation + out-of-fold threshold tuning -------------
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=RNG)
    oof_proba = np.zeros(len(y_tr), dtype=float)
    fold_aucs = []
    for fold, (idx_tr, idx_va) in enumerate(skf.split(X_tr_s, y_tr), start=1):
        model = build_model(pos_weight)
        model.fit(X_tr_s.iloc[idx_tr], y_tr.iloc[idx_tr])
        proba = model.predict_proba(X_tr_s.iloc[idx_va])[:, 1]
        oof_proba[idx_va] = proba
        auc = roc_auc_score(y_tr.iloc[idx_va], proba)
        fold_aucs.append(auc)
        print(f"fold {fold}/{args.folds}: AUC={auc:.4f}")

    threshold = tune_threshold(y_tr.to_numpy(), oof_proba)
    print(f"CV AUC: {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f} | "
          f"tuned threshold={threshold:.2f}")

    # ---- 3. final fit + held-out evaluation --------------------------------
    final_model = build_model(pos_weight)
    final_model.fit(X_tr_s, y_tr)
    test_proba = final_model.predict_proba(X_te_s)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)

    metrics = {
        "trained_on": str(date.today()),
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "class_balance": {str(k): int(v) for k, v in y.value_counts().items()},
        "cv": {
            "folds": args.folds,
            "roc_auc_mean": round(float(np.mean(fold_aucs)), 4),
            "roc_auc_std": round(float(np.std(fold_aucs)), 4),
        },
        "tuned_threshold": round(threshold, 2),
        "test": {
            "roc_auc": round(float(roc_auc_score(y_te, test_proba)), 4),
            "accuracy": round(float(accuracy_score(y_te, test_pred)), 4),
            "precision": round(float(precision_score(y_te, test_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_te, test_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_te, test_pred, zero_division=0)), 4),
        },
    }
    print(json.dumps(metrics["test"], indent=2))

    # ---- 4. persist ---------------------------------------------------------
    MODEL_DIR.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    bundle = {
        "model": final_model,
        "scaler": scaler,
        "columns": list(X.columns),
        "threshold": threshold,
        "metrics": metrics,
    }
    joblib.dump(bundle, MODEL_DIR / "swap_hedging_xgb.joblib")
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved {MODEL_DIR / 'swap_hedging_xgb.joblib'} and {REPORTS / 'metrics.json'}")


if __name__ == "__main__":
    main()
