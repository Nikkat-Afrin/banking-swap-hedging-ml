"""
Banking Swap-Hedging Adoption — reproducible model-comparison pipeline.

Enhancement layer on top of the exploratory notebook. Trains a panel of
classifiers with explicit class-imbalance handling (~8% positive class),
then emits a side-by-side metrics table and three figures:
    reports/model_comparison.md
    reports/figures/roc_curves.png
    reports/figures/confusion_matrix_xgboost.png
    reports/figures/shap_summary_xgboost.png

Run from the repo root:   python src/model_comparison.py
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)
import xgboost as xgb

warnings.filterwarnings("ignore")
RNG = 47
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load_and_prepare():
    df = pd.read_csv(ROOT / "data" / "Mevliutov_Data.csv")
    # Drop identifier / junk columns if present
    df = df.drop(columns=[c for c in ["Bank", "Reg", "Unnamed: 15"] if c in df.columns])
    y = df["Hedge_indicator"].astype(int)
    X = df.drop(columns=["Hedge_indicator"])
    # Percent-formatted strings (e.g. "32.4%") -> float
    for c in X.select_dtypes(include="object").columns:
        s = X[c].dropna().astype(str)
        if s.str.contains("%").any():
            X[c] = (X[c].astype(str).str.replace("%", "", regex=False)
                    .replace({"nan": np.nan, "None": np.nan}).astype(float))
    # One-hot the remaining categoricals (Status)
    X = pd.get_dummies(X, columns=list(X.select_dtypes(include="object").columns),
                       drop_first=True)
    # KNN imputation for the few missing numeric values
    X = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(X),
                     columns=X.columns, index=X.index)
    return X, y


def evaluate(name, model, X_tr, X_te, y_tr, y_te, roc_store):
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    roc_store[name] = roc_curve(y_te, proba)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_te, pred),
        "Precision": precision_score(y_te, pred, zero_division=0),
        "Recall": recall_score(y_te, pred, zero_division=0),
        "F1": f1_score(y_te, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_te, proba),
    }, model


def main():
    X, y = load_and_prepare()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RNG)

    scaler = StandardScaler().fit(X_tr)
    X_tr = pd.DataFrame(scaler.transform(X_tr), columns=X.columns, index=X_tr.index)
    X_te = pd.DataFrame(scaler.transform(X_te), columns=X.columns, index=X_te.index)

    # Imbalance ratio for cost-sensitive learning
    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    print(f"Class balance (train): {y_tr.value_counts().to_dict()} | scale_pos_weight={pos_weight:.1f}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=RNG),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RNG, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                                     subsample=0.9, colsample_bytree=0.9,
                                     scale_pos_weight=pos_weight, eval_metric="logloss",
                                     random_state=RNG),
    }

    roc_store, rows, fitted = {}, [], {}
    for name, mdl in models.items():
        row, m = evaluate(name, mdl, X_tr, X_te, y_tr, y_te, roc_store)
        rows.append(row)
        fitted[name] = m

    # Soft-voting ensemble of the three strongest base learners
    ens = VotingClassifier(estimators=[("lr", models["Logistic Regression"]),
                                       ("rf", models["Random Forest"]),
                                       ("xgb", models["XGBoost"])], voting="soft")
    row, _ = evaluate("Soft-Voting Ensemble", ens, X_tr, X_te, y_tr, y_te, roc_store)
    rows.append(row)

    results = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    print("\n=== Test-set model comparison ===")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # --- markdown table for the README (no external deps) ---
    (ROOT / "reports").mkdir(exist_ok=True)
    cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    fmt = lambda v: v if isinstance(v, str) else f"{v:.3f}"
    lines = ["# Test-set model comparison", "",
             "| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in results.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    (ROOT / "reports" / "model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- ROC overlay ---
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, _) in roc_store.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, fitted.get(name, ens).predict_proba(X_te)[:,1]):.3f})"
                 if name in fitted else f"{name}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Swap-Hedging Adoption"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(FIG / "roc_curves.png", dpi=120); plt.close()

    # --- confusion matrix for XGBoost ---
    xgb_pred = fitted["XGBoost"].predict(X_te)
    ConfusionMatrixDisplay(confusion_matrix(y_te, xgb_pred),
                           display_labels=["No Hedge", "Hedge"]).plot(cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix — XGBoost (test)")
    plt.tight_layout(); plt.savefig(FIG / "confusion_matrix_xgboost.png", dpi=120); plt.close()

    # --- SHAP interpretation (tree explainer on XGBoost) ---
    try:
        import shap
        expl = shap.TreeExplainer(fitted["XGBoost"])
        sv = expl.shap_values(X_te)
        shap.summary_plot(sv, X_te, show=False, plot_size=(8, 6))
        plt.title("SHAP feature importance — XGBoost")
        plt.tight_layout(); plt.savefig(FIG / "shap_summary_xgboost.png", dpi=120); plt.close()
        print("SHAP summary written.")
    except Exception as e:  # SHAP is optional; never fail the run on it
        print(f"[SHAP skipped: {e}]")

    print(f"\nFigures + table written to {ROOT/'reports'}")


if __name__ == "__main__":
    main()
