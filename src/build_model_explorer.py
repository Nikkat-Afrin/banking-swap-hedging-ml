"""Build an interactive model-output explorer (docs/index.html).

Loads the persisted XGBoost bundle, scores the held-out test split, and
renders a self-contained interactive page:

  * threshold explorer - a slider sweeps the decision threshold and updates
    the confusion matrix and precision / recall / F1 live (all states are
    precomputed, so the page needs no server)
  * ROC curve with the tuned operating point marked
  * score distributions for hedgers vs non-hedgers
  * top feature importances of the deployed model

Run from the repo root (after `python src/train.py`):
    python src/build_model_explorer.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from data_prep import load_and_prepare

RNG = 47
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "index.html"
BUNDLE = ROOT / "models" / "swap_hedging_xgb.joblib"

CARD_CSS = """
 body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
        background: #f4f6f9; color: #1c2733; }
 header { padding: 26px 34px 6px; }
 h1 { margin: 0 0 4px; font-size: 25px; }
 .sub { color: #5c6b7a; font-size: 14px; max-width: 900px; }
 .kpis { display: flex; gap: 14px; padding: 16px 34px 0; flex-wrap: wrap; }
 .kpi { background: white; border-radius: 10px; padding: 12px 20px;
        box-shadow: 0 1px 4px rgba(20,40,80,.08); min-width: 130px; }
 .kpi .v { font-size: 22px; font-weight: 700; color: #0b5fff; }
 .kpi .l { font-size: 12px; color: #5c6b7a; }
 .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(470px, 1fr));
         gap: 18px; padding: 18px 34px 36px; }
 .card { background: white; border-radius: 10px; padding: 6px;
         box-shadow: 0 1px 4px rgba(20,40,80,.08); }
 .wide { grid-column: 1 / -1; }
 footer { padding: 0 34px 26px; color: #8595a5; font-size: 13px; }
"""


def main() -> None:
    bundle = joblib.load(BUNDLE)
    model, scaler, threshold = bundle["model"], bundle["scaler"], bundle["threshold"]

    X, y = load_and_prepare()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RNG)
    proba = model.predict_proba(scaler.transform(X_te))[:, 1]
    y_arr = y_te.to_numpy()
    auc = roc_auc_score(y_arr, proba)

    # ---- threshold explorer (precomputed slider states) --------------------
    thresholds = np.round(np.linspace(0.05, 0.95, 19), 2)
    frames = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_arr, pred).ravel()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        frames.append((t, tn, fp, fn, tp, prec, rec, f1))

    fig_thr = go.Figure()
    steps = []
    for i, (t, tn, fp, fn, tp, prec, rec, f1) in enumerate(frames):
        fig_thr.add_trace(go.Heatmap(
            z=[[fn, tp], [tn, fp]],
            x=["Predicted: no hedge", "Predicted: hedge"],
            y=["Actual: hedge", "Actual: no hedge"],
            text=[[fn, tp], [tn, fp]], texttemplate="%{text}",
            colorscale="Blues", showscale=False,
            visible=(t == 0.30)))
        steps.append(dict(
            method="update", label=f"{t:.2f}",
            args=[{"visible": [j == i for j in range(len(frames))]},
                  {"title": (f"Threshold {t:.2f} - precision {prec:.2f} · "
                             f"recall {rec:.2f} · F1 {f1:.2f}")}]))
    active = int(np.argmin(np.abs(thresholds - 0.30)))
    fig_thr.update_layout(
        sliders=[dict(active=active, currentvalue={"prefix": "Decision threshold: "},
                      steps=steps, pad={"t": 34})],
        title=(f"Threshold {frames[active][0]:.2f} - precision {frames[active][5]:.2f} · "
               f"recall {frames[active][6]:.2f} · F1 {frames[active][7]:.2f}"),
        margin=dict(l=30, r=30, t=60, b=20), height=460)

    # ---- ROC ---------------------------------------------------------------
    fpr, tpr, thr = roc_curve(y_arr, proba)
    op_idx = int(np.argmin(np.abs(thr - threshold)))
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"XGBoost (AUC {auc:.3f})",
                                 line=dict(color="#0b5fff", width=2.5)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 name="Chance", line=dict(dash="dash", color="#98a5b3")))
    fig_roc.add_trace(go.Scatter(x=[fpr[op_idx]], y=[tpr[op_idx]], mode="markers+text",
                                 marker=dict(size=12, color="#e8590c"),
                                 text=[f"tuned threshold {threshold:.2f}"],
                                 textposition="bottom right", name="Operating point"))
    fig_roc.update_layout(title="ROC curve (held-out test set)",
                          xaxis_title="False positive rate", yaxis_title="True positive rate",
                          margin=dict(l=50, r=20, t=50, b=45), height=430)

    # ---- score distribution -------------------------------------------------
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=proba[y_arr == 0], nbinsx=40, name="No hedge",
                                    marker_color="#98a5b3", opacity=0.75))
    fig_dist.add_trace(go.Histogram(x=proba[y_arr == 1], nbinsx=40, name="Hedge",
                                    marker_color="#0b5fff", opacity=0.75))
    fig_dist.add_vline(x=threshold, line_dash="dash", line_color="#e8590c",
                       annotation_text=f"threshold {threshold:.2f}")
    fig_dist.update_layout(barmode="overlay", title="Predicted probability by actual class",
                           xaxis_title="P(hedging adoption)", yaxis_title="Banks",
                           margin=dict(l=50, r=20, t=50, b=45), height=430)

    # ---- feature importance --------------------------------------------------
    imp = pd.Series(model.feature_importances_, index=bundle["columns"]).nlargest(12)[::-1]
    fig_imp = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                               marker=dict(color=imp.values, colorscale="Blues")))
    fig_imp.update_layout(title="Top feature importances (gain)",
                          margin=dict(l=140, r=20, t=50, b=40), height=430)

    # ---- page ---------------------------------------------------------------
    tn, fp, fn, tp = confusion_matrix(y_arr, (proba >= threshold).astype(int)).ravel()
    prec = tp / (tp + fp); rec = tp / (tp + fn)
    kpis = [(f"{auc:.3f}", "Test ROC-AUC"), (f"{threshold:.2f}", "Tuned threshold"),
            (f"{prec:.2f}", "Precision @ threshold"), (f"{rec:.2f}", "Recall @ threshold"),
            (f"{len(y_arr):,}", "Test bank-years")]
    kpi_html = "".join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>'
                       for v, l in kpis)
    charts = []
    for i, fig in enumerate([fig_thr, fig_roc, fig_dist, fig_imp]):
        cls = "card wide" if i == 0 else "card"
        inner = fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False,
                            div_id=f"chart-{i}")
        charts.append(f'<div class="{cls}">{inner}</div>')
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Swap-Hedging Model Explorer</title><style>{CARD_CSS}</style></head><body>
<header><h1>Swap-Hedging Adoption - Model Explorer</h1>
<div class="sub">Interactive view of the deployed XGBoost classifier on the held-out
test split (2,354 bank-year panel, ~8% positive class). Drag the threshold slider to
see the business trade-off between catching hedgers (recall) and false alarms.</div>
</header>
<div class="kpis">{kpi_html}</div>
<div class="grid">{''.join(charts)}</div>
<footer>Regenerate: <code>python src/train.py && python src/build_model_explorer.py</code></footer>
</body></html>"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"Explorer -> {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
