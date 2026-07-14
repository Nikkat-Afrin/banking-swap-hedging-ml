# Model Card — Swap-Hedging Adoption Classifier

## Overview
XGBoost binary classifier predicting whether a Russian bank adopts interest-rate
swap hedging in a given year, trained on a 2,354 bank-year panel (2016–2021).

- **Artifact:** `swap_hedging_xgb.joblib` — self-contained bundle: fitted model,
  fitted `StandardScaler`, training feature columns, and tuned decision threshold.
- **Training entry point:** `python src/train.py`
- **Batch scoring:** `python src/predict.py <input.csv> --out predictions.csv`

## Performance (5-fold stratified CV + 20% held-out test)
| Metric | Value |
|---|---|
| CV ROC-AUC | 0.963 ± 0.011 |
| Test ROC-AUC | 0.953 |
| Test F1 (threshold 0.28) | 0.75 |
| Test precision / recall | 0.71 / 0.79 |

The decision threshold is tuned on out-of-fold predictions to maximize F1 —
with an ~8% positive class the default 0.5 cutoff sacrifices most of the recall.
Full metrics: `reports/metrics.json`.

## Features
Bank financial ratios and balance-sheet aggregates (capital adequacy Н1.0, ROE,
ROA, log loans/securities/deposits/equity/EBT, H4 liquidity, overdue-debt ratio)
plus bank status. Identifiers (bank name, registration number) are dropped;
percent-formatted strings are parsed; missing values are KNN-imputed
(see `src/data_prep.py` — single source of truth shared by training and scoring).

## Intended use & limitations
Research/portfolio model on a historical public panel. Not for production credit
or trading decisions: the panel covers one market and period, and hedging
adoption is influenced by unobserved factors (management policy, group treasury
structures). Class imbalance means precision degrades if base rates shift.
