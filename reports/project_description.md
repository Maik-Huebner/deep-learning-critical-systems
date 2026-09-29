# Project Description

## Objective

This project is a reproducible Deep-Learning research prototype for classifying
short-horizon changes in the Office of Financial Research Financial Stress Index
(OFR FSI). A 60-trading-day window of nine published OFR series is mapped to one
of three classes describing the mean FSI movement over the next five observations.

The deliverable is an auditable experiment: provenance checks, chronological
splits, train-only preprocessing, validation-only selection, held-out evaluation,
machine-readable evidence and explicit limitations. It is not a financial advice,
trading or operational alerting product.

## Dataset and target

The official OFR CSV is downloaded at runtime and is not stored in Git. The
reference experiment uses 6,730 observations through 2026-08-05. Its raw download
and canonical cleaned snapshot are recorded by SHA-256 in
`reports/evidence/data_snapshot.json`.

Features are OFR FSI, Credit, Equity valuation, Safe assets, Funding, Volatility,
United States, Other advanced economies and Emerging markets. Required columns,
dates, duplicates and missing values are validated before use.

For target date `t`, the label source is:

```text
mean(OFR FSI[t+1], ..., OFR FSI[t+5]) - OFR FSI[t]
```

Training-set tertiles create decrease, stable and increase classes. Targets are
computed within each split; incomplete horizons are removed.

## Experimental design

| Split | Boundary | Sequences |
|---|---|---:|
| Train | through 2016 | 4,213 |
| Validation | 2017–2019 | 749 |
| Test | 2020–2026-07-29 prediction dates | 1,694 |

The scaler is fit only on training observations. Validation and test windows may
use earlier observations as feature history, never later dates. Hyperparameters
are selected on validation Macro-F1 with validation loss as tie-breaker. The test
set is used once for final descriptive evaluation and later post-hoc analyses.

## Implemented models

- MLP baseline: flattened 60 x 9 input, hidden sizes 128 and 64.
- LSTM: one 64-unit recurrent layer and a 32-unit classifier layer.
- Transformer: 64-dimensional input projection, four heads, two encoder blocks,
  128-unit feed-forward layers, mean pooling and a 32-unit classifier layer.

The selected LSTM L1 is also compared with finalist L9 across three seeds. The
selected Transformer is T1. Both choices precede test evaluation.

## Results

| Model | Accuracy | Macro-F1 | Increase recall | Macro ROC-AUC |
|---|---:|---:|---:|---:|
| MLP | **43.92%** | 33.44% | 0.37% | 0.5801 |
| LSTM | 41.79% | 36.43% | 6.89% | 0.5970 |
| Transformer | 42.44% | **37.38%** | **8.19%** | **0.6183** |
| Training-majority baseline | 30.64% | 15.63% | 0.00% | n/a |

The Transformer leads on the balanced headline metrics but remains weak in
absolute terms. Its confusion matrix is:

```text
[[368, 122, 29],
 [279, 307, 52],
 [256, 237, 44]]
```

It misses 493 of 537 stress increases. The MLP's higher Accuracy is not a better
stress detector: it correctly identifies only two stress increases.

## Supporting analyses

- ROC-AUC evaluates probability ranking after model selection.
- Calendar-year slices expose performance variation and nonstationarity.
- LSTM multi-seed comparison exposes selection sensitivity.
- Attention plots provide descriptive token-relation views, not causal or direct
  feature-importance explanations.
- Gaussian input noise is a bounded sensitivity test, not a market-shock model.

The exact results live in `reports/evidence/` and are protected by a SHA-256
manifest. `python scripts/reproduce_evidence.py` validates them offline.

## Engineering controls

The project pins Python 3.13 dependencies, formats and lints all source/tests/
scripts with Ruff, runs automated tests, builds an sdist and wheel, performs an
isolated wheel import smoke test, and audits tracked files for common local
artifacts, checkpoints, raw data, oversized binaries and obvious secret patterns.
The audit is intentionally described as a lightweight guard, not a professional
secret scanner.

For full methodology, data constraints and use boundaries, see `docs/`.
