# Methodology

## Research question

Given the previous 60 OFR Financial Stress Index trading days, classify whether
the mean OFR FSI over the next five observations is lower, stable or higher
relative to the current value. This is a three-class research task, not a return
forecast or a trading strategy.

## Temporal protocol

The cleaned reference snapshot contains 6,730 daily observations from
2000-01-03 through 2026-08-05. Splits are chronological:

| Split | Boundary | Prediction dates | Sequences |
|---|---|---|---:|
| Train | through 2016-12-31 | 2000-03-28–2016-12-22 | 4,213 |
| Validation | 2017-01-01–2019-12-31 | 2017-01-03–2019-12-23 | 749 |
| Test | from 2020-01-01 | 2020-01-02–2026-07-29 | 1,694 |

Targets are calculated separately inside each split, so the last five dates of
one split cannot borrow outcomes from the next. Validation and test windows may
use earlier observations as feature context because those values were already
available at the prediction date. They never use later observations.

The lower and upper class thresholds are the training target tertiles
(-0.1388 and 0.08627). `StandardScaler` is fit on training features only and
then applied unchanged to validation and test data. Data loaders preserve order
with `shuffle=False`.

## Models and selection

The comparison covers exactly three implemented neural models: a flattened MLP
baseline, a single-layer LSTM and a two-block Transformer encoder. Candidate
hyperparameters are evaluated on validation data. Test results are read only
after selection. LSTM finalists L1 and L9 additionally use seeds 42, 123 and
2026; L1 is retained by mean validation Macro-F1 with validation loss as the
tie-breaker. The Transformer uses T1, selected by validation Macro-F1 and loss.

The same 1,694 test targets feed every final evaluator. Reported metrics include
Accuracy, Macro-Precision, Macro-Recall, Macro-F1, per-class scores, confusion
matrices and one-vs-rest ROC-AUC. Calendar-year, attention and Gaussian-noise
analyses are descriptive post-hoc checks and do not trigger reselection.

## Leakage audit

The implementation is designed to avoid the leakage modes checked here:

- no random temporal split;
- future target horizons stop at split boundaries;
- thresholds and scaler parameters are learned on training data only;
- validation selects hyperparameters; test data does not;
- held-out metrics, yearly slices, ROC-AUC and robustness are post-selection;
- sequence construction uses only values at or before each prediction date.

This is a bounded implementation claim, not proof that every possible form of
financial-data leakage has been eliminated. OFR can revise source data, and the
experiment is not a point-in-time vintage database.

## Reproduction levels

1. `make evidence-check` verifies the versioned evidence without network,
   raw data or checkpoints.
2. `make reproduce-evidence` regenerates the comparison figure from that
   evidence.
3. Maintainers with the frozen raw CSV and ignored `.pt` files can run the
   evaluator modules and `scripts/export_reference_evidence.py` to reconstruct
   and re-hash the evidence bundle.
4. Full retraining is intentionally not a CI job because it is materially more
   expensive and sensitive to hardware-specific numerical behavior.
