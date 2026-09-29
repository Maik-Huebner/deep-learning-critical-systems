# Model Card: Financial Stress Regime Classifiers

## Scope

This repository contains MLP, LSTM and Transformer research classifiers for a
three-class, five-observation OFR FSI direction target. It is a reproducible
portfolio prototype, not an operational early-warning, investment or trading
system.

## Final held-out results

| Model | Accuracy | Macro-F1 | Stress-increase recall | Macro ROC-AUC |
|---|---:|---:|---:|---:|
| MLP | **43.92%** | 33.44% | 0.37% | 0.5801 |
| LSTM | 41.79% | 36.43% | 6.89% | 0.5970 |
| Transformer | 42.44% | **37.38%** | **8.19%** | **0.6183** |
| Training-majority baseline | 30.64% | 15.63% | 0.00% | n/a |

The Transformer leads the neural models on the selected balanced metrics, but
its absolute performance is modest. It correctly identifies only 44 of 537
stress-increase cases. Accuracy alone favors the MLP and obscures its two correct
stress-increase predictions.

## Intended and out-of-scope uses

Appropriate uses are education, method review, reproducibility exercises and
research comparisons under the documented snapshot. Out-of-scope uses include
capital allocation, automated trading, regulatory reporting, credit decisions,
real-time alerts or any safety-critical decision without an independently
validated system and human controls.

## Interpretability and robustness

Transformer attention maps are descriptive views of internal token-to-token
weighting. They do not establish causality and are not direct feature importance.
The noise study adds Gaussian perturbations to standardized inputs; it probes one
narrow sensitivity axis and does not simulate market crises or distribution shift.

## Temporal behavior and uncertainty

Calendar-year Macro-F1 and stress-increase recall vary substantially. In several
years all models have near-zero increase recall, while 2021 is materially better
for sequential models. The partial 2026 slice is weak. Seed sensitivity is also
visible in the LSTM validation study. No statistical confidence interval can
remove the dependence among overlapping time windows, so results should be read
as one fixed historical experiment rather than a population guarantee.
