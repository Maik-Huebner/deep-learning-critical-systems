# Discussion and Error Analysis

## What the comparison shows

The three architectures trade headline Accuracy for class balance. The MLP has
the highest Accuracy (43.92%) but a Macro-F1 of 33.44% and essentially no ability
to detect increases (2/537). The LSTM improves Macro-F1 to 36.43% and detects
37 increases. The Transformer reaches the highest Macro-F1 (37.38%), increase
recall (44/537, 8.19%) and Macro ROC-AUC (0.6183), but those values remain modest.

This supports a narrow conclusion: on this fixed experiment the selected
Transformer ranks and balances the three classes slightly better than the other
neural models. It does not support reliable financial forecasting.

## Class-specific failure modes

Transformer errors are concentrated in three routes:

| Actual -> predicted | Count | Interpretation |
|---|---:|---|
| Stable -> Decrease | 279 | stable periods are often read as relief |
| Increase -> Decrease | 256 | most consequential directional inversion |
| Increase -> Stable | 237 | escalation is muted into the middle class |
| Decrease -> Stable | 122 | easing is often undercalled |

Only 44 of 537 actual increases are correct. The test support is reasonably
distributed (519 decrease, 638 stable, 537 increase), so this is not explained by
an extremely rare increase class. The model's decision surface instead favors
decrease and stable outcomes under the selected objective and data regime.

The MLP makes the imbalance even clearer: 546 stable cases are correct, while all
but two increases are assigned to decrease or stable. Its 43.92% Accuracy is thus
a poor proxy for the project's risk-sensitive objective.

## Probability ranking versus decisions

Stress-increase ROC-AUC is 0.5534 for MLP, 0.5632 for LSTM and 0.5481 for the
Transformer. These values show only weak ranking ability. The Transformer's best
macro ROC-AUC comes from decrease (0.6863) and stable (0.6207), not the critical
increase class. Threshold analysis or calibration might change operating points,
but neither was used post hoc because that would require a separately reserved
selection set.

## Temporal variation

Yearly slices reveal substantial regime dependence. Transformer increase recall
is 41.57% in 2021 but zero in 2023, 2025 and the partial 2026 period. LSTM exceeds
Transformer Macro-F1 in 2025, while Transformer leads in other years. The partial
2026 period is weak for all three models. These are descriptive slices of the same
test set, not independent replications.

The overlapping 60-day windows also create serial dependence. Treating all 1,694
sequences as independent observations would make conventional confidence
intervals misleading, so none are claimed here. A future study should use blocked
bootstrap or walk-forward folds designed for dependent time series.

## Selection stability

Across seeds 42, 123 and 2026, L1 achieves mean validation Macro-F1
0.2875 ± 0.0715 and L9 achieves 0.2813 ± 0.0768. The standard deviations are
large relative to the difference between means. L1 remains selected by the
predeclared rule, but the outcome should be read as seed-sensitive rather than a
clear architectural win.

## Explainability limits

Attention weights show how tokens participate in a specific forward pass. The
correct increase example has only 36.83% confidence; a missed increase assigned
to decrease has 58.34% confidence. These examples illustrate internal behavior
and confidence failure, but attention does not establish causal influence or
direct feature importance.

## Robustness limits

At Gaussian input-noise standard deviation 0.05, 96.69% of predictions agree with
clean inputs and Macro-F1 is 36.88%. At 0.50, agreement falls to 73.32% and
Macro-F1 to 32.94%. This checks numerical sensitivity around standardized inputs.
It does not represent liquidity shocks, missing feeds, historical revisions,
adversarial manipulation or distribution shift.

## Remaining threats to validity

- OFR data is a current series rather than point-in-time historical vintages.
- Only nine aggregate OFR series are used.
- One fixed split cannot establish broad temporal generalization.
- Five-day targets and 60-day windows are design choices, not universal optima.
- Tuning and seed checks are limited in breadth.
- No calibration, costs, returns, drawdowns or decision utility are evaluated.
- No independent external dataset is used.

The appropriate next research step is predeclared walk-forward evaluation with
point-in-time data and a metric/threshold strategy aligned to missed increases.
