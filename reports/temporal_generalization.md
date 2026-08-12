# Zeitliche Generalisierung 2020–2026

Diese Analyse untersucht die bereits final festgelegten Modelle getrennt nach Kalenderjahr im gehaltenen Testzeitraum.

Die Jahresanalyse dient ausschließlich der Beschreibung der zeitlichen Generalisierung. Sie wurde **nicht** für weitere Modellauswahl, Hyperparameter-Optimierung oder Retraining genutzt.

Das Jahr **2026 ist ein Teiljahr**. Der eingefrorene OFR-Datensatz endet am 05.08.2026; der letzte verfügbare Vorhersagetag ist der 29.07.2026.

## Jahresergebnisse

| Jahr | Modell | Fälle | Accuracy | Macro-F1 | Stress-Increase-Recall |
|---:|---|---:|---:|---:|---:|
| 2020 | MLP | 253 | 43.08 % | 35.49 % | 2.53 % |
| 2020 | LSTM | 253 | 39.13 % | 36.66 % | 15.19 % |
| 2020 | Transformer | 253 | 46.25 % | 38.00 % | 2.53 % |
| 2021 | MLP | 252 | 38.89 % | 18.67 % | 0.00 % |
| 2021 | LSTM | 252 | 38.89 % | 32.68 % | 25.84 % |
| 2021 | Transformer | 252 | 37.30 % | 37.67 % | 41.57 % |
| 2022 | MLP | 257 | 40.08 % | 25.53 % | 0.00 % |
| 2022 | LSTM | 257 | 42.80 % | 28.72 % | 1.01 % |
| 2022 | Transformer | 257 | 44.36 % | 31.03 % | 1.01 % |
| 2023 | MLP | 260 | 50.00 % | 35.41 % | 0.00 % |
| 2023 | LSTM | 260 | 38.08 % | 28.82 % | 0.00 % |
| 2023 | Transformer | 260 | 48.46 % | 36.17 % | 0.00 % |
| 2024 | MLP | 261 | 52.87 % | 23.06 % | 0.00 % |
| 2024 | LSTM | 261 | 45.59 % | 33.70 % | 0.00 % |
| 2024 | Transformer | 261 | 46.36 % | 38.38 % | 5.80 % |
| 2025 | MLP | 261 | 45.98 % | 30.64 % | 0.00 % |
| 2025 | LSTM | 261 | 49.43 % | 37.96 % | 0.00 % |
| 2025 | Transformer | 261 | 37.93 % | 29.49 % | 0.00 % |
| 2026* | MLP | 150 | 30.67 % | 15.65 % | 0.00 % |
| 2026* | LSTM | 150 | 36.00 % | 29.07 % | 1.67 % |
| 2026* | Transformer | 150 | 32.00 % | 26.37 % | 0.00 % |

## Interpretation

Die jahresweise Betrachtung macht sichtbar, ob die Modelle über unterschiedliche Marktphasen hinweg ähnlich stabil arbeiten oder ob ihre Leistung zeitlich deutlich schwankt.

Eine solche Schwankung ist bei Finanzzeitreihen besonders relevant, weil sich Marktregime, Volatilität und strukturelle Zusammenhänge über die Zeit verändern können.

Die Jahresmetriken werden deshalb als Ergänzung zur Gesamt-Testauswertung verstanden und nicht als Grundlage für nachträgliche Modelloptimierung.

## Abbildungen

- `figures/temporal_generalization_macro_f1.png`
- `figures/temporal_generalization_increase_recall.png`
