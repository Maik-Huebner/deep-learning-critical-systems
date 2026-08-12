# ROC-AUC-Auswertung

Die finalen Modelle MLP, LSTM und Transformer wurden zusätzlich mit Multi-Class ROC-AUC nach dem One-vs-Rest-Verfahren (OvR) auf dem gehaltenen Testdatensatz bewertet.

**Testfälle:** 1694

Diese Auswertung ist ausschließlich eine nachgelagerte Beschreibung der bereits festgelegten Modelle. Die Ergebnisse wurden **nicht** für weitere Modellauswahl, Hyperparameter-Optimierung oder Retraining verwendet.

## Ergebnisse

| Modell | Macro ROC-AUC | Stressrückgang | Stabil | Stressanstieg |
|---|---:|---:|---:|---:|
| MLP | 0.5801 | 0.6296 | 0.5572 | 0.5534 |
| LSTM | 0.5970 | 0.6333 | 0.5944 | 0.5632 |
| Transformer | 0.6183 | 0.6863 | 0.6207 | 0.5481 |

## Einordnung

ROC-AUC ergänzt Accuracy, Precision, Recall und Macro-F1 um eine schwellenunabhängige Betrachtung der Trennfähigkeit.

Ein ROC-AUC-Wert von 0,5 entspricht näherungsweise einer zufälligen Rangordnung. Höhere Werte zeigen, dass das Modell positive und negative Fälle einer Klasse über verschiedene Entscheidungsschwellen hinweg besser voneinander trennt.

Die ROC-AUC-Ergebnisse ersetzen die bisherigen Metriken nicht. Insbesondere bei diesem Projekt bleiben Macro-F1 und der Recall für Stressanstiege wichtig, weil die tatsächliche harte Drei-Klassen-Entscheidung für die fachliche Bewertung relevant ist.

Die Majority-Class-Baseline wird hier nicht aufgenommen, da sie keine modellierten Klassenwahrscheinlichkeiten bereitstellt.

## Abbildung

- `figures/roc_auc_comparison.png`
