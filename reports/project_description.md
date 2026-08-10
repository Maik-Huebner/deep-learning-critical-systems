# Projektbeschreibung

## Financial Stress Regime Forecasting with PyTorch

### 1. Ausgangssituation

Finanzmärkte verändern sich über die Zeit und können Phasen erhöhten oder sinkenden Finanzstresses durchlaufen. Solche Stressphasen können unter anderem durch Veränderungen bei Kreditrisiken, Liquidität, Volatilität, Bewertungen oder makroökonomischen Rahmenbedingungen entstehen.

Das Projekt untersucht eine Deep-Learning-basierte Klassifikation zukünftiger Finanzstressregime. Im Mittelpunkt steht dabei nicht die Entwicklung eines direkten Trading-Systems, sondern die methodisch saubere Modellierung einer realen Finanzzeitreihe mit PyTorch.

---

## 2. Forschungsfrage

> Kann ein PyTorch-Transformer anhand der vergangenen 60 Handelstage multivariater Finanzstressdaten klassifizieren, ob der Finanzstress über die folgenden fünf Handelstage sinkt, stabil bleibt oder steigt?

---

## 3. Projektziel

Ziel des Projekts ist die Entwicklung und Evaluation eines reproduzierbaren Deep-Learning-Workflows für eine multivariate Finanzzeitreihe.

Dabei werden drei neuronale Modellansätze miteinander verglichen:

1. MLP als einfache neuronale Baseline
2. LSTM als sequenzielles Vergleichsmodell
3. Transformer als Hauptmodell

Neben der reinen Modellleistung werden insbesondere folgende Punkte betrachtet:

- korrekte chronologische Datenaufteilung
- Vermeidung von Data Leakage
- reproduzierbare Datenvorbereitung
- Early Stopping
- Hyperparameter-Tuning
- geeignete Multi-Class-Metriken
- Confusion Matrices
- Fehleranalyse
- Attention-basierte Explainability
- Robustheitsanalyse
- Limitationen
- Fairness-Einordnung
- Responsible AI

---

## 4. Erfolgskriterien

Das Projekt gilt fachlich als erfolgreich, wenn ein vollständiger und nachvollziehbarer Deep-Learning-Prozess umgesetzt wird und die Modelle auf einem zeitlich späteren, unbekannten Testzeitraum sinnvoll miteinander verglichen werden können.

Die Bewertung erfolgt nicht ausschließlich anhand der Accuracy.

Wichtige Erfolgskriterien sind:

- ein vollständig chronologischer Train-/Validation-/Test-Split
- kein Data Leakage zwischen den Zeitabschnitten
- Scaling ausschließlich anhand der Trainingsdaten
- mindestens eine einfache Baseline
- Vergleich mehrerer Deep-Learning-Architekturen
- reproduzierbares Training
- Early Stopping gegen unnötiges Overfitting
- Hyperparameter-Auswahl ausschließlich anhand der Validation-Daten
- finale Evaluation auf einem zuvor nicht für die Modellauswahl verwendeten Testzeitraum
- Auswertung von Accuracy, Precision, Recall und F1
- besondere Betrachtung der Klasse `Stress Increase`
- Analyse korrekter und falscher Vorhersagen
- Explainability
- Robustheitsbetrachtung
- nachvollziehbare Diskussion von Limitationen und nächsten Schritten

Ein produktionsreifes Finanzprognosemodell oder eine profitable Trading-Strategie ist ausdrücklich kein Erfolgskriterium dieses Abschlussprojekts.

---

## 5. Datenquelle

Verwendet wird der Financial Stress Index des Office of Financial Research (OFR).

Offizielle Quelle:

https://www.financialresearch.gov/

Der im Projekt verwendete Datensatz umfasst:

- 6.730 Beobachtungen
- Zeitraum vom 03.01.2000 bis 05.08.2026
- neun numerische Features
- keine fehlenden Werte
- keine doppelten Datumswerte
- chronologisch sortierte Beobachtungen

Die Rohdaten werden lokal gespeichert und nicht in das Git-Repository eingecheckt.

---

## 6. Verwendete Features

Die Modellierung verwendet folgende neun Variablen:

1. `OFR FSI`
2. `Credit`
3. `Equity valuation`
4. `Safe assets`
5. `Funding`
6. `Volatility`
7. `United States`
8. `Other advanced economies`
9. `Emerging markets`

Die einzelnen Variablen bilden unterschiedliche Komponenten und regionale Beiträge des Financial Stress Index ab.

---

## 7. Explorative Datenanalyse

Vor der Modellierung wird der Datensatz explorativ untersucht.

Dazu gehören unter anderem:

- zeitlicher Verlauf des OFR FSI
- Verteilung der zukünftigen Stressänderungen
- Korrelationen zwischen den Features
- Klassenverteilungen
- Kontrolle auf fehlende Werte
- Kontrolle auf doppelte Beobachtungen
- Kontrolle der chronologischen Sortierung

Die explorative Analyse zeigt unter anderem starke Korrelationen zwischen mehreren OFR-Komponenten.

Diese Korrelationen werden nicht als Kausalität interpretiert.

---

## 8. Zielvariable

Die Zielvariable basiert auf der Veränderung des OFR Financial Stress Index über die folgenden fünf Handelstage.

Für jeden Zeitpunkt wird zunächst der Mittelwert der nächsten fünf OFR-FSI-Werte berechnet.

Anschließend wird der aktuelle OFR-FSI-Wert abgezogen:

```text
zukünftige Stressänderung
=
Mittelwert der nächsten fünf OFR-FSI-Werte
-
aktueller OFR-FSI-Wert
```

Die daraus entstehenden Werte werden in drei Klassen überführt:

```text
0 = Stress Decrease
1 = Stable
2 = Stress Increase
```

Die Klassengrenzen werden ausschließlich anhand des Trainingszeitraums bestimmt.

Verwendete Trainingsgrenzen:

```text
untere Grenze ≈ -0,1388
obere Grenze ≈  0,0863
```

Dadurch werden Validation- und Testdaten nicht für die Definition der Zielklassen verwendet.

---

## 9. Chronologische Datenaufteilung

Die Daten werden nicht zufällig aufgeteilt.

Stattdessen erfolgt eine zeitlich geordnete Aufteilung:

```text
Training:
bis Ende 2016

Validation:
2017 bis Ende 2019

Test:
ab 2020
```

Der tatsächlich letzte Trainingstag im Datensatz ist der 30.12.2016.

Die chronologische Aufteilung ist für Finanzzeitreihen wichtig, weil zukünftige Informationen nicht in frühere Trainingsphasen gelangen dürfen.

---

## 10. Vermeidung von Data Leakage

Zur Vermeidung von Data Leakage werden mehrere Maßnahmen umgesetzt.

### Chronologische Splits

Vergangenheit und Zukunft werden nicht zufällig vermischt.

### Zielberechnung innerhalb der Splits

Die zukünftige Fünf-Tage-Zielvariable wird separat innerhalb der einzelnen Datenabschnitte berechnet.

Dadurch überschreiten Zielwerte keine Split-Grenzen.

### Scaling nur auf Trainingsdaten

Der StandardScaler wird ausschließlich auf dem Trainingszeitraum angepasst.

Validation und Test werden anschließend nur mit den auf dem Training gelernten Parametern transformiert.

### Historischer Kontext

Für die ersten Vorhersagen innerhalb von Validation und Test dürfen bereits bekannte historische Beobachtungen aus dem jeweils vorherigen Zeitraum als Eingabekontext verwendet werden.

Dabei werden ausschließlich vergangene Beobachtungen verwendet.

---

## 11. Sliding-Window-Aufbereitung

Jede Modellvorhersage basiert auf den vergangenen 60 Handelstagen.

Jeder Handelstag besitzt neun Features.

Damit ergibt sich pro Sequenz:

```text
60 Handelstage × 9 Features
```

Tensorform:

```text
(samples, timesteps, features)
```

Finale Sequenzgrößen:

```text
Training:
(4213, 60, 9)

Validation:
(749, 60, 9)

Test:
(1694, 60, 9)
```

Klassenverteilungen:

```text
Training:
[1406, 1402, 1405]

Validation:
[233, 303, 213]

Test:
[519, 638, 537]
```

Die DataLoader verwenden `shuffle=False`, damit die chronologische Reihenfolge nicht unnötig verändert wird.

---

## 12. Modell 1: MLP-Baseline

Das MLP dient als einfache neuronale Baseline.

Die 60 Handelstage mit neun Features werden zu einem flachen Vektor zusammengeführt.

Architektur:

```text
60 × 9
  ↓
Flatten
  ↓
540
  ↓
Linear 540 → 128
ReLU
Dropout
  ↓
Linear 128 → 64
ReLU
Dropout
  ↓
Linear 64 → 3
```

Trainierbare Parameter:

```text
77.699
```

Das MLP besitzt keine speziell für zeitliche Abhängigkeiten entwickelte Architektur.

---

## 13. Modell 2: LSTM

Das LSTM verarbeitet die Daten als Sequenz.

Architektur:

```text
9 Features
  ↓
LSTM
Hidden Size 64
  ↓
letzter Sequenzzustand
  ↓
Linear 64 → 32
ReLU
Dropout
  ↓
Linear 32 → 3
```

Trainierbare Parameter:

```text
21.379
```

Das LSTM dient als sequenzielles Vergleichsmodell zum Transformer.

---

## 14. Modell 3: Transformer

Der Transformer ist das Hauptmodell des Projekts.

Die Eingabefeatures werden zunächst in eine höhere Modelldimension projiziert.

Anschließend wird ein sinusoidales Positional Encoding ergänzt.

Die Sequenz wird danach durch zwei selbst implementierte Transformer-Encoder-Blöcke verarbeitet.

Architektur:

```text
9 Features
  ↓
Input Projection 9 → 64
  ↓
Sinusoidal Positional Encoding
  ↓
Transformer Encoder Block
  ↓
Transformer Encoder Block
  ↓
Mean Pooling
  ↓
Linear 64 → 32
ReLU
Dropout
  ↓
Linear 32 → 3
```

Finale Hyperparameter:

```text
Model Dimension:       64
Attention Heads:        4
Encoder Layers:         2
Feed-Forward Size:    128
Classifier Hidden:     32
Dropout:             0,20
Learning Rate:     0,0005
Batch Size:            64
```

Trainierbare Parameter:

```text
69.763
```

---

## 15. Training

Alle Modelle werden mit einem gemeinsamen PyTorch-Training-Workflow trainiert.

Verwendet werden unter anderem:

- `CrossEntropyLoss`
- Adam Optimizer
- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy
- Early Stopping
- Wiederherstellung des besten Modellzustands
- fester Random Seed
- automatische Device-Auswahl

Device-Priorität:

```text
CUDA
→ MPS
→ CPU
```

Auf dem verwendeten System wurde Apple Metal Performance Shaders (MPS) genutzt.

Early-Stopping-Konfiguration:

```text
Maximum Epochs: 50
Patience:        7
Min Delta:       0,0001
```

---

## 16. Transformer-Hyperparameter-Tuning

Das Transformer-Tuning wird ausschließlich mit Trainings- und Validation-Daten durchgeführt.

Das Testset wird nicht für die Hyperparameter-Auswahl verwendet.

Primäres Auswahlkriterium:

```text
Validation Macro-F1
```

Sekundäres Kriterium:

```text
Validation Loss
```

Die Ausgangskonfiguration T0 erreicht:

- Validation Accuracy: 48,46 %
- Validation Macro-F1: 39,63 %
- Recall Stress Increase: 7,04 %

In zwei Tuning-Stufen werden insgesamt die Konfigurationen T0 bis T11 untersucht.

Die beste Konfiguration nach dem vorher festgelegten Auswahlkriterium ist T1.

T1 verwendet:

```text
Model Dimension:     64
Attention Heads:      4
Encoder Layers:       2
Feed Forward:       128
Dropout:            0,20
Learning Rate:    0,0005
```

Validation-Ergebnis:

- Best Epoch: 12
- Validation Loss: 1,0518
- Validation Accuracy: 47,66 %
- Validation Macro-F1: 41,91 %
- Recall Stress Increase: 18,31 %

---

## 17. Finale Modellevaluation

### Majority-Baseline

- Accuracy: 30,64 %
- Macro-F1: 15,63 %
- Recall Stress Increase: 0,00 %

### MLP

- Accuracy: 43,92 %
- Macro-F1: 33,44 %
- Recall Stress Increase: 0,37 %

### LSTM

- Accuracy: 44,33 %
- Macro-F1: 37,90 %
- Recall Stress Increase: 6,15 %

### Transformer

- Accuracy: 42,44 %
- Macro-F1: 37,38 %
- Recall Stress Increase: 8,19 %

Das LSTM erreicht damit den besten finalen Macro-F1.

Der Transformer erreicht den höchsten Recall für die Klasse `Stress Increase` unter den drei neuronalen Modellen.

Das Ergebnis zeigt, dass eine komplexere Architektur nicht automatisch besser auf unbekannte zukünftige Daten generalisiert.

---

## 18. Transformer Confusion Matrix

Finale Confusion Matrix des Transformers:

```text
[[368, 122,  29],
 [279, 307,  52],
 [256, 237,  44]]
```

Klassenspezifischer Recall:

```text
Stress Decrease: 70,91 %
Stable:          48,12 %
Stress Increase:  8,19 %
```

Die wichtigste Schwäche des Modells ist die geringe Erkennung steigenden Finanzstresses.

Von 537 tatsächlichen Stressanstiegen werden nur 44 korrekt erkannt.

---

## 19. Explainability

Für den Transformer werden Attention-Gewichte untersucht.

Analysiert werden unter anderem:

- ein korrekt erkannter Stressanstieg
- ein falsch klassifizierter tatsächlicher Stressanstieg
- klassenspezifische Fehler
- häufige Fehlklassifikationen

Ein korrekt erkanntes Beispiel besitzt folgende Modellwahrscheinlichkeiten:

```text
Stress Decrease: 27,77 %
Stable:          35,40 %
Stress Increase: 36,83 %
```

Ein falsch klassifizierter tatsächlicher Stressanstieg wird dagegen mit folgenden Wahrscheinlichkeiten vorhergesagt:

```text
Stress Decrease: 58,34 %
Stable:          13,84 %
Stress Increase: 27,81 %
```

Das zweite Beispiel zeigt, dass eine relativ hohe Modellkonfidenz keine Garantie für eine korrekte Vorhersage ist.

Attention-Gewichte werden dabei als Interpretationshilfe verstanden.

Sie beweisen keine Kausalität und sind nicht automatisch mit Feature Importance gleichzusetzen.

---

## 20. Robustheitsanalyse

Der finale Transformer wird zusätzlich mit künstlichem Gaußschem Rauschen auf den standardisierten Testfeatures untersucht.

Das Modell wird dabei nicht neu trainiert und nicht weiter optimiert.

Ergebnisse:

| Rauschen | Accuracy | Macro-F1 | Recall Stress Increase | Prediction Agreement |
|---:|---:|---:|---:|---:|
| 0 % | 42,44 % | 37,38 % | 8,19 % | 100,00 % |
| 5 % | 42,09 % | 36,88 % | 7,64 % | 96,69 % |
| 10 % | 42,38 % | 37,31 % | 8,19 % | 92,50 % |
| 20 % | 41,15 % | 35,87 % | 7,45 % | 85,66 % |
| 50 % | 39,37 % | 32,94 % | 4,66 % | 73,32 % |

Kleine Störungen verändern die Modellleistung nur begrenzt.

Bei stärkeren Eingabestörungen sinken die Leistungswerte zunehmend.

Die konkrete Verwendung von Gaußschem Rauschen ist eine praktische Robustheitsimplementierung und keine Simulation eines bestimmten realen Finanzmarktschocks.

---

## 21. Limitationen

Das Projekt besitzt mehrere klare Einschränkungen.

### Begrenzte Modellleistung

Der beste Test-Macro-F1 liegt unter 40 %.

Die Modelle sind damit nicht produktionsreif.

### Schwache Erkennung steigenden Finanzstresses

Die Klasse `Stress Increase` bleibt bei allen Modellen schwierig.

### Nichtstationäre Finanzmärkte

Zusammenhänge zwischen Features können sich über unterschiedliche Marktregime hinweg verändern.

### Begrenzte Datenbasis

Verwendet werden ausschließlich die neun OFR-Variablen.

Weitere makroökonomische, fundamentale oder textbasierte Informationen werden nicht einbezogen.

### Fester Prognosehorizont

Das Projekt verwendet einen festen Prognosehorizont von fünf Handelstagen.

### Festes Eingabefenster

Alle Modelle verwenden 60 historische Handelstage.

### Attention ist keine Kausalität

Attention-Gewichte erlauben Einblicke in interne Modellbeziehungen, beweisen aber keine Ursache-Wirkungs-Beziehungen.

### Synthetische Robustheitsanalyse

Gaußsches Rauschen bildet reale Finanzkrisen oder strukturelle Marktbrüche nicht vollständig ab.

### Keine Trading-Evaluation

Es werden keine Renditen, Transaktionskosten, Slippage, Drawdowns oder Portfolio-Kennzahlen untersucht.

---

## 22. Fairness

Der OFR-Datensatz enthält aggregierte Finanzmarktinformationen und keine personenbezogenen Merkmale.

Klassische demografische Fairnessmetriken sind deshalb für die aktuelle Aufgabenstellung nicht direkt anwendbar.

Sollten zukünftige Systeme die Modelloutputs für Entscheidungen über einzelne Personen verwenden, beispielsweise bei Kredit- oder Versicherungsentscheidungen, wäre eine separate Fairnessanalyse erforderlich.

---

## 23. Responsible AI und Sicherheit

Das Modell darf nicht als autonomes Finanzentscheidungssystem verstanden werden.

Vorhersagen können falsch sein.

Für eine produktive Anwendung wären zusätzliche technische und organisatorische Maßnahmen notwendig, beispielsweise:

- Data Quality Monitoring
- Model Monitoring
- Drift Detection
- Model Versioning
- reproduzierbare Inferenz
- Confidence Monitoring
- Fallback-Regeln
- menschliche Kontrolle bei kritischen Entscheidungen

---

## 24. Nächste Schritte

### 1. Stress-Increase-Erkennung verbessern

Die wichtigste Weiterentwicklung ist die Verbesserung des Recalls für steigenden Finanzstress.

Mögliche Ansätze sind zusätzliche Features, andere historische Fenstergrößen, alternative Zielgrenzen oder weitere Sequenzmodelle.

### 2. Datenbasis erweitern

Zukünftige Versionen könnten zusätzliche Finanz- und Makrodaten integrieren, beispielsweise:

- Leitzinsen
- Staatsanleiherenditen
- Zinsstrukturkurven
- Inflation
- Arbeitsmarktdaten
- Kreditmarktindikatoren
- zusätzliche Volatilitätsdaten

### 3. Walk-Forward-Evaluation

Eine spätere Version sollte mehrere chronologisch aufeinanderfolgende Trainings- und Testperioden verwenden.

Dadurch könnte die Stabilität des Modells über unterschiedliche Marktregime hinweg realistischer untersucht werden.

---

## 25. Technischer Stack

- Python 3.13
- PyTorch 2.13
- NumPy
- pandas
- scikit-learn
- Matplotlib
- seaborn
- pytest
- certifi
- Git
- GitHub
- Apple Metal Performance Shaders

---

## 26. Reproduzierbarkeit

Das Projekt verwendet einen festen Random Seed von:

```text
42
```

Weitere Maßnahmen zur Reproduzierbarkeit:

- chronologische Datenaufteilung
- train-only Scaling
- fest definierte Zielgrenzen
- dokumentierte Modellarchitekturen
- dokumentierte Hyperparameter
- gespeicherte Trainingsverläufe
- gespeicherte Checkpoints
- reproduzierbare Tuning-Runs
- automatisierte Tests

Aktueller Teststand:

```text
58 passed
```

---

## 27. Projektabgrenzung

Das Projekt ist ein Deep-Learning-Forschungsprototyp für Financial AI.

Es ist ausdrücklich:

- kein autonomes Handelssystem
- keine Anlageberatung
- keine produktive Risikoplattform
- kein Nachweis einer profitablen Trading-Strategie

Der Schwerpunkt liegt auf einem nachvollziehbaren, reproduzierbaren und kritisch evaluierten Deep-Learning-Workflow.

---

## 28. Fazit

Das Projekt zeigt einen vollständigen Deep-Learning-Prozess für eine reale Finanzzeitreihen-Klassifikation.

Umgesetzt wurden:

- Datenbeschaffung
- explorative Datenanalyse
- chronologische Datenaufteilung
- Leakage-Vermeidung
- train-only Scaling
- Sliding Windows
- PyTorch Dataset und DataLoader
- MLP-Baseline
- LSTM
- Transformer
- Early Stopping
- Hyperparameter-Tuning
- Multi-Class-Metriken
- Confusion Matrices
- Modellvergleich
- Explainability
- Fehleranalyse
- Robustheitsanalyse
- Limitationen
- Fairness-Einordnung
- Responsible AI
- konkrete nächste Entwicklungsschritte

Das LSTM erreicht auf dem unbekannten Testzeitraum den besten Macro-F1.

Der Transformer erreicht unter den neuronalen Modellen den höchsten Recall für steigenden Finanzstress.

Damit liefert das Projekt gleichzeitig eine wichtige fachliche Erkenntnis:

> Eine komplexere Deep-Learning-Architektur ist nicht automatisch die beste Architektur für unbekannte zukünftige Finanzdaten.

---

## 29. Quellen

### Datensatz

Office of Financial Research (OFR) — Financial Stress Index

https://www.financialresearch.gov/

### Technische Grundlage

PyTorch: https://pytorch.org/

scikit-learn: https://scikit-learn.org/

### Projekt-Repository

https://github.com/Maik-Huebner/deep-learning-critical-systems
