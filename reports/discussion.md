# Diskussion, Limitationen und Robustheit

## 1. Projektziel

Dieses Projekt untersucht, ob ein PyTorch-Transformer anhand der
vergangenen 60 Handelstage multivariater Finanzstressdaten vorhersagen
kann, ob sich der Finanzstress in den folgenden fünf Handelstagen:

- verringert,
- stabil bleibt oder
- erhöht.

Als Datengrundlage wird der Financial Stress Index des
Office of Financial Research (OFR) verwendet.

Das Modell verarbeitet neun numerische Merkmale:

- OFR FSI
- Credit
- Equity valuation
- Safe assets
- Funding
- Volatility
- United States
- Other advanced economies
- Emerging markets

Die Zielklassen lauten:

- `0 = Stress Decrease`
- `1 = Stable`
- `2 = Stress Increase`

Neben dem Transformer wurden zwei Vergleichsmodelle entwickelt:

- MLP als einfache Baseline
- LSTM als sequenzielles Vergleichsmodell
- Transformer als Hauptmodell

---

## 2. Methodischer Aufbau

Die Daten wurden chronologisch aufgeteilt.

- Training: bis Ende 2016
- Validation: 2017 bis Ende 2019
- Test: ab 2020

Es wurde bewusst kein zufälliger Train-Test-Split verwendet.

Bei Finanzzeitreihen könnte ein zufälliger Split dazu führen, dass
Informationen aus späteren Marktphasen indirekt in das Training
gelangen.

Der StandardScaler wurde ausschließlich auf den Trainingsdaten
angepasst.

Validation- und Testdaten wurden anschließend nur mit den auf dem
Training gelernten Skalierungsparametern transformiert.

Die Eingabesequenzen besitzen eine Länge von 60 Handelstagen.

Für Validation und Test wurden ausschließlich bereits bekannte
historische Beobachtungen als Kontext verwendet. Dadurch konnten auch
die ersten Tage eines neuen Datenabschnitts vorhergesagt werden, ohne
zukünftige Informationen zu verwenden.

---

## 3. Zielvariable

Die Zielvariable basiert auf der Veränderung des OFR Financial Stress
Index über die folgenden fünf Handelstage.

Dazu wird der Mittelwert der nächsten fünf OFR-FSI-Werte berechnet und
mit dem aktuellen OFR-FSI-Wert verglichen.

Die Klassengrenzen wurden ausschließlich aus dem Trainingszeitraum
bestimmt.

Dadurch wird verhindert, dass Informationen aus Validation oder Test
für die Definition der Zielklassen verwendet werden.

Die drei Zielklassen sind:

- Stress Decrease
- Stable
- Stress Increase

---

## 4. Modellvergleich

Die drei neuronalen Modelle wurden auf demselben chronologisch
zurückgehaltenen Testzeitraum verglichen.

| Modell | Accuracy | Macro-F1 | Recall Stress Increase |
|---|---:|---:|---:|
| MLP | 43,92 % | 33,44 % | 0,37 % |
| LSTM | 44,33 % | 37,90 % | 6,15 % |
| Transformer | 42,44 % | 37,38 % | 8,19 % |
| Majority-Baseline | 30,64 % | 15,63 % | 0,00 % |

Das LSTM erreicht die höchste Test-Accuracy und den höchsten
Test-Macro-F1.

Der Transformer erreicht dagegen den höchsten Recall für die besonders
schwierige Klasse `Stress Increase`.

Das Ergebnis zeigt, dass ein komplexeres Modell nicht automatisch eine
bessere Generalisierung liefert.

Der Transformer besitzt mehr Möglichkeiten zur Modellierung komplexer
zeitlicher Zusammenhänge, erreicht im unbekannten Testzeitraum aber
nicht automatisch bessere Gesamtergebnisse als das LSTM.

---

## 5. MLP-Baseline

Das MLP dient als einfache Referenz.

Die 60 Handelstage mit jeweils neun Merkmalen werden zu einem flachen
Eingabevektor zusammengefasst.

Das Modell besitzt keine spezielle Architektur zur Verarbeitung
zeitlicher Abhängigkeiten.

Testresultate:

- Accuracy: 43,92 %
- Macro-F1: 33,44 %
- Recall Stress Increase: 0,37 %

Das Modell erkennt die Klasse `Stress Increase` nahezu gar nicht.

Die MLP-Baseline zeigt damit, dass die reine Verarbeitung des gesamten
Fensters als flacher Vektor für diese Aufgabe nur begrenzt geeignet ist.

---

## 6. LSTM

Das LSTM verarbeitet die 60 Handelstage als echte Sequenz.

Dadurch kann das Modell zeitliche Beziehungen innerhalb des
historischen Fensters berücksichtigen.

Testresultate:

- Accuracy: 44,33 %
- Macro-F1: 37,90 %
- Recall Stress Increase: 6,15 %

Das LSTM erreicht den besten Gesamtwert beim Macro-F1.

Im Vergleich zum MLP verbessert es insbesondere die Erkennung der
Klasse `Stress Increase`.

Trotzdem bleibt auch beim LSTM die Erkennung steigenden Finanzstresses
schwach.

---

## 7. Transformer

Der Transformer verwendet:

- lineare Input-Projektion
- sinusoidales Positional Encoding
- Multi-Head Self-Attention
- Residual Connections
- Layer Normalization
- Feed-Forward-Netzwerke
- Dropout
- Mean Pooling
- Klassifikationskopf

Die finale Architektur besitzt:

- Model Dimension: 64
- Attention Heads: 4
- Encoder Layers: 2
- Feed-Forward Size: 128
- Dropout: 0,20
- Learning Rate: 0,0005
- Batch Size: 64
- trainierbare Parameter: 69.763

---

## 8. Hyperparameter-Tuning

Das Hyperparameter-Tuning wurde ausschließlich mit Trainings- und
Validierungsdaten durchgeführt.

Das Testset wurde während der Modellauswahl nicht verwendet.

Die ursprüngliche Transformer-Konfiguration T0 war:

- Model Dimension: 64
- Attention Heads: 4
- Encoder Layers: 2
- Feed-Forward Size: 128
- Dropout: 0,20
- Learning Rate: 0,001

T0 erreichte:

- Validation Accuracy: 48,46 %
- Validation Macro-F1: 39,63 %
- Recall Stress Increase: 7,04 %

### Stage 1

In der ersten Tuning-Stufe wurde jeweils ein zentraler Hyperparameter
gegenüber der Ausgangskonfiguration verändert.

Untersucht wurden:

- Learning Rate
- Model Dimension
- Anzahl Attention Heads
- Anzahl Encoder Layers
- Feed-Forward-Größe
- Dropout

Die beste Verbesserung des Macro-F1 entstand durch eine Reduktion der
Learning Rate von 0,001 auf 0,0005.

Diese Konfiguration wurde als T1 bezeichnet.

T1 erreichte:

- Validation Accuracy: 47,66 %
- Validation Macro-F1: 41,91 %
- Recall Stress Increase: 18,31 %

### Stage 2

In der zweiten Tuning-Stufe wurde der Bereich um T1 genauer untersucht.

Getestet wurden unter anderem:

- Learning Rate 0,00025
- Learning Rate 0,00075
- Learning Rate 0,0005 mit geringerem Dropout
- Learning Rate 0,0005 mit nur einem Encoder-Layer
- Kombination aus geringerem Dropout und nur einem Encoder-Layer

Der knappste Konkurrent war T9.

T9 erreichte:

- Validation Accuracy: 47,66 %
- Validation Macro-F1: 41,73 %
- Recall Stress Increase: 20,66 %

T1 blieb trotzdem die finale Konfiguration.

Der Grund ist, dass vor dem finalen Test festgelegt wurde, den
Validation Macro-F1 als primäres Auswahlkriterium zu verwenden.

Die Auswahlregel wurde nachträglich nicht verändert.

---

## 9. Warum Macro-F1 wichtig ist

Die drei Zielklassen werden vom Modell sehr unterschiedlich gut
erkannt.

Eine reine Accuracy kann deshalb irreführend sein.

Ein Modell könnte beispielsweise häufig die Klasse `Stable`
vorhersagen und dadurch eine akzeptable Accuracy erreichen, obwohl
andere Klassen kaum erkannt werden.

Macro-F1 berechnet den F1-Wert für jede Klasse separat und behandelt
anschließend alle Klassen gleichgewichtet.

Dadurch ist die Kennzahl für dieses Projekt aussagekräftiger als die
Accuracy allein.

---

## 10. Generalisierung

Der finale Transformer erreichte:

- Validation Macro-F1: 41,91 %
- Test Macro-F1: 37,38 %

Die niedrigere Leistung im späteren Testzeitraum zeigt, dass die
Generalisierung nicht perfekt ist.

Das ist bei Finanzzeitreihen besonders relevant.

Marktbeziehungen können sich durch unterschiedliche wirtschaftliche
Regime verändern.

Beispiele dafür sind:

- Finanzkrisen
- starke Zinsänderungen
- Inflationsphasen
- Pandemien
- geopolitische Schocks
- Veränderungen der Marktstruktur

Ein Muster, das im Trainingszeitraum funktioniert, muss deshalb nicht
unverändert in späteren Marktphasen bestehen bleiben.

---

## 11. Klassenspezifische Leistung des Transformers

Recall auf dem Testset:

- Stress Decrease: 70,91 %
- Stable: 48,12 %
- Stress Increase: 8,19 %

Die größte Schwäche ist eindeutig die Klasse `Stress Increase`.

Von 537 tatsächlichen Stressanstiegen wurden nur 44 korrekt erkannt.

Damit wurden 493 tatsächliche Stressanstiege nicht korrekt
klassifiziert.

Die häufigsten Fehlklassifikationen waren:

- Stable → Stress Decrease: 279
- Stress Increase → Stress Decrease: 256
- Stress Increase → Stable: 237
- Stress Decrease → Stable: 122
- Stable → Stress Increase: 52
- Stress Decrease → Stress Increase: 29

Für ein reales Finanzrisikosystem wäre insbesondere das Verpassen
steigenden Stresses problematisch.

Das Modell ist deshalb nicht produktionsreif.

---

## 12. Explainability und Attention

Für den finalen Transformer wurden Attention-Gewichte ausgewertet.

Untersucht wurden zwei reproduzierbar ausgewählte Beispiele:

1. eine korrekt erkannte `Stress Increase`-Beobachtung
2. eine falsch klassifizierte tatsächliche `Stress Increase`-Beobachtung

### Korrektes Beispiel

Das korrekt erkannte Beispiel wurde mit folgenden
Wahrscheinlichkeiten klassifiziert:

- Stress Decrease: 27,77 %
- Stable: 35,40 %
- Stress Increase: 36,83 %

Obwohl die Vorhersage korrekt war, war sich das Modell also nur knapp
sicherer für `Stress Increase` als für `Stable`.

### Falsches Beispiel

Ein tatsächlicher Stressanstieg wurde mit folgenden
Wahrscheinlichkeiten klassifiziert:

- Stress Decrease: 58,34 %
- Stable: 13,84 %
- Stress Increase: 27,81 %

Das Modell lag hier falsch und war gleichzeitig relativ sicher in
seiner falschen Entscheidung.

Das zeigt:

Eine hohe Modellkonfidenz garantiert keine korrekte Vorhersage.

---

## 13. Bedeutung der Attention-Gewichte

Die Attention-Visualisierung zeigt, wie stark unterschiedliche
Positionen des 60-Tage-Fensters innerhalb des Transformers miteinander
in Beziehung gesetzt werden.

Die Attention war bei den untersuchten Beispielen über mehrere
historische Positionen verteilt.

Es gab nicht einen einzelnen Handelstag, der die gesamte Attention
dominierte.

Wichtig:

Attention ist keine Kausalitätsanalyse.

Ein hoher Attention-Wert bedeutet nicht automatisch, dass ein
bestimmter Handelstag eine zukünftige Entwicklung verursacht hat.

Attention darf ebenfalls nicht direkt mit Feature Importance
gleichgesetzt werden.

Die Visualisierung ist daher eine Interpretationshilfe für die interne
Verarbeitung des Modells.

---

## 14. Robustheitsanalyse

Der finale, bereits eingefrorene Transformer wurde zusätzlich einem
kontrollierten Robustheitstest unterzogen.

Dazu wurde künstliches Gaußsches Rauschen zu den standardisierten
Testmerkmalen hinzugefügt.

Das Modell wurde dabei:

- nicht neu trainiert
- nicht weiter getunt
- nicht auf Basis der Robustheitsergebnisse verändert

Die verwendeten Störungsstufen waren:

- 0 %
- 5 %
- 10 %
- 20 %
- 50 %

Die Prozentangaben beziehen sich auf die standardisierte Feature-Skala.

Sie dürfen nicht als prozentuale Kurs- oder Marktbewegungen
interpretiert werden.

| Rauschen | Accuracy | Macro-F1 | Recall Stress Increase | Übereinstimmung mit Original |
|---:|---:|---:|---:|---:|
| 0 % | 42,44 % | 37,38 % | 8,19 % | 100,00 % |
| 5 % | 42,09 % | 36,88 % | 7,64 % | 96,69 % |
| 10 % | 42,38 % | 37,31 % | 8,19 % | 92,50 % |
| 20 % | 41,15 % | 35,87 % | 7,45 % | 85,66 % |
| 50 % | 39,37 % | 32,94 % | 4,66 % | 73,32 % |

Bei kleinen Störungen von 5 bis 10 % bleibt die Modellleistung
vergleichsweise stabil.

Bei 10 % Rauschen bleiben mehr als 92 % der ursprünglichen
Vorhersagen unverändert.

Bei stärkeren Störungen sinkt die Leistung zunehmend.

Bei 50 % Rauschen fällt:

- Accuracy von 42,44 % auf 39,37 %
- Macro-F1 von 37,38 % auf 32,94 %
- Recall Stress Increase von 8,19 % auf 4,66 %
- Prediction Agreement auf 73,32 %

Der Transformer besitzt damit eine gewisse Stabilität gegenüber kleinen
Eingabestörungen, reagiert aber sichtbar auf stärkere Veränderungen.

Die konkrete Verwendung künstlichen Gaußschen Rauschens ist eine
praktische Umsetzung eines Robustheitstests und wird nicht als
Kausalitäts- oder Realmarktsimulation interpretiert.

---

## 15. Limitationen

### 15.1 Begrenzte Vorhersageleistung

Die Gesamtleistung der Modelle bleibt moderat.

Selbst der beste Test-Macro-F1 liegt unter 40 %.

Das Projekt demonstriert deshalb einen vollständigen und methodisch
sauberen Deep-Learning-Prozess, aber kein produktionsreifes
Finanzprognosesystem.

### 15.2 Schwache Erkennung steigenden Finanzstresses

Die wichtigste Modellschwäche ist der geringe Recall der Klasse
`Stress Increase`.

Gerade für ein Risikofrühwarnsystem wäre diese Klasse besonders
wichtig.

### 15.3 Nichtstationäre Finanzmärkte

Finanzmärkte verändern sich im Zeitverlauf.

Historische Beziehungen zwischen Merkmalen können in zukünftigen
Marktphasen schwächer werden oder sich vollständig verändern.

### 15.4 Begrenzte Datenbasis

Das Projekt verwendet ausschließlich die neun Variablen des
OFR-FSI-Datensatzes.

Nicht enthalten sind beispielsweise:

- zusätzliche Zinsdaten
- Inflation
- Wirtschaftswachstum
- Zinsstrukturkurven
- externe Credit Spreads
- zusätzliche Volatilitätsdaten
- Unternehmensfundamentaldaten
- Nachrichten oder andere Textdaten

### 15.5 Fester Prognosehorizont

Das Projekt verwendet einen Prognosehorizont von fünf Handelstagen.

Andere Horizonte könnten zu anderen Ergebnissen führen.

### 15.6 Festes historisches Fenster

Alle Modelle verwenden 60 Handelstage als Eingabefenster.

Andere Fenstergrößen wurden innerhalb dieses Projekts nicht
systematisch untersucht.

### 15.7 Attention ist keine Kausalität

Attention-Plots ermöglichen Einblicke in die interne zeitliche
Verarbeitung.

Sie beweisen keine Ursache-Wirkungs-Beziehung.

### 15.8 Synthetischer Robustheitstest

Künstliches Gaußsches Rauschen ist nur ein kontrollierter Stresstest.

Es bildet reale Marktkrisen, strukturelle Brüche, Datenfehler oder
fehlende Daten nicht vollständig ab.

### 15.9 Keine Trading-Strategie

Das Projekt prognostiziert Finanzstressregime.

Es untersucht nicht:

- Handelsrenditen
- Transaktionskosten
- Slippage
- Positionsgrößen
- Drawdowns
- Sharpe Ratio
- Portfolioallokation

Die Ergebnisse sind daher kein Nachweis für eine profitable
Trading-Strategie.

### 15.10 Keine Produktionsumgebung

Das Projekt ist ein Forschungsprototyp.

Nicht enthalten sind beispielsweise:

- Live-Datenaufnahme
- automatisches Retraining
- Produktionsmonitoring
- Model Registry
- Drift Detection
- Echtzeit-Inferenz
- produktive Alarmierung

---

## 16. Fairness

Der verwendete OFR-Datensatz enthält aggregierte Finanzmarktdaten.

Er enthält keine personenbezogenen Merkmale wie:

- Alter
- Geschlecht
- ethnische Zugehörigkeit
- Religion
- Behinderung
- individuelles Einkommen
- persönliche Bonitätsdaten

Klassische demografische Fairnessmetriken sind deshalb für die
aktuelle Aufgabe nicht direkt anwendbar.

Fairness würde jedoch deutlich wichtiger, wenn ein zukünftiges System
die Modellvorhersagen für Entscheidungen über einzelne Personen
verwenden würde.

Das könnte beispielsweise betreffen:

- Kreditentscheidungen
- Versicherungsentscheidungen
- Kundensegmentierung
- Risikobewertungen einzelner Personen

In solchen Anwendungen müsste eine separate Fairnessanalyse
durchgeführt werden.

---

## 17. Responsible AI und Sicherheit

Das Modell darf nicht als autonomes Finanzentscheidungssystem
verstanden werden.

Vorhersagen können falsch sein.

Die Explainability-Analyse hat zusätzlich gezeigt, dass auch relativ
selbstsichere Vorhersagen falsch sein können.

Für eine zukünftige produktive Anwendung wären deshalb unter anderem
notwendig:

- Datenqualitätsprüfungen
- Modellmonitoring
- Drift Detection
- Versionsverwaltung von Modellen
- reproduzierbare Inferenz
- Überwachung der Modellkonfidenz
- klar definierte Fallback-Regeln
- menschliche Kontrolle bei kritischen Entscheidungen

---

## 18. Drei zentrale nächste Schritte

### Next Step 1: Stress-Increase-Erkennung verbessern

Die höchste Priorität ist eine bessere Erkennung steigenden
Finanzstresses.

Mögliche zukünftige Experimente:

- alternative Klassengrenzen
- zusätzliche Finanz- und Makromerkmale
- andere historische Fenstergrößen
- weitere Sequenzmodelle
- andere Ansätze zum Umgang mit schwierigen Klassen

Ziel wäre eine Verbesserung von Recall und Macro-F1 ohne Verlust der
Generalisierungsfähigkeit.

### Next Step 2: Datenbasis erweitern

Die neun OFR-Merkmale bilden nur einen Ausschnitt des Finanzsystems ab.

Eine spätere Version könnte zusätzliche Daten integrieren:

- Leitzinsen
- Staatsanleiherenditen
- Zinsstrukturkurve
- Inflation
- Arbeitsmarktdaten
- Kreditmarktindikatoren
- weitere Volatilitätsindikatoren
- makroökonomische Daten

Dadurch könnte das Modell mehr Informationen über entstehende
Finanzstressregime erhalten.

### Next Step 3: Walk-Forward-Evaluation und Monitoring

Eine spätere Version sollte nicht nur einen einzigen festen
Train-Validation-Test-Split verwenden.

Eine Walk-Forward-Evaluation könnte wiederholt:

1. auf historischen Daten trainieren,
2. auf einer späteren Phase validieren,
3. auf der nächsten unbekannten Phase testen,
4. das gesamte Zeitfenster anschließend nach vorne verschieben.

Dadurch ließe sich besser untersuchen, wie stabil das Modell über
unterschiedliche Marktregime hinweg arbeitet.

Für eine produktionsorientierte Weiterentwicklung sollte dies später
mit Daten- und Model-Drift-Monitoring kombiniert werden.

---

## 19. Fazit

Das Projekt demonstriert einen vollständigen Deep-Learning-Workflow für
eine reale Finanzzeitreihen-Klassifikationsaufgabe.

Umgesetzt wurden:

- reale Finanzmarktdaten
- explorative Datenanalyse
- chronologische Datenaufteilung
- Leakage-Vermeidung
- train-only Scaling
- Sliding-Window-Sequenzen
- MLP-Baseline
- LSTM
- PyTorch-Transformer
- Early Stopping
- Hyperparameter-Tuning
- Accuracy, Precision, Recall und F1
- Confusion Matrices
- Trainings- und Validierungskurven
- Modellvergleich
- Attention-Visualisierung
- Analyse korrekter und falscher Vorhersagen
- Robustheitsanalyse
- Limitationen
- Fairness-Einordnung
- Responsible-AI-Betrachtung
- konkrete nächste Entwicklungsschritte

Das LSTM erreicht auf dem unbekannten Testzeitraum den besten
Macro-F1.

Der Transformer erreicht innerhalb der drei neuronalen Modelle den
höchsten Recall für steigenden Finanzstress.

Damit zeigt das Projekt auch ein wichtiges Ergebnis:

Eine komplexere Modellarchitektur ist nicht automatisch die beste
Architektur für unbekannte zukünftige Daten.

Der aktuelle Stand ist deshalb als reproduzierbarer
Financial-AI-Forschungsprototyp zu verstehen und nicht als fertiges
Trading-, Investment- oder Risikoproduktionssystem.
