"""Zeitliche Generalisierungsanalyse der finalen Modelle.

Die bereits festgelegten Modelle MLP, LSTM und Transformer werden
getrennt für jedes Kalenderjahr des gehaltenen Testzeitraums
ausgewertet.

Diese Analyse ist rein beschreibend. Sie wird nicht für weitere
Modellauswahl, Hyperparameter-Tuning oder Retraining verwendet.

Das Jahr 2026 ist ein Teiljahr, da der eingefrorene OFR-Datensatz
am 05.08.2026 endet und der letzte verfügbare Vorhersagetag
der 29.07.2026 ist.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from deep_learning_critical_systems.data.datasets import (
    create_data_loaders,
)
from deep_learning_critical_systems.data.prepare_ofr_fsi import (
    CLASS_NAMES,
    prepare_ofr_data,
)
from deep_learning_critical_systems.evaluation.evaluate_lstm import (
    load_model as load_lstm_model,
    predict as predict_lstm,
)
from deep_learning_critical_systems.evaluation.evaluate_mlp import (
    load_model as load_mlp_model,
    predict as predict_mlp,
)
from deep_learning_critical_systems.evaluation.evaluate_transformer import (
    CHECKPOINT_PATH as TRANSFORMER_CHECKPOINT_PATH,
    collect_predictions as predict_transformer,
    load_model as load_transformer_model,
)
from deep_learning_critical_systems.training.trainer import (
    select_device,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

LOG_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "logs"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
)

FIGURE_DIR = (
    REPORT_DIR
    / "figures"
)

RESULTS_JSON_PATH = (
    LOG_DIR
    / "temporal_generalization.json"
)

RESULTS_CSV_PATH = (
    LOG_DIR
    / "temporal_generalization.csv"
)

REPORT_PATH = (
    REPORT_DIR
    / "temporal_generalization.md"
)

MACRO_F1_FIGURE_PATH = (
    FIGURE_DIR
    / "temporal_generalization_macro_f1.png"
)

INCREASE_RECALL_FIGURE_PATH = (
    FIGURE_DIR
    / "temporal_generalization_increase_recall.png"
)

BATCH_SIZE = 64

MODEL_ORDER = [
    "MLP",
    "LSTM",
    "Transformer",
]


def calculate_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Berechne vergleichbare Klassifikationsmetriken."""

    class_recalls = recall_score(
        targets,
        predictions,
        labels=[
            0,
            1,
            2,
        ],
        average=None,
        zero_division=0,
    )

    return {
        "accuracy": float(
            accuracy_score(
                targets,
                predictions,
            )
        ),
        "macro_precision": float(
            precision_score(
                targets,
                predictions,
                labels=[
                    0,
                    1,
                    2,
                ],
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                targets,
                predictions,
                labels=[
                    0,
                    1,
                    2,
                ],
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=[
                    0,
                    1,
                    2,
                ],
                average="macro",
                zero_division=0,
            )
        ),
        "recall_stress_decrease": float(
            class_recalls[0]
        ),
        "recall_stable": float(
            class_recalls[1]
        ),
        "recall_stress_increase": float(
            class_recalls[2]
        ),
    }


def collect_model_predictions(
    test_loader,
    device,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
]:
    """Erzeuge Vorhersagen aller drei finalen Modelle."""

    mlp_model = load_mlp_model(
        device
    )

    (
        mlp_targets,
        mlp_predictions,
    ) = predict_mlp(
        model=mlp_model,
        test_loader=test_loader,
        device=device,
    )

    (
        lstm_model,
        _,
    ) = load_lstm_model(
        device
    )

    (
        lstm_targets,
        lstm_predictions,
    ) = predict_lstm(
        model=lstm_model,
        test_loader=test_loader,
        device=device,
    )

    (
        transformer_model,
        _,
    ) = load_transformer_model(
        TRANSFORMER_CHECKPOINT_PATH,
        device,
    )

    (
        transformer_targets,
        transformer_predictions,
    ) = predict_transformer(
        model=transformer_model,
        data_loader=test_loader,
        device=device,
    )

    if not np.array_equal(
        mlp_targets,
        lstm_targets,
    ):
        raise RuntimeError(
            "MLP- und LSTM-Testziele stimmen nicht überein."
        )

    if not np.array_equal(
        mlp_targets,
        transformer_targets,
    ):
        raise RuntimeError(
            "MLP- und Transformer-Testziele stimmen nicht überein."
        )

    return (
        mlp_targets,
        {
            "MLP": mlp_predictions,
            "LSTM": lstm_predictions,
            "Transformer": transformer_predictions,
        },
    )


def build_yearly_results(
    dates: pd.DatetimeIndex,
    targets: np.ndarray,
    predictions_by_model: dict[str, np.ndarray],
) -> list[dict]:
    """Erzeuge Jahresmetriken für alle finalen Modelle."""

    results = []

    years = sorted(
        dates.year.unique()
    )

    final_year = max(
        years
    )

    for year in years:
        mask = np.asarray(
            dates.year == year
        )

        year_dates = dates[
            mask
        ]

        year_targets = targets[
            mask
        ]

        supports = np.bincount(
            year_targets,
            minlength=len(
                CLASS_NAMES
            ),
        )

        partial_year = (
            year == final_year
            and (
                year_dates.max().month < 12
                or year_dates.max().day < 31
            )
        )

        for model_name in MODEL_ORDER:
            year_predictions = (
                predictions_by_model[
                    model_name
                ][
                    mask
                ]
            )

            metrics = calculate_metrics(
                year_targets,
                year_predictions,
            )

            results.append(
                {
                    "year": int(
                        year
                    ),
                    "model": model_name,
                    "partial_year": bool(
                        partial_year
                    ),
                    "period_start": (
                        year_dates
                        .min()
                        .strftime(
                            "%Y-%m-%d"
                        )
                    ),
                    "period_end": (
                        year_dates
                        .max()
                        .strftime(
                            "%Y-%m-%d"
                        )
                    ),
                    "sample_count": int(
                        len(
                            year_targets
                        )
                    ),
                    "support_stress_decrease": int(
                        supports[0]
                    ),
                    "support_stable": int(
                        supports[1]
                    ),
                    "support_stress_increase": int(
                        supports[2]
                    ),
                    **metrics,
                }
            )

    return results


def save_json_and_csv(
    results: list[dict],
) -> None:
    """Speichere die Jahresergebnisse als JSON und CSV."""

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "analysis": (
            "Zeitliche Generalisierung "
            "der finalen Testmodelle"
        ),
        "purpose": (
            "Beschreibende Post-hoc-Analyse; "
            "keine weitere Modellauswahl oder Optimierung."
        ),
        "models": MODEL_ORDER,
        "results": results,
    }

    RESULTS_JSON_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with RESULTS_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            results
        )

    print(
        "Gespeichert:",
        RESULTS_JSON_PATH,
    )

    print(
        "Gespeichert:",
        RESULTS_CSV_PATH,
    )


def plot_metric(
    results: list[dict],
    metric_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Erzeuge eine Jahresgrafik für eine ausgewählte Metrik."""

    years = sorted(
        {
            result[
                "year"
            ]
            for result in results
        }
    )

    final_year = max(
        years
    )

    figure, axis = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    for model_name in MODEL_ORDER:
        model_results = [
            result
            for result in results
            if result[
                "model"
            ]
            == model_name
        ]

        model_results = sorted(
            model_results,
            key=lambda result: result[
                "year"
            ],
        )

        axis.plot(
            [
                result[
                    "year"
                ]
                for result in model_results
            ],
            [
                result[
                    metric_key
                ]
                * 100
                for result in model_results
            ],
            marker="o",
            label=model_name,
        )

    axis.set_title(
        title
    )

    axis.set_xlabel(
        "Testjahr"
    )

    axis.set_ylabel(
        ylabel
    )

    axis.set_xticks(
        years
    )

    axis.set_xticklabels(
        [
            (
                f"{year}*"
                if year == final_year
                else str(
                    year
                )
            )
            for year in years
        ]
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend()

    axis.text(
        0.01,
        -0.16,
        "* 2026 ist ein Teiljahr.",
        transform=axis.transAxes,
    )

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        "Gespeichert:",
        output_path,
    )


def save_markdown_report(
    results: list[dict],
) -> None:
    """Erzeuge einen deutschen Kurzbericht zur Jahresanalyse."""

    lines = [
        "# Zeitliche Generalisierung 2020–2026",
        "",
        "Diese Analyse untersucht die bereits final festgelegten "
        "Modelle getrennt nach Kalenderjahr im gehaltenen Testzeitraum.",
        "",
        "Die Jahresanalyse dient ausschließlich der Beschreibung der "
        "zeitlichen Generalisierung. Sie wurde **nicht** für weitere "
        "Modellauswahl, Hyperparameter-Optimierung oder Retraining genutzt.",
        "",
        "Das Jahr **2026 ist ein Teiljahr**. Der eingefrorene OFR-Datensatz "
        "endet am 05.08.2026; der letzte verfügbare Vorhersagetag ist "
        "der 29.07.2026.",
        "",
        "## Jahresergebnisse",
        "",
        "| Jahr | Modell | Fälle | Accuracy | Macro-F1 | Stress-Increase-Recall |",
        "|---:|---|---:|---:|---:|---:|",
    ]

    for result in results:
        year_label = (
            f"{result['year']}*"
            if result[
                "partial_year"
            ]
            else str(
                result[
                    "year"
                ]
            )
        )

        lines.append(
            "| "
            f"{year_label} | "
            f"{result['model']} | "
            f"{result['sample_count']} | "
            f"{result['accuracy'] * 100:.2f} % | "
            f"{result['macro_f1'] * 100:.2f} % | "
            f"{result['recall_stress_increase'] * 100:.2f} % |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Die jahresweise Betrachtung macht sichtbar, ob die Modelle "
            "über unterschiedliche Marktphasen hinweg ähnlich stabil "
            "arbeiten oder ob ihre Leistung zeitlich deutlich schwankt.",
            "",
            "Eine solche Schwankung ist bei Finanzzeitreihen besonders "
            "relevant, weil sich Marktregime, Volatilität und strukturelle "
            "Zusammenhänge über die Zeit verändern können.",
            "",
            "Die Jahresmetriken werden deshalb als Ergänzung zur "
            "Gesamt-Testauswertung verstanden und nicht als Grundlage "
            "für nachträgliche Modelloptimierung.",
            "",
            "## Abbildungen",
            "",
            "- `figures/temporal_generalization_macro_f1.png`",
            "- `figures/temporal_generalization_increase_recall.png`",
            "",
        ]
    )

    REPORT_PATH.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    print(
        "Gespeichert:",
        REPORT_PATH,
    )


def print_results(
    results: list[dict],
) -> None:
    """Gib die wichtigsten Jahresmetriken kompakt aus."""

    print()
    print(
        "=== ZEITLICHE GENERALISIERUNG 2020–2026 ==="
    )

    print()

    print(
        "Jahr | Modell      | Fälle | Accuracy | Macro-F1 | Increase Recall"
    )

    print(
        "-" * 76
    )

    for result in results:
        year_label = (
            f"{result['year']}*"
            if result[
                "partial_year"
            ]
            else str(
                result[
                    "year"
                ]
            )
        )

        print(
            f"{year_label:>5} | "
            f"{result['model']:<11} | "
            f"{result['sample_count']:>5} | "
            f"{result['accuracy'] * 100:>7.2f}% | "
            f"{result['macro_f1'] * 100:>7.2f}% | "
            f"{result['recall_stress_increase'] * 100:>14.2f}%"
        )

    print()

    print(
        "* 2026 ist ein Teiljahr."
    )


def main() -> None:
    """Führe die vollständige zeitliche Generalisierungsanalyse aus."""

    prepared_data = prepare_ofr_data()

    (
        _,
        _,
        test_loader,
    ) = create_data_loaders(
        prepared_data,
        batch_size=BATCH_SIZE,
    )

    dates = pd.DatetimeIndex(
        pd.to_datetime(
            prepared_data.dates_test
        )
    )

    if len(
        dates
    ) != len(
        prepared_data.y_test
    ):
        raise RuntimeError(
            "Anzahl der Testdaten stimmt nicht mit "
            "der Anzahl der Testziele überein."
        )

    device = select_device()

    print()
    print(
        "=== ANALYSE DER ZEITLICHEN GENERALISIERUNG ==="
    )

    print(
        "Gerät:",
        device,
    )

    print(
        "Testfälle gesamt:",
        len(
            prepared_data.y_test
        ),
    )

    print(
        "Testzeitraum:",
        dates.min().strftime(
            "%Y-%m-%d"
        ),
        "bis",
        dates.max().strftime(
            "%Y-%m-%d"
        ),
    )

    print(
        "Weitere Modellauswahl:",
        "NEIN",
    )

    print(
        "Weitere Hyperparameter-Optimierung:",
        "NEIN",
    )

    (
        targets,
        predictions_by_model,
    ) = collect_model_predictions(
        test_loader,
        device,
    )

    results = build_yearly_results(
        dates=dates,
        targets=targets,
        predictions_by_model=predictions_by_model,
    )

    print_results(
        results
    )

    save_json_and_csv(
        results
    )

    plot_metric(
        results=results,
        metric_key="macro_f1",
        title=(
            "Zeitliche Generalisierung – Macro-F1 nach Testjahr"
        ),
        ylabel="Macro-F1 (%)",
        output_path=MACRO_F1_FIGURE_PATH,
    )

    plot_metric(
        results=results,
        metric_key="recall_stress_increase",
        title=(
            "Zeitliche Generalisierung – "
            "Recall für Stressanstiege nach Testjahr"
        ),
        ylabel="Stress-Increase-Recall (%)",
        output_path=INCREASE_RECALL_FIGURE_PATH,
    )

    save_markdown_report(
        results
    )

    print()
    print(
        "Zeitliche Generalisierungsanalyse abgeschlossen."
    )


if __name__ == "__main__":
    main()
