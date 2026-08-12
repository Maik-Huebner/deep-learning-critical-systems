"""ROC-AUC-Auswertung der finalen neuronalen Modelle.

Die bereits final ausgewählten Modelle MLP, LSTM und Transformer
werden auf dem gehaltenen Testdatensatz mit Multi-Class ROC-AUC
ausgewertet.

Die Analyse ist rein beschreibend und wird nicht für weitere
Modellauswahl, Hyperparameter-Optimierung oder Retraining verwendet.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
)

from deep_learning_critical_systems.data.datasets import (
    create_data_loaders,
)
from deep_learning_critical_systems.data.prepare_ofr_fsi import (
    prepare_ofr_data,
)
from deep_learning_critical_systems.evaluation.evaluate_lstm import (
    load_model as load_lstm_model,
)
from deep_learning_critical_systems.evaluation.evaluate_mlp import (
    load_model as load_mlp_model,
)
from deep_learning_critical_systems.evaluation.evaluate_transformer import (
    CHECKPOINT_PATH as TRANSFORMER_CHECKPOINT_PATH,
    load_model as load_transformer_model,
)
from deep_learning_critical_systems.training.trainer import (
    select_device,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

LOG_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "logs"
    / "roc_auc_test.json"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "roc_auc.md"
)

FIGURE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "roc_auc_comparison.png"
)

CLASS_NAMES_DE = [
    "Stressrückgang",
    "Stabil",
    "Stressanstieg",
]

MODEL_ORDER = [
    "MLP",
    "LSTM",
    "Transformer",
]

BATCH_SIZE = 64


def collect_probabilities(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Sammle Testziele und vorhergesagte Klassenwahrscheinlichkeiten."""

    targets = []
    probabilities = []

    model.eval()

    with torch.no_grad():
        for features, batch_targets in data_loader:
            features = features.to(
                device
            )

            logits = model(
                features
            )

            batch_probabilities = torch.softmax(
                logits,
                dim=1,
            )

            targets.extend(
                batch_targets.numpy().tolist()
            )

            probabilities.extend(
                batch_probabilities
                .cpu()
                .numpy()
                .tolist()
            )

    return (
        np.asarray(
            targets,
            dtype=np.int64,
        ),
        np.asarray(
            probabilities,
            dtype=np.float64,
        ),
    )


def calculate_roc_auc(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """Berechne Macro-OvR-ROC-AUC und klassenweise ROC-AUC-Werte."""

    per_class = {}

    for class_id, class_name in enumerate(
        CLASS_NAMES_DE
    ):
        binary_targets = (
            targets == class_id
        ).astype(
            np.int64
        )

        class_auc = roc_auc_score(
            binary_targets,
            probabilities[
                :,
                class_id,
            ],
        )

        per_class[
            class_name
        ] = float(
            class_auc
        )

    macro_auc = roc_auc_score(
        targets,
        probabilities,
        multi_class="ovr",
        average="macro",
    )

    return {
        "macro_ovr_roc_auc": float(
            macro_auc
        ),
        "per_class_roc_auc": per_class,
    }


def load_final_models(
    device: torch.device,
) -> dict[str, torch.nn.Module]:
    """Lade die drei finalen Modelle."""

    mlp_model = load_mlp_model(
        device
    )

    (
        lstm_model,
        _,
    ) = load_lstm_model(
        device
    )

    (
        transformer_model,
        _,
    ) = load_transformer_model(
        TRANSFORMER_CHECKPOINT_PATH,
        device,
    )

    return {
        "MLP": mlp_model,
        "LSTM": lstm_model,
        "Transformer": transformer_model,
    }


def save_results(
    results: dict[str, dict],
    sample_count: int,
) -> None:
    """Speichere die ROC-AUC-Ergebnisse als JSON."""

    LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "analysis": (
            "ROC-AUC-Auswertung der finalen Testmodelle"
        ),
        "method": (
            "Multi-Class One-vs-Rest ROC-AUC"
        ),
        "test_samples": sample_count,
        "used_for_model_selection": False,
        "results": results,
    }

    LOG_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "Gespeichert:",
        LOG_PATH,
    )


def save_report(
    results: dict[str, dict],
    sample_count: int,
) -> None:
    """Erzeuge einen deutschen Markdown-Kurzbericht."""

    lines = [
        "# ROC-AUC-Auswertung",
        "",
        "Die finalen Modelle MLP, LSTM und Transformer wurden zusätzlich "
        "mit Multi-Class ROC-AUC nach dem One-vs-Rest-Verfahren (OvR) "
        "auf dem gehaltenen Testdatensatz bewertet.",
        "",
        f"**Testfälle:** {sample_count}",
        "",
        "Diese Auswertung ist ausschließlich eine nachgelagerte "
        "Beschreibung der bereits festgelegten Modelle. Die Ergebnisse "
        "wurden **nicht** für weitere Modellauswahl, Hyperparameter-"
        "Optimierung oder Retraining verwendet.",
        "",
        "## Ergebnisse",
        "",
        "| Modell | Macro ROC-AUC | Stressrückgang | Stabil | Stressanstieg |",
        "|---|---:|---:|---:|---:|",
    ]

    for model_name in MODEL_ORDER:
        model_result = results[
            model_name
        ]

        class_result = model_result[
            "per_class_roc_auc"
        ]

        lines.append(
            "| "
            f"{model_name} | "
            f"{model_result['macro_ovr_roc_auc']:.4f} | "
            f"{class_result['Stressrückgang']:.4f} | "
            f"{class_result['Stabil']:.4f} | "
            f"{class_result['Stressanstieg']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Einordnung",
            "",
            "ROC-AUC ergänzt Accuracy, Precision, Recall und Macro-F1 um "
            "eine schwellenunabhängige Betrachtung der Trennfähigkeit.",
            "",
            "Ein ROC-AUC-Wert von 0,5 entspricht näherungsweise einer "
            "zufälligen Rangordnung. Höhere Werte zeigen, dass das Modell "
            "positive und negative Fälle einer Klasse über verschiedene "
            "Entscheidungsschwellen hinweg besser voneinander trennt.",
            "",
            "Die ROC-AUC-Ergebnisse ersetzen die bisherigen Metriken nicht. "
            "Insbesondere bei diesem Projekt bleiben Macro-F1 und der Recall "
            "für Stressanstiege wichtig, weil die tatsächliche harte "
            "Drei-Klassen-Entscheidung für die fachliche Bewertung relevant ist.",
            "",
            "Die Majority-Class-Baseline wird hier nicht aufgenommen, da sie "
            "keine modellierten Klassenwahrscheinlichkeiten bereitstellt.",
            "",
            "## Abbildung",
            "",
            "- `figures/roc_auc_comparison.png`",
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


def save_figure(
    targets: np.ndarray,
    probabilities_by_model: dict[str, np.ndarray],
    results: dict[str, dict],
) -> None:
    """Erzeuge ROC-Kurven für alle drei finalen Modelle."""

    FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(
            18,
            5.5,
        ),
        sharex=True,
        sharey=True,
    )

    for axis, model_name in zip(
        axes,
        MODEL_ORDER,
        strict=True,
    ):
        probabilities = probabilities_by_model[
            model_name
        ]

        for class_id, class_name in enumerate(
            CLASS_NAMES_DE
        ):
            binary_targets = (
                targets == class_id
            ).astype(
                np.int64
            )

            false_positive_rate, true_positive_rate, _ = (
                roc_curve(
                    binary_targets,
                    probabilities[
                        :,
                        class_id,
                    ],
                )
            )

            class_auc = results[
                model_name
            ][
                "per_class_roc_auc"
            ][
                class_name
            ]

            axis.plot(
                false_positive_rate,
                true_positive_rate,
                label=(
                    f"{class_name} "
                    f"(AUC {class_auc:.3f})"
                ),
            )

        axis.plot(
            [
                0,
                1,
            ],
            [
                0,
                1,
            ],
            linestyle="--",
            linewidth=1,
            label="Zufallsniveau",
        )

        macro_auc = results[
            model_name
        ][
            "macro_ovr_roc_auc"
        ]

        axis.set_title(
            f"{model_name}\n"
            f"Macro ROC-AUC: {macro_auc:.3f}"
        )

        axis.set_xlabel(
            "Falsch-Positiv-Rate"
        )

        axis.grid(
            alpha=0.25
        )

        axis.legend(
            fontsize=8
        )

    axes[0].set_ylabel(
        "Richtig-Positiv-Rate"
    )

    figure.suptitle(
        "ROC-AUC der finalen Modelle auf dem Testdatensatz",
        fontsize=14,
    )

    figure.tight_layout()

    figure.savefig(
        FIGURE_PATH,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        "Gespeichert:",
        FIGURE_PATH,
    )


def print_results(
    results: dict[str, dict],
) -> None:
    """Gib die ROC-AUC-Ergebnisse kompakt aus."""

    print()
    print(
        "=== ROC-AUC-AUSWERTUNG ==="
    )

    print()

    print(
        "Modell      | Macro AUC | Rückgang | Stabil | Anstieg"
    )

    print(
        "-" * 62
    )

    for model_name in MODEL_ORDER:
        model_result = results[
            model_name
        ]

        class_result = model_result[
            "per_class_roc_auc"
        ]

        print(
            f"{model_name:<11} | "
            f"{model_result['macro_ovr_roc_auc']:.4f}    | "
            f"{class_result['Stressrückgang']:.4f}   | "
            f"{class_result['Stabil']:.4f} | "
            f"{class_result['Stressanstieg']:.4f}"
        )


def main() -> None:
    """Führe die vollständige ROC-AUC-Auswertung aus."""

    prepared_data = prepare_ofr_data()

    (
        _,
        _,
        test_loader,
    ) = create_data_loaders(
        prepared_data,
        batch_size=BATCH_SIZE,
    )

    device = select_device()

    print()
    print(
        "=== ROC-AUC-ANALYSE DER FINALEN MODELLE ==="
    )

    print(
        "Gerät:",
        device,
    )

    print(
        "Testfälle:",
        len(
            prepared_data.y_test
        ),
    )

    print(
        "Weitere Modellauswahl: NEIN"
    )

    print(
        "Weitere Hyperparameter-Optimierung: NEIN"
    )

    models = load_final_models(
        device
    )

    common_targets = None
    probabilities_by_model = {}
    results = {}

    for model_name in MODEL_ORDER:
        (
            targets,
            probabilities,
        ) = collect_probabilities(
            model=models[
                model_name
            ],
            data_loader=test_loader,
            device=device,
        )

        if common_targets is None:
            common_targets = targets
        elif not np.array_equal(
            common_targets,
            targets,
        ):
            raise RuntimeError(
                "Die Testziele der Modelle stimmen nicht überein."
            )

        probabilities_by_model[
            model_name
        ] = probabilities

        results[
            model_name
        ] = calculate_roc_auc(
            targets,
            probabilities,
        )

    if common_targets is None:
        raise RuntimeError(
            "Es wurden keine Testvorhersagen erzeugt."
        )

    print_results(
        results
    )

    save_results(
        results=results,
        sample_count=len(
            common_targets
        ),
    )

    save_report(
        results=results,
        sample_count=len(
            common_targets
        ),
    )

    save_figure(
        targets=common_targets,
        probabilities_by_model=probabilities_by_model,
        results=results,
    )

    print()
    print(
        "ROC-AUC-Auswertung abgeschlossen."
    )


if __name__ == "__main__":
    main()
