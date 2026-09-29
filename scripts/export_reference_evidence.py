"""Export compact reference evidence from local data and checkpoints.

This maintainer command intentionally requires ignored raw data, checkpoints and
experiment logs. Its output under ``reports/evidence`` is small and versioned.
Ordinary users and CI validate that output offline instead of needing model files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from deep_learning_critical_systems.data.datasets import create_data_loaders
from deep_learning_critical_systems.data.load_ofr_fsi import (
    OFR_DATA_URL,
    RAW_DATA_FILE,
    calculate_file_sha256,
)
from deep_learning_critical_systems.data.prepare_ofr_fsi import (
    ANALYSIS_END,
    FEATURE_COLUMNS,
    REFERENCE_SNAPSHOT_ROWS,
    REFERENCE_SNAPSHOT_SHA256,
    prepare_ofr_data,
)
from deep_learning_critical_systems.evaluation.evaluate_lstm import (
    load_model as load_lstm,
)
from deep_learning_critical_systems.evaluation.evaluate_lstm import (
    predict as predict_lstm,
)
from deep_learning_critical_systems.evaluation.evaluate_mlp import (
    load_model as load_mlp,
)
from deep_learning_critical_systems.evaluation.evaluate_mlp import (
    predict as predict_mlp,
)
from deep_learning_critical_systems.evaluation.evaluate_transformer import (
    CHECKPOINT_PATH as TRANSFORMER_CHECKPOINT,
)
from deep_learning_critical_systems.evaluation.evaluate_transformer import (
    collect_predictions as predict_transformer,
)
from deep_learning_critical_systems.evaluation.evaluate_transformer import (
    load_model as load_transformer,
)
from deep_learning_critical_systems.evaluation.evidence import EVIDENCE_FILES
from deep_learning_critical_systems.training.trainer import select_device

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_LOGS = PROJECT_ROOT / "artifacts" / "logs"
OUTPUT = PROJECT_ROOT / "reports" / "evidence"


def read_local_json(name: str) -> Any:
    """Read one ignored experiment log."""

    path = LOCAL_LOGS / name
    if not path.is_file():
        raise FileNotFoundError(f"Required local experiment log not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(name: str, payload: Any) -> None:
    """Write normalized JSON to the versioned evidence directory."""

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def model_result(targets: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    """Build a complete, internally checkable result for one model."""

    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=[0, 1, 2],
        zero_division=0,
    )
    matrix = confusion_matrix(targets, predictions, labels=[0, 1, 2])
    class_names = ("Stress Decrease", "Stable", "Stress Increase")
    return {
        "metrics": {
            "accuracy": float(accuracy_score(targets, predictions)),
            "macro_precision": float(
                precision_score(targets, predictions, average="macro", zero_division=0)
            ),
            "macro_recall": float(
                recall_score(targets, predictions, average="macro", zero_division=0)
            ),
            "macro_f1": float(
                f1_score(targets, predictions, average="macro", zero_division=0)
            ),
            "stress_increase_recall": float(recall[2]),
        },
        "per_class": {
            class_name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, class_name in enumerate(class_names)
        },
        "confusion_matrix": matrix.tolist(),
    }


def export_final_metrics() -> None:
    """Re-run all final checkpoints and export their held-out metrics."""

    prepared = prepare_ofr_data()
    _, _, test_loader = create_data_loaders(prepared, batch_size=64)
    device = select_device()

    mlp = load_mlp(device)
    targets, mlp_predictions = predict_mlp(mlp, test_loader, device)

    lstm, lstm_checkpoint = load_lstm(device)
    lstm_targets, lstm_predictions = predict_lstm(lstm, test_loader, device)

    transformer, transformer_checkpoint = load_transformer(
        TRANSFORMER_CHECKPOINT,
        device,
    )
    transformer_targets, transformer_predictions = predict_transformer(
        transformer,
        test_loader,
        device,
    )

    if not (
        np.array_equal(targets, lstm_targets)
        and np.array_equal(targets, transformer_targets)
    ):
        raise ValueError("Final evaluators did not use identical test targets")

    majority_class = int(np.bincount(prepared.y_train).argmax())
    majority_predictions = np.full_like(targets, majority_class)
    baseline = model_result(targets, majority_predictions)

    write_json(
        "final_metrics.json",
        {
            "evaluation_split": "held_out_test_set",
            "test_period": {"start": "2020-01-02", "end": "2026-07-29"},
            "test_samples": int(len(targets)),
            "used_for_model_selection": False,
            "models": {
                "MLP": model_result(targets, mlp_predictions),
                "LSTM": {
                    **model_result(targets, lstm_predictions),
                    "selected_run": lstm_checkpoint.get("selected_run"),
                    "canonical_seed": lstm_checkpoint.get("canonical_model_seed"),
                },
                "Transformer": {
                    **model_result(targets, transformer_predictions),
                    "selected_run": transformer_checkpoint.get("selected_run"),
                    "canonical_seed": 42,
                },
            },
            "majority_baseline": {
                "source": "training_set_majority_class",
                "class_id": majority_class,
                "metrics": baseline["metrics"],
                "confusion_matrix": baseline["confusion_matrix"],
            },
        },
    )


def export_other_evidence() -> None:
    """Normalize already-computed post-hoc analyses and tuning summaries."""

    prepared = prepare_ofr_data()
    write_json(
        "data_snapshot.json",
        {
            "source_name": "Office of Financial Research Financial Stress Index",
            "source_url": OFR_DATA_URL,
            "retrieved_for_reference_run": "2026-08-10",
            "raw_file_sha256": calculate_file_sha256(RAW_DATA_FILE),
            "analysis_end": ANALYSIS_END,
            "rows": REFERENCE_SNAPSHOT_ROWS,
            "canonical_sha256": REFERENCE_SNAPSHOT_SHA256,
            "features": FEATURE_COLUMNS,
            "window_size": 60,
            "forecast_horizon": 5,
            "split_boundaries": {
                "train_end": "2016-12-31",
                "validation_end": "2019-12-31",
                "test_start": "2020-01-01",
            },
            "sequence_shapes": {
                "train": list(prepared.X_train.shape),
                "validation": list(prepared.X_validation.shape),
                "test": list(prepared.X_test.shape),
            },
            "class_thresholds": {
                "lower_training_tertile": prepared.low_threshold,
                "upper_training_tertile": prepared.high_threshold,
            },
            "data_distribution": (
                "The source CSV is downloaded directly from OFR and is intentionally not "
                "redistributed by this repository. OFR source terms and upstream indicator "
                "rights remain separate from the MIT-licensed source code."
            ),
        },
    )

    copies = {
        "roc_auc.json": "roc_auc_test.json",
        "lstm_stability.json": "lstm_stability_summary.json",
        "temporal_generalization.json": "temporal_generalization.json",
        "transformer_explainability.json": "transformer_explainability.json",
        "transformer_robustness.json": "transformer_robustness.json",
    }
    for output_name, local_name in copies.items():
        write_json(output_name, read_local_json(local_name))

    lstm_runs = read_local_json("lstm_tuning_results_final.json")
    transformer_runs = read_local_json("transformer_tuning_results_final.json")
    write_json(
        "tuning_summary.json",
        {
            "selection_uses_test_set": False,
            "selection_split": "validation",
            "primary_metric": "macro_f1",
            "tie_breaker": "validation_loss",
            "lstm": {
                "candidate_count": len(lstm_runs),
                "selected_run": "L1",
                "seed_stability_finalists": ["L1", "L9"],
                "selected_seed": 42,
            },
            "transformer": {
                "candidate_count": len(transformer_runs),
                "selected_run": "T1",
                "selected_seed": 42,
            },
        },
    )


def export_manifest() -> None:
    """Hash every evidence file after generation."""

    import hashlib

    files = {}
    for name in EVIDENCE_FILES:
        files[name] = hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest()
    write_json(
        "manifest.json",
        {
            "schema_version": 1,
            "purpose": "Integrity manifest for versioned, checkpoint-derived evidence",
            "files": files,
        },
    )


def main() -> None:
    """Export all reference evidence."""

    export_final_metrics()
    export_other_evidence()
    export_manifest()
    print(f"Exported and hashed {len(EVIDENCE_FILES)} evidence files in {OUTPUT}")


if __name__ == "__main__":
    main()
