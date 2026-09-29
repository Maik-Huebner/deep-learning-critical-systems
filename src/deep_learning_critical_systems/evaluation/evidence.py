"""Validate the small, versioned evidence bundle used by project reports."""

from __future__ import annotations

import hashlib
import json
import math
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EVIDENCE_FILES = (
    "data_snapshot.json",
    "final_metrics.json",
    "roc_auc.json",
    "lstm_stability.json",
    "temporal_generalization.json",
    "transformer_explainability.json",
    "transformer_robustness.json",
    "tuning_summary.json",
)

MODEL_NAMES = ("MLP", "LSTM", "Transformer")
CLASS_NAMES = ("Stress Decrease", "Stable", "Stress Increase")
METRIC_NAMES = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "stress_increase_recall",
)


@dataclass(frozen=True)
class EvidenceAudit:
    """Result of an evidence validation run."""

    files_checked: int
    test_samples: int
    models: tuple[str, ...]


def load_json(path: Path) -> Any:
    """Load UTF-8 JSON from *path*."""

    return json.loads(path.read_text(encoding="utf-8"))


def calculate_sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_confusion_matrix(
    matrix: Any,
    sample_count: int,
    result_name: str,
) -> list[list[int]]:
    """Return a validated non-negative integer 3x3 confusion matrix."""

    _require(
        isinstance(matrix, list)
        and len(matrix) == 3
        and all(isinstance(row, list) and len(row) == 3 for row in matrix),
        f"Invalid confusion matrix for {result_name}",
    )
    _require(
        all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for row in matrix
            for value in row
        ),
        f"Confusion matrix must contain non-negative integers for {result_name}",
    )
    _require(
        sum(sum(row) for row in matrix) == sample_count,
        f"Confusion matrix total differs for {result_name}",
    )
    return matrix


def _calculate_confusion_metrics(
    matrix: list[list[int]],
) -> tuple[dict[str, float], list[dict[str, int | float]]]:
    """Recalculate aggregate and per-class metrics from a confusion matrix."""

    total = sum(sum(row) for row in matrix)
    per_class = []

    for class_index in range(3):
        true_positive = matrix[class_index][class_index]
        support = sum(matrix[class_index])
        predicted = sum(row[class_index] for row in matrix)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class.append(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    metrics = {
        "accuracy": sum(matrix[index][index] for index in range(3)) / total,
        "macro_precision": sum(item["precision"] for item in per_class) / 3,
        "macro_recall": sum(item["recall"] for item in per_class) / 3,
        "macro_f1": sum(item["f1"] for item in per_class) / 3,
        "stress_increase_recall": per_class[2]["recall"],
    }
    return metrics, per_class


def _validate_metric(value: Any, expected: float, description: str) -> None:
    """Require a finite unit-interval metric equal to its recomputed value."""

    _require(
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0,
        f"Invalid {description}",
    )
    _require(
        math.isclose(float(value), expected, rel_tol=1e-12, abs_tol=1e-12),
        f"{description} differs from confusion matrix",
    )


def validate_final_metrics(payload: dict[str, Any]) -> None:
    """Validate model metrics, supports and confusion matrices."""

    _require(payload.get("evaluation_split") == "held_out_test_set", "Invalid split")
    sample_count = payload.get("test_samples")
    _require(isinstance(sample_count, int) and sample_count > 0, "Invalid sample count")

    models = payload.get("models")
    _require(isinstance(models, dict), "Model results must be an object")
    _require(tuple(models) == MODEL_NAMES, "Unexpected model ordering or names")

    for model_name, result in models.items():
        metrics = result.get("metrics")
        _require(isinstance(metrics, dict), f"Missing metrics for {model_name}")
        matrix = _validate_confusion_matrix(
            result.get("confusion_matrix"), sample_count, model_name
        )
        expected_metrics, expected_per_class = _calculate_confusion_metrics(matrix)

        for metric_name in METRIC_NAMES:
            _validate_metric(
                metrics.get(metric_name),
                expected_metrics[metric_name],
                f"{metric_name} for {model_name}",
            )

        per_class = result.get("per_class")
        _require(
            isinstance(per_class, dict), f"Missing per-class results for {model_name}"
        )
        _require(
            tuple(per_class) == CLASS_NAMES,
            f"Unexpected per-class ordering or names for {model_name}",
        )
        for class_name, expected in zip(CLASS_NAMES, expected_per_class, strict=True):
            class_result = per_class[class_name]
            _require(
                isinstance(class_result, dict),
                f"Invalid per-class result for {model_name}/{class_name}",
            )
            _require(
                class_result.get("support") == expected["support"],
                f"Support differs from matrix for {model_name}/{class_name}",
            )
            for metric_name in ("precision", "recall", "f1"):
                _validate_metric(
                    class_result.get(metric_name),
                    expected[metric_name],
                    f"{metric_name} for {model_name}/{class_name}",
                )

    majority = payload.get("majority_baseline", {})
    _require(
        majority.get("source") == "training_set_majority_class", "Invalid baseline"
    )
    majority_matrix = _validate_confusion_matrix(
        majority.get("confusion_matrix"), sample_count, "majority baseline"
    )
    expected_majority_metrics, _ = _calculate_confusion_metrics(majority_matrix)
    majority_metrics = majority.get("metrics")
    _require(isinstance(majority_metrics, dict), "Missing majority-baseline metrics")
    for metric_name in ("accuracy", "macro_f1", "stress_increase_recall"):
        _validate_metric(
            majority_metrics.get(metric_name),
            expected_majority_metrics[metric_name],
            f"{metric_name} for majority baseline",
        )


def validate_manifest(evidence_directory: Path) -> None:
    """Verify that every evidence file matches the versioned manifest."""

    if not evidence_directory.is_dir():
        raise FileNotFoundError(
            f"Evidence directory not found: {evidence_directory.resolve()}"
        )

    manifest_path = evidence_directory / "manifest.json"
    manifest = load_json(manifest_path)
    expected_files = manifest.get("files")
    _require(isinstance(expected_files, dict), "Evidence manifest has no file map")
    _require(tuple(expected_files) == EVIDENCE_FILES, "Evidence manifest is incomplete")

    for file_name, expected_sha256 in expected_files.items():
        path = evidence_directory / file_name
        _require(path.is_file(), f"Missing evidence file: {file_name}")
        _require(
            calculate_sha256(path) == expected_sha256,
            f"Evidence hash mismatch: {file_name}",
        )


def verify_evidence(evidence_directory: Path) -> EvidenceAudit:
    """Validate the complete evidence bundle and cross-file invariants."""

    validate_manifest(evidence_directory)

    data_snapshot = load_json(evidence_directory / "data_snapshot.json")
    _require(data_snapshot.get("rows") == 6730, "Unexpected snapshot row count")
    _require(data_snapshot.get("analysis_end") == "2026-08-05", "Unexpected cutoff")
    _require(len(data_snapshot.get("features", [])) == 9, "Unexpected feature count")
    _require(
        data_snapshot.get("canonical_sha256")
        == "38535be9eadd819493c3b77e11885deb14e344d97007551f87c76700cc829c9c",
        "Unexpected canonical snapshot hash",
    )

    final_metrics = load_json(evidence_directory / "final_metrics.json")
    validate_final_metrics(final_metrics)

    roc_auc = load_json(evidence_directory / "roc_auc.json")
    _require(roc_auc.get("used_for_model_selection") is False, "ROC-AUC selection flag")
    _require(tuple(roc_auc.get("results", {})) == MODEL_NAMES, "ROC-AUC model mismatch")

    temporal = load_json(evidence_directory / "temporal_generalization.json")
    rows = temporal.get("results", [])
    _require(len(rows) == 21, "Temporal evidence must contain 3 models x 7 years")
    _require(
        sum(row["sample_count"] for row in rows if row["model"] == "MLP") == 1694,
        "Temporal sample totals differ",
    )

    robustness = load_json(evidence_directory / "transformer_robustness.json")
    _require(len(robustness.get("results", [])) == 5, "Unexpected robustness grid")

    explainability = load_json(evidence_directory / "transformer_explainability.json")
    note = explainability.get("interpretation_note", "").lower()
    _require(
        "causality" in note and "feature importance" in note, "Missing attention caveat"
    )

    stability = load_json(evidence_directory / "lstm_stability.json")
    _require(len(stability) == 2, "Unexpected LSTM stability finalists")

    tuning = load_json(evidence_directory / "tuning_summary.json")
    _require(
        tuning.get("selection_uses_test_set") is False, "Tuning uses test evidence"
    )

    return EvidenceAudit(
        files_checked=len(EVIDENCE_FILES),
        test_samples=final_metrics["test_samples"],
        models=MODEL_NAMES,
    )


def main(evidence_directory: Path) -> None:
    """Run the offline evidence audit."""

    result = verify_evidence(evidence_directory)
    print(
        f"Evidence audit passed: {result.files_checked} files, "
        f"{result.test_samples} held-out samples, {', '.join(result.models)}."
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("evidence_directory", type=Path)
    main(parser.parse_args().evidence_directory)
