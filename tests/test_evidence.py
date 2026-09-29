"""Tests for versioned, checkpoint-derived report evidence."""

from __future__ import annotations

import json
import shutil

import pytest

from deep_learning_critical_systems.evaluation.evidence import (
    EVIDENCE_DIRECTORY,
    calculate_sha256,
    load_json,
    validate_final_metrics,
    verify_evidence,
)


def update_manifest_hash(evidence_directory, file_name: str) -> None:
    """Update one copied manifest entry after an intentional test mutation."""

    manifest_path = evidence_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][file_name] = calculate_sha256(evidence_directory / file_name)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_versioned_evidence_bundle_is_internally_consistent() -> None:
    result = verify_evidence()

    assert result.files_checked == 8
    assert result.test_samples == 1694
    assert result.models == ("MLP", "LSTM", "Transformer")


def test_published_metrics_match_confusion_matrices() -> None:
    payload = load_json(EVIDENCE_DIRECTORY / "final_metrics.json")

    validate_final_metrics(payload)
    assert payload["models"]["MLP"]["confusion_matrix"] == [
        [196, 323, 0],
        [92, 546, 0],
        [93, 442, 2],
    ]
    assert payload["models"]["LSTM"]["confusion_matrix"] == [
        [299, 174, 46],
        [231, 372, 35],
        [205, 295, 37],
    ]
    assert payload["models"]["Transformer"]["confusion_matrix"] == [
        [368, 122, 29],
        [279, 307, 52],
        [256, 237, 44],
    ]


def test_manifest_detects_modified_evidence(tmp_path) -> None:
    copied_evidence = tmp_path / "evidence"
    shutil.copytree(EVIDENCE_DIRECTORY, copied_evidence)
    metrics_path = copied_evidence / "final_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["test_samples"] = 1
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        verify_evidence(copied_evidence)


def test_content_validation_rejects_modified_macro_f1_with_valid_hash(
    tmp_path,
) -> None:
    copied_evidence = tmp_path / "evidence"
    shutil.copytree(EVIDENCE_DIRECTORY, copied_evidence)
    metrics_path = copied_evidence / "final_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["models"]["Transformer"]["metrics"]["macro_f1"] += 0.01
    metrics_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    update_manifest_hash(copied_evidence, "final_metrics.json")

    with pytest.raises(ValueError, match="macro_f1.*differs from confusion matrix"):
        verify_evidence(copied_evidence)


def test_content_validation_rejects_modified_baseline_metric_with_valid_hash(
    tmp_path,
) -> None:
    copied_evidence = tmp_path / "evidence"
    shutil.copytree(EVIDENCE_DIRECTORY, copied_evidence)
    metrics_path = copied_evidence / "final_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["majority_baseline"]["metrics"]["accuracy"] += 0.01
    metrics_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    update_manifest_hash(copied_evidence, "final_metrics.json")

    with pytest.raises(ValueError, match="accuracy.*differs from confusion matrix"):
        verify_evidence(copied_evidence)


def test_content_validation_rejects_modified_class_metric_with_valid_hash(
    tmp_path,
) -> None:
    copied_evidence = tmp_path / "evidence"
    shutil.copytree(EVIDENCE_DIRECTORY, copied_evidence)
    metrics_path = copied_evidence / "final_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["models"]["MLP"]["per_class"]["Stress Increase"]["precision"] = 0.5
    metrics_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    update_manifest_hash(copied_evidence, "final_metrics.json")

    with pytest.raises(ValueError, match="precision.*differs from confusion matrix"):
        verify_evidence(copied_evidence)
