"""Tests for claims and CI behavior that span repository files."""

from pathlib import Path

from deep_learning_critical_systems.evaluation.evidence import load_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIRECTORY = PROJECT_ROOT / "reports" / "evidence"


def test_readme_final_table_matches_versioned_evidence() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    evidence = load_json(EVIDENCE_DIRECTORY / "final_metrics.json")

    for model_name, result in evidence["models"].items():
        metrics = result["metrics"]
        expected_row = (
            f"| {model_name} | "
            f"{'**' if model_name == 'MLP' else ''}{metrics['accuracy']:.2%}"
        )
        assert expected_row in readme
        assert f"{metrics['macro_f1']:.2%}" in readme
        assert f"{metrics['stress_increase_recall']:.2%}" in readme

    assert "44.33%" not in readme
    assert "37.90%" not in readme


def test_ci_uses_offline_evidence_and_never_trains_models() -> None:
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "quality.yml").read_text(
        encoding="utf-8"
    )

    assert "make evidence-check" in workflow
    assert "make quality" in workflow
    assert "make test" in workflow
    assert "make build" in workflow
    assert "train_mlp" not in workflow
    assert "train_lstm" not in workflow
    assert "train_transformer" not in workflow
    assert "export_reference_evidence" not in workflow
