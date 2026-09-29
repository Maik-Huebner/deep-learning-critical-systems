"""Tests for the lightweight repository-hygiene guard."""

from deep_learning_critical_systems.repository_audit import audit_paths


def test_audit_accepts_small_source_and_gitkeep(tmp_path) -> None:
    source = tmp_path / "src" / "example.py"
    marker = tmp_path / "data" / "raw" / ".gitkeep"
    source.parent.mkdir(parents=True)
    marker.parent.mkdir(parents=True)
    source.write_text("value = 1\n", encoding="utf-8")
    marker.write_text("", encoding="utf-8")

    assert audit_paths(tmp_path, ["src/example.py", "data/raw/.gitkeep"]) == []


def test_audit_rejects_checkpoint_raw_data_and_secret(tmp_path) -> None:
    checkpoint = tmp_path / "artifacts" / "model.pt"
    raw_data = tmp_path / "data" / "raw" / "source.csv"
    secret = tmp_path / "config.txt"
    checkpoint.parent.mkdir(parents=True)
    raw_data.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    raw_data.write_text("x\n1\n", encoding="utf-8")
    secret.write_text("AKIA" + "ABCDEFGHIJKLMNOP" + "\n", encoding="utf-8")

    findings = audit_paths(
        tmp_path,
        ["artifacts/model.pt", "data/raw/source.csv", "config.txt"],
    )
    reasons = {finding.reason for finding in findings}

    assert "checkpoint/cache binary is tracked" in reasons
    assert "raw or processed data is tracked" in reasons
    assert "possible AWS access key" in reasons
