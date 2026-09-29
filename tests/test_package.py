from importlib.metadata import version as distribution_version
from pathlib import Path
from tomllib import load

from deep_learning_critical_systems import __version__

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_version() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        pyproject_version = load(pyproject_file)["project"]["version"]

    assert __version__ == distribution_version("deep-learning-critical-systems")
    assert __version__ == pyproject_version
