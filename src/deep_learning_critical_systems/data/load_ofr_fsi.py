"""Download and load the OFR Financial Stress Index data."""

import hashlib
import ssl
from pathlib import Path
from urllib.request import urlopen

import certifi
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_FILE = RAW_DATA_DIR / "ofr_fsi.csv"

OFR_DATA_URL = "https://www.financialresearch.gov/financial-stress-index/data/fsi.csv"

REFERENCE_RAW_FILE_SHA256 = (
    "2d4a955fb0d72993fae454a731628d1deb4aca980a19121b989e80de09bf8478"
)


def calculate_file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""

    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def verify_reference_raw_file(path: Path = RAW_DATA_FILE) -> str:
    """Verify the exact raw CSV used for the published reference run.

    The live OFR file grows and may be revised. This check is therefore an
    explicit reference-snapshot check, not a requirement for downloading a
    newer source file whose rows through the analysis cutoff still match the
    canonical snapshot.
    """

    if not path.is_file():
        raise FileNotFoundError(f"Raw OFR data file not found: {path}")

    actual_sha256 = calculate_file_sha256(path)
    if actual_sha256 != REFERENCE_RAW_FILE_SHA256:
        raise ValueError(
            "Raw OFR file does not match the published reference file: "
            f"expected {REFERENCE_RAW_FILE_SHA256}, received {actual_sha256}."
        )

    return actual_sha256


def download_ofr_fsi() -> Path:
    """
    Download the OFR Financial Stress Index CSV.

    The file is downloaded only if it does not already exist locally.
    HTTPS certificate verification remains enabled and uses the
    certificate bundle provided by certifi.
    """

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_DATA_FILE.exists() and RAW_DATA_FILE.stat().st_size > 0:
        print(f"Raw data already exists: {RAW_DATA_FILE}")
        return RAW_DATA_FILE

    print("Downloading OFR Financial Stress Index...")

    ssl_context = ssl.create_default_context(cafile=certifi.where())

    with urlopen(
        OFR_DATA_URL,
        context=ssl_context,
        timeout=30,
    ) as response:
        RAW_DATA_FILE.write_bytes(response.read())

    print(f"Saved raw data to: {RAW_DATA_FILE}")

    return RAW_DATA_FILE


def load_ofr_fsi() -> pd.DataFrame:
    """
    Load the raw OFR Financial Stress Index CSV into a pandas DataFrame.
    """

    csv_path = download_ofr_fsi()

    data = pd.read_csv(csv_path)

    print()
    print("Dataset shape:")
    print(data.shape)

    print()
    print("Columns:")
    print(data.columns.tolist())

    print()
    print("First rows:")
    print(data.head())

    return data


if __name__ == "__main__":
    load_ofr_fsi()
