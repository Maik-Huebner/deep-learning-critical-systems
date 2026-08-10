"""Download and load the OFR Financial Stress Index data."""

import ssl
from pathlib import Path
from urllib.request import urlopen

import certifi
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_FILE = RAW_DATA_DIR / "ofr_fsi.csv"

OFR_DATA_URL = (
    "https://www.financialresearch.gov/"
    "financial-stress-index/data/fsi.csv"
)


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

    ssl_context = ssl.create_default_context(
        cafile=certifi.where()
    )

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