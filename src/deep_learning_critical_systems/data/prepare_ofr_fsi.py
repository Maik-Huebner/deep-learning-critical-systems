"""Prepare OFR Financial Stress Index data for Deep Learning.

This module performs the complete preprocessing pipeline used by the
project:

1. Load the raw OFR Financial Stress Index data.
2. Create chronological train, validation and test splits.
3. Create the five-day future stress-change target inside each split.
4. Learn class thresholds exclusively from the training data.
5. Standardize all numerical features using training statistics only.
6. Convert the time series into 60-day sliding windows.

The preprocessing is intentionally designed to avoid data leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from deep_learning_critical_systems.data.load_ofr_fsi import load_ofr_fsi


# ---------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------

WINDOW_SIZE = 60
FORECAST_HORIZON = 5

TRAIN_END = "2016-12-31"
VALIDATION_END = "2019-12-31"

DATE_COLUMN = "Date"

FEATURE_COLUMNS = [
    "OFR FSI",
    "Credit",
    "Equity valuation",
    "Safe assets",
    "Funding",
    "Volatility",
    "United States",
    "Other advanced economies",
    "Emerging markets",
]

TARGET_COLUMN = "target"
FUTURE_CHANGE_COLUMN = "future_stress_change"

CLASS_NAMES = [
    "Stress Decrease",
    "Stable",
    "Stress Increase",
]


# ---------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------


@dataclass
class PreparedOFRData:
    """Container holding all prepared arrays and preprocessing metadata."""

    X_train: np.ndarray
    y_train: np.ndarray
    dates_train: np.ndarray

    X_validation: np.ndarray
    y_validation: np.ndarray
    dates_validation: np.ndarray

    X_test: np.ndarray
    y_test: np.ndarray
    dates_test: np.ndarray

    feature_names: list[str]

    low_threshold: float
    high_threshold: float

    scaler: StandardScaler


# ---------------------------------------------------------------------
# Basic preparation
# ---------------------------------------------------------------------


def clean_raw_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and prepare the raw OFR dataset.

    The downloaded OFR dataset is already very clean. This function
    nevertheless performs explicit checks so that the project does not
    silently rely on assumptions about the source data.
    """

    data = data.copy()

    required_columns = [DATE_COLUMN, *FEATURE_COLUMNS]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns are missing from the OFR dataset: "
            + ", ".join(missing_columns)
        )

    data[DATE_COLUMN] = pd.to_datetime(
        data[DATE_COLUMN],
        errors="raise",
    )

    data = (
        data
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )

    if data[DATE_COLUMN].duplicated().any():
        raise ValueError(
            "Duplicate dates were found in the OFR dataset."
        )

    if data[required_columns].isna().any().any():
        raise ValueError(
            "Missing values were found in the required OFR columns."
        )

    return data[required_columns]


# ---------------------------------------------------------------------
# Chronological splitting
# ---------------------------------------------------------------------


def split_chronologically(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset into chronological train, validation and test sets."""

    train = data.loc[
        data[DATE_COLUMN] <= TRAIN_END
    ].copy()

    validation = data.loc[
        (data[DATE_COLUMN] > TRAIN_END)
        & (data[DATE_COLUMN] <= VALIDATION_END)
    ].copy()

    test = data.loc[
        data[DATE_COLUMN] > VALIDATION_END
    ].copy()

    if train.empty:
        raise ValueError("Training split is empty.")

    if validation.empty:
        raise ValueError("Validation split is empty.")

    if test.empty:
        raise ValueError("Test split is empty.")

    return train, validation, test


# ---------------------------------------------------------------------
# Target creation
# ---------------------------------------------------------------------


def add_future_stress_change(
    split: pd.DataFrame,
    horizon: int = FORECAST_HORIZON,
) -> pd.DataFrame:
    """Calculate future financial-stress change inside one split.

    For every observation, the mean OFR FSI of the following five
    observations is calculated.

    The current OFR FSI is then subtracted:

        future_change = mean(next 5 OFR FSI values) - current OFR FSI

    Positive values therefore represent increasing future financial
    stress, while negative values represent decreasing stress.

    Target calculation is intentionally performed separately for each
    chronological split. This prevents future observations from one
    split being used to create targets in another split.
    """

    split = split.copy()

    future_values = pd.concat(
        [
            split["OFR FSI"].shift(-step)
            for step in range(1, horizon + 1)
        ],
        axis=1,
    )

    # skipna=False ensures that all five future observations must exist.
    future_mean = future_values.mean(
        axis=1,
        skipna=False,
    )

    split[FUTURE_CHANGE_COLUMN] = (
        future_mean - split["OFR FSI"]
    )

    return split


def calculate_training_thresholds(
    train: pd.DataFrame,
) -> tuple[float, float]:
    """Learn the three class boundaries exclusively from training data."""

    training_changes = train[
        FUTURE_CHANGE_COLUMN
    ].dropna()

    if training_changes.empty:
        raise ValueError(
            "No valid future stress changes are available in training data."
        )

    low_threshold = float(
        training_changes.quantile(1 / 3)
    )

    high_threshold = float(
        training_changes.quantile(2 / 3)
    )

    return low_threshold, high_threshold


def add_target_classes(
    split: pd.DataFrame,
    low_threshold: float,
    high_threshold: float,
) -> pd.DataFrame:
    """Convert continuous future stress changes into three classes.

    Classes:

    0 = Stress Decrease
    1 = Stable
    2 = Stress Increase
    """

    split = split.copy()

    target = pd.Series(
        pd.NA,
        index=split.index,
        dtype="Int64",
    )

    valid_mask = split[
        FUTURE_CHANGE_COLUMN
    ].notna()

    changes = split.loc[
        valid_mask,
        FUTURE_CHANGE_COLUMN,
    ]

    target.loc[
        valid_mask & (
            split[FUTURE_CHANGE_COLUMN]
            <= low_threshold
        )
    ] = 0

    target.loc[
        valid_mask & (
            split[FUTURE_CHANGE_COLUMN]
            > low_threshold
        ) & (
            split[FUTURE_CHANGE_COLUMN]
            < high_threshold
        )
    ] = 1

    target.loc[
        valid_mask & (
            split[FUTURE_CHANGE_COLUMN]
            >= high_threshold
        )
    ] = 2

    split[TARGET_COLUMN] = target

    return split


# ---------------------------------------------------------------------
# Feature scaling
# ---------------------------------------------------------------------


def scale_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    StandardScaler,
]:
    """Standardize all features without leaking validation/test information."""

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(
        train[FEATURE_COLUMNS]
    )

    validation_scaled = scaler.transform(
        validation[FEATURE_COLUMNS]
    )

    test_scaled = scaler.transform(
        test[FEATURE_COLUMNS]
    )

    return (
        train_scaled,
        validation_scaled,
        test_scaled,
        scaler,
    )


# ---------------------------------------------------------------------
# Sliding windows
# ---------------------------------------------------------------------


def create_sequences(
    features: np.ndarray,
    targets: pd.Series,
    dates: pd.Series,
    window_size: int = WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create fixed-length time-series windows.

    Every sample contains the previous 60 observations including the
    observation on the prediction date.

    Output shape:

        X = (samples, timesteps, features)
        y = (samples,)
    """

    X = []
    y = []
    sample_dates = []

    target_values = targets.to_numpy()
    date_values = dates.to_numpy()

    for end_index in range(
        window_size - 1,
        len(features),
    ):
        target_value = target_values[end_index]

        # The final five observations of each split have no valid target.
        if pd.isna(target_value):
            continue

        start_index = (
            end_index - window_size + 1
        )

        window = features[
            start_index:end_index + 1
        ]

        X.append(window)
        y.append(int(target_value))
        sample_dates.append(
            date_values[end_index]
        )

    return (
        np.asarray(
            X,
            dtype=np.float32,
        ),
        np.asarray(
            y,
            dtype=np.int64,
        ),
        np.asarray(sample_dates),
    )


# ---------------------------------------------------------------------
# Complete preprocessing pipeline
# ---------------------------------------------------------------------


def prepare_ofr_data(
    window_size: int = WINDOW_SIZE,
    forecast_horizon: int = FORECAST_HORIZON,
) -> PreparedOFRData:
    """Run the complete preprocessing pipeline."""

    raw_data = load_ofr_fsi()

    data = clean_raw_data(
        raw_data
    )

    train, validation, test = (
        split_chronologically(data)
    )

    train = add_future_stress_change(
        train,
        horizon=forecast_horizon,
    )

    validation = add_future_stress_change(
        validation,
        horizon=forecast_horizon,
    )

    test = add_future_stress_change(
        test,
        horizon=forecast_horizon,
    )

    low_threshold, high_threshold = (
        calculate_training_thresholds(
            train
        )
    )

    train = add_target_classes(
        train,
        low_threshold,
        high_threshold,
    )

    validation = add_target_classes(
        validation,
        low_threshold,
        high_threshold,
    )

    test = add_target_classes(
        test,
        low_threshold,
        high_threshold,
    )

    (
        train_scaled,
        validation_scaled,
        test_scaled,
        scaler,
    ) = scale_features(
        train,
        validation,
        test,
    )

    (
        X_train,
        y_train,
        dates_train,
    ) = create_sequences(
        train_scaled,
        train[TARGET_COLUMN],
        train[DATE_COLUMN],
        window_size,
    )

    (
        X_validation,
        y_validation,
        dates_validation,
    ) = create_sequences(
        validation_scaled,
        validation[TARGET_COLUMN],
        validation[DATE_COLUMN],
        window_size,
    )

    (
        X_test,
        y_test,
        dates_test,
    ) = create_sequences(
        test_scaled,
        test[TARGET_COLUMN],
        test[DATE_COLUMN],
        window_size,
    )

    return PreparedOFRData(
        X_train=X_train,
        y_train=y_train,
        dates_train=dates_train,

        X_validation=X_validation,
        y_validation=y_validation,
        dates_validation=dates_validation,

        X_test=X_test,
        y_test=y_test,
        dates_test=dates_test,

        feature_names=FEATURE_COLUMNS.copy(),

        low_threshold=low_threshold,
        high_threshold=high_threshold,

        scaler=scaler,
    )


# ---------------------------------------------------------------------
# Manual inspection
# ---------------------------------------------------------------------


def print_summary(
    prepared: PreparedOFRData,
) -> None:
    """Print the most important preprocessing results."""

    print()
    print("=== PREPARED DATA ===")

    print()
    print("Feature count:")
    print(len(prepared.feature_names))

    print()
    print("Features:")
    for feature in prepared.feature_names:
        print(f"- {feature}")

    print()
    print("Training thresholds:")
    print(
        "Stress Decrease / Stable:",
        round(prepared.low_threshold, 4),
    )
    print(
        "Stable / Stress Increase:",
        round(prepared.high_threshold, 4),
    )

    print()
    print("Sequence shapes:")
    print(
        "Train:",
        prepared.X_train.shape,
        prepared.y_train.shape,
    )
    print(
        "Validation:",
        prepared.X_validation.shape,
        prepared.y_validation.shape,
    )
    print(
        "Test:",
        prepared.X_test.shape,
        prepared.y_test.shape,
    )

    print()
    print("Class distribution:")

    for split_name, labels in [
        ("Train", prepared.y_train),
        ("Validation", prepared.y_validation),
        ("Test", prepared.y_test),
    ]:
        counts = np.bincount(
            labels,
            minlength=len(CLASS_NAMES),
        )

        print(
            f"{split_name}:",
            counts,
        )

    print()
    print("First and last prediction dates:")

    print(
        "Train:",
        prepared.dates_train[0],
        "to",
        prepared.dates_train[-1],
    )

    print(
        "Validation:",
        prepared.dates_validation[0],
        "to",
        prepared.dates_validation[-1],
    )

    print(
        "Test:",
        prepared.dates_test[0],
        "to",
        prepared.dates_test[-1],
    )


if __name__ == "__main__":
    prepared_data = prepare_ofr_data()
    print_summary(prepared_data)