"""Prepare OFR Financial Stress Index data for Deep Learning.

This module performs the complete preprocessing pipeline used by the
project:

1. Load and validate the raw OFR Financial Stress Index data.
2. Create chronological train, validation and test splits.
3. Create the five-day future stress-change target inside each split.
4. Learn class thresholds exclusively from the training data.
5. Standardize all numerical features using training statistics only.
6. Convert the time series into 60-day sliding windows.

Validation and test windows may use observations from the immediately
preceding split as historical context. Only the target date determines
which split a sample belongs to.

This preserves all possible prediction dates without introducing
future information.
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
# Raw-data validation
# ---------------------------------------------------------------------


def clean_raw_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and prepare the raw OFR dataset."""

    data = data.copy()

    required_columns = [
        DATE_COLUMN,
        *FEATURE_COLUMNS,
    ]

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
        raise ValueError(
            "Training split is empty."
        )

    if validation.empty:
        raise ValueError(
            "Validation split is empty."
        )

    if test.empty:
        raise ValueError(
            "Test split is empty."
        )

    return train, validation, test


# ---------------------------------------------------------------------
# Target creation
# ---------------------------------------------------------------------


def add_future_stress_change(
    split: pd.DataFrame,
    horizon: int = FORECAST_HORIZON,
) -> pd.DataFrame:
    """Calculate future financial-stress change inside one split.

    For each observation, the average OFR FSI of the following
    ``horizon`` observations is calculated.

    The current OFR FSI is then subtracted:

        future_change =
            mean(next n OFR FSI values) - current OFR FSI

    Positive values therefore represent increasing future financial
    stress. Negative values represent decreasing future stress.

    Target calculation is performed separately inside every
    chronological split. This prevents a training target from using
    validation observations or a validation target from using test
    observations.
    """

    split = split.copy()

    future_values = pd.concat(
        [
            split["OFR FSI"].shift(-step)
            for step in range(
                1,
                horizon + 1,
            )
        ],
        axis=1,
    )

    # All future observations must exist.
    future_mean = future_values.mean(
        axis=1,
        skipna=False,
    )

    split[FUTURE_CHANGE_COLUMN] = (
        future_mean
        - split["OFR FSI"]
    )

    return split


def calculate_training_thresholds(
    train: pd.DataFrame,
) -> tuple[float, float]:
    """Learn class boundaries exclusively from training targets."""

    training_changes = train[
        FUTURE_CHANGE_COLUMN
    ].dropna()

    if training_changes.empty:
        raise ValueError(
            "No valid future stress changes are available "
            "in the training data."
        )

    low_threshold = float(
        training_changes.quantile(
            1 / 3
        )
    )

    high_threshold = float(
        training_changes.quantile(
            2 / 3
        )
    )

    return (
        low_threshold,
        high_threshold,
    )


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

    decrease_mask = (
        valid_mask
        & (
            split[FUTURE_CHANGE_COLUMN]
            <= low_threshold
        )
    )

    stable_mask = (
        valid_mask
        & (
            split[FUTURE_CHANGE_COLUMN]
            > low_threshold
        )
        & (
            split[FUTURE_CHANGE_COLUMN]
            < high_threshold
        )
    )

    increase_mask = (
        valid_mask
        & (
            split[FUTURE_CHANGE_COLUMN]
            >= high_threshold
        )
    )

    target.loc[
        decrease_mask
    ] = 0

    target.loc[
        stable_mask
    ] = 1

    target.loc[
        increase_mask
    ] = 2

    split[
        TARGET_COLUMN
    ] = target

    return split


# ---------------------------------------------------------------------
# Feature scaling
# ---------------------------------------------------------------------


def fit_scaler(
    train: pd.DataFrame,
) -> StandardScaler:
    """Fit the feature scaler exclusively on training observations."""

    scaler = StandardScaler()

    scaler.fit(
        train[
            FEATURE_COLUMNS
        ]
    )

    return scaler


def transform_features(
    data: pd.DataFrame,
    scaler: StandardScaler,
) -> np.ndarray:
    """Apply an already fitted scaler to feature columns."""

    return scaler.transform(
        data[
            FEATURE_COLUMNS
        ]
    )


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
    """Standardize all features without validation/test leakage."""

    scaler = fit_scaler(
        train
    )

    train_scaled = transform_features(
        train,
        scaler,
    )

    validation_scaled = transform_features(
        validation,
        scaler,
    )

    test_scaled = transform_features(
        test,
        scaler,
    )

    return (
        train_scaled,
        validation_scaled,
        test_scaled,
        scaler,
    )


# ---------------------------------------------------------------------
# Sliding-window helpers
# ---------------------------------------------------------------------


def create_sequences(
    features: np.ndarray,
    targets: pd.Series,
    dates: pd.Series,
    window_size: int = WINDOW_SIZE,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Create windows when no earlier external context is available.

    This function is mainly used for the training split and for unit
    tests.

    Output shape:

        X = (samples, timesteps, features)
        y = (samples,)
    """

    X = []
    y = []
    sample_dates = []

    target_values = (
        targets
        .reset_index(drop=True)
        .to_numpy()
    )

    date_values = (
        dates
        .reset_index(drop=True)
        .to_numpy()
    )

    for end_index in range(
        window_size - 1,
        len(features),
    ):
        target_value = (
            target_values[
                end_index
            ]
        )

        if pd.isna(
            target_value
        ):
            continue

        start_index = (
            end_index
            - window_size
            + 1
        )

        window = features[
            start_index:
            end_index + 1
        ]

        X.append(
            window
        )

        y.append(
            int(
                target_value
            )
        )

        sample_dates.append(
            date_values[
                end_index
            ]
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
        np.asarray(
            sample_dates,
        ),
    )


def create_sequences_with_history(
    historical_features: np.ndarray,
    current_features: np.ndarray,
    current_targets: pd.Series,
    current_dates: pd.Series,
    window_size: int = WINDOW_SIZE,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Create windows for validation/test using earlier observations.

    A validation or test prediction may use observations that occurred
    before the split boundary because those observations were already
    known at prediction time.

    Only the target date determines split membership.

    Example
    -------
    The first test observation on 2020-01-02 may use the final
    59 trading days of 2019 as historical context. It may never use
    observations after 2020-01-02.
    """

    if len(
        historical_features
    ) < (
        window_size - 1
    ):
        raise ValueError(
            "Not enough historical observations are available "
            "to build the requested window."
        )

    history = historical_features[
        -(
            window_size - 1
        ):
    ]

    combined_features = np.concatenate(
        [
            history,
            current_features,
        ],
        axis=0,
    )

    target_values = (
        current_targets
        .reset_index(drop=True)
        .to_numpy()
    )

    date_values = (
        current_dates
        .reset_index(drop=True)
        .to_numpy()
    )

    X = []
    y = []
    sample_dates = []

    history_length = len(
        history
    )

    for current_index in range(
        len(
            current_features
        )
    ):
        target_value = (
            target_values[
                current_index
            ]
        )

        if pd.isna(
            target_value
        ):
            continue

        end_index = (
            history_length
            + current_index
        )

        start_index = (
            end_index
            - window_size
            + 1
        )

        window = combined_features[
            start_index:
            end_index + 1
        ]

        if len(
            window
        ) != window_size:
            raise ValueError(
                "A generated sequence has an invalid window length."
            )

        X.append(
            window
        )

        y.append(
            int(
                target_value
            )
        )

        sample_dates.append(
            date_values[
                current_index
            ]
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
        np.asarray(
            sample_dates,
        ),
    )


# ---------------------------------------------------------------------
# Complete preprocessing pipeline
# ---------------------------------------------------------------------


def prepare_ofr_data(
    window_size: int = WINDOW_SIZE,
    forecast_horizon: int = FORECAST_HORIZON,
) -> PreparedOFRData:
    """Run the complete leakage-safe preprocessing pipeline."""

    raw_data = (
        load_ofr_fsi()
    )

    data = clean_raw_data(
        raw_data
    )

    (
        train,
        validation,
        test,
    ) = split_chronologically(
        data
    )

    train = (
        add_future_stress_change(
            train,
            horizon=forecast_horizon,
        )
    )

    validation = (
        add_future_stress_change(
            validation,
            horizon=forecast_horizon,
        )
    )

    test = (
        add_future_stress_change(
            test,
            horizon=forecast_horizon,
        )
    )

    (
        low_threshold,
        high_threshold,
    ) = calculate_training_thresholds(
        train
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

    # Training can only use observations from the training split.
    (
        X_train,
        y_train,
        dates_train,
    ) = create_sequences(
        train_scaled,
        train[
            TARGET_COLUMN
        ],
        train[
            DATE_COLUMN
        ],
        window_size,
    )

    # Validation can use the final training observations as historical
    # context because they occurred before every validation target date.
    (
        X_validation,
        y_validation,
        dates_validation,
    ) = create_sequences_with_history(
        historical_features=train_scaled,
        current_features=validation_scaled,
        current_targets=validation[
            TARGET_COLUMN
        ],
        current_dates=validation[
            DATE_COLUMN
        ],
        window_size=window_size,
    )

    # Test can use the final validation observations as historical
    # context because they occurred before every test target date.
    (
        X_test,
        y_test,
        dates_test,
    ) = create_sequences_with_history(
        historical_features=validation_scaled,
        current_features=test_scaled,
        current_targets=test[
            TARGET_COLUMN
        ],
        current_dates=test[
            DATE_COLUMN
        ],
        window_size=window_size,
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

        feature_names=(
            FEATURE_COLUMNS.copy()
        ),

        low_threshold=(
            low_threshold
        ),

        high_threshold=(
            high_threshold
        ),

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
    print(
        "=== PREPARED DATA ==="
    )

    print()
    print(
        "Feature count:"
    )

    print(
        len(
            prepared.feature_names
        )
    )

    print()
    print(
        "Features:"
    )

    for feature in (
        prepared.feature_names
    ):
        print(
            f"- {feature}"
        )

    print()
    print(
        "Training thresholds:"
    )

    print(
        "Stress Decrease / Stable:",
        round(
            prepared.low_threshold,
            4,
        ),
    )

    print(
        "Stable / Stress Increase:",
        round(
            prepared.high_threshold,
            4,
        ),
    )

    print()
    print(
        "Sequence shapes:"
    )

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
    print(
        "Class distribution:"
    )

    for (
        split_name,
        labels,
    ) in [
        (
            "Train",
            prepared.y_train,
        ),
        (
            "Validation",
            prepared.y_validation,
        ),
        (
            "Test",
            prepared.y_test,
        ),
    ]:
        counts = np.bincount(
            labels,
            minlength=len(
                CLASS_NAMES
            ),
        )

        print(
            f"{split_name}:",
            counts,
        )

    print()
    print(
        "First and last prediction dates:"
    )

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
    prepared_data = (
        prepare_ofr_data()
    )

    print_summary(
        prepared_data
    )