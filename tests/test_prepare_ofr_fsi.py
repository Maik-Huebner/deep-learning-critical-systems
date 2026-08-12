"""Tests for the OFR Financial Stress Index preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from deep_learning_critical_systems.data.prepare_ofr_fsi import (
    ANALYSIS_END,
    FEATURE_COLUMNS,
    REFERENCE_SNAPSHOT_ROWS,
    REFERENCE_SNAPSHOT_SHA256,
    add_future_stress_change,
    add_target_classes,
    calculate_snapshot_sha256,
    calculate_training_thresholds,
    clean_raw_data,
    create_sequences,
    create_sequences_with_history,
    scale_features,
    select_analysis_snapshot,
    split_chronologically,
    verify_reference_snapshot,
)


def create_test_dataframe() -> pd.DataFrame:
    """Create a small deterministic OFR-like dataset for unit tests."""

    dates = pd.date_range(
        start="2010-01-01",
        end="2021-12-31",
        freq="B",
    )

    n_rows = len(dates)

    base_signal = np.linspace(
        -2.0,
        2.0,
        n_rows,
    )

    data = pd.DataFrame(
        {
            "Date": dates,
            "OFR FSI": base_signal,
            "Credit": base_signal * 0.5,
            "Equity valuation": base_signal * 0.3,
            "Safe assets": base_signal * -0.2,
            "Funding": base_signal * 0.4,
            "Volatility": base_signal * 0.6,
            "United States": base_signal * 0.7,
            "Other advanced economies": base_signal * 0.2,
            "Emerging markets": base_signal * 0.1,
        }
    )

    return data


def create_snapshot_test_dataframe() -> pd.DataFrame:
    """Create deterministic OFR-like rows around the analysis cutoff."""

    dates = pd.date_range(
        start="2026-08-03",
        end="2026-08-10",
        freq="B",
    )

    n_rows = len(dates)

    base_signal = np.linspace(
        -1.0,
        1.0,
        n_rows,
    )

    return pd.DataFrame(
        {
            "Date": dates,
            "OFR FSI": base_signal,
            "Credit": base_signal * 0.5,
            "Equity valuation": base_signal * 0.3,
            "Safe assets": base_signal * -0.2,
            "Funding": base_signal * 0.4,
            "Volatility": base_signal * 0.6,
            "United States": base_signal * 0.7,
            "Other advanced economies": base_signal * 0.2,
            "Emerging markets": base_signal * 0.1,
        }
    )


def test_clean_raw_data_keeps_required_columns():
    """Cleaning should preserve exactly the required project columns."""

    data = create_test_dataframe()

    cleaned = clean_raw_data(data)

    expected_columns = [
        "Date",
        *FEATURE_COLUMNS,
    ]

    assert cleaned.columns.tolist() == expected_columns
    assert cleaned.isna().sum().sum() == 0
    assert cleaned["Date"].is_monotonic_increasing


def test_reference_snapshot_metadata_is_frozen():
    """Reference metadata should match the documented experiment."""

    assert ANALYSIS_END == "2026-08-05"
    assert REFERENCE_SNAPSHOT_ROWS == 6730
    assert (
        REFERENCE_SNAPSHOT_SHA256
        == "38535be9eadd819493c3b77e11885deb14e344d97007551f87c76700cc829c9c"
    )


def test_analysis_snapshot_truncates_newer_observations():
    """Rows after the documented analysis cutoff must be excluded."""

    data = clean_raw_data(
        create_snapshot_test_dataframe()
    )

    snapshot = select_analysis_snapshot(
        data
    )

    cutoff = pd.Timestamp(
        ANALYSIS_END
    )

    assert snapshot["Date"].max() == cutoff
    assert (
        snapshot["Date"] <= cutoff
    ).all()
    assert len(snapshot) < len(data)


def test_analysis_snapshot_requires_documented_cutoff_date():
    """The snapshot must contain the documented final observation."""

    data = clean_raw_data(
        create_snapshot_test_dataframe()
    )

    data = data.loc[
        data["Date"] != pd.Timestamp(
            ANALYSIS_END
        )
    ].copy()

    with pytest.raises(
        ValueError,
        match="analysis end date",
    ):
        select_analysis_snapshot(
            data
        )


def test_snapshot_hash_is_deterministic():
    """The canonical snapshot hash should be stable for identical data."""

    data = clean_raw_data(
        create_snapshot_test_dataframe()
    )

    snapshot = select_analysis_snapshot(
        data
    )

    first_hash = calculate_snapshot_sha256(
        snapshot
    )

    second_hash = calculate_snapshot_sha256(
        snapshot.copy()
    )

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_reference_snapshot_verification_accepts_matching_data():
    """Matching row count and canonical hash should pass verification."""

    data = clean_raw_data(
        create_snapshot_test_dataframe()
    )

    snapshot = select_analysis_snapshot(
        data
    )

    expected_hash = calculate_snapshot_sha256(
        snapshot
    )

    verified_hash = verify_reference_snapshot(
        snapshot,
        expected_rows=len(snapshot),
        expected_sha256=expected_hash,
    )

    assert verified_hash == expected_hash


def test_reference_snapshot_verification_rejects_wrong_row_count():
    """A changed snapshot length must fail provenance verification."""

    data = clean_raw_data(
        create_snapshot_test_dataframe()
    )

    snapshot = select_analysis_snapshot(
        data
    )

    expected_hash = calculate_snapshot_sha256(
        snapshot
    )

    with pytest.raises(
        ValueError,
        match="row count differs",
    ):
        verify_reference_snapshot(
            snapshot,
            expected_rows=len(snapshot) + 1,
            expected_sha256=expected_hash,
        )


def test_reference_snapshot_verification_rejects_modified_data():
    """A historical value change must fail canonical hash verification."""

    data = clean_raw_data(
        create_snapshot_test_dataframe()
    )

    snapshot = select_analysis_snapshot(
        data
    )

    expected_hash = calculate_snapshot_sha256(
        snapshot
    )

    modified_snapshot = snapshot.copy()
    modified_snapshot.loc[
        0,
        "Credit",
    ] += 0.001

    with pytest.raises(
        ValueError,
        match="canonical SHA-256",
    ):
        verify_reference_snapshot(
            modified_snapshot,
            expected_rows=len(modified_snapshot),
            expected_sha256=expected_hash,
        )


def test_chronological_split_has_no_overlap():
    """Train, validation and test periods must remain chronologically separate."""

    data = clean_raw_data(
        create_test_dataframe()
    )

    train, validation, test = split_chronologically(
        data
    )

    assert train["Date"].max() < validation["Date"].min()
    assert validation["Date"].max() < test["Date"].min()

    assert train["Date"].max() <= pd.Timestamp(
        "2016-12-31"
    )

    assert validation["Date"].max() <= pd.Timestamp(
        "2019-12-31"
    )

    assert test["Date"].min() > pd.Timestamp(
        "2019-12-31"
    )


def test_future_target_requires_complete_horizon():
    """The final five observations must not receive incomplete targets."""

    data = clean_raw_data(
        create_test_dataframe()
    )

    train, _, _ = split_chronologically(
        data
    )

    train = add_future_stress_change(
        train,
        horizon=5,
    )

    assert train[
        "future_stress_change"
    ].tail(5).isna().all()

    assert pd.notna(
        train[
            "future_stress_change"
        ].iloc[-6]
    )


def test_thresholds_are_calculated_from_training_data():
    """Training quantiles should define the two class boundaries."""

    data = clean_raw_data(
        create_test_dataframe()
    )

    train, _, _ = split_chronologically(
        data
    )

    train = add_future_stress_change(
        train,
        horizon=5,
    )

    low_threshold, high_threshold = (
        calculate_training_thresholds(
            train
        )
    )

    valid_changes = train[
        "future_stress_change"
    ].dropna()

    expected_low = valid_changes.quantile(
        1 / 3
    )

    expected_high = valid_changes.quantile(
        2 / 3
    )

    assert np.isclose(
        low_threshold,
        expected_low,
    )

    assert np.isclose(
        high_threshold,
        expected_high,
    )

    assert low_threshold < high_threshold


def test_target_classes_are_valid():
    """Every valid target must belong to one of the three project classes."""

    data = clean_raw_data(
        create_test_dataframe()
    )

    train, _, _ = split_chronologically(
        data
    )

    train = add_future_stress_change(
        train,
        horizon=5,
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

    valid_targets = train[
        "target"
    ].dropna()

    assert set(
        valid_targets.unique()
    ).issubset(
        {0, 1, 2}
    )

    assert train[
        "target"
    ].tail(5).isna().all()


def test_scaler_is_fit_only_on_training_data():
    """Scaler statistics must come exclusively from the training split."""

    data = clean_raw_data(
        create_test_dataframe()
    )

    train, validation, test = split_chronologically(
        data
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

    expected_training_mean = (
        train[FEATURE_COLUMNS]
        .mean()
        .to_numpy()
    )

    assert np.allclose(
        scaler.mean_,
        expected_training_mean,
    )

    assert train_scaled.shape == (
        len(train),
        len(FEATURE_COLUMNS),
    )

    assert validation_scaled.shape == (
        len(validation),
        len(FEATURE_COLUMNS),
    )

    assert test_scaled.shape == (
        len(test),
        len(FEATURE_COLUMNS),
    )


def test_create_sequences_has_correct_shape():
    """Sliding windows must produce samples x timesteps x features."""

    n_rows = 100
    n_features = len(FEATURE_COLUMNS)
    window_size = 60

    features = np.arange(
        n_rows * n_features,
        dtype=np.float32,
    ).reshape(
        n_rows,
        n_features,
    )

    targets = pd.Series(
        [1] * 95 + [pd.NA] * 5,
        dtype="Int64",
    )

    dates = pd.Series(
        pd.date_range(
            start="2020-01-01",
            periods=n_rows,
            freq="B",
        )
    )

    X, y, sample_dates = create_sequences(
        features,
        targets,
        dates,
        window_size=window_size,
    )

    # Valid prediction positions:
    # index 59 through index 94 inclusive.
    expected_samples = 36

    assert X.shape == (
        expected_samples,
        window_size,
        n_features,
    )

    assert y.shape == (
        expected_samples,
    )

    assert sample_dates.shape == (
        expected_samples,
    )

    assert X.dtype == np.float32
    assert y.dtype == np.int64

    assert sample_dates[0] == dates.iloc[59]
    assert sample_dates[-1] == dates.iloc[94]


def test_sequences_with_history_use_only_past_context():
    """Validation/test windows may use earlier split observations."""

    n_features = len(FEATURE_COLUMNS)
    window_size = 60

    historical_features = np.arange(
        59 * n_features,
        dtype=np.float32,
    ).reshape(
        59,
        n_features,
    )

    current_features = np.vstack(
        [
            np.full(
                n_features,
                1000.0,
                dtype=np.float32,
            ),
            np.full(
                n_features,
                2000.0,
                dtype=np.float32,
            ),
            np.full(
                n_features,
                3000.0,
                dtype=np.float32,
            ),
        ]
    )

    current_targets = pd.Series(
        [0, 2, pd.NA],
        dtype="Int64",
    )

    current_dates = pd.Series(
        pd.to_datetime(
            [
                "2020-01-02",
                "2020-01-03",
                "2020-01-06",
            ]
        )
    )

    X, y, sample_dates = create_sequences_with_history(
        historical_features=historical_features,
        current_features=current_features,
        current_targets=current_targets,
        current_dates=current_dates,
        window_size=window_size,
    )

    assert X.shape == (
        2,
        60,
        n_features,
    )

    assert y.tolist() == [
        0,
        2,
    ]

    assert sample_dates.tolist() == [
        current_dates.iloc[0],
        current_dates.iloc[1],
    ]

    # First prediction:
    # 59 historical rows + first current observation.
    assert np.array_equal(
        X[0][:-1],
        historical_features,
    )

    assert np.array_equal(
        X[0][-1],
        current_features[0],
    )

    # Second prediction:
    # Oldest history row drops out and the previous current observation
    # becomes part of the historical context.
    assert np.array_equal(
        X[1][:-2],
        historical_features[1:],
    )

    assert np.array_equal(
        X[1][-2],
        current_features[0],
    )

    assert np.array_equal(
        X[1][-1],
        current_features[1],
    )