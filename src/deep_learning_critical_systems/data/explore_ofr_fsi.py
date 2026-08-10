"""Exploratory analysis of the OFR Financial Stress Index dataset.

The analysis focuses on the questions that are relevant for the
financial-stress forecasting project:

1. How has financial stress developed over time?
2. How strongly are the available OFR features related?
3. How is the five-day future stress-change target distributed?
4. How do the three target classes differ across train, validation
   and test periods?

All target thresholds are calculated from the training period only.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deep_learning_critical_systems.data.load_ofr_fsi import (
    load_ofr_fsi,
)
from deep_learning_critical_systems.data.prepare_ofr_fsi import (
    CLASS_NAMES,
    FEATURE_COLUMNS,
    FUTURE_CHANGE_COLUMN,
    TARGET_COLUMN,
    TRAIN_END,
    VALIDATION_END,
    add_future_stress_change,
    add_target_classes,
    calculate_training_thresholds,
    clean_raw_data,
    split_chronologically,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

FIGURES_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)


def prepare_eda_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    float,
    float,
]:
    """Prepare the chronological splits and project targets for EDA."""

    raw_data = load_ofr_fsi()

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

    train = add_future_stress_change(
        train
    )

    validation = add_future_stress_change(
        validation
    )

    test = add_future_stress_change(
        test
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

    return (
        data,
        train,
        validation,
        test,
        low_threshold,
        high_threshold,
    )


def plot_financial_stress_over_time(
    data: pd.DataFrame,
) -> None:
    """Plot the complete OFR Financial Stress Index time series."""

    figure, axis = plt.subplots(
        figsize=(14, 6)
    )

    axis.plot(
        data["Date"],
        data["OFR FSI"],
        linewidth=1.0,
    )

    axis.axhline(
        y=0,
        linestyle="--",
        linewidth=1.0,
    )

    axis.axvline(
        pd.Timestamp(TRAIN_END),
        linestyle="--",
        linewidth=1.2,
        label="End of training period",
    )

    axis.axvline(
        pd.Timestamp(VALIDATION_END),
        linestyle="--",
        linewidth=1.2,
        label="End of validation period",
    )

    axis.set_title(
        "OFR Financial Stress Index over Time"
    )

    axis.set_xlabel(
        "Date"
    )

    axis.set_ylabel(
        "OFR FSI"
    )

    axis.legend()

    axis.grid(
        alpha=0.3
    )

    figure.tight_layout()

    output_path = (
        FIGURES_DIR
        / "ofr_fsi_over_time.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"Saved: {output_path}"
    )


def plot_feature_correlations(
    train: pd.DataFrame,
) -> None:
    """Plot feature correlations using training observations only."""

    correlations = (
        train[
            FEATURE_COLUMNS
        ]
        .corr()
    )

    figure, axis = plt.subplots(
        figsize=(11, 9)
    )

    image = axis.imshow(
        correlations,
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )

    axis.set_xticks(
        np.arange(
            len(FEATURE_COLUMNS)
        )
    )

    axis.set_yticks(
        np.arange(
            len(FEATURE_COLUMNS)
        )
    )

    axis.set_xticklabels(
        FEATURE_COLUMNS,
        rotation=45,
        ha="right",
    )

    axis.set_yticklabels(
        FEATURE_COLUMNS
    )

    for row in range(
        len(FEATURE_COLUMNS)
    ):
        for column in range(
            len(FEATURE_COLUMNS)
        ):
            axis.text(
                column,
                row,
                f"{correlations.iloc[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    figure.colorbar(
        image,
        ax=axis,
        label="Correlation",
    )

    axis.set_title(
        "Training-Period Feature Correlations"
    )

    figure.tight_layout()

    output_path = (
        FIGURES_DIR
        / "feature_correlation_matrix.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"Saved: {output_path}"
    )


def plot_future_change_distribution(
    train: pd.DataFrame,
    low_threshold: float,
    high_threshold: float,
) -> None:
    """Plot the distribution used to define the target classes."""

    valid_changes = (
        train[
            FUTURE_CHANGE_COLUMN
        ]
        .dropna()
    )

    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    axis.hist(
        valid_changes,
        bins=80,
        edgecolor="black",
        linewidth=0.3,
    )

    axis.axvline(
        low_threshold,
        linestyle="--",
        linewidth=1.5,
        label=(
            "Decrease / Stable threshold "
            f"({low_threshold:.3f})"
        ),
    )

    axis.axvline(
        high_threshold,
        linestyle="--",
        linewidth=1.5,
        label=(
            "Stable / Increase threshold "
            f"({high_threshold:.3f})"
        ),
    )

    axis.set_title(
        "Training Distribution of Five-Day Future Stress Change"
    )

    axis.set_xlabel(
        "Future mean OFR FSI minus current OFR FSI"
    )

    axis.set_ylabel(
        "Number of observations"
    )

    axis.legend()

    axis.grid(
        alpha=0.3
    )

    figure.tight_layout()

    output_path = (
        FIGURES_DIR
        / "future_stress_change_distribution.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"Saved: {output_path}"
    )


def plot_class_distribution(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Compare target-class distributions across the three splits."""

    split_data = {
        "Train": train,
        "Validation": validation,
        "Test": test,
    }

    class_counts = []

    for split_name, split in split_data.items():
        counts = (
            split[
                TARGET_COLUMN
            ]
            .dropna()
            .astype(int)
            .value_counts()
            .reindex(
                range(
                    len(CLASS_NAMES)
                ),
                fill_value=0,
            )
        )

        class_counts.append(
            counts.to_numpy()
        )

    values = np.asarray(
        class_counts
    )

    x_positions = np.arange(
        len(CLASS_NAMES)
    )

    bar_width = 0.25

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    for split_index, split_name in enumerate(
        split_data
    ):
        axis.bar(
            x_positions
            + split_index * bar_width,
            values[split_index],
            width=bar_width,
            label=split_name,
        )

    axis.set_xticks(
        x_positions
        + bar_width
    )

    axis.set_xticklabels(
        CLASS_NAMES
    )

    axis.set_title(
        "Target Class Distribution by Data Split"
    )

    axis.set_xlabel(
        "Target class"
    )

    axis.set_ylabel(
        "Number of observations"
    )

    axis.legend()

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()

    output_path = (
        FIGURES_DIR
        / "class_distribution_by_split.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"Saved: {output_path}"
    )


def print_numerical_summary(
    data: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    low_threshold: float,
    high_threshold: float,
) -> None:
    """Print the key numerical findings of the exploratory analysis."""

    print()
    print(
        "=== EDA SUMMARY ==="
    )

    print()
    print(
        "Dataset:"
    )

    print(
        "Rows:",
        len(data),
    )

    print(
        "Features:",
        len(FEATURE_COLUMNS),
    )

    print(
        "Date range:",
        data["Date"].min(),
        "to",
        data["Date"].max(),
    )

    print()
    print(
        "OFR FSI summary:"
    )

    print(
        data["OFR FSI"]
        .describe()
        .round(3)
    )

    print()
    print(
        "Training target thresholds:"
    )

    print(
        "Decrease / Stable:",
        round(
            low_threshold,
            4,
        ),
    )

    print(
        "Stable / Increase:",
        round(
            high_threshold,
            4,
        ),
    )

    print()
    print(
        "Target class distributions:"
    )

    for split_name, split in [
        (
            "Train",
            train,
        ),
        (
            "Validation",
            validation,
        ),
        (
            "Test",
            test,
        ),
    ]:
        counts = (
            split[
                TARGET_COLUMN
            ]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
        )

        print()
        print(
            split_name
        )

        for class_index, class_name in enumerate(
            CLASS_NAMES
        ):
            print(
                f"  {class_name}:",
                counts.get(
                    class_index,
                    0,
                ),
            )

    print()
    print(
        "Largest OFR FSI observations:"
    )

    print(
        data[
            [
                "Date",
                "OFR FSI",
            ]
        ]
        .nlargest(
            10,
            "OFR FSI",
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Largest future stress increases in training:"
    )

    print(
        train[
            [
                "Date",
                "OFR FSI",
                FUTURE_CHANGE_COLUMN,
            ]
        ]
        .nlargest(
            10,
            FUTURE_CHANGE_COLUMN,
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Strongest absolute training correlations:"
    )

    correlations = (
        train[
            FEATURE_COLUMNS
        ]
        .corr()
    )

    correlation_pairs = []

    for first_index in range(
        len(FEATURE_COLUMNS)
    ):
        for second_index in range(
            first_index + 1,
            len(FEATURE_COLUMNS),
        ):
            first_feature = (
                FEATURE_COLUMNS[
                    first_index
                ]
            )

            second_feature = (
                FEATURE_COLUMNS[
                    second_index
                ]
            )

            correlation = (
                correlations.loc[
                    first_feature,
                    second_feature,
                ]
            )

            correlation_pairs.append(
                (
                    first_feature,
                    second_feature,
                    correlation,
                )
            )

    correlation_pairs.sort(
        key=lambda item: abs(
            item[2]
        ),
        reverse=True,
    )

    for (
        first_feature,
        second_feature,
        correlation,
    ) in correlation_pairs[:10]:
        print(
            f"  {first_feature} <-> {second_feature}: "
            f"{correlation:.3f}"
        )


def run_eda() -> None:
    """Run the complete exploratory analysis."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        data,
        train,
        validation,
        test,
        low_threshold,
        high_threshold,
    ) = prepare_eda_data()

    print_numerical_summary(
        data,
        train,
        validation,
        test,
        low_threshold,
        high_threshold,
    )

    print()
    print(
        "=== CREATING FIGURES ==="
    )

    plot_financial_stress_over_time(
        data
    )

    plot_feature_correlations(
        train
    )

    plot_future_change_distribution(
        train,
        low_threshold,
        high_threshold,
    )

    plot_class_distribution(
        train,
        validation,
        test,
    )

    print()
    print(
        "EDA completed."
    )


if __name__ == "__main__":
    run_eda()
