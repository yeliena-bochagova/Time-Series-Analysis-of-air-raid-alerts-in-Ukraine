import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.paths import VOLUNTEER_PANEL_PATH


VALIDATION_DAYS = 30
TEST_DAYS = 30


def load_panel(path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Build the daily panel first: python -m src.prepare_daily_panel"
        )

    panel = pd.read_csv(path, parse_dates=["date"])
    required_columns = {"region", "date", "alert_starts", "has_alert_next_day"}
    missing_columns = required_columns - set(panel.columns)
    if missing_columns:
        raise ValueError(f"Panel is missing required columns: {sorted(missing_columns)}")

    return panel


def chronological_split(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    dates = pd.Index(sorted(panel["date"].unique()))
    needed_days = VALIDATION_DAYS + TEST_DAYS
    if len(dates) < needed_days:
        raise ValueError(f"Need at least {needed_days} forecast days, found {len(dates)}.")

    test_dates = dates[-TEST_DAYS:]
    validation_dates = dates[-(VALIDATION_DAYS + TEST_DAYS) : -TEST_DAYS]
    train_dates = dates[: -(VALIDATION_DAYS + TEST_DAYS)]

    return {
        "train": panel[panel["date"].isin(train_dates)].copy(),
        "validation": panel[panel["date"].isin(validation_dates)].copy(),
        "test": panel[panel["date"].isin(test_dates)].copy(),
    }


def print_split_ranges(splits: dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 90)
    print("CHRONOLOGICAL SPLITS")
    print("=" * 90)

    for name in ("train", "validation", "test"):
        split = splits[name]
        unique_days = split["date"].nunique()
        start = split["date"].min().date() if not split.empty else "none"
        end = split["date"].max().date() if not split.empty else "none"
        print(f"{name}: {start} to {end} ({unique_days:,} forecast days, {len(split):,} rows)")


def baseline_predictions(split: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "always_alert": pd.Series(1, index=split.index),
        "persistence": split["alert_starts"].gt(0).astype(int),
    }


def metric_row(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def print_confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    matrix_df = pd.DataFrame(
        matrix,
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    )
    print(matrix_df.to_string())


def evaluate_split(name: str, split: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print(f"{name.upper()} BASELINE RESULTS")
    print("=" * 90)

    y_true = split["has_alert_next_day"].astype(int)

    for baseline_name, y_pred in baseline_predictions(split).items():
        metrics = metric_row(y_true, y_pred)
        print(f"\n{baseline_name}")
        print(
            "accuracy={accuracy:.3f}  precision={precision:.3f}  recall={recall:.3f}  "
            "f1={f1:.3f}  balanced_accuracy={balanced_accuracy:.3f}".format(**metrics)
        )
        print("Confusion matrix:")
        print_confusion_matrix(y_true, y_pred)


def run_baseline_evaluation() -> dict[str, pd.DataFrame]:
    panel = load_panel(VOLUNTEER_PANEL_PATH)
    splits = chronological_split(panel)

    print_split_ranges(splits)
    evaluate_split("validation", splits["validation"])
    evaluate_split("test", splits["test"])

    return splits


def main() -> None:
    run_baseline_evaluation()


if __name__ == "__main__":
    main()
