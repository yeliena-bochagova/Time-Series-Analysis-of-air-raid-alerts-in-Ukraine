import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.paths import VOLUNTEER_PANEL_PATH


VALIDATION_DAYS = 30
TEST_DAYS = 30

NUMERIC_FEATURES = [
    "alert_starts_today",
    "has_alert_today",
    "alert_starts_last_3d",
    "alert_days_last_3d",
    "alert_starts_last_7d",
    "alert_days_last_7d",
    "alert_starts_last_14d",
    "alert_days_last_14d",
    "alert_starts_last_30d",
    "alert_days_last_30d",
    "days_since_last_alert",
    "day_of_week",
    "is_weekend",
    "month",
]
NATIONAL_CONTEXT_FEATURES = [
    "national_regions_with_alert_today",
    "national_alert_starts_today",
    "national_regions_with_alert_last_3d",
    "national_alert_starts_last_3d",
]
CATEGORICAL_FEATURES = ["region"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
FEATURE_COLUMNS_WITH_NATIONAL_CONTEXT = NUMERIC_FEATURES + NATIONAL_CONTEXT_FEATURES + CATEGORICAL_FEATURES


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


def add_leakage_safe_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["region", "date"]).copy()
    panel["alert_starts_today"] = panel["alert_starts"].astype(int)
    panel["has_alert_today"] = panel["alert_starts_today"].gt(0).astype(int)

    for days in (3, 7, 14, 30):
        panel[f"alert_starts_last_{days}d"] = (
            panel.groupby("region")["alert_starts_today"]
            .rolling(window=days, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        panel[f"alert_days_last_{days}d"] = (
            panel.groupby("region")["has_alert_today"]
            .rolling(window=days, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

    alert_dates = panel["date"].where(panel["has_alert_today"].eq(1))
    panel["last_alert_date"] = alert_dates.groupby(panel["region"]).ffill()
    panel["days_since_last_alert"] = (panel["date"] - panel["last_alert_date"]).dt.days
    panel["days_since_last_alert"] = panel["days_since_last_alert"].fillna(999).astype(int)
    panel = panel.drop(columns=["last_alert_date"])

    panel["day_of_week"] = panel["date"].dt.dayofweek
    panel["is_weekend"] = panel["day_of_week"].isin([5, 6]).astype(int)
    panel["month"] = panel["date"].dt.month
    panel = add_national_context_features(panel)

    return panel


def add_national_context_features(panel: pd.DataFrame) -> pd.DataFrame:
    daily_national = (
        panel.groupby("date", as_index=False)
        .agg(
            national_regions_with_alert_today=("has_alert_today", "sum"),
            national_alert_starts_today=("alert_starts_today", "sum"),
        )
        .sort_values("date")
    )

    regional_recent_alert = panel[["region", "date", "has_alert_today"]].copy()
    regional_recent_alert["has_alert_in_last_3d"] = (
        regional_recent_alert.sort_values(["region", "date"])
        .groupby("region")["has_alert_today"]
        .rolling(window=3, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .gt(0)
        .astype(int)
    )
    regions_with_alert_last_3d = (
        regional_recent_alert.groupby("date", as_index=False)["has_alert_in_last_3d"]
        .sum()
        .rename(columns={"has_alert_in_last_3d": "national_regions_with_alert_last_3d"})
    )

    daily_national["national_alert_starts_last_3d"] = (
        daily_national["national_alert_starts_today"].rolling(window=3, min_periods=1).sum()
    )
    daily_national = daily_national.merge(regions_with_alert_last_3d, on="date", how="left")

    return panel.merge(daily_national, on="date", how="left")


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


def make_logistic_regression(numeric_features: list[str] | None = None) -> Pipeline:
    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("region", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


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


def print_feature_list() -> None:
    print("\n" + "=" * 90)
    print("LOGISTIC REGRESSION FEATURES")
    print("=" * 90)
    print("Numeric features:")
    for feature in NUMERIC_FEATURES:
        print(f"- {feature}")
    print("Categorical features:")
    for feature in CATEGORICAL_FEATURES:
        print(f"- {feature} one-hot encoded")


def print_national_context_feature_list() -> None:
    print("\nNational context features tested:")
    for feature in NATIONAL_CONTEXT_FEATURES:
        print(f"- {feature}")


def evaluate_model_split(name: str, split: pd.DataFrame, model: Pipeline) -> None:
    print("\n" + "=" * 90)
    print(f"{name.upper()} PERSISTENCE VS LOGISTIC REGRESSION")
    print("=" * 90)

    y_true = split["has_alert_next_day"].astype(int)
    predictions = {
        "persistence": split["alert_starts"].gt(0).astype(int),
        "logistic_regression": pd.Series(model.predict(split[FEATURE_COLUMNS]), index=split.index),
    }

    for model_name, y_pred in predictions.items():
        metrics = metric_row(y_true, y_pred)
        print(f"\n{model_name}")
        print(
            "accuracy={accuracy:.3f}  precision={precision:.3f}  recall={recall:.3f}  "
            "f1={f1:.3f}  balanced_accuracy={balanced_accuracy:.3f}".format(**metrics)
        )
        print("Confusion matrix:")
        print_confusion_matrix(y_true, y_pred)


def confusion_counts(y_true: pd.Series, y_pred: pd.Series) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def print_validation_shared_threat_experiment(splits: dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 90)
    print("VALIDATION SHARED-THREAT FEATURE EXPERIMENT")
    print("=" * 90)
    print("Same logistic regression, same train/validation split, same 0.5 threshold.")
    print_national_context_feature_list()

    y_train = splits["train"]["has_alert_next_day"].astype(int)
    validation = splits["validation"]
    y_validation = validation["has_alert_next_day"].astype(int)

    current_model = make_logistic_regression(NUMERIC_FEATURES)
    current_model.fit(splits["train"][FEATURE_COLUMNS], y_train)
    current_pred = pd.Series(current_model.predict(validation[FEATURE_COLUMNS]), index=validation.index)

    national_model = make_logistic_regression(NUMERIC_FEATURES + NATIONAL_CONTEXT_FEATURES)
    national_model.fit(splits["train"][FEATURE_COLUMNS_WITH_NATIONAL_CONTEXT], y_train)
    national_pred = pd.Series(
        national_model.predict(validation[FEATURE_COLUMNS_WITH_NATIONAL_CONTEXT]),
        index=validation.index,
    )

    rows = []
    for name, prediction in (
        ("current_logistic_regression", current_pred),
        ("with_national_context", national_pred),
    ):
        metrics = metric_row(y_validation, prediction)
        counts = confusion_counts(y_validation, prediction)
        rows.append(
            {
                "model": name,
                "balanced_accuracy": metrics["balanced_accuracy"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "false_positives": counts["fp"],
                "false_negatives": counts["fn"],
                "true_negatives": counts["tn"],
                "true_positives": counts["tp"],
            }
        )

    summary = pd.DataFrame(rows)
    print("\nValidation comparison:")
    print(
        summary.to_string(
            index=False,
            formatters={
                "balanced_accuracy": "{:.3f}".format,
                "accuracy": "{:.3f}".format,
                "precision": "{:.3f}".format,
                "recall": "{:.3f}".format,
                "f1": "{:.3f}".format,
            },
        )
    )

    changed = validation.copy()
    changed["current_pred"] = current_pred
    changed["national_pred"] = national_pred
    changed["y"] = y_validation
    recovered_fn = changed[
        changed["current_pred"].eq(0) & changed["national_pred"].eq(1) & changed["y"].eq(1)
    ]
    new_fp = changed[
        changed["current_pred"].eq(0) & changed["national_pred"].eq(1) & changed["y"].eq(0)
    ]
    lost_tp = changed[
        changed["current_pred"].eq(1) & changed["national_pred"].eq(0) & changed["y"].eq(1)
    ]
    corrected_fp = changed[
        changed["current_pred"].eq(1) & changed["national_pred"].eq(0) & changed["y"].eq(0)
    ]

    print("\nPrediction changes from adding national context:")
    print(f"- recovered false negatives: {len(recovered_fn):,}")
    print(f"- new false positives: {len(new_fp):,}")
    print(f"- lost true positives: {len(lost_tp):,}")
    print(f"- corrected false positives: {len(corrected_fp):,}")

    if not recovered_fn.empty:
        print("\nRecovered false negatives by region:")
        print(recovered_fn["region"].value_counts().head(10).to_string())
    if not new_fp.empty:
        print("\nNew false positives by region:")
        print(new_fp["region"].value_counts().head(10).to_string())
    if not lost_tp.empty:
        print("\nLost true positives by region:")
        print(lost_tp["region"].value_counts().head(10).to_string())
    if not corrected_fp.empty:
        print("\nCorrected false positives by region:")
        print(corrected_fp["region"].value_counts().head(10).to_string())


def run_baseline_evaluation() -> dict[str, pd.DataFrame]:
    panel = add_leakage_safe_features(load_panel(VOLUNTEER_PANEL_PATH))
    splits = chronological_split(panel)

    print_split_ranges(splits)
    evaluate_split("validation", splits["validation"])
    evaluate_split("test", splits["test"])

    print_feature_list()
    model = make_logistic_regression()
    model.fit(splits["train"][FEATURE_COLUMNS], splits["train"]["has_alert_next_day"].astype(int))
    evaluate_model_split("validation", splits["validation"], model)
    evaluate_model_split("test", splits["test"], model)
    print_validation_shared_threat_experiment(splits)

    return splits


def main() -> None:
    run_baseline_evaluation()


if __name__ == "__main__":
    main()
