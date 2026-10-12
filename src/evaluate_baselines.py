import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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
VALIDATION_SUPPORTED_LOGISTIC_THRESHOLD = 0.50
EXCLUDED_V1_REGIONS = ["Luhanska oblast"]

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
CATEGORICAL_FEATURES = ["region"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def filter_v1_modeling_scope(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[~panel["region"].isin(EXCLUDED_V1_REGIONS)].copy()


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

    panel["target_date"] = panel["date"] + pd.Timedelta(days=1)
    return panel


def chronological_split(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    dates = pd.Index(sorted(panel["date"].unique()))
    needed_days = VALIDATION_DAYS + TEST_DAYS + 2
    if len(dates) < needed_days:
        raise ValueError(f"Need at least {needed_days} forecast days, found {len(dates)}.")

    test_dates = dates[-TEST_DAYS:]
    test_start = test_dates[0]

    validation_candidates = dates[dates < test_start - pd.Timedelta(days=1)]
    if len(validation_candidates) < VALIDATION_DAYS:
        raise ValueError("Not enough forecast days before the test purge gap for validation.")
    validation_dates = validation_candidates[-VALIDATION_DAYS:]
    validation_start = validation_dates[0]

    train_dates = dates[dates < validation_start - pd.Timedelta(days=1)]
    if train_dates.empty:
        raise ValueError("Not enough forecast days before the validation purge gap for training.")

    splits = {
        "train": panel[panel["date"].isin(train_dates)].copy(),
        "validation": panel[panel["date"].isin(validation_dates)].copy(),
        "test": panel[panel["date"].isin(test_dates)].copy(),
    }
    assert_no_split_target_overlap(splits)
    return splits


def assert_no_split_target_overlap(splits: dict[str, pd.DataFrame]) -> None:
    train_target_end = splits["train"]["target_date"].max()
    validation_start = splits["validation"]["date"].min()
    validation_target_end = splits["validation"]["target_date"].max()
    test_start = splits["test"]["date"].min()

    if train_target_end >= validation_start:
        raise ValueError("Training labels overlap the validation forecast period.")
    if validation_target_end >= test_start:
        raise ValueError("Validation labels overlap the test forecast period.")


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

    return panel


def print_split_ranges(splits: dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 90)
    print("CHRONOLOGICAL SPLITS")
    print("=" * 90)

    for name in ("train", "validation", "test"):
        split = splits[name]
        unique_days = split["date"].nunique()
        start = split["date"].min().date() if not split.empty else "none"
        end = split["date"].max().date() if not split.empty else "none"
        target_start = split["target_date"].min().date() if not split.empty else "none"
        target_end = split["target_date"].max().date() if not split.empty else "none"
        print(
            f"{name}: forecast {start} to {end}; target {target_start} to {target_end} "
            f"({unique_days:,} forecast days, {len(split):,} rows)"
        )


def baseline_predictions(split: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "always_alert": pd.Series(1, index=split.index),
        "persistence": split["alert_starts"].gt(0).astype(int),
    }


def make_logistic_regression() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("region", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def make_constrained_random_forest() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("region", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=4,
                    min_samples_leaf=50,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
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


def print_validation_threshold_tuning(splits: dict[str, pd.DataFrame], model: Pipeline) -> None:
    print("\n" + "=" * 90)
    print("VALIDATION LOGISTIC REGRESSION THRESHOLD TUNING")
    print("=" * 90)
    print("Original logistic regression features and model; validation only.")

    validation = splits["validation"]
    y_validation = validation["has_alert_next_day"].astype(int)
    probabilities = pd.Series(
        model.predict_proba(validation[FEATURE_COLUMNS])[:, 1],
        index=validation.index,
    )

    thresholds = [round(threshold, 2) for threshold in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]]
    rows = []
    for threshold in thresholds:
        prediction = probabilities.ge(threshold).astype(int)
        metrics = metric_row(y_validation, prediction)
        counts = confusion_counts(y_validation, prediction)
        rows.append(
            {
                "threshold": threshold,
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
    print("\nValidation threshold sweep:")
    print(
        summary.to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "balanced_accuracy": "{:.3f}".format,
                "accuracy": "{:.3f}".format,
                "precision": "{:.3f}".format,
                "recall": "{:.3f}".format,
                "f1": "{:.3f}".format,
            },
        )
    )

    best = summary.sort_values(
        ["balanced_accuracy", "f1", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]
    default = summary[summary["threshold"].eq(0.50)].iloc[0]

    print("\nBest validation threshold by balanced accuracy:")
    print(
        f"threshold={best['threshold']:.2f}, balanced_accuracy={best['balanced_accuracy']:.3f}, "
        f"false_positives={int(best['false_positives'])}, false_negatives={int(best['false_negatives'])}"
    )
    print(
        "Default 0.50 threshold for comparison: "
        f"balanced_accuracy={default['balanced_accuracy']:.3f}, "
        f"false_positives={int(default['false_positives'])}, "
        f"false_negatives={int(default['false_negatives'])}"
    )


def validation_result_row(name: str, y_true: pd.Series, y_pred: pd.Series) -> dict:
    metrics = metric_row(y_true, y_pred)
    counts = confusion_counts(y_true, y_pred)
    return {
        "model": name,
        "balanced_accuracy": metrics["balanced_accuracy"],
        "alert_recall": metrics["recall"],
        "precision": metrics["precision"],
        "f1": metrics["f1"],
        "false_positives": counts["fp"],
        "false_negatives": counts["fn"],
        "true_negatives": counts["tn"],
        "true_positives": counts["tp"],
    }


def print_validation_final_model_experiment(splits: dict[str, pd.DataFrame], logistic_model: Pipeline) -> None:
    print("\n" + "=" * 90)
    print("VALIDATION FINAL MODEL EXPERIMENT")
    print("=" * 90)
    print("Validation only. Same local features and chronological split.")
    print(
        "Nonlinear model: RandomForestClassifier("
        "n_estimators=200, max_depth=4, min_samples_leaf=50, class_weight='balanced_subsample')."
    )

    train = splits["train"]
    validation = splits["validation"]
    y_train = train["has_alert_next_day"].astype(int)
    y_validation = validation["has_alert_next_day"].astype(int)

    logistic_probabilities = logistic_model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
    logistic_prediction = pd.Series(
        logistic_probabilities,
        index=validation.index,
    ).ge(VALIDATION_SUPPORTED_LOGISTIC_THRESHOLD).astype(int)

    random_forest = make_constrained_random_forest()
    random_forest.fit(train[FEATURE_COLUMNS], y_train)
    random_forest_prediction = pd.Series(
        random_forest.predict(validation[FEATURE_COLUMNS]),
        index=validation.index,
    )

    persistence_prediction = validation["alert_starts"].gt(0).astype(int)
    rows = [
        validation_result_row("persistence", y_validation, persistence_prediction),
        validation_result_row(
            f"logistic_regression_threshold_{VALIDATION_SUPPORTED_LOGISTIC_THRESHOLD:.2f}",
            y_validation,
            logistic_prediction,
        ),
        validation_result_row("constrained_random_forest", y_validation, random_forest_prediction),
    ]
    summary = pd.DataFrame(rows)

    print("\nValidation comparison:")
    print(
        summary.to_string(
            index=False,
            formatters={
                "balanced_accuracy": "{:.3f}".format,
                "alert_recall": "{:.3f}".format,
                "precision": "{:.3f}".format,
                "f1": "{:.3f}".format,
            },
        )
    )


def run_baseline_evaluation() -> dict[str, pd.DataFrame]:
    panel = add_leakage_safe_features(filter_v1_modeling_scope(load_panel(VOLUNTEER_PANEL_PATH)))
    splits = chronological_split(panel)

    print_split_ranges(splits)
    evaluate_split("validation", splits["validation"])
    evaluate_split("test", splits["test"])

    print_feature_list()
    model = make_logistic_regression()
    model.fit(splits["train"][FEATURE_COLUMNS], splits["train"]["has_alert_next_day"].astype(int))
    evaluate_model_split("validation", splits["validation"], model)
    evaluate_model_split("test", splits["test"], model)
    print_validation_threshold_tuning(splits, model)
    print_validation_final_model_experiment(splits, model)

    return splits


def main() -> None:
    run_baseline_evaluation()


if __name__ == "__main__":
    main()
