import pandas as pd
import matplotlib.pyplot as plt

from src.evaluate_baselines import (
    FEATURE_COLUMNS,
    add_leakage_safe_features,
    confusion_counts,
    load_panel,
    make_logistic_regression,
    metric_row,
)
from src.paths import OUTPUTS_DIR, VOLUNTEER_PANEL_PATH


THRESHOLDS = [round(threshold, 2) for threshold in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)]


def build_purged_fold(
    panel: pd.DataFrame,
    eval_end_idx: int,
    validation_days: int,
    evaluation_days: int,
) -> dict[str, pd.DataFrame | list[pd.Timestamp]]:
    dates = pd.Index(sorted(panel["date"].unique()))

    eval_start_idx = eval_end_idx - evaluation_days + 1
    validation_end_idx = eval_start_idx - 2
    validation_start_idx = validation_end_idx - validation_days + 1
    train_end_idx = validation_start_idx - 2

    if min(eval_start_idx, validation_end_idx, validation_start_idx, train_end_idx) < 0:
        raise ValueError("Not enough history to build this fold.")

    train_dates = dates[: train_end_idx + 1]
    validation_dates = dates[validation_start_idx : validation_end_idx + 1]
    evaluation_dates = dates[eval_start_idx : eval_end_idx + 1]
    dropped_dates = [dates[train_end_idx + 1], dates[validation_end_idx + 1]]

    fold = {
        "train": panel[panel["date"].isin(train_dates)].copy(),
        "validation": panel[panel["date"].isin(validation_dates)].copy(),
        "evaluation": panel[panel["date"].isin(evaluation_dates)].copy(),
        "dropped_dates": dropped_dates,
    }
    assert_purged_fold(fold)
    return fold


def assert_purged_fold(fold: dict[str, pd.DataFrame | list[pd.Timestamp]]) -> None:
    train = fold["train"]
    validation = fold["validation"]
    evaluation = fold["evaluation"]

    if train["target_date"].max() >= validation["date"].min():
        raise ValueError("Training labels overlap the validation forecast period.")
    if validation["target_date"].max() >= evaluation["date"].min():
        raise ValueError("Validation labels overlap the evaluation forecast period.")


def select_threshold(
    probabilities: pd.Series,
    y_true: pd.Series,
) -> tuple[float, dict[str, float], dict[str, int]]:
    rows = []
    for threshold in THRESHOLDS:
        prediction = probabilities.ge(threshold).astype(int)
        metrics = metric_row(y_true, prediction)
        counts = confusion_counts(y_true, prediction)
        rows.append(
            {
                "threshold": threshold,
                "balanced_accuracy": metrics["balanced_accuracy"],
                "f1": metrics["f1"],
                "false_positives": counts["fp"],
                "false_negatives": counts["fn"],
                "metrics": metrics,
                "counts": counts,
            }
        )

    best = sorted(
        rows,
        key=lambda row: (row["balanced_accuracy"], row["f1"], -row["threshold"]),
        reverse=True,
    )[0]
    return best["threshold"], best["metrics"], best["counts"]


def model_result_row(name: str, y_true: pd.Series, y_pred: pd.Series) -> dict:
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


def evaluate_fold(
    panel: pd.DataFrame,
    fold_id: int,
    eval_end_idx: int,
    validation_days: int,
    evaluation_days: int,
) -> dict:
    fold = build_purged_fold(panel, eval_end_idx, validation_days, evaluation_days)
    train = fold["train"]
    validation = fold["validation"]
    evaluation = fold["evaluation"]

    model = make_logistic_regression()
    model.fit(train[FEATURE_COLUMNS], train["has_alert_next_day"].astype(int))

    validation_y = validation["has_alert_next_day"].astype(int)
    validation_probabilities = pd.Series(
        model.predict_proba(validation[FEATURE_COLUMNS])[:, 1],
        index=validation.index,
    )
    threshold, validation_metrics, validation_counts = select_threshold(
        validation_probabilities,
        validation_y,
    )

    evaluation_y = evaluation["has_alert_next_day"].astype(int)
    evaluation_probabilities = pd.Series(
        model.predict_proba(evaluation[FEATURE_COLUMNS])[:, 1],
        index=evaluation.index,
    )
    logistic_prediction = evaluation_probabilities.ge(threshold).astype(int)
    persistence_prediction = evaluation["alert_starts"].gt(0).astype(int)

    logistic_result = model_result_row("logistic_regression", evaluation_y, logistic_prediction)
    persistence_result = model_result_row("persistence", evaluation_y, persistence_prediction)

    return {
        "fold": fold_id,
        "validation_days": validation_days,
        "evaluation_days": evaluation_days,
        "selected_threshold": threshold,
        "train_forecast_start": train["date"].min().date(),
        "train_forecast_end": train["date"].max().date(),
        "validation_forecast_start": validation["date"].min().date(),
        "validation_forecast_end": validation["date"].max().date(),
        "evaluation_forecast_start": evaluation["date"].min().date(),
        "evaluation_forecast_end": evaluation["date"].max().date(),
        "dropped_forecast_dates": ", ".join(date.date().isoformat() for date in fold["dropped_dates"]),
        "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
        "validation_alert_recall": validation_metrics["recall"],
        "validation_false_positives": validation_counts["fp"],
        "validation_false_negatives": validation_counts["fn"],
        "eval_logistic_balanced_accuracy": logistic_result["balanced_accuracy"],
        "eval_logistic_alert_recall": logistic_result["alert_recall"],
        "eval_logistic_false_positives": logistic_result["false_positives"],
        "eval_logistic_false_negatives": logistic_result["false_negatives"],
        "eval_persistence_balanced_accuracy": persistence_result["balanced_accuracy"],
        "eval_persistence_alert_recall": persistence_result["alert_recall"],
        "eval_persistence_false_positives": persistence_result["false_positives"],
        "eval_persistence_false_negatives": persistence_result["false_negatives"],
    }


def run_backtest(
    panel: pd.DataFrame,
    validation_days: int,
    evaluation_days: int,
    fold_count: int,
    step_days: int,
) -> pd.DataFrame:
    dates = pd.Index(sorted(panel["date"].unique()))
    rows = []

    for fold_id in range(1, fold_count + 1):
        eval_end_idx = len(dates) - 1 - ((fold_id - 1) * step_days)
        rows.append(
            evaluate_fold(
                panel,
                fold_id,
                eval_end_idx,
                validation_days,
                evaluation_days,
            )
        )

    return pd.DataFrame(rows)


def print_fold_table(title: str, results: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

    columns = [
        "fold",
        "train_forecast_end",
        "validation_forecast_start",
        "validation_forecast_end",
        "evaluation_forecast_start",
        "evaluation_forecast_end",
        "dropped_forecast_dates",
        "selected_threshold",
        "validation_balanced_accuracy",
        "validation_false_positives",
        "validation_false_negatives",
        "eval_logistic_balanced_accuracy",
        "eval_logistic_alert_recall",
        "eval_logistic_false_positives",
        "eval_logistic_false_negatives",
        "eval_persistence_balanced_accuracy",
        "eval_persistence_alert_recall",
        "eval_persistence_false_positives",
        "eval_persistence_false_negatives",
    ]
    print(
        results[columns].to_string(
            index=False,
            formatters={
                "selected_threshold": "{:.2f}".format,
                "validation_balanced_accuracy": "{:.3f}".format,
                "eval_logistic_balanced_accuracy": "{:.3f}".format,
                "eval_logistic_alert_recall": "{:.3f}".format,
                "eval_persistence_balanced_accuracy": "{:.3f}".format,
                "eval_persistence_alert_recall": "{:.3f}".format,
            },
        )
    )


def print_compact_comparison(title: str, results: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

    compact = results[
        [
            "fold",
            "evaluation_forecast_start",
            "evaluation_forecast_end",
            "selected_threshold",
            "eval_logistic_balanced_accuracy",
            "eval_persistence_balanced_accuracy",
            "eval_logistic_alert_recall",
            "eval_persistence_alert_recall",
            "eval_logistic_false_positives",
            "eval_logistic_false_negatives",
            "eval_persistence_false_positives",
            "eval_persistence_false_negatives",
        ]
    ].copy()
    compact = compact.rename(
        columns={
            "evaluation_forecast_start": "eval_start",
            "evaluation_forecast_end": "eval_end",
            "selected_threshold": "threshold",
            "eval_logistic_balanced_accuracy": "log_bal_acc",
            "eval_persistence_balanced_accuracy": "pers_bal_acc",
            "eval_logistic_alert_recall": "log_recall",
            "eval_persistence_alert_recall": "pers_recall",
            "eval_logistic_false_positives": "log_fp",
            "eval_logistic_false_negatives": "log_fn",
            "eval_persistence_false_positives": "pers_fp",
            "eval_persistence_false_negatives": "pers_fn",
        }
    )
    print(
        compact.to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "log_bal_acc": "{:.3f}".format,
                "pers_bal_acc": "{:.3f}".format,
                "log_recall": "{:.3f}".format,
                "pers_recall": "{:.3f}".format,
            },
        )
    )


def print_summary(title: str, results: pd.DataFrame) -> None:
    print("\n" + title)
    print("-" * len(title))

    summary = results.agg(
        {
            "eval_logistic_balanced_accuracy": ["mean", "min", "max"],
            "eval_logistic_alert_recall": ["mean", "min", "max"],
            "eval_logistic_false_positives": ["mean", "sum"],
            "eval_logistic_false_negatives": ["mean", "sum"],
            "eval_persistence_balanced_accuracy": ["mean", "min", "max"],
            "eval_persistence_alert_recall": ["mean", "min", "max"],
            "eval_persistence_false_positives": ["mean", "sum"],
            "eval_persistence_false_negatives": ["mean", "sum"],
        }
    )
    print(summary.round(3).to_string())


def save_results_csv(results: pd.DataFrame, filename: str) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / filename
    results.to_csv(path, index=False)
    print(f"\nSaved detailed fold results: {path}")


def save_balanced_accuracy_plot(results: pd.DataFrame, filename: str, title: str) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / filename

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(results["fold"], results["eval_logistic_balanced_accuracy"], marker="o", label="Logistic regression")
    ax.plot(results["fold"], results["eval_persistence_balanced_accuracy"], marker="o", label="Persistence")
    ax.set_title(title)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Balanced accuracy")
    ax.set_xticks(results["fold"])
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {path}")


def save_fp_fn_plot(results: pd.DataFrame, filename: str, title: str) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / filename

    labels = [str(fold) for fold in results["fold"]]
    x = range(len(results))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar([idx - 1.5 * width for idx in x], results["eval_logistic_false_positives"], width, label="Logistic FP")
    ax.bar([idx - 0.5 * width for idx in x], results["eval_logistic_false_negatives"], width, label="Logistic FN")
    ax.bar([idx + 0.5 * width for idx in x], results["eval_persistence_false_positives"], width, label="Persistence FP")
    ax.bar([idx + 1.5 * width for idx in x], results["eval_persistence_false_negatives"], width, label="Persistence FN")
    ax.set_title(title)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Count")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {path}")


def save_backtest_outputs(results: pd.DataFrame, prefix: str, title: str) -> None:
    save_results_csv(results, f"{prefix}_fold_results.csv")
    save_balanced_accuracy_plot(results, f"{prefix}_balanced_accuracy_by_fold.png", f"{title}: Balanced Accuracy")
    save_fp_fn_plot(results, f"{prefix}_fp_fn_by_fold.png", f"{title}: FP/FN Comparison")


def load_feature_panel() -> pd.DataFrame:
    return add_leakage_safe_features(load_panel(VOLUNTEER_PANEL_PATH))


def run_latest_period_evaluation(
    panel: pd.DataFrame,
    validation_days: int,
    evaluation_days: int,
) -> pd.DataFrame:
    results = run_backtest(
        panel,
        validation_days=validation_days,
        evaluation_days=evaluation_days,
        fold_count=1,
        step_days=evaluation_days,
    )
    print_compact_comparison("LATEST-PERIOD EVALUATION", results)
    print_summary("Latest-period summary", results)
    save_backtest_outputs(results, "latest_period_evaluation", "Latest-Period Evaluation")
    return results


def run_historical_backtests(
    validation_days: int = 30,
    evaluation_days: int = 30,
    fold_count: int = 6,
    step_days: int = 90,
    sensitivity_specs: list[tuple[str, int, int, int, int]] | None = None,
) -> None:
    panel = load_feature_panel()

    main_results = run_backtest(
        panel,
        validation_days=validation_days,
        evaluation_days=evaluation_days,
        fold_count=fold_count,
        step_days=step_days,
    )
    label = f"{validation_days}-day validation / {evaluation_days}-day evaluation"
    print_compact_comparison(f"HISTORICAL BACKTEST: {label.upper()}", main_results)
    print_summary(f"{label} historical summary", main_results)
    save_backtest_outputs(main_results, "historical_backtest", "Historical Backtest")

    if sensitivity_specs is None:
        sensitivity_specs = [
            ("14-day validation / 14-day evaluation", 14, 14, 6, 60),
            ("60-day validation / 60-day evaluation", 60, 60, 4, 120),
        ]

    sensitivity_rows = []

    for label, validation_days, evaluation_days, fold_count, step_days in sensitivity_specs:
        results = run_backtest(
            panel,
            validation_days=validation_days,
            evaluation_days=evaluation_days,
            fold_count=fold_count,
            step_days=step_days,
        )
        print_compact_comparison(f"SENSITIVITY CHECK: {label.upper()}", results)
        print_summary(f"{label} summary", results)
        prefix = label.replace(" ", "_").replace("/", "per").replace("-", "_")
        save_backtest_outputs(results, f"sensitivity_{prefix}", f"Sensitivity: {label}")
        sensitivity_rows.append(
            {
                "window": label,
                "folds": len(results),
                "mean_logistic_balanced_accuracy": results["eval_logistic_balanced_accuracy"].mean(),
                "mean_logistic_alert_recall": results["eval_logistic_alert_recall"].mean(),
                "total_logistic_false_positives": results["eval_logistic_false_positives"].sum(),
                "total_logistic_false_negatives": results["eval_logistic_false_negatives"].sum(),
                "mean_persistence_balanced_accuracy": results["eval_persistence_balanced_accuracy"].mean(),
                "mean_persistence_alert_recall": results["eval_persistence_alert_recall"].mean(),
                "total_persistence_false_positives": results["eval_persistence_false_positives"].sum(),
                "total_persistence_false_negatives": results["eval_persistence_false_negatives"].sum(),
            }
        )

    print("\n" + "=" * 90)
    print("WINDOW-SIZE SENSITIVITY SUMMARY")
    print("=" * 90)
    sensitivity_summary = pd.DataFrame(sensitivity_rows)
    print(
        sensitivity_summary.to_string(
            index=False,
            formatters={
                "mean_logistic_balanced_accuracy": "{:.3f}".format,
                "mean_logistic_alert_recall": "{:.3f}".format,
                "mean_persistence_balanced_accuracy": "{:.3f}".format,
                "mean_persistence_alert_recall": "{:.3f}".format,
            },
        )
    )
    save_results_csv(sensitivity_summary, "window_size_sensitivity_summary.csv")


def run_cli_pipeline(
    mode: str,
    validation_days: int,
    evaluation_days: int,
    folds: int,
    step_days: int,
    include_sensitivity: bool,
) -> None:
    panel = load_feature_panel()

    if mode in ("latest", "all"):
        run_latest_period_evaluation(panel, validation_days, evaluation_days)

    if mode in ("backtest", "all"):
        sensitivity_specs = None if include_sensitivity else []
        main_results = run_backtest(
            panel,
            validation_days=validation_days,
            evaluation_days=evaluation_days,
            fold_count=folds,
            step_days=step_days,
        )
        label = f"{validation_days}-day validation / {evaluation_days}-day evaluation"
        print_compact_comparison(f"HISTORICAL BACKTEST: {label.upper()}", main_results)
        print_summary(f"{label} historical summary", main_results)
        save_backtest_outputs(main_results, "historical_backtest", "Historical Backtest")

        if sensitivity_specs is None:
            sensitivity_specs = [
                ("14-day validation / 14-day evaluation", 14, 14, 6, 60),
                ("60-day validation / 60-day evaluation", 60, 60, 4, 120),
            ]
        if sensitivity_specs:
            sensitivity_rows = []
            for label, sens_validation_days, sens_evaluation_days, sens_folds, sens_step_days in sensitivity_specs:
                results = run_backtest(
                    panel,
                    validation_days=sens_validation_days,
                    evaluation_days=sens_evaluation_days,
                    fold_count=sens_folds,
                    step_days=sens_step_days,
                )
                print_compact_comparison(f"SENSITIVITY CHECK: {label.upper()}", results)
                print_summary(f"{label} summary", results)
                prefix = label.replace(" ", "_").replace("/", "per").replace("-", "_")
                save_backtest_outputs(results, f"sensitivity_{prefix}", f"Sensitivity: {label}")
                sensitivity_rows.append(
                    {
                        "window": label,
                        "folds": len(results),
                        "mean_logistic_balanced_accuracy": results["eval_logistic_balanced_accuracy"].mean(),
                        "mean_logistic_alert_recall": results["eval_logistic_alert_recall"].mean(),
                        "total_logistic_false_positives": results["eval_logistic_false_positives"].sum(),
                        "total_logistic_false_negatives": results["eval_logistic_false_negatives"].sum(),
                        "mean_persistence_balanced_accuracy": results["eval_persistence_balanced_accuracy"].mean(),
                        "mean_persistence_alert_recall": results["eval_persistence_alert_recall"].mean(),
                        "total_persistence_false_positives": results["eval_persistence_false_positives"].sum(),
                        "total_persistence_false_negatives": results["eval_persistence_false_negatives"].sum(),
                    }
                )

            sensitivity_summary = pd.DataFrame(sensitivity_rows)
            print("\n" + "=" * 90)
            print("WINDOW-SIZE SENSITIVITY SUMMARY")
            print("=" * 90)
            print(
                sensitivity_summary.to_string(
                    index=False,
                    formatters={
                        "mean_logistic_balanced_accuracy": "{:.3f}".format,
                        "mean_logistic_alert_recall": "{:.3f}".format,
                        "mean_persistence_balanced_accuracy": "{:.3f}".format,
                        "mean_persistence_alert_recall": "{:.3f}".format,
                    },
                )
            )
            save_results_csv(sensitivity_summary, "window_size_sensitivity_summary.csv")


def main() -> None:
    run_historical_backtests()


if __name__ == "__main__":
    main()
