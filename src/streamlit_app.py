import pandas as pd
import streamlit as st

from src.backtest_model import build_purged_fold, load_feature_panel, select_threshold
from src.evaluate_baselines import (
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    add_leakage_safe_features,
    confusion_counts,
    make_logistic_regression,
)
from src.paths import DATASETS, KYIV_TZ, OUTPUTS_DIR
from src.prepare_daily_panel import latest_complete_local_date
from src.raw_data_audit import load_csv as load_raw_csv


DEFAULT_VALIDATION_DAYS = 30
DEFAULT_EVALUATION_DAYS = 30


@st.cache_data
def load_cached_panel() -> pd.DataFrame:
    panel = load_feature_panel()
    panel["date"] = pd.to_datetime(panel["date"])
    panel["target_date"] = pd.to_datetime(panel["target_date"])
    return panel


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def score_forecast_day(
    panel: pd.DataFrame,
    region: str,
    forecast_date: str,
    validation_days: int = DEFAULT_VALIDATION_DAYS,
) -> dict:
    dates = pd.Index(sorted(panel["date"].unique()))
    selected_date = pd.Timestamp(forecast_date)
    eval_end_idx = dates.get_loc(selected_date)
    fold = build_purged_fold(
        panel,
        eval_end_idx=eval_end_idx,
        validation_days=validation_days,
        evaluation_days=1,
    )

    train = fold["train"]
    validation = fold["validation"]
    evaluation = fold["evaluation"]

    model = make_logistic_regression()
    model.fit(train[FEATURE_COLUMNS], train["has_alert_next_day"].astype(int))

    validation_probabilities = pd.Series(
        model.predict_proba(validation[FEATURE_COLUMNS])[:, 1],
        index=validation.index,
    )
    threshold, _, _ = select_threshold(
        validation_probabilities,
        validation["has_alert_next_day"].astype(int),
    )

    row = evaluation[
        evaluation["region"].eq(region) & evaluation["date"].eq(selected_date)
    ].iloc[0]
    model_score = float(model.predict_proba(row.to_frame().T[FEATURE_COLUMNS])[:, 1][0])
    logistic_prediction = int(model_score >= threshold)
    persistence_prediction = int(row["alert_starts"] > 0)
    actual = None if pd.isna(row["has_alert_next_day"]) else int(row["has_alert_next_day"])

    return {
        "model_score": model_score,
        "threshold": threshold,
        "logistic_prediction": logistic_prediction,
        "persistence_prediction": persistence_prediction,
        "actual": actual,
        "target_date": row["target_date"].date().isoformat(),
        "row": row.to_dict(),
        "train_start": train["date"].min().date().isoformat(),
        "train_end": train["date"].max().date().isoformat(),
        "validation_start": validation["date"].min().date().isoformat(),
        "validation_end": validation["date"].max().date().isoformat(),
        "evaluation_date": selected_date.date().isoformat(),
        "dropped_dates": [date.date().isoformat() for date in fold["dropped_dates"]],
    }


@st.cache_data
def score_historical_target_day(
    region: str,
    target_date: str,
    validation_days: int = DEFAULT_VALIDATION_DAYS,
) -> dict:
    panel = load_cached_panel()
    forecast_date = pd.Timestamp(target_date) - pd.Timedelta(days=1)
    return score_forecast_day(
        panel,
        region,
        forecast_date.date().isoformat(),
        validation_days=validation_days,
    )


@st.cache_data
def load_latest_input_panel() -> pd.DataFrame:
    panel = load_cached_panel()[["region", "date", "alert_starts", "has_alert_next_day"]].copy()
    raw = load_raw_csv(DATASETS["volunteer"]["path"])

    raw["started_at_utc"] = pd.to_datetime(raw["started_at"], errors="coerce", utc=True)
    raw = raw.dropna(subset=["region", "started_at_utc"]).copy()
    raw["started_at_kyiv"] = raw["started_at_utc"].dt.tz_convert(KYIV_TZ)
    raw["date"] = raw["started_at_kyiv"].dt.normalize().dt.tz_localize(None)

    latest_input_date = latest_complete_local_date(raw["date"])
    if latest_input_date > panel["date"].max():
        counts = (
            raw[raw["date"].eq(latest_input_date)]
            .groupby("region")
            .size()
            .to_dict()
        )
        latest_rows = [
            {
                "region": region,
                "date": latest_input_date,
                "alert_starts": int(counts.get(region, 0)),
                "has_alert_next_day": pd.NA,
            }
            for region in sorted(panel["region"].unique())
        ]
        panel = pd.concat([panel, pd.DataFrame(latest_rows)], ignore_index=True)

    panel["target_date"] = panel["date"] + pd.Timedelta(days=1)
    return add_leakage_safe_features(panel)


@st.cache_data
def score_latest_available_day(
    region: str,
    validation_days: int = DEFAULT_VALIDATION_DAYS,
) -> dict:
    panel = load_latest_input_panel()
    forecast_date = panel["date"].max().date().isoformat()
    return score_forecast_day(
        panel,
        region,
        forecast_date,
        validation_days=validation_days,
    )


def rates_from_counts(counts: dict[str, int]) -> dict[str, float | None]:
    alert_total = counts["tp"] + counts["fn"]
    quiet_total = counts["tn"] + counts["fp"]
    alert_recall = counts["tp"] / alert_total if alert_total else None
    quiet_recall = counts["tn"] / quiet_total if quiet_total else None
    balanced_accuracy = (
        (alert_recall + quiet_recall) / 2
        if alert_recall is not None and quiet_recall is not None
        else None
    )
    return {
        "balanced_accuracy": balanced_accuracy,
        "alert_recall": alert_recall,
        "quiet_recall": quiet_recall,
    }


@st.cache_data
def latest_per_region_comparison(
    validation_days: int = DEFAULT_VALIDATION_DAYS,
    evaluation_days: int = DEFAULT_EVALUATION_DAYS,
) -> tuple[pd.DataFrame, dict]:
    panel = load_cached_panel()
    dates = pd.Index(sorted(panel["date"].unique()))
    fold = build_purged_fold(
        panel,
        eval_end_idx=len(dates) - 1,
        validation_days=validation_days,
        evaluation_days=evaluation_days,
    )

    train = fold["train"]
    validation = fold["validation"]
    evaluation = fold["evaluation"].copy()

    model = make_logistic_regression()
    model.fit(train[FEATURE_COLUMNS], train["has_alert_next_day"].astype(int))

    validation_probabilities = pd.Series(
        model.predict_proba(validation[FEATURE_COLUMNS])[:, 1],
        index=validation.index,
    )
    threshold, _, _ = select_threshold(
        validation_probabilities,
        validation["has_alert_next_day"].astype(int),
    )

    evaluation["actual"] = evaluation["has_alert_next_day"].astype(int)
    evaluation["logistic_model_score"] = model.predict_proba(evaluation[FEATURE_COLUMNS])[:, 1]
    evaluation["logistic_prediction"] = evaluation["logistic_model_score"].ge(threshold).astype(int)
    evaluation["persistence_prediction"] = evaluation["alert_starts"].gt(0).astype(int)

    rows = []
    for region, group in evaluation.groupby("region"):
        y_true = group["actual"]
        logistic_counts = confusion_counts(y_true, group["logistic_prediction"])
        persistence_counts = confusion_counts(y_true, group["persistence_prediction"])
        logistic_rates = rates_from_counts(logistic_counts)
        persistence_rates = rates_from_counts(persistence_counts)
        rows.append(
            {
                "region": region,
                "forecast_days": len(group),
                "actual_alert_rate": y_true.mean(),
                "logistic_balanced_accuracy": logistic_rates["balanced_accuracy"],
                "persistence_balanced_accuracy": persistence_rates["balanced_accuracy"],
                "logistic_alert_recall": logistic_rates["alert_recall"],
                "persistence_alert_recall": persistence_rates["alert_recall"],
                "logistic_fp": logistic_counts["fp"],
                "logistic_fn": logistic_counts["fn"],
                "persistence_fp": persistence_counts["fp"],
                "persistence_fn": persistence_counts["fn"],
            }
        )

    summary = {
        "threshold": threshold,
        "evaluation_start": evaluation["date"].min().date().isoformat(),
        "evaluation_end": evaluation["date"].max().date().isoformat(),
    }
    return pd.DataFrame(rows).sort_values("region"), summary


def format_yes_no(value: int) -> str:
    return "Yes" if value else "No"


def get_scoreable_dates(panel: pd.DataFrame) -> pd.Index:
    dates = pd.Index(sorted(panel["date"].unique()))
    first_scoreable_idx = DEFAULT_VALIDATION_DAYS + 3
    return dates[first_scoreable_idx:]


def show_historical_inspection(panel: pd.DataFrame, selected_region: str) -> None:
    st.header("Historical Inspection")
    scoreable_dates = get_scoreable_dates(panel)
    scoreable_target_dates = scoreable_dates + pd.Timedelta(days=1)

    selected_target_date = st.date_input(
        "Day to predict",
        value=scoreable_target_dates[-1].date(),
        min_value=scoreable_target_dates[0].date(),
        max_value=scoreable_target_dates[-1].date(),
    )
    selected_target_timestamp = pd.Timestamp(selected_target_date)
    forecast_timestamp = selected_target_timestamp - pd.Timedelta(days=1)

    if scoreable_target_dates.get_indexer([selected_target_timestamp])[0] == -1:
        st.error("Selected date is not available in the processed panel.")
        return

    score = score_historical_target_day(
        selected_region,
        selected_target_timestamp.date().isoformat(),
    )
    st.caption(
        f"Prediction target: {selected_target_timestamp.date().isoformat()}. "
        f"Information cutoff: end of {forecast_timestamp.date().isoformat()} in Europe/Kyiv. "
        "The actual outcome is known here because this is historical inspection."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Logistic model score", f"{score['model_score']:.3f}")
    col2.metric("Threshold", f"{score['threshold']:.2f}")
    col3.metric("Logistic prediction", format_yes_no(score["logistic_prediction"]))
    col4.metric("Persistence prediction", format_yes_no(score["persistence_prediction"]))

    st.metric(
        f"Actual outcome on {score['target_date']}",
        format_yes_no(score["actual"]),
    )

    with st.expander("Chronological windows used for this scored day", expanded=False):
        st.write(
            pd.DataFrame(
                [
                    {
                        "train_forecast_start": score["train_start"],
                        "train_forecast_end": score["train_end"],
                        "validation_forecast_start": score["validation_start"],
                        "validation_forecast_end": score["validation_end"],
                        "information_cutoff_day": score["evaluation_date"],
                        "target_day": score["target_date"],
                        "dropped_forecast_dates": ", ".join(score["dropped_dates"]),
                    }
                ]
            )
        )

    st.subheader("Recent-History Inputs")
    feature_values = pd.DataFrame(
        [{"feature": feature, "value": score["row"][feature]} for feature in NUMERIC_FEATURES]
    )
    st.dataframe(feature_values, use_container_width=True, hide_index=True)

    history_start = forecast_timestamp - pd.Timedelta(days=30)
    history = panel[
        panel["region"].eq(selected_region)
        & panel["date"].between(history_start, forecast_timestamp)
    ][["date", "alert_starts", "has_alert_next_day"]].copy()
    history["date"] = history["date"].dt.date

    st.bar_chart(history.set_index("date")["alert_starts"])
    st.dataframe(history.tail(15), use_container_width=True, hide_index=True)


def show_latest_forecast(selected_region: str) -> None:
    st.header("Latest-Available Forecast")
    latest_panel = load_latest_input_panel()
    score = score_latest_available_day(selected_region)

    st.caption(
        f"Prediction target: {score['target_date']}. "
        f"Information cutoff: end of {score['evaluation_date']} in Europe/Kyiv. "
        "Later target dates are not offered because they would require alert history from days "
        "that are not complete or not in the data yet."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Logistic model score", f"{score['model_score']:.3f}")
    col2.metric("Threshold", f"{score['threshold']:.2f}")
    col3.metric("Logistic prediction", format_yes_no(score["logistic_prediction"]))
    col4.metric("Persistence prediction", format_yes_no(score["persistence_prediction"]))

    actual_text = "Not available yet" if score["actual"] is None else format_yes_no(score["actual"])
    st.metric(f"Actual outcome on {score['target_date']}", actual_text)

    with st.expander("Chronological windows used for this latest forecast", expanded=False):
        st.write(
            pd.DataFrame(
                [
                    {
                        "train_forecast_start": score["train_start"],
                        "train_forecast_end": score["train_end"],
                        "validation_forecast_start": score["validation_start"],
                        "validation_forecast_end": score["validation_end"],
                        "information_cutoff_day": score["evaluation_date"],
                        "target_day": score["target_date"],
                        "dropped_forecast_dates": ", ".join(score["dropped_dates"]),
                    }
                ]
            )
        )

    st.subheader("Latest Input Row")
    latest_inputs = pd.DataFrame(
        [{"feature": feature, "value": score["row"][feature]} for feature in NUMERIC_FEATURES]
    )
    st.dataframe(latest_inputs, use_container_width=True, hide_index=True)

    cutoff = pd.Timestamp(score["evaluation_date"])
    history_start = cutoff - pd.Timedelta(days=30)
    history = latest_panel[
        latest_panel["region"].eq(selected_region)
        & latest_panel["date"].between(history_start, cutoff)
    ][["date", "alert_starts", "has_alert_next_day"]].copy()
    history["date"] = history["date"].dt.date
    st.bar_chart(history.set_index("date")["alert_starts"])
    st.dataframe(history.tail(15), use_container_width=True, hide_index=True)


def show_backtest_artifacts() -> None:
    st.header("Saved Backtest Artifacts")
    if not OUTPUTS_DIR.exists():
        st.info("No outputs directory found. Run the CLI evaluation first to create saved results.")
        return

    csv_paths = sorted(OUTPUTS_DIR.glob("*.csv"))
    png_paths = sorted(OUTPUTS_DIR.glob("*.png"))

    if csv_paths:
        selected_csv = st.selectbox("Saved CSV result", [path.name for path in csv_paths])
        csv_path = next(path for path in csv_paths if path.name == selected_csv)
        st.dataframe(load_csv(str(csv_path)), use_container_width=True)
    else:
        st.info("No saved CSV results found under outputs/.")

    if png_paths:
        selected_plot = st.selectbox("Saved plot", [path.name for path in png_paths])
        plot_path = next(path for path in png_paths if path.name == selected_plot)
        st.image(str(plot_path), caption=selected_plot, use_container_width=True)
    else:
        st.info("No saved plots found under outputs/.")


def show_per_region_comparison() -> None:
    st.header("Latest-Period Per-Region Comparison")
    comparison, summary = latest_per_region_comparison()
    st.caption(
        "Computed with the same latest-period 30-day validation / 30-day evaluation procedure "
        "used by the CLI. This is for inspection only."
    )
    st.write(
        f"Evaluation forecast dates: {summary['evaluation_start']} to {summary['evaluation_end']}. "
        f"Selected threshold: {summary['threshold']:.2f}."
    )

    display = comparison.copy()
    percent_columns = [
        "actual_alert_rate",
        "logistic_balanced_accuracy",
        "persistence_balanced_accuracy",
        "logistic_alert_recall",
        "persistence_alert_recall",
    ]
    for column in percent_columns:
        display[column] = display[column].map(lambda value: None if pd.isna(value) else round(value, 3))
    st.dataframe(display, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="Air-Raid Alert Forecast Inspector",
        layout="wide",
    )
    st.title("Historical Forecasting Prototype")
    st.warning(
        "Inspection/demo layer only. This is not a live alert system and must not be used "
        "for safety decisions."
    )
    st.info(
        "V1 modeling excludes Luhanska oblast because its permanent siren is a different process "
        "not represented correctly by the event-start dataset. Crimea remains out of scope because "
        "it is absent from the volunteer dataset."
    )

    panel = load_cached_panel()
    regions = sorted(panel["region"].unique())
    selected_region = st.sidebar.selectbox("Region", regions)
    st.sidebar.caption(
        f"Historical panel input days: {panel['date'].min().date()} to {panel['date'].max().date()}, "
        f"{panel['region'].nunique()} regions"
    )

    tab_history, tab_latest, tab_regions, tab_outputs = st.tabs(
        ["Historical Inspection", "Latest Forecast", "Per-Region Comparison", "Saved Results"]
    )
    with tab_history:
        show_historical_inspection(panel, selected_region)
    with tab_latest:
        show_latest_forecast(selected_region)
    with tab_regions:
        show_per_region_comparison()
    with tab_outputs:
        show_backtest_artifacts()


if __name__ == "__main__":
    main()
