import pandas as pd

from src.paths import DATASETS, KYIV_TZ, VOLUNTEER_PANEL_PATH
from src.raw_data_audit import load_csv


def latest_complete_local_date(local_dates: pd.Series) -> pd.Timestamp:
    latest_observed_date = local_dates.max()
    today_local = pd.Timestamp.now(tz=KYIV_TZ).normalize().tz_localize(None)

    if latest_observed_date >= today_local:
        return today_local - pd.Timedelta(days=1)

    return latest_observed_date


def build_volunteer_daily_panel(path, output_path) -> pd.DataFrame:
    print("\n" + "=" * 90)
    print("VOLUNTEER DAILY REGION-DATE PANEL")
    print("=" * 90)

    df = load_csv(path)
    required_columns = {"region", "started_at"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Volunteer data is missing required columns: {sorted(missing_columns)}")

    df = df.copy()
    df["started_at_utc"] = pd.to_datetime(df["started_at"], errors="coerce", utc=True)
    invalid_started_at = int(df["started_at_utc"].isna().sum())
    if invalid_started_at:
        raise ValueError(f"Could not parse {invalid_started_at:,} started_at values.")

    df["started_at_kyiv"] = df["started_at_utc"].dt.tz_convert(KYIV_TZ)
    df["date"] = df["started_at_kyiv"].dt.normalize().dt.tz_localize(None)

    regions = sorted(df["region"].dropna().unique())
    first_date = df["date"].min()
    last_complete_start_date = latest_complete_local_date(df["date"])
    last_forecast_date = last_complete_start_date - pd.Timedelta(days=1)

    if first_date > last_forecast_date:
        raise ValueError("Not enough complete local days to build next-day targets.")

    daily_counts = (
        df.groupby(["region", "date"], as_index=False)
        .size()
        .rename(columns={"size": "alert_starts"})
    )

    full_index = pd.MultiIndex.from_product(
        [regions, pd.date_range(first_date, last_complete_start_date, freq="D")],
        names=["region", "date"],
    )

    panel = (
        daily_counts.set_index(["region", "date"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    panel["alert_starts"] = panel["alert_starts"].astype(int)
    panel["has_alert_next_day"] = (
        panel.groupby("region")["alert_starts"].shift(-1).fillna(0).gt(0).astype(int)
    )
    panel = panel[panel["date"] <= last_forecast_date].copy()
    panel["date"] = panel["date"].dt.strftime("%Y-%m-%d")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)

    print(f"Saved panel to {output_path}")
    print(f"Date range: {panel['date'].min()} to {panel['date'].max()} (Europe/Kyiv)")
    print(f"Number of regions: {panel['region'].nunique():,}")
    print(f"Total rows: {len(panel):,}")
    print(f"Positive target rate: {panel['has_alert_next_day'].mean():.2%}")

    example_region = regions[0]
    print(f"\nLast 5 rows for example region: {example_region}")
    print(panel[panel["region"] == example_region].tail(5).to_string(index=False))

    return panel


def main() -> None:
    volunteer_path = DATASETS["volunteer"]["path"]
    if not volunteer_path.exists():
        raise FileNotFoundError(
            f"Missing {volunteer_path}. Run the raw data stage first: "
            "python -m src.raw_data_audit"
        )

    build_volunteer_daily_panel(volunteer_path, VOLUNTEER_PANEL_PATH)


if __name__ == "__main__":
    main()
