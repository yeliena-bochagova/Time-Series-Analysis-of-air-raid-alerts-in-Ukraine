import pandas as pd
import requests

from src.paths import DATASETS


REGION_HINTS = (
    "region",
    "oblast",
    "raion",
    "district",
    "city",
    "location",
    "area",
    "hromada",
    "community",
    "level",
)

TIMESTAMP_HINTS = ("started_at", "finished_at", "created_at", "updated_at", "date", "time", "timestamp")


def download_file(name: str, url: str, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name}: {url}")

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    path.write_bytes(response.content)
    print(f"Saved to {path} ({path.stat().st_size:,} bytes)")


def load_csv(path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def find_columns(df: pd.DataFrame, hints: tuple[str, ...]) -> list[str]:
    return [col for col in df.columns if any(hint in col.lower() for hint in hints)]


def parse_timestamp_columns(df: pd.DataFrame) -> dict[str, pd.Series]:
    parsed = {}

    for column in find_columns(df, TIMESTAMP_HINTS):
        values = pd.to_datetime(df[column], errors="coerce", utc=True)
        if values.notna().any():
            parsed[column] = values

    return parsed


def pick_primary_timestamp(parsed_timestamps: dict[str, pd.Series]) -> str | None:
    if not parsed_timestamps:
        return None

    preferred_names = ("started_at", "start", "created_at", "date", "timestamp", "finished_at", "end")
    lower_to_original = {column.lower(): column for column in parsed_timestamps}

    for preferred in preferred_names:
        for lower_name, original_name in lower_to_original.items():
            if preferred in lower_name:
                return original_name

    return next(iter(parsed_timestamps))


def print_basic_schema(df: pd.DataFrame) -> None:
    print("\nSchema and dtypes:")
    schema = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "non_null": [df[column].notna().sum() for column in df.columns],
        }
    )
    print(schema.to_string(index=False))
    print(f"\nRow count: {len(df):,}")
    print(f"Column count: {len(df.columns):,}")


def print_timestamp_ranges(parsed_timestamps: dict[str, pd.Series]) -> None:
    print("\nTimestamp ranges:")
    if not parsed_timestamps:
        print("No timestamp-like columns were detected.")
        return

    for column, values in parsed_timestamps.items():
        valid = values.dropna()
        print(f"- {column}: {valid.min()} to {valid.max()} ({len(valid):,} parsed values)")


def print_region_fields(df: pd.DataFrame) -> None:
    region_columns = find_columns(df, REGION_HINTS)
    print("\nRegion-related fields:")

    if not region_columns:
        print("No region-like columns were detected.")
        return

    for column in region_columns:
        values = df[column].dropna().astype(str)
        unique_values = sorted(values.unique())
        shown_values = unique_values[:20]
        suffix = "" if len(unique_values) <= 20 else " ..."
        print(f"- {column}: {len(unique_values):,} unique values")
        print(f"  {shown_values}{suffix}")


def print_missing_values(df: pd.DataFrame) -> None:
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    print("\nMissing values:")
    if missing.empty:
        print("No missing values detected.")
        return

    summary = pd.DataFrame(
        {
            "missing": missing,
            "percent": (missing / len(df) * 100).round(2),
        }
    )
    print(summary.to_string())


def print_duplicates(df: pd.DataFrame) -> None:
    duplicate_count = int(df.duplicated().sum())
    print(f"\nExact duplicate rows: {duplicate_count:,}")


def print_alert_types(df: pd.DataFrame) -> None:
    alert_type_columns = [
        column
        for column in df.columns
        if column.lower() in ("type", "alert_type", "alarm_type")
        or ("type" in column.lower() and ("alert" in column.lower() or "alarm" in column.lower()))
    ]

    print("\nAlert types:")
    if not alert_type_columns:
        print("No alert type column was detected.")
        return

    for column in alert_type_columns:
        print(f"- {column}")
        print(df[column].value_counts(dropna=False).head(20).to_string())


def print_recent_month_counts(df: pd.DataFrame, parsed_timestamps: dict[str, pd.Series], months: int = 12) -> None:
    primary_timestamp = pick_primary_timestamp(parsed_timestamps)
    print("\nRecords by month, most recent months:")

    if primary_timestamp is None:
        print("No timestamp-like column is available for monthly counts.")
        return

    month_values = parsed_timestamps[primary_timestamp].dt.tz_convert(None).dt.to_period("M")
    month_counts = month_values.value_counts().sort_index().tail(months)

    print(f"Using timestamp column: {primary_timestamp}")
    print(month_counts.to_string())


def print_sample_rows(df: pd.DataFrame) -> None:
    print("\nSample rows:")
    if df.empty:
        print("Dataset is empty.")
        return

    print(df.head(5).to_string(index=False))


def inspect_dataset(name: str, path) -> dict:
    print("\n" + "=" * 90)
    print(f"{name.upper()} DATASET")
    print("=" * 90)

    df = load_csv(path)
    parsed_timestamps = parse_timestamp_columns(df)
    region_columns = find_columns(df, REGION_HINTS)
    primary_timestamp = pick_primary_timestamp(parsed_timestamps)

    print_basic_schema(df)
    print_timestamp_ranges(parsed_timestamps)
    print_region_fields(df)
    print_missing_values(df)
    print_duplicates(df)
    print_alert_types(df)
    print_recent_month_counts(df, parsed_timestamps)
    print_sample_rows(df)

    primary_values = parsed_timestamps.get(primary_timestamp) if primary_timestamp else None
    primary_min = primary_values.min() if primary_values is not None else None
    primary_max = primary_values.max() if primary_values is not None else None

    return {
        "name": name,
        "rows": len(df),
        "columns": list(df.columns),
        "region_columns": region_columns,
        "primary_timestamp": primary_timestamp,
        "primary_min": primary_min,
        "primary_max": primary_max,
        "region_unique_counts": {
            column: int(df[column].dropna().astype(str).nunique()) for column in region_columns
        },
    }


def print_comparison(summaries: dict[str, dict]) -> None:
    print("\n" + "=" * 90)
    print("DATASET COMPARISON")
    print("=" * 90)

    official = summaries["official"]
    volunteer = summaries["volunteer"]

    print(f"Official rows:  {official['rows']:,}")
    print(f"Volunteer rows: {volunteer['rows']:,}")

    print("\nDate coverage:")
    for summary in (official, volunteer):
        print(
            f"- {summary['name']}: {summary['primary_timestamp']} "
            f"{summary['primary_min']} to {summary['primary_max']}"
        )

    print("\nRegion-related columns:")
    for summary in (official, volunteer):
        print(f"- {summary['name']}: {summary['region_columns']}")

    print("\nRegion detalization hints:")
    for summary in (official, volunteer):
        if not summary["region_unique_counts"]:
            print(f"- {summary['name']}: no region-like columns detected")
            continue

        counts = ", ".join(
            f"{column}={count:,}" for column, count in summary["region_unique_counts"].items()
        )
        print(f"- {summary['name']}: {counts}")

    official_only_columns = sorted(set(official["columns"]) - set(volunteer["columns"]))
    volunteer_only_columns = sorted(set(volunteer["columns"]) - set(official["columns"]))

    print("\nColumn differences:")
    print(f"- Only in official: {official_only_columns if official_only_columns else 'none'}")
    print(f"- Only in volunteer: {volunteer_only_columns if volunteer_only_columns else 'none'}")

    print("\nShort interpretation:")
    print(
        "- The official dataset has separate administrative columns, so its regional "
        "detail should be judged from fields such as oblast, raion, hromada, and level."
    )
    print(
        "- The volunteer dataset has a simpler single region field, which makes its "
        "regional structure easier to inspect but less detailed."
    )
    print(
        "- No aggregation, feature engineering, modeling, or date splitting has been done here."
    )


def run_raw_data_audit() -> dict[str, dict]:
    summaries = {}

    for name, config in DATASETS.items():
        download_file(name, config["url"], config["path"])
        summaries[name] = inspect_dataset(name, config["path"])

    print_comparison(summaries)
    return summaries


def main() -> None:
    run_raw_data_audit()


if __name__ == "__main__":
    main()
