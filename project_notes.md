# Project Notes

## Established Goal

- V1 should use `volunteer_data_en.csv` as the primary dataset because it is the most current source observed so far.
- The official dataset should be kept only as a secondary reference for now.
- The intended prediction task is: for each region/oblast and local calendar day `D`, use only information available by the end of `D` to predict whether at least one new air-raid alert will start in that region during day `D+1`.

## Data Sources Observed

- Volunteer CSV path: `data/raw/volunteer_data_en.csv`.
- Official CSV path: `data/raw/official_data_en.csv`.
- Volunteer schema observed: `region`, `started_at`, `finished_at`, `naive`.
- Official schema observed: `oblast`, `raion`, `hromada`, `level`, `started_at`, `finished_at`, `source`.
- Volunteer data downloaded on 2026-10-12 covered `2022-02-25 16:36:22+00:00` through `2026-10-12 04:49:48+00:00` by `started_at`.
- Official data downloaded on 2026-10-12 covered `2022-03-15 16:10:34+00:00` through `2026-09-07 05:28:22+00:00` by `started_at`.
- The official dataset was therefore stale relative to the volunteer dataset at the time inspected.

## Target Definition Decisions

- Calendar days should be defined in `Europe/Kyiv` local time, not UTC.
- The target should count only new alert starts on `D+1`.
- An alert that started on `D` and continued into `D+1` should not make the `D+1` target positive by itself.
- Continuing alerts may be valid input information later if they are known by the end of `D`, but they are not part of the target definition.

## Leakage Notes

- No feature, model input, or training information may use data from after the forecast time.
- For forecast day `D`, the cutoff is the end of `D` in `Europe/Kyiv` local time.
- `finished_at` can leak future information if an alert has not ended by the forecast cutoff.
- The volunteer `naive` column indicates inferred end times, so duration-based analysis should treat those rows carefully.
- The latest incomplete local day should be handled carefully before evaluation.

## Current Implementation State

- `start.py` downloads both CSVs into `data/raw/` and prints inspection summaries.
- No aggregation, feature engineering, model training, validation/test splitting, plotting, or tests have been implemented yet.
- Python execution for this project should use:

```text
C:\ProgramData\anaconda3\envs\kse_agentic\python.exe
```
