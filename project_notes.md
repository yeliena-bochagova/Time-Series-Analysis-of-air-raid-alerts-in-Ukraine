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

- Raw download/audit is implemented in `src/raw_data_audit.py`.
- Daily volunteer region-date panel preparation is implemented in `src/prepare_daily_panel.py`.
- Baseline and logistic-regression evaluation is implemented in `src/evaluate_baselines.py`.
- `start.py` currently runs the evaluation stage.
- Processed volunteer panel path: `data/processed/volunteer_daily_region_panel.csv`.
- The processed panel has one row per region per complete `Europe/Kyiv` forecast day, with `alert_starts` and `has_alert_next_day`.
- Current processed panel summary: `2022-02-25` to `2026-10-10`, 25 regions, 42,225 rows, positive target rate 70.46%.
- No UI, plots, complex models, or tests have been implemented yet.
- Python execution for this project should use:

```text
C:\ProgramData\anaconda3\envs\kse_agentic\python.exe
```

## Evaluation Setup

- Chronological split uses the latest 30 complete forecast days as test, the preceding 30 days as validation, and everything earlier as training/history.
- Current split:
  - Train: `2022-02-25` to `2026-08-11`.
  - Validation: `2026-08-12` to `2026-09-10`.
  - Test: `2026-09-11` to `2026-10-10`.
- Baselines implemented:
  - Always predict alert tomorrow.
  - Persistence: predict alert tomorrow if the current day has at least one alert.

## Model Findings So Far

- Persistence is a strong baseline.
- Test persistence result: balanced accuracy 0.893, false positives 21, false negatives 26.
- Logistic regression with local history, calendar features, and region identity improves test balanced accuracy to 0.940 and reduces false positives to 5, but increases false negatives to 50.
- Logistic regression is therefore more conservative: it distinguishes quiet days better, but misses more alert days.
- Validation error analysis showed the main failure mode: logistic regression misses sudden alert returns in quieter/intermittent regions, especially after sparse recent alert history.
- Validation false negatives were concentrated in regions including Rivnenska, Khmelnytska, Chernivetska, Zhytomyrska, Ternopilska, and Vinnytska oblasts.
- A simple national-context feature experiment on validation did not help this failure mode.
- National context features tested: regions with alerts on day `D`, total national alert starts on day `D`, regions with alerts in the last 3 days, and national alert starts in the last 3 days.
- Validation comparison:
  - Current logistic regression: balanced accuracy 0.898, false positives 18, false negatives 62.
  - With national context: balanced accuracy 0.896, false positives 16, false negatives 70.
- The simple shared-threat feature set made the model slightly more conservative and recovered no validation false negatives.

## Parked Ideas

- Combining official and volunteer datasets is low priority for now.
- If revisited, first compare overlap-date agreement between volunteer region/day labels and official oblast/day labels before merging.
