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
- Because the label for forecast day `D` comes from `D+1`, chronological evaluation must avoid split-boundary target leakage.
- The evaluation now uses purge gaps before validation and test so that training labels do not overlap validation forecast dates, and validation labels do not overlap test forecast dates.

## Current Implementation State

- Raw download/audit is implemented in `src/raw_data_audit.py`.
- Daily volunteer region-date panel preparation is implemented in `src/prepare_daily_panel.py`.
- Baseline and logistic-regression evaluation is implemented in `src/evaluate_baselines.py`.
- Historical backtesting and CLI-facing evaluation helpers are implemented in `src/backtest_model.py`.
- `start.py` is the user-facing CLI entry point for latest-period evaluation and historical backtests.
- Detailed fold results and plots are saved under `outputs/`.
- Processed volunteer panel path: `data/processed/volunteer_daily_region_panel.csv`.
- The processed panel has one row per region per complete `Europe/Kyiv` forecast day, with `alert_starts` and `has_alert_next_day`.
- Current processed panel summary: `2022-02-25` to `2026-10-10`, 25 regions, 42,225 rows, positive target rate 70.46%.
- No UI, plots, complex models, or tests have been implemented yet.
- Python execution for this project should use:

```text
C:\ProgramData\anaconda3\envs\kse_agentic\python.exe
```

Example CLI runs:

```text
C:\ProgramData\anaconda3\envs\kse_agentic\python.exe start.py --mode latest
C:\ProgramData\anaconda3\envs\kse_agentic\python.exe start.py --mode backtest --folds 6 --step-days 90
C:\ProgramData\anaconda3\envs\kse_agentic\python.exe start.py --mode all --validation-days 30 --evaluation-days 30
```

## Evaluation Setup

- Chronological split uses the latest 30 complete forecast days as test, 30 validation forecast days before a one-day purge gap, and training/history before another one-day purge gap.
- Current split:
  - Train forecast dates: `2022-02-25` to `2026-08-09`; target dates: `2022-02-26` to `2026-08-10`.
  - Validation forecast dates: `2026-08-11` to `2026-09-09`; target dates: `2026-08-12` to `2026-09-10`.
  - Test forecast dates: `2026-09-11` to `2026-10-10`; target dates: `2026-09-12` to `2026-10-11`.
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
- National context features should not be kept in the current model.
- After fixing split-boundary leakage, validation-only threshold tuning for the original logistic regression found the best balanced accuracy at the default threshold 0.50.
- Corrected validation result for logistic regression at threshold 0.50: balanced accuracy 0.905, false positives 16, false negatives 60.
- Final constrained nonlinear validation experiment did not beat logistic regression.
- Constrained random forest validation result: balanced accuracy 0.901, false positives 16, false negatives 64.
- Current modeling conclusion: keep the local-feature logistic regression at threshold 0.50 and stop model experimentation for now.

## Metric Selection Notes

- F1 and balanced accuracy disagree because alert days are common.
- F1 focuses on the positive class and ignores true negatives, so it can favor alert-heavy predictions that do not identify quiet days well.
- Balanced accuracy averages alert recall and no-alert recall, so it better reflects whether the model distinguishes both classes under class imbalance.
- Balanced accuracy is a reasonable primary model-selection metric for this project, but it should not be used alone.
- Use alert-day recall / false negatives as a guardrail, because missing alerts may be more costly than false alarms.
- Current practical stance: prefer balanced accuracy as the main comparison metric, while inspecting confusion matrices and avoiding threshold choices that reduce false negatives only by creating too many false positives.

## Parked Ideas

- Combining official and volunteer datasets is low priority for now.
- If revisited, first compare overlap-date agreement between volunteer region/day labels and official oblast/day labels before merging.
