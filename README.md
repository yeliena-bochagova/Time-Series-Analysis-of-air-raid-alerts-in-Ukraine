# Time-Series Analysis of Ukrainian Air-Raid Alerts

Historical forecasting prototype for Ukrainian air-raid alert starts. This is not a live alert system and must not be used for safety decisions.

## Prediction Task

For each modeled region/oblast and each local calendar day `D`, use only information available by the end of `D` in `Europe/Kyiv` time to predict whether at least one new air-raid alert will start in that region on day `D+1`.

The user-facing date is the target day being predicted. The information cutoff is the previous local day. Ongoing alerts that started before the target day do not make the target positive by themselves; the target is only new alert starts.

## Data

The raw-data stage downloads both source CSVs:

- Volunteer dataset: `data/raw/volunteer_data_en.csv`
- Official dataset: `data/raw/official_data_en.csv`

V1 uses the volunteer dataset as the primary modeling source because it was the most current dataset observed during inspection. The official dataset is kept as a secondary reference because it was stale relative to the volunteer data at the time checked.

The processed panel is saved to `data/processed/volunteer_daily_region_panel.csv`. It has one row per region and complete Kyiv-local forecast day, including days with zero alert starts.

## V1 Modeling Scope

Modeling and evaluation exclude `Luhanska oblast` because the source documentation notes a permanent siren there that is not represented correctly as ordinary event starts. Crimea remains out of scope because it is absent from the volunteer dataset. The raw source files and processed panel are left unchanged; the exclusion is applied only before modeling/evaluation/UI inspection.

Final v1 modeling panel:

- 24 regions
- Forecast dates: `2022-02-25` to `2026-10-10`
- 40,536 region-day rows

## Model and Features

The frozen v1 model is logistic regression with class balancing. Features are intentionally simple and leakage-safe:

- Local recent history: alert starts today, alert today, rolling alert-start counts over 3/7/14/30 days, rolling alert-day counts over 3/7/14/30 days, days since last alert.
- Calendar fields: day of week, weekend flag, month.
- Region identity: one-hot encoded region.

All history features for forecast day `D` use only that region's data through `D`; no future dates and no cross-region rolling contamination are used.

Baselines:

- Always predict alert tomorrow.
- Persistence: predict alert tomorrow if the current day has at least one alert.

## Leakage Controls

Evaluation is chronological. For the latest-period evaluation, the latest 30 complete forecast days are the evaluation period, the preceding 30 forecast days are validation, and everything earlier is training/history.

Because the label for forecast day `D` comes from target day `D+1`, split boundaries use a one-forecast-day purge gap:

- `max(train target_date) < min(validation forecast_date)`
- `max(validation target_date) < min(evaluation forecast_date)`

Preprocessing is fitted on training rows only inside each fold. Threshold selection is done on the validation period only, then evaluated on the following unseen period.

## Metrics

Balanced accuracy is the primary comparison metric because alert days are common and plain accuracy/F1 can reward alert-heavy predictions while ignoring quiet-day discrimination. Alert recall and false negatives are still inspected as guardrails because missed alert starts are important.

Confusion-matrix counts are reported alongside balanced accuracy, precision, recall, and F1 in the CLI outputs.

## Experiments Kept Out of V1

Two small experiments were rejected before freezing the modeling approach:

- National context features, such as number of regions with alerts on day `D`, did not recover the missed sudden alert returns and made the model slightly more conservative.
- A constrained random forest did not produce a meaningful validation improvement over logistic regression.

The final approach remains the local-feature logistic regression with validation-selected threshold. No more complex models or extra feature hypotheses are used in v1.

## Results

The latest period was inspected during development, so it should not be treated as the strongest evidence of generalization. The rolling historical backtest is the stronger check because it repeats the same procedure across multiple earlier timeframes.

Latest-period evaluation:

| Model | Balanced accuracy | Alert recall | False positives | False negatives |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 0.932 | 0.908 | 4 | 58 |
| Persistence | 0.865 | 0.959 | 21 | 26 |

Six-fold rolling historical backtest, 30-day validation / 30-day evaluation windows:

| Model | Mean balanced accuracy | Mean alert recall | Total false positives | Total false negatives |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 0.855 | 0.816 | 116 | 594 |
| Persistence | 0.764 | 0.880 | 379 | 386 |

Sensitivity checks with 14-day and 60-day windows are available from the CLI. In general, logistic regression improves balanced accuracy by reducing false positives, but it is more conservative and misses more alert starts than persistence.

## Per-Region Inspection

The Streamlit UI includes a latest-period per-region comparison table. This is mainly diagnostic: a 30-day window is short, and some regions have only one observed target class during that period, so per-region balanced accuracy can be unstable.

In the latest-period evaluation, the most visible logistic-regression misses are concentrated in quieter or intermittent regions where alerts return after sparse recent history. Inspect per-region false positives/false negatives rather than relying only on per-region balanced accuracy.

## Limitations

- This is a historical forecasting prototype, not operational safety software.
- The model predicts new alert starts, not whether any alert or permanent siren is active.
- Luhansk is excluded from v1 modeling because the permanent-siren process is not represented by the event-start dataset.
- Crimea is absent from the volunteer dataset and is out of scope.
- The model uses alert history only; text/media/news features were considered optional but not implemented.
- The latest-period result may be biased by development-time inspection; rolling backtests are the better generalization check.

## Reproduce

Commands below use the project interpreter used during development. If your environment lives elsewhere, replace the interpreter path consistently.

Install dependencies:

```powershell
& "C:\ProgramData\anaconda3\envs\kse_agentic\python.exe" -m pip install -r requirements.txt
```

Download and audit raw data:

```powershell
& "C:\ProgramData\anaconda3\envs\kse_agentic\python.exe" -m src.raw_data_audit
```

Prepare the daily region-date panel:

```powershell
& "C:\ProgramData\anaconda3\envs\kse_agentic\python.exe" -m src.prepare_daily_panel
```

Run latest-period evaluation and historical backtests:

```powershell
& "C:\ProgramData\anaconda3\envs\kse_agentic\python.exe" start.py --mode all
```

Useful CLI options:

```powershell
& "C:\ProgramData\anaconda3\envs\kse_agentic\python.exe" start.py --mode latest
& "C:\ProgramData\anaconda3\envs\kse_agentic\python.exe" start.py --mode backtest --folds 6 --step-days 90
& "C:\ProgramData\anaconda3\envs\kse_agentic\python.exe" start.py --mode all --validation-days 30 --evaluation-days 30
```

Run tests:

```powershell
& "C:\ProgramData\anaconda3\envs\kse_agentic\python.exe" -m unittest discover -v
```

Run the optional inspection UI:

```powershell
& "C:\ProgramData\anaconda3\envs\kse_agentic\python.exe" -m streamlit run src/streamlit_app.py
```

Generated raw/processed data and evaluation artifacts are written under `data/` and `outputs/`. These are ignored by Git.
