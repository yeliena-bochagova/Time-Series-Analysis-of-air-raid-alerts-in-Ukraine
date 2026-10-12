# Time-Series-Analysis-of-air-raid-alerts-in-Ukraine

Historical forecasting prototype for Ukrainian air-raid alert starts.

## Setup

Use the project interpreter explicitly:

```powershell
& "C:\ProgramData\anaconda3\envs\kse_agentic\python.exe" -m pip install -r requirements.txt
```

Fetch/audit the raw data and prepare the daily region-date panel:

```powershell
& "C:\ProgramData\anaconda3\envs\kse_agentic\python.exe" -m src.raw_data_audit
& "C:\ProgramData\anaconda3\envs\kse_agentic\python.exe" -m src.prepare_daily_panel
```

Generated data and evaluation artifacts are written under `data/` and `outputs/`.

## Commands

Run the CLI evaluation/backtest:

```powershell
& "C:\ProgramData\anaconda3\envs\kse_agentic\python.exe" start.py --mode all
```

Run the optional Streamlit inspection UI:

```powershell
& "C:\ProgramData\anaconda3\envs\kse_agentic\python.exe" -m streamlit run src/streamlit_app.py
```

The UI is only an inspection/demo layer over saved and existing pipeline outputs. It is not a live alert system.
It is target-date oriented: historical inspection selects the day being predicted, while the latest forecast uses the newest complete input day available.

## V1 Scope Limitation

Modeling and evaluation exclude `Luhanska oblast` because the source documentation notes a permanent siren there that is not represented correctly as ordinary event starts. Crimea remains out of scope because it is absent from the volunteer dataset. The target remains unchanged: predict new alert starts on the target day, not whether any permanent/ongoing siren is active.
