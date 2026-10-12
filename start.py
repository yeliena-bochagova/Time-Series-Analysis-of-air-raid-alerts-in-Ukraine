from src.paths import DATASETS, VOLUNTEER_PANEL_PATH
from src.prepare_daily_panel import build_volunteer_daily_panel
from src.raw_data_audit import run_raw_data_audit
from src.evaluate_baselines import run_baseline_evaluation


def main() -> None:
    run_raw_data_audit()
    # build_volunteer_daily_panel(DATASETS["volunteer"]["path"], VOLUNTEER_PANEL_PATH)
    # run_baseline_evaluation()



if __name__ == "__main__":
    main()
