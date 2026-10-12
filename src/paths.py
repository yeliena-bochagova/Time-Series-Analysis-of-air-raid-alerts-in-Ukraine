from pathlib import Path


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
OUTPUTS_DIR = Path("outputs")
KYIV_TZ = "Europe/Kyiv"

DATASETS = {
    "official": {
        "url": "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/official_data_en.csv",
        "path": RAW_DIR / "official_data_en.csv",
    },
    "volunteer": {
        "url": "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/volunteer_data_en.csv",
        "path": RAW_DIR / "volunteer_data_en.csv",
    },
}

VOLUNTEER_PANEL_PATH = PROCESSED_DIR / "volunteer_daily_region_panel.csv"
