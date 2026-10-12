import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.backtest_model import build_purged_fold, run_backtest
from src.evaluate_baselines import (
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    add_leakage_safe_features,
    assert_no_split_target_overlap,
    chronological_split,
    make_logistic_regression,
)
from src.paths import KYIV_TZ
from src.prepare_daily_panel import build_volunteer_daily_panel


def make_daily_panel(regions: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for region in regions:
        for date in dates:
            next_date = date + pd.Timedelta(days=1)
            rows.append(
                {
                    "region": region,
                    "date": date,
                    "alert_starts": int(date.day % 3 == 0),
                    "has_alert_next_day": int(next_date.day % 3 == 0),
                    "target_date": next_date,
                }
            )
    return pd.DataFrame(rows)


class PanelPreparationTests(unittest.TestCase):
    def test_panel_target_is_next_day_by_region_and_grid_is_complete(self) -> None:
        raw = pd.DataFrame(
            [
                {"region": "A", "started_at": "2026-01-01 12:00:00+00:00", "finished_at": "2026-01-01 12:30:00+00:00", "naive": False},
                {"region": "A", "started_at": "2026-01-03 12:00:00+00:00", "finished_at": "2026-01-03 12:30:00+00:00", "naive": False},
                {"region": "B", "started_at": "2026-01-02 12:00:00+00:00", "finished_at": "2026-01-02 12:30:00+00:00", "naive": False},
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "volunteer.csv"
            output_path = Path(tmpdir) / "panel.csv"
            raw.to_csv(raw_path, index=False)

            with contextlib.redirect_stdout(io.StringIO()):
                panel = build_volunteer_daily_panel(raw_path, output_path)

        self.assertEqual(len(panel), 4)
        self.assertEqual(panel[["region", "date"]].duplicated().sum(), 0)
        self.assertEqual(set(panel["region"]), {"A", "B"})
        self.assertEqual(set(panel["date"]), {"2026-01-01", "2026-01-02"})

        keyed = panel.set_index(["region", "date"])
        self.assertEqual(keyed.loc[("A", "2026-01-01"), "alert_starts"], 1)
        self.assertEqual(keyed.loc[("A", "2026-01-01"), "has_alert_next_day"], 0)
        self.assertEqual(keyed.loc[("A", "2026-01-02"), "alert_starts"], 0)
        self.assertEqual(keyed.loc[("A", "2026-01-02"), "has_alert_next_day"], 1)
        self.assertEqual(keyed.loc[("B", "2026-01-01"), "alert_starts"], 0)
        self.assertEqual(keyed.loc[("B", "2026-01-01"), "has_alert_next_day"], 1)
        self.assertEqual(keyed.loc[("B", "2026-01-02"), "alert_starts"], 1)
        self.assertEqual(keyed.loc[("B", "2026-01-02"), "has_alert_next_day"], 0)

    def test_incomplete_latest_local_days_are_excluded(self) -> None:
        today = pd.Timestamp.now(tz=KYIV_TZ).normalize()
        local_datetimes = [
            today - pd.Timedelta(days=2) + pd.Timedelta(hours=12),
            today - pd.Timedelta(days=1) + pd.Timedelta(hours=12),
            today + pd.Timedelta(hours=12),
        ]
        raw = pd.DataFrame(
            [
                {
                    "region": "A",
                    "started_at": local_time.tz_convert("UTC").isoformat(),
                    "finished_at": (local_time + pd.Timedelta(minutes=30)).tz_convert("UTC").isoformat(),
                    "naive": False,
                }
                for local_time in local_datetimes
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "volunteer.csv"
            output_path = Path(tmpdir) / "panel.csv"
            raw.to_csv(raw_path, index=False)

            with contextlib.redirect_stdout(io.StringIO()):
                panel = build_volunteer_daily_panel(raw_path, output_path)

        expected_last_forecast_date = (today - pd.Timedelta(days=2)).date().isoformat()
        self.assertEqual(panel["date"].max(), expected_last_forecast_date)


class FeatureEngineeringTests(unittest.TestCase):
    def test_rolling_features_use_same_region_and_dates_up_to_d_only(self) -> None:
        dates = pd.date_range("2026-01-01", periods=4, freq="D")
        panel = pd.DataFrame(
            [
                {"region": "A", "date": dates[0], "alert_starts": 0, "has_alert_next_day": 0, "target_date": dates[1]},
                {"region": "A", "date": dates[1], "alert_starts": 0, "has_alert_next_day": 1, "target_date": dates[2]},
                {"region": "A", "date": dates[2], "alert_starts": 10, "has_alert_next_day": 0, "target_date": dates[3]},
                {"region": "A", "date": dates[3], "alert_starts": 0, "has_alert_next_day": 0, "target_date": dates[3] + pd.Timedelta(days=1)},
                {"region": "B", "date": dates[0], "alert_starts": 5, "has_alert_next_day": 0, "target_date": dates[1]},
                {"region": "B", "date": dates[1], "alert_starts": 0, "has_alert_next_day": 0, "target_date": dates[2]},
                {"region": "B", "date": dates[2], "alert_starts": 0, "has_alert_next_day": 0, "target_date": dates[3]},
                {"region": "B", "date": dates[3], "alert_starts": 0, "has_alert_next_day": 0, "target_date": dates[3] + pd.Timedelta(days=1)},
            ]
        )

        featured = add_leakage_safe_features(panel).set_index(["region", "date"])

        self.assertEqual(featured.loc[("A", dates[0]), "alert_starts_last_3d"], 0)
        self.assertEqual(featured.loc[("A", dates[1]), "alert_starts_last_3d"], 0)
        self.assertEqual(featured.loc[("A", dates[2]), "alert_starts_last_3d"], 10)
        self.assertEqual(featured.loc[("A", dates[3]), "alert_starts_last_3d"], 10)
        self.assertEqual(featured.loc[("B", dates[2]), "alert_starts_last_3d"], 5)

        self.assertEqual(featured.loc[("A", dates[1]), "has_alert_today"], 0)
        self.assertEqual(featured.loc[("A", dates[1]), "days_since_last_alert"], 999)
        self.assertEqual(featured.loc[("B", dates[2]), "days_since_last_alert"], 2)


class PurgedSplitTests(unittest.TestCase):
    def test_chronological_split_has_purge_gaps_and_no_label_overlap(self) -> None:
        dates = pd.date_range("2026-01-01", periods=70, freq="D")
        panel = make_daily_panel(["A"], dates)

        splits = chronological_split(panel)
        used_dates = set(pd.concat([splits[name]["date"] for name in splits]).dt.strftime("%Y-%m-%d"))
        dropped_dates = [date.strftime("%Y-%m-%d") for date in dates if date.strftime("%Y-%m-%d") not in used_dates]

        self.assertEqual(splits["validation"]["date"].nunique(), 30)
        self.assertEqual(splits["test"]["date"].nunique(), 30)
        self.assertLess(splits["train"]["target_date"].max(), splits["validation"]["date"].min())
        self.assertLess(splits["validation"]["target_date"].max(), splits["test"]["date"].min())
        self.assertEqual(dropped_dates, ["2026-01-09", "2026-02-09"])

    def test_negative_split_without_purge_is_rejected(self) -> None:
        dates = pd.date_range("2026-01-01", periods=70, freq="D")
        panel = make_daily_panel(["A"], dates)
        bad_splits = {
            "train": panel[panel["date"].le(pd.Timestamp("2026-01-10"))],
            "validation": panel[
                panel["date"].between(pd.Timestamp("2026-01-11"), pd.Timestamp("2026-02-09"))
            ],
            "test": panel[panel["date"].ge(pd.Timestamp("2026-02-10"))],
        }

        with self.assertRaises(ValueError):
            assert_no_split_target_overlap(bad_splits)

    def test_backtest_fold_preserves_chronology_and_embargo_dates(self) -> None:
        dates = pd.date_range("2026-01-01", periods=90, freq="D")
        panel = make_daily_panel(["A", "B"], dates)
        fold = build_purged_fold(panel, eval_end_idx=len(dates) - 1, validation_days=5, evaluation_days=5)

        train = fold["train"]
        validation = fold["validation"]
        evaluation = fold["evaluation"]
        self.assertLess(train["target_date"].max(), validation["date"].min())
        self.assertLess(validation["target_date"].max(), evaluation["date"].min())
        self.assertEqual(validation["date"].nunique(), 5)
        self.assertEqual(evaluation["date"].nunique(), 5)
        self.assertEqual(fold["dropped_dates"][0], validation["date"].min() - pd.Timedelta(days=1))
        self.assertEqual(fold["dropped_dates"][1], evaluation["date"].min() - pd.Timedelta(days=1))

    def test_every_backtest_fold_is_independently_chronological(self) -> None:
        dates = pd.date_range("2026-01-01", periods=100, freq="D")
        panel = add_leakage_safe_features(make_daily_panel(["A", "B"], dates))
        results = run_backtest(panel, validation_days=5, evaluation_days=5, fold_count=3, step_days=10)

        self.assertEqual(len(results), 3)
        for _, row in results.iterrows():
            self.assertLess(pd.Timestamp(row["train_forecast_end"]), pd.Timestamp(row["validation_forecast_start"]))
            self.assertLess(pd.Timestamp(row["validation_forecast_end"]), pd.Timestamp(row["evaluation_forecast_start"]))
            dropped_dates = row["dropped_forecast_dates"].split(", ")
            self.assertEqual(dropped_dates[0], (pd.Timestamp(row["validation_forecast_start"]) - pd.Timedelta(days=1)).date().isoformat())
            self.assertEqual(dropped_dates[1], (pd.Timestamp(row["evaluation_forecast_start"]) - pd.Timedelta(days=1)).date().isoformat())


class PreprocessingTests(unittest.TestCase):
    def test_preprocessor_is_fit_on_training_rows_only(self) -> None:
        train = pd.DataFrame(
            [
                {"region": "A", "has_alert_next_day": 0, **{feature: 0 for feature in NUMERIC_FEATURES}},
                {"region": "A", "has_alert_next_day": 1, **{feature: 2 for feature in NUMERIC_FEATURES}},
                {"region": "A", "has_alert_next_day": 0, **{feature: 4 for feature in NUMERIC_FEATURES}},
                {"region": "A", "has_alert_next_day": 1, **{feature: 6 for feature in NUMERIC_FEATURES}},
            ]
        )
        validation = pd.DataFrame(
            [
                {"region": "B", "has_alert_next_day": 1, **{feature: 1000 for feature in NUMERIC_FEATURES}},
            ]
        )

        model = make_logistic_regression()
        model.fit(train[FEATURE_COLUMNS], train["has_alert_next_day"])

        numeric_scaler = model.named_steps["preprocessor"].named_transformers_["numeric"]
        expected_train_mean = train[NUMERIC_FEATURES].mean().to_numpy()
        self.assertTrue((numeric_scaler.mean_ == expected_train_mean).all())

        region_encoder = model.named_steps["preprocessor"].named_transformers_["region"]
        self.assertEqual(region_encoder.categories_[0].tolist(), ["A"])
        model.predict(validation[FEATURE_COLUMNS])


class CliSmokeTests(unittest.TestCase):
    def test_cli_latest_mode_smoke(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "start.py",
                "--mode",
                "latest",
                "--validation-days",
                "5",
                "--evaluation-days",
                "5",
            ],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            timeout=45,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("LATEST-PERIOD EVALUATION", result.stdout)
        self.assertIn("Saved detailed fold results", result.stdout)


if __name__ == "__main__":
    unittest.main()
