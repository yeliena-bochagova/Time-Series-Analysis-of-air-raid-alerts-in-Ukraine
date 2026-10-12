import argparse

from src.backtest_model import run_cli_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ukrainian air-raid alert latest-period evaluation and historical backtests."
    )
    parser.add_argument(
        "--mode",
        choices=["latest", "backtest", "all"],
        default="all",
        help="Which evaluation to run. Default: all.",
    )
    parser.add_argument(
        "--validation-days",
        type=int,
        default=30,
        help="Number of forecast days in each validation window. Default: 30.",
    )
    parser.add_argument(
        "--evaluation-days",
        type=int,
        default=30,
        help="Number of forecast days in each evaluation window. Default: 30.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=6,
        help="Number of historical backtest folds. Default: 6.",
    )
    parser.add_argument(
        "--step-days",
        type=int,
        default=90,
        help="Days to step backward between historical folds. Default: 90.",
    )
    parser.add_argument(
        "--no-sensitivity",
        action="store_true",
        help="Skip the built-in 14-day and 60-day window sensitivity checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_cli_pipeline(
        mode=args.mode,
        validation_days=args.validation_days,
        evaluation_days=args.evaluation_days,
        folds=args.folds,
        step_days=args.step_days,
        include_sensitivity=not args.no_sensitivity,
    )


if __name__ == "__main__":
    main()
