"""CLI entrypoint for AutoPortfolio.

Examples:
    python run_pipeline.py --portfolio banking
    python run_pipeline.py --all
    python run_pipeline.py --all --parallel
    python run_pipeline.py --portfolio it --timesteps 50000 --trials 10
"""
import argparse
import json
import logging

from config import list_portfolio_names
from scheduler.pipeline import run_all_portfolios, run_nightly_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AutoPortfolio nightly pipeline.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--portfolio", choices=list_portfolio_names(), help="Run a single portfolio.")
    target.add_argument("--all", action="store_true", help="Run every portfolio in config/portfolios.yaml.")

    parser.add_argument("--parallel", action="store_true", help="With --all, run portfolios concurrently.")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total training timesteps per agent.")
    parser.add_argument("--trials", type=int, default=20, help="Optuna trials per agent during hyperopt.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.portfolio:
        result = run_nightly_pipeline(args.portfolio, total_timesteps=args.timesteps, n_trials=args.trials)
    else:
        result = run_all_portfolios(
            parallel=args.parallel, total_timesteps=args.timesteps, n_trials=args.trials
        )

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
