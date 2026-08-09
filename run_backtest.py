"""
Run: python run_backtest.py [--config config/config.yaml] [--source csv|mt5]
"""

import argparse
import os
import yaml

from logging_utils.logger import configure_logging, get_logger
from backtest.engine import run_backtest, save_results

log = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(BASE_DIR, "config", "config.yaml"))
    parser.add_argument("--results-dir", default=os.path.join(BASE_DIR, "results"))
    parser.add_argument("--source", choices=["csv", "mt5"], default="csv")
    parser.add_argument("--initial-cash", type=float, default=None, help="Override config's initial_cash for this run.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.initial_cash is not None:
        config["initial_cash"] = args.initial_cash

    configure_logging(**config["logging"])
    log.info(f"Loaded config from {args.config}")

    try:
        results = run_backtest(config, data_source=args.source)
    except (RuntimeError, ValueError) as e:
        log.error(f"Backtest aborted: {e}")
        print(f"\n Backtest aborted: {e}\n")
        raise SystemExit(1)

    save_results(results, args.results_dir)
    print(f"\nFinal equity: {results['equity_df']['equity'].iloc[-1]:.2f} | Survived: {results['survived']}")


if __name__ == "__main__":
    main()
