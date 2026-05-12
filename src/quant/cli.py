import argparse
import logging
from collections.abc import Sequence

from quant.backtest import compute_quantile_backtest
from quant.bootstrap import compute_bootstrap_ic
from quant.costs import compute_cost_analysis
from quant.data import download_data
from quant.factors import compute_factors
from quant.labels import compute_labels
from quant.metrics import compute_ic_analysis
from quant.preprocess import preprocess_data
from quant.report import generate_report


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(prog="quant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download-data")
    download_parser.add_argument("--config", default="config.yaml")

    preprocess_parser = subparsers.add_parser("preprocess-data")
    preprocess_parser.add_argument("--config", default="config.yaml")

    factors_parser = subparsers.add_parser("compute-factors")
    factors_parser.add_argument("--config", default="config.yaml")

    labels_parser = subparsers.add_parser("compute-labels")
    labels_parser.add_argument("--config", default="config.yaml")

    ic_parser = subparsers.add_parser("compute-ic")
    ic_parser.add_argument("--config", default="config.yaml")

    backtest_parser = subparsers.add_parser("run-backtest")
    backtest_parser.add_argument("--config", default="config.yaml")

    costs_parser = subparsers.add_parser("analyze-costs")
    costs_parser.add_argument("--config", default="config.yaml")

    bootstrap_parser = subparsers.add_parser("bootstrap-ic")
    bootstrap_parser.add_argument("--config", default="config.yaml")

    report_parser = subparsers.add_parser("generate-report")
    report_parser.add_argument("--config", default="config.yaml")

    args = parser.parse_args(argv)
    if args.command == "download-data":
        paths = download_data(config_path=args.config)
        for name, path in paths.items():
            print(f"{name}: {path}")
    elif args.command == "preprocess-data":
        path = preprocess_data(config_path=args.config)
        print(f"clean_panel: {path}")
    elif args.command == "compute-factors":
        path = compute_factors(config_path=args.config)
        print(f"factor_panel: {path}")
    elif args.command == "compute-labels":
        path = compute_labels(config_path=args.config)
        print(f"label_panel: {path}")
    elif args.command == "compute-ic":
        paths = compute_ic_analysis(config_path=args.config)
        for name, path in paths.items():
            print(f"{name}: {path}")
    elif args.command == "run-backtest":
        paths = compute_quantile_backtest(config_path=args.config)
        for name, path in paths.items():
            print(f"{name}: {path}")
    elif args.command == "analyze-costs":
        paths = compute_cost_analysis(config_path=args.config)
        for name, path in paths.items():
            print(f"{name}: {path}")
    elif args.command == "bootstrap-ic":
        paths = compute_bootstrap_ic(config_path=args.config)
        for name, path in paths.items():
            print(f"{name}: {path}")
    elif args.command == "generate-report":
        paths = generate_report(config_path=args.config)
        for name, path in paths.items():
            print(f"{name}: {path}")
