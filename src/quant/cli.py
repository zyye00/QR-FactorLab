import argparse
from collections.abc import Sequence

from quant.data import download_data
from quant.factors import compute_factors
from quant.preprocess import preprocess_data


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="quant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download-data")
    download_parser.add_argument("--config", default="config.yaml")
    download_parser.add_argument("--adjust", default="qfq")

    preprocess_parser = subparsers.add_parser("preprocess-data")
    preprocess_parser.add_argument("--config", default="config.yaml")

    factors_parser = subparsers.add_parser("compute-factors")
    factors_parser.add_argument("--config", default="config.yaml")

    args = parser.parse_args(argv)
    if args.command == "download-data":
        paths = download_data(
            config_path=args.config,
            adjust=args.adjust,
        )
        for name, path in paths.items():
            print(f"{name}: {path}")
    elif args.command == "preprocess-data":
        path = preprocess_data(config_path=args.config)
        print(f"clean_panel: {path}")
    elif args.command == "compute-factors":
        path = compute_factors(config_path=args.config)
        print(f"factor_panel: {path}")
