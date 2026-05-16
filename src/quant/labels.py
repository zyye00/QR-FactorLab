from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import yaml

from quant.fetch import PANEL_INDEX

DEFAULT_HORIZONS = [5, 20]
LABEL_COLUMNS = ["fwd_excess_ret_5d", "fwd_excess_ret_20d"]


def compute_forward_return(
    panel: pd.DataFrame,
    horizon: int,
    price_column: str = "close",
) -> pd.Series:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    data = _normalize_panel(panel).reset_index().sort_values(PANEL_INDEX)
    forward = (
        data.groupby("ticker", group_keys=False)[price_column].shift(-horizon)
        / data[price_column]
        - 1
    )
    forward.index = pd.MultiIndex.from_frame(data[PANEL_INDEX])
    forward.name = f"fwd_ret_{horizon}d"
    return forward.sort_index()


def compute_forward_excess_return(
    stock_panel: pd.DataFrame,
    benchmark_panel: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    stock_forward = compute_forward_return(stock_panel, horizon)
    benchmark_forward = _benchmark_forward_by_date(
        compute_forward_return(benchmark_panel, horizon)
    )

    result = stock_forward.to_frame()
    benchmark_column = f"benchmark_fwd_ret_{horizon}d"
    excess_column = f"fwd_excess_ret_{horizon}d"
    dates = result.index.get_level_values("date")
    result[benchmark_column] = benchmark_forward.reindex(dates).to_numpy()
    result[excess_column] = result[stock_forward.name] - result[benchmark_column]
    return result.sort_index()


def build_label_panel(
    stock_panel: pd.DataFrame,
    benchmark_panel: pd.DataFrame | None = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    use_excess_return: bool = True,
) -> pd.DataFrame:
    label_frames = []
    for horizon in horizons:
        if use_excess_return:
            if benchmark_panel is None:
                raise ValueError("benchmark_panel is required for excess returns.")
            horizon_labels = compute_forward_excess_return(
                stock_panel,
                benchmark_panel,
                horizon,
            )[[f"fwd_excess_ret_{horizon}d"]]
        else:
            horizon_labels = compute_forward_return(stock_panel, horizon).to_frame()
        label_frames.append(horizon_labels)

    return pd.concat(label_frames, axis=1).sort_index()


def align_factor_and_label(
    factor_panel: pd.DataFrame,
    label_panel: pd.DataFrame,
    dropna: bool = True,
) -> pd.DataFrame:
    aligned = _normalize_panel(factor_panel).join(
        _normalize_panel(label_panel),
        how="inner",
    )
    if dropna:
        aligned = aligned.dropna()
    return aligned.sort_index()


def compute_labels(config_path: str = "config.yaml") -> Path:
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    source_dir = Path(config["data"]["source_dir"])
    work_dir = Path(config["data"]["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    clean_panel = pd.read_parquet(work_dir / "clean_panel.parquet")
    use_excess_return = config["labels"].get("use_excess_return", True)
    benchmark_panel = None
    if use_excess_return:
        benchmark = config["data"]["benchmark"]
        benchmark_panel = pd.read_parquet(
            source_dir / f"benchmark_{benchmark}_ohlcv.parquet"
        )

    label_panel = build_label_panel(
        clean_panel,
        benchmark_panel=benchmark_panel,
        horizons=[int(horizon) for horizon in config["labels"]["horizons"]],
        use_excess_return=use_excess_return,
    )
    path = work_dir / "label_panel.parquet"
    label_panel.to_parquet(path)
    return path


def _normalize_panel(panel: pd.DataFrame) -> pd.DataFrame:
    if set(PANEL_INDEX).issubset(panel.index.names):
        data = panel.reset_index()
    else:
        data = panel.copy()
    missing_columns = [column for column in PANEL_INDEX if column not in data.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Panel is missing required column(s): {missing}.")

    data["date"] = pd.to_datetime(data["date"])
    data["ticker"] = data["ticker"].astype(str).str.zfill(6)
    return data.set_index(PANEL_INDEX).sort_index()


def _benchmark_forward_by_date(forward_return: pd.Series) -> pd.Series:
    frame = forward_return.reset_index()[["date", forward_return.name]]
    if frame["date"].duplicated().any():
        raise ValueError("Benchmark forward returns must have one row per date.")
    return frame.set_index("date")[forward_return.name].sort_index()
