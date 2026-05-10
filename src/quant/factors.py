from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from quant.data import PANEL_INDEX, save_parquet

FACTOR_COLUMNS = [
    "reversal_5",
    "momentum_20",
    "low_volatility_20",
    "turnover_change_20",
    "liquidity_20",
]


def compute_factor_columns(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.reset_index().sort_values(["ticker", "date"]).copy()
    by_ticker = data.groupby("ticker", group_keys=False)

    data["reversal_5"] = -data["ret_5d"]
    data["momentum_20"] = data["ret_20d"]
    data["low_volatility_20"] = -by_ticker["ret_1d"].transform(
        lambda values: values.rolling(20, min_periods=20).std()
    )
    mean_turnover_20 = by_ticker["turnover"].transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    data["turnover_change_20"] = data["turnover"] / mean_turnover_20 - 1
    mean_amount_20 = by_ticker["amount"].transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    data["liquidity_20"] = np.log(mean_amount_20)

    return data.set_index(PANEL_INDEX).sort_index()


def winsorize_cross_section(
    panel: pd.DataFrame,
    columns: list[str],
    lower: float,
    upper: float,
) -> pd.DataFrame:
    return panel.groupby(level="date", group_keys=False).apply(
        lambda group: _winsorize_group(group, columns, lower, upper)
    )


def zscore_cross_section(panel: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return panel.groupby(level="date", group_keys=False).apply(
        lambda group: _zscore_group(group, columns)
    )


def build_factor_panel(
    clean_panel: pd.DataFrame,
    winsorize_lower: float,
    winsorize_upper: float,
) -> pd.DataFrame:
    factors = compute_factor_columns(clean_panel)[FACTOR_COLUMNS]
    factors = winsorize_cross_section(
        factors,
        FACTOR_COLUMNS,
        lower=winsorize_lower,
        upper=winsorize_upper,
    )
    factors = zscore_cross_section(factors, FACTOR_COLUMNS)
    return factors.sort_index()


def compute_factors(config_path: str = "config.yaml") -> Path:
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    processed_dir = Path(config["data"]["processed_dir"])
    clean_panel = pd.read_parquet(processed_dir / "clean_panel.parquet")
    factor_panel = build_factor_panel(
        clean_panel,
        winsorize_lower=config["preprocess"]["winsorize_lower"],
        winsorize_upper=config["preprocess"]["winsorize_upper"],
    )
    return save_parquet(factor_panel, processed_dir / "factor_panel.parquet")


def _winsorize_group(
    group: pd.DataFrame,
    columns: list[str],
    lower: float,
    upper: float,
) -> pd.DataFrame:
    result = group.copy()
    for column in columns:
        lower_value = result[column].quantile(lower)
        upper_value = result[column].quantile(upper)
        result[column] = result[column].clip(lower=lower_value, upper=upper_value)
    return result


def _zscore_group(group: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = group.copy()
    for column in columns:
        mean = result[column].mean()
        std = result[column].std()
        if pd.isna(std) or std == 0:
            result[column] = pd.NA
        else:
            result[column] = (result[column] - mean) / std
    return result
