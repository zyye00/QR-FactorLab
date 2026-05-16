from pathlib import Path

import pandas as pd
import yaml

from quant.fetch import OHLCV_COLUMNS, PANEL_INDEX

RETURN_COLUMNS = ["ret_1d", "ret_5d", "ret_20d"]


def clean_panel(panel: pd.DataFrame, min_history_days: int = 0) -> pd.DataFrame:
    cleaned = panel.reset_index()
    cleaned["date"] = pd.to_datetime(cleaned["date"])
    cleaned["ticker"] = cleaned["ticker"].astype(str).str.zfill(6)
    cleaned = cleaned.drop_duplicates(PANEL_INDEX, keep="last")
    cleaned = cleaned.dropna(subset=["close"])
    cleaned = cleaned[cleaned["close"] > 0]
    cleaned = cleaned[cleaned["volume"] > 0]

    if min_history_days > 0:
        counts = cleaned.groupby("ticker")["date"].transform("size")
        cleaned = cleaned[counts >= min_history_days]

    return cleaned[PANEL_INDEX + OHLCV_COLUMNS].set_index(PANEL_INDEX).sort_index()


def add_return_columns(panel: pd.DataFrame) -> pd.DataFrame:
    enriched = panel.sort_index().copy()
    close_by_ticker = enriched.groupby(level="ticker")["close"]
    enriched["ret_1d"] = close_by_ticker.pct_change(1)
    enriched["ret_5d"] = close_by_ticker.pct_change(5)
    enriched["ret_20d"] = close_by_ticker.pct_change(20)
    return enriched


def preprocess_panel(panel: pd.DataFrame, min_history_days: int = 0) -> pd.DataFrame:
    return add_return_columns(clean_panel(panel, min_history_days=min_history_days))


def preprocess_data(config_path: str = "config.yaml") -> Path:
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    source_dir = Path(config["data"]["source_dir"])
    work_dir = Path(config["data"]["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    min_history_days = config["preprocess"]["min_history_days"]
    raw_panel = pd.read_parquet(source_dir / "stock_ohlcv.parquet")
    clean = preprocess_panel(raw_panel, min_history_days=min_history_days)
    path = work_dir / "clean_panel.parquet"
    clean.to_parquet(path)
    return path
