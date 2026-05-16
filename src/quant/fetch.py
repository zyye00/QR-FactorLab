import logging
from abc import ABC, abstractmethod
from pathlib import Path

import akshare as ak
import pandas as pd
import yaml

LOGGER: logging.Logger = logging.getLogger(__name__)
LOG_NAME: str = "download.log"
STOCK_ADJUST: str = "hfq"
PANEL_INDEX = ["date", "ticker"]
OHLCV_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover",
]
OHLCV_MAP: dict[str, str] = {
    "日期": "date",
    "股票代码": "ticker",
    "开盘价": "open",
    "最高价": "high",
    "最低价": "low",
    "收盘价": "close",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover",
}


class FetchData(ABC):
    def __init__(self, start_date: str, end_date: str | None, **_):
        self.start_date: pd.Timestamp = pd.to_datetime(start_date)
        if end_date is not None:
            self.end_date: pd.Timestamp = pd.to_datetime(end_date)
        else:
            self.end_date: pd.Timestamp = pd.Timestamp.today()

    @abstractmethod
    def fetch(self, *args, **kwargs) -> pd.Series | pd.DataFrame: ...

    @abstractmethod
    def _normalize(self, df: pd.DataFrame) -> pd.Series | pd.DataFrame: ...


class FetchUniverse(FetchData):
    def fetch(self, symbol: str) -> pd.Series:
        LOGGER.info(
            f"Starting data download: benchmark={symbol} "
            f"start_date={self.start_date} end_date={self.end_date} "
            f"stock_adjust={STOCK_ADJUST}",
        )
        return self._normalize(ak.index_stock_cons_csindex(symbol=symbol))

    def _normalize(self, df: pd.DataFrame) -> pd.Series:
        df = df.rename(columns={"成分券代码": "ticker"})
        return df["ticker"].str.zfill(6)


class FetchStocks(FetchData):
    @staticmethod
    def _fetch_eastmoney(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """df = ak.stock_zh_a_hist(
            symbol=symbol, start_date=start_date, end_date=end_date, adjust=STOCK_ADJUST
        )
        df["成交量"] *= 100  # Convert from hand to share
        return df"""
        return pd.DataFrame()

    @staticmethod
    def _fetch_cdr(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        symbol = ("sh" if symbol.startswith(("5", "6", "9")) else "sz") + symbol
        df = ak.stock_zh_a_cdr_daily(
            symbol=symbol, start_date=start_date, end_date=end_date
        )
        df["volume"] *= 100  # Convert from hand to share
        return df

    @staticmethod
    def _fetch_sina(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        symbol = ("sh" if symbol.startswith(("5", "6", "9")) else "sz") + symbol
        return ak.stock_zh_a_daily(
            symbol=symbol, start_date=start_date, end_date=end_date, adjust=STOCK_ADJUST
        )

    def fetch(self, tickers: pd.Series) -> pd.DataFrame:
        df: list[pd.DataFrame] = []
        failures: list[str] = []
        for i, ticker in enumerate(tickers, start=1):
            LOGGER.info(
                f"[{i}/{len(tickers)}] Fetching OHLCV for {ticker} "
                f"from {self.start_date} to {self.end_date}",
            )
            for fetcher in (self._fetch_eastmoney, self._fetch_cdr, self._fetch_sina):
                try:
                    frame = fetcher(
                        symbol=ticker,
                        start_date=self.start_date.strftime("%Y%m%d"),
                        end_date=self.end_date.strftime("%Y%m%d"),
                    )
                    if frame is not None and not frame.empty:
                        df.append(self._normalize(frame, ticker=ticker))
                        break
                except Exception as exc:
                    LOGGER.warning(
                        f"{fetcher.__name__} failed for {ticker}: {exc}",
                    )
            else:
                failures.append(ticker)
                LOGGER.warning(f"Failed to download OHLCV for {ticker}")

        if failures:
            failed = ", ".join(ticker for ticker in failures[:10])
            LOGGER.warning(f"Skipped {len(failures)} failed ticker(s): {failed}")
        if not df:
            return pd.DataFrame(columns=[*PANEL_INDEX, *OHLCV_COLUMNS])
        return pd.concat(df, ignore_index=True)

    def _normalize(self, df: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
        df = df.rename(columns=OHLCV_MAP)
        panel_columns = [*PANEL_INDEX, *OHLCV_COLUMNS]
        if ticker is not None:
            df["ticker"] = ticker
        for col in OHLCV_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = df["ticker"].astype(str).str.zfill(6)
        return df[panel_columns].sort_values(PANEL_INDEX)


class FetchIndex(FetchStocks):
    def fetch(self, symbol: str) -> pd.DataFrame:
        frames = ak.stock_zh_index_daily_tx(
            symbol=f"sh{symbol}",
            start_date=self.start_date.strftime("%Y%m%d"),
            end_date=self.end_date.strftime("%Y%m%d"),
        )
        frames["date"] = pd.to_datetime(frames["date"])
        frames["ticker"] = symbol
        frames["date"] = pd.to_datetime(frames["date"])
        return frames


def download_data(config_path: str = "config.yaml") -> dict[str, Path]:
    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    data_config = config["data"]

    data_dir = Path(data_config["dir"])
    source_dir = Path(data_config["source_dir"])
    source_dir.mkdir(parents=True, exist_ok=True)
    log_path = _configure_download_file_logging(data_dir)

    tickers: pd.Series = FetchUniverse(**data_config).fetch(
        symbol=data_config["benchmark"]
    )
    stock_panel: pd.DataFrame = FetchStocks(**data_config).fetch(tickers=tickers)
    stock_panel_path = source_dir / "stock_ohlcv.parquet"
    stock_panel.to_parquet(stock_panel_path, index=False)
    LOGGER.info(
        f"Saved stock OHLCV panel with {len(stock_panel)} rows to {stock_panel_path}",
    )

    benchmark_panel: pd.DataFrame = FetchIndex(**data_config).fetch(
        symbol=data_config["benchmark"]
    )
    benchmark_panel_path = (
        source_dir / f"benchmark_{data_config['benchmark']}_ohlcv.parquet"
    )
    benchmark_panel.to_parquet(benchmark_panel_path, index=False)
    LOGGER.info(
        f"Saved benchmark OHLCV panel with {len(benchmark_panel)} rows to "
        f"{benchmark_panel_path}",
    )

    return {
        "download_log": log_path,
        "stock_ohlcv": stock_panel_path,
        "benchmark_ohlcv": benchmark_panel_path,
    }


def _configure_download_file_logging(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path = data_dir / LOG_NAME
    for handler in list(LOGGER.handlers):
        if getattr(handler, "_qr_download_file_handler", False):
            LOGGER.removeHandler(handler)
            handler.close()
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler._qr_download_file_handler = True
    LOGGER.addHandler(handler)
    return log_path
