import logging
import sys
import types
from pathlib import Path

import pandas as pd

from quant import fetch

sys.modules.setdefault("akshare", types.SimpleNamespace())


def test_fetch_universe_downloads_and_zero_pads_tickers(monkeypatch) -> None:
    calls = []

    def fake_index_stock_cons_csindex(symbol: str) -> pd.DataFrame:
        calls.append(symbol)
        return pd.DataFrame({"ticker": ["1", "600000"]})

    monkeypatch.setattr(
        fetch.ak,
        "index_stock_cons_csindex",
        fake_index_stock_cons_csindex,
        raising=False,
    )

    tickers = fetch.FetchUniverse(
        start_date="2024-01-01",
        end_date="2024-01-31",
    ).fetch(symbol="000905")

    assert calls == ["000905"]
    assert tickers.tolist() == ["000001", "600000"]


def test_fetch_stocks_falls_back_to_sina_and_normalizes_panel(monkeypatch) -> None:
    calls = []

    def fake_cdr(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        calls.append(("cdr", symbol, start_date, end_date))
        return pd.DataFrame()

    def fake_sina(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        calls.append(("sina", symbol, start_date, end_date))
        return _source_ohlcv_frame(ticker=None, date="2024-01-03")

    monkeypatch.setattr(fetch.FetchStocks, "_fetch_cdr", staticmethod(fake_cdr))
    monkeypatch.setattr(fetch.FetchStocks, "_fetch_sina", staticmethod(fake_sina))

    panel = fetch.FetchStocks(
        start_date="2024-01-01",
        end_date="2024-01-31",
    ).fetch(tickers=pd.Series(["1"]))

    assert calls == [
        ("cdr", "1", "20240101", "20240131"),
        ("sina", "1", "20240101", "20240131"),
    ]
    assert list(panel.columns) == ["date", "ticker", *fetch.OHLCV_COLUMNS]
    assert panel.loc[0, "date"] == pd.Timestamp("2024-01-03")
    assert panel.loc[0, "ticker"] == "000001"
    assert panel.loc[0, "close"] == 10.2


def test_fetch_cdr_prefixes_market_and_converts_volume(monkeypatch) -> None:
    calls = []

    def fake_stock_zh_a_cdr_daily(
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        calls.append((symbol, start_date, end_date))
        return pd.DataFrame({"volume": [10]})

    monkeypatch.setattr(
        fetch.ak,
        "stock_zh_a_cdr_daily",
        fake_stock_zh_a_cdr_daily,
        raising=False,
    )

    frame = fetch.FetchStocks._fetch_cdr("689009", "20240101", "20240131")

    assert calls == [("sh689009", "20240101", "20240131")]
    assert frame.loc[0, "volume"] == 1000


def test_fetch_sina_prefixes_market_and_uses_hfq_adjust(monkeypatch) -> None:
    calls = []
    raw = pd.DataFrame({"date": ["2024-01-03"]})

    def fake_stock_zh_a_daily(
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str,
    ) -> pd.DataFrame:
        calls.append((symbol, start_date, end_date, adjust))
        return raw

    monkeypatch.setattr(
        fetch.ak,
        "stock_zh_a_daily",
        fake_stock_zh_a_daily,
        raising=False,
    )

    frame = fetch.FetchStocks._fetch_sina("000001", "20240101", "20240131")

    assert calls == [("sz000001", "20240101", "20240131", fetch.STOCK_ADJUST)]
    assert frame is raw


def test_fetch_index_downloads_with_sh_prefix_and_sets_ticker(monkeypatch) -> None:
    calls = []

    def fake_stock_zh_index_daily_tx(
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        calls.append((symbol, start_date, end_date))
        return pd.DataFrame({"date": ["2024-01-03"], "close": [5000.0]})

    monkeypatch.setattr(
        fetch.ak,
        "stock_zh_index_daily_tx",
        fake_stock_zh_index_daily_tx,
        raising=False,
    )

    frame = fetch.FetchIndex(
        start_date="2024-01-01",
        end_date="2024-01-31",
    ).fetch(symbol="000905")

    assert calls == [("sh000905", "20240101", "20240131")]
    assert frame.loc[0, "date"] == pd.Timestamp("2024-01-03")
    assert frame.loc[0, "ticker"] == "000905"


def test_download_data_writes_stock_and_benchmark_parquet(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  dir: data",
                "  source_dir: data/source",
                "  start_date: '2024-01-01'",
                "  end_date: '2024-01-31'",
                "  benchmark: '000905'",
            ]
        ),
        encoding="utf-8",
    )
    stock_panel = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-03")],
            "ticker": ["000001"],
            "close": [10.2],
        }
    )
    benchmark_panel = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-03")],
            "ticker": ["000905"],
            "close": [5000.0],
        }
    )

    monkeypatch.setattr(
        fetch.FetchUniverse,
        "fetch",
        lambda self, symbol: pd.Series(["000001"]),
    )
    monkeypatch.setattr(
        fetch.FetchStocks,
        "fetch",
        lambda self, tickers: stock_panel,
    )
    monkeypatch.setattr(
        fetch.FetchIndex,
        "fetch",
        lambda self, symbol: benchmark_panel,
    )

    try:
        paths = fetch.download_data(str(config_path))
    finally:
        _close_fetch_file_handlers()

    assert set(paths) == {"download_log", "stock_ohlcv", "benchmark_ohlcv"}
    assert paths["download_log"] == Path("data/download.log")
    assert paths["stock_ohlcv"] == Path("data/source/stock_ohlcv.parquet")
    assert paths["benchmark_ohlcv"] == Path(
        "data/source/benchmark_000905_ohlcv.parquet"
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(paths["stock_ohlcv"]),
        stock_panel,
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(paths["benchmark_ohlcv"]),
        benchmark_panel,
    )


def _source_ohlcv_frame(ticker: str | None, date: str) -> pd.DataFrame:
    values = {
        "date": date,
        "open": 10.0,
        "high": 11.0,
        "low": 9.8,
        "close": 10.2,
        "volume": 1000,
        "amount": 10200.0,
        "turnover": 0.5,
    }
    if ticker is not None:
        values["ticker"] = ticker
    return pd.DataFrame(
        {_source_column(normalized): [value] for normalized, value in values.items()}
    )


def _source_column(normalized: str) -> str:
    return next(
        source for source, column in fetch.OHLCV_MAP.items() if column == normalized
    )


def _close_fetch_file_handlers() -> None:
    for handler in list(fetch.LOGGER.handlers):
        if isinstance(handler, logging.FileHandler):
            fetch.LOGGER.removeHandler(handler)
            handler.close()
