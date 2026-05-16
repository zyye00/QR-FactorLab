import numpy as np
import pandas as pd
import pytest

from quant.factors import (
    FACTOR_COLUMNS,
    build_factor_panel,
    compute_factor_columns,
    winsorize_cross_section,
    zscore_cross_section,
)


def test_compute_factor_columns_uses_trailing_data_only() -> None:
    panel = _sample_clean_panel(n_days=25, tickers=["000001"])

    factors = compute_factor_columns(panel)

    row = factors.loc[(pd.Timestamp("2024-01-21"), "000001")]
    assert row["reversal_5"] == pytest.approx(-(121 / 116 - 1))
    assert row["momentum_20"] == pytest.approx(121 / 101 - 1)
    assert row["low_volatility_20"] == pytest.approx(
        -panel.xs("000001", level="ticker")["ret_1d"].iloc[1:21].std()
    )
    assert row["turnover_change_20"] == pytest.approx(
        row["turnover"] / np.mean(range(3, 23)) - 1
    )
    assert row["liquidity_20"] == pytest.approx(np.log(np.mean(range(1002, 1022))))


def test_winsorize_and_zscore_cross_section() -> None:
    panel = pd.DataFrame(
        {
            "date": ["2024-01-02"] * 4,
            "ticker": ["000001", "000002", "000003", "000004"],
            "factor": [1.0, 2.0, 3.0, 100.0],
        }
    ).set_index(["date", "ticker"])

    winsorized = winsorize_cross_section(panel, ["factor"], lower=0.25, upper=0.75)
    standardized = zscore_cross_section(winsorized, ["factor"])

    assert winsorized["factor"].max() < 100
    assert standardized["factor"].mean() == pytest.approx(0)
    assert standardized["factor"].std() == pytest.approx(1)


def test_build_factor_panel_returns_standardized_factor_columns() -> None:
    panel = _sample_clean_panel(n_days=25, tickers=["000001", "000002", "000003"])

    factors = build_factor_panel(panel, winsorize_lower=0.01, winsorize_upper=0.99)
    date_slice = factors.xs(pd.Timestamp("2024-01-21"), level="date")

    assert list(factors.columns) == FACTOR_COLUMNS
    assert factors.index.names == ["date", "ticker"]
    assert date_slice["momentum_20"].mean() == pytest.approx(0)
    assert date_slice["momentum_20"].std() == pytest.approx(1)


def _sample_clean_panel(n_days: int, tickers: list[str]) -> pd.DataFrame:
    rows = []
    for ticker_index, ticker in enumerate(tickers):
        base = 100 + ticker_index * 10
        for day in range(1, n_days + 1):
            close = base + day
            rows.append(
                {
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day - 1),
                    "ticker": ticker,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 1000 + day,
                    "amount": 1000 + day + ticker_index * 100,
                    "turnover": day + 1 + ticker_index,
                }
            )
    panel = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
    by_ticker = panel.groupby(level="ticker")["close"]
    panel["ret_1d"] = by_ticker.pct_change(1)
    panel["ret_5d"] = by_ticker.pct_change(5)
    panel["ret_20d"] = by_ticker.pct_change(20)
    return panel
