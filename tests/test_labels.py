import numpy as np
import pandas as pd
import pytest

from quant.labels import (
    LABEL_COLUMNS,
    align_factor_and_label,
    build_label_panel,
    compute_forward_excess_return,
    compute_forward_return,
)


def test_compute_forward_return_aligns_label_to_current_date() -> None:
    panel = _price_panel({"000001": [100.0, 110.0, 121.0]})

    forward = compute_forward_return(panel, horizon=1)

    assert forward.loc[(pd.Timestamp("2024-01-01"), "000001")] == pytest.approx(0.10)
    assert forward.loc[(pd.Timestamp("2024-01-02"), "000001")] == pytest.approx(0.10)
    assert np.isnan(forward.loc[(pd.Timestamp("2024-01-03"), "000001")])


def test_compute_forward_excess_return_subtracts_benchmark_by_date() -> None:
    stock_panel = _price_panel(
        {
            "000001": [100.0, 110.0, 121.0],
            "000002": [200.0, 210.0, 231.0],
        }
    )
    benchmark_panel = _price_panel({"000905": [100.0, 105.0, 110.25]})

    labels = compute_forward_excess_return(stock_panel, benchmark_panel, horizon=1)

    assert labels.loc[
        (pd.Timestamp("2024-01-01"), "000001"), "fwd_excess_ret_1d"
    ] == pytest.approx(0.05)
    assert labels.loc[
        (pd.Timestamp("2024-01-01"), "000002"), "fwd_excess_ret_1d"
    ] == pytest.approx(0.00)
    assert np.isnan(
        labels.loc[(pd.Timestamp("2024-01-03"), "000001"), "fwd_excess_ret_1d"]
    )


def test_build_label_panel_returns_configured_excess_label_columns() -> None:
    stock_panel = _price_panel({"000001": [float(100 + day) for day in range(25)]})
    benchmark_panel = _price_panel(
        {"000905": [float(100 + day / 2) for day in range(25)]}
    )

    labels = build_label_panel(
        stock_panel,
        benchmark_panel=benchmark_panel,
        horizons=[5, 20],
        use_excess_return=True,
    )

    assert list(labels.columns) == LABEL_COLUMNS
    assert labels.index.names == ["date", "ticker"]
    assert labels.loc[
        (pd.Timestamp("2024-01-01"), "000001"), "fwd_excess_ret_5d"
    ] == pytest.approx((105 / 100 - 1) - (102.5 / 100 - 1))
    assert np.isnan(
        labels.loc[(pd.Timestamp("2024-01-21"), "000001"), "fwd_excess_ret_5d"]
    )


def test_align_factor_and_label_joins_same_date_and_drops_missing_rows() -> None:
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-01"), "000001"),
            (pd.Timestamp("2024-01-02"), "000001"),
        ],
        names=["date", "ticker"],
    )
    factors = pd.DataFrame({"momentum_20": [1.0, 2.0]}, index=index)
    labels = pd.DataFrame({"fwd_excess_ret_5d": [0.03, np.nan]}, index=index)

    aligned = align_factor_and_label(factors, labels)

    assert aligned.index.tolist() == [(pd.Timestamp("2024-01-01"), "000001")]
    assert aligned.loc[(pd.Timestamp("2024-01-01"), "000001"), "momentum_20"] == 1.0
    assert aligned.loc[
        (pd.Timestamp("2024-01-01"), "000001"), "fwd_excess_ret_5d"
    ] == pytest.approx(0.03)


def _price_panel(prices_by_ticker: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for ticker, prices in prices_by_ticker.items():
        for offset, close in enumerate(prices):
            rows.append(
                {
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=offset),
                    "ticker": ticker,
                    "close": close,
                }
            )
    return pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
