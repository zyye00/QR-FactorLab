import pandas as pd
import pytest

from quant.preprocess import RETURN_COLUMNS, clean_panel, preprocess_panel


def test_clean_panel_filters_bad_rows_and_duplicates() -> None:
    panel = pd.DataFrame(
        {
            "date": [
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
            ],
            "ticker": ["1", "1", "1", "1"],
            "open": [10.0, 10.1, 10.2, 10.3],
            "high": [11.0, 11.1, 11.2, 11.3],
            "low": [9.0, 9.1, 9.2, 9.3],
            "close": [10.0, 10.1, None, 10.3],
            "volume": [100.0, 101.0, 102.0, 0.0],
            "amount": [1000.0, 1010.0, 1020.0, 1030.0],
            "turnover": [0.1, 0.1, 0.1, 0.1],
        }
    ).set_index(["date", "ticker"])

    cleaned = clean_panel(panel)

    assert cleaned.index.names == ["date", "ticker"]
    assert cleaned.index.is_unique
    assert cleaned.index.get_level_values("ticker").tolist() == ["000001"]
    assert cleaned.iloc[0]["close"] == 10.1


def test_preprocess_panel_adds_returns_by_ticker() -> None:
    rows = []
    for day in range(1, 23):
        rows.append(
            {
                "date": f"2024-01-{day:02d}",
                "ticker": "000001",
                "open": 100 + day,
                "high": 100 + day,
                "low": 100 + day,
                "close": 100 + day,
                "volume": 1000,
                "amount": 10000,
                "turnover": 0.1,
            }
        )
    panel = pd.DataFrame(rows).set_index(["date", "ticker"])

    clean = preprocess_panel(panel)

    assert all(column in clean.columns for column in RETURN_COLUMNS)
    assert clean.loc[(pd.Timestamp("2024-01-02"), "000001"), "ret_1d"] == pytest.approx(
        1 / 101
    )
    assert clean.loc[(pd.Timestamp("2024-01-06"), "000001"), "ret_5d"] == pytest.approx(
        5 / 101
    )
    assert clean.loc[
        (pd.Timestamp("2024-01-21"), "000001"), "ret_20d"
    ] == pytest.approx(20 / 101)
