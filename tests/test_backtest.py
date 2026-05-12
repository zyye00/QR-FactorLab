from pathlib import Path

import pandas as pd
import pytest
import yaml

from quant.backtest import (
    assign_quantile_groups,
    compute_cumulative_returns,
    compute_long_only_returns,
    compute_long_short_returns,
    compute_quantile_backtest,
    compute_quantile_returns,
    factor_directions_from_ic_summary,
    infer_factor_directions,
    select_rebalance_dates,
    summarize_backtest,
)


def test_assign_quantile_groups_by_date() -> None:
    values = _series_by_date(
        {
            "2024-01-01": [1.0, 2.0, 3.0, 4.0, 5.0],
            "2024-01-02": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )

    groups = assign_quantile_groups(values, n_quantiles=5)

    assert groups.loc[(pd.Timestamp("2024-01-01"), "000001")] == 1
    assert groups.loc[(pd.Timestamp("2024-01-01"), "000005")] == 5
    assert groups.loc[(pd.Timestamp("2024-01-02"), "000001")] == 5
    assert groups.loc[(pd.Timestamp("2024-01-02"), "000005")] == 1


def test_compute_quantile_returns_and_portfolio_returns() -> None:
    factors, labels = _sample_factor_label_panels()

    quantile_returns = compute_quantile_returns(
        factors,
        labels,
        n_quantiles=5,
    )
    long_short = compute_long_short_returns(quantile_returns, n_quantiles=5)
    long_only = compute_long_only_returns(quantile_returns, long_quantile=5)

    q1 = "factor_a__fwd_excess_ret_5d__Q1"
    q5 = "factor_a__fwd_excess_ret_5d__Q5"
    pair = "factor_a__fwd_excess_ret_5d"
    assert quantile_returns.loc[pd.Timestamp("2024-01-01"), q1] == pytest.approx(0.01)
    assert quantile_returns.loc[pd.Timestamp("2024-01-01"), q5] == pytest.approx(0.05)
    assert long_short.loc[pd.Timestamp("2024-01-01"), pair] == pytest.approx(0.04)
    assert long_short.loc[pd.Timestamp("2024-01-02"), pair] == pytest.approx(-0.04)
    assert long_only.loc[pd.Timestamp("2024-01-01"), pair] == pytest.approx(0.05)


def test_portfolio_returns_reverse_negative_factor_direction() -> None:
    dates = pd.to_datetime(["2024-01-01"])
    quantile_returns = pd.DataFrame(
        {
            "factor_a__fwd_excess_ret_5d__Q1": [0.05],
            "factor_a__fwd_excess_ret_5d__Q5": [0.01],
        },
        index=dates,
    )
    directions = {"factor_a__fwd_excess_ret_5d": -1}

    long_short = compute_long_short_returns(
        quantile_returns,
        n_quantiles=5,
        directions=directions,
    )
    long_only = compute_long_only_returns(
        quantile_returns,
        long_quantile=5,
        directions=directions,
    )

    assert long_short.loc[
        pd.Timestamp("2024-01-01"), "factor_a__fwd_excess_ret_5d"
    ] == pytest.approx(0.04)
    assert long_only.loc[
        pd.Timestamp("2024-01-01"), "factor_a__fwd_excess_ret_5d"
    ] == pytest.approx(0.05)


def test_factor_directions_use_rank_ic_mean() -> None:
    summary = pd.DataFrame(
        {
            "factor_label": ["factor_a__label", "factor_b__label"],
            "rank_ic_mean": [-0.1, 0.2],
            "ic_mean": [0.3, -0.4],
        }
    )

    directions = factor_directions_from_ic_summary(summary)

    assert directions == {"factor_a__label": -1, "factor_b__label": 1}


def test_infer_factor_directions_uses_daily_rank_ic() -> None:
    index = pd.MultiIndex.from_product(
        [
            pd.to_datetime(["2024-01-01", "2024-01-02"]),
            ["000001", "000002", "000003"],
        ],
        names=["date", "ticker"],
    )
    factors = pd.DataFrame(
        {"factor_a": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]},
        index=index,
    )
    labels = pd.DataFrame(
        {"fwd_excess_ret_5d": [3.0, 2.0, 1.0, 3.0, 2.0, 1.0]},
        index=index,
    )

    directions = infer_factor_directions(factors, labels)

    assert directions == {"factor_a__fwd_excess_ret_5d": -1}


def test_compute_cumulative_returns_compounds_returns() -> None:
    returns = pd.DataFrame(
        {"factor_a__label": [0.10, -0.10]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    cumulative = compute_cumulative_returns(returns)

    assert cumulative.loc[pd.Timestamp("2024-01-01"), "factor_a__label"] == (
        pytest.approx(0.10)
    )
    assert cumulative.loc[pd.Timestamp("2024-01-02"), "factor_a__label"] == (
        pytest.approx(-0.01)
    )


def test_select_rebalance_dates_keeps_last_date_per_period() -> None:
    returns = pd.DataFrame(
        {"factor_a__label": [0.01, 0.02, 0.03]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-08"]),
    )

    weekly = select_rebalance_dates(returns, "W")

    assert weekly.index.tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-08"),
    ]


def test_select_rebalance_dates_can_follow_label_horizon() -> None:
    dates = pd.bdate_range("2024-01-01", periods=25)
    returns = pd.DataFrame(
        {
            "factor_a__fwd_excess_ret_5d__Q1": range(25),
            "factor_a__fwd_excess_ret_20d__Q1": range(100, 125),
        },
        index=dates,
    )

    selected = select_rebalance_dates(returns, "label_horizon")

    five_day_dates = selected.index[
        selected["factor_a__fwd_excess_ret_5d__Q1"].notna()
    ].tolist()
    twenty_day_dates = selected.index[
        selected["factor_a__fwd_excess_ret_20d__Q1"].notna()
    ].tolist()
    assert five_day_dates == dates[[0, 5, 10, 15, 20]].tolist()
    assert twenty_day_dates == dates[[0, 20]].tolist()


def test_summarize_backtest_reports_core_statistics() -> None:
    returns = pd.DataFrame(
        {"factor_a__label": [0.10, -0.05]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    summary = summarize_backtest(returns, returns)

    assert summary.loc["factor_a__label", "long_short_mean"] == pytest.approx(0.025)
    assert summary.loc["factor_a__label", "long_short_hit_rate"] == pytest.approx(0.5)
    assert summary.loc[
        "factor_a__label", "long_short_cumulative_return"
    ] == pytest.approx(0.045)


def test_compute_quantile_backtest_writes_outputs(tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    factors, labels = _sample_factor_label_panels()
    factors.to_parquet(processed_dir / "factor_panel.parquet")
    labels.to_parquet(processed_dir / "label_panel.parquet")
    config_path = _write_config(tmp_path, processed_dir)

    paths = compute_quantile_backtest(config_path=str(config_path))

    assert paths.keys() == {
        "quantile_returns",
        "long_short_returns",
        "long_only_returns",
        "backtest_summary",
        "cumulative_return_figure",
    }
    assert all(path.exists() for path in paths.values())
    summary = pd.read_csv(paths["backtest_summary"])
    assert summary.loc[0, "factor_label"] == "factor_a__fwd_excess_ret_5d"
    assert paths["cumulative_return_figure"].stat().st_size > 0


def _sample_factor_label_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    tickers = [f"{ticker:06d}" for ticker in range(1, 6)]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    factors = pd.DataFrame(
        {
            "factor_a": [1.0, 2.0, 3.0, 4.0, 5.0] * 2,
        },
        index=index,
    )
    labels = pd.DataFrame(
        {
            "fwd_excess_ret_5d": [
                0.01,
                0.02,
                0.03,
                0.04,
                0.05,
                0.05,
                0.04,
                0.03,
                0.02,
                0.01,
            ],
        },
        index=index,
    )
    return factors, labels


def _series_by_date(values_by_date: dict[str, list[float]]) -> pd.Series:
    rows = []
    for date, values in values_by_date.items():
        for position, value in enumerate(values, start=1):
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "ticker": f"{position:06d}",
                    "value": value,
                }
            )
    return pd.DataFrame(rows).set_index(["date", "ticker"])["value"]


def _write_config(tmp_path: Path, processed_dir: Path) -> Path:
    config = {
        "data": {"processed_dir": str(processed_dir)},
        "features": {"factors": ["factor_a"]},
        "labels": {"horizons": [5], "use_excess_return": True},
        "backtest": {"n_quantiles": 5},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path
