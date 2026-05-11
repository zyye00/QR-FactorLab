from pathlib import Path

import pandas as pd
import pytest
import yaml

from quant.costs import (
    apply_transaction_cost,
    build_quantile_portfolio_weights,
    compute_cost_analysis,
    compute_turnover,
    run_cost_sensitivity,
)


def test_compute_turnover_from_weight_changes() -> None:
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-01"), "000001"),
            (pd.Timestamp("2024-01-01"), "000002"),
            (pd.Timestamp("2024-01-02"), "000001"),
            (pd.Timestamp("2024-01-02"), "000002"),
        ],
        names=["date", "ticker"],
    )
    weights = pd.DataFrame({"factor_a": [0.5, 0.5, 1.0, 0.0]}, index=index)

    turnover = compute_turnover(weights)

    assert turnover.loc[pd.Timestamp("2024-01-01"), "factor_a"] == pytest.approx(1.0)
    assert turnover.loc[pd.Timestamp("2024-01-02"), "factor_a"] == pytest.approx(1.0)


def test_apply_transaction_cost_subtracts_turnover_times_rate() -> None:
    returns = pd.DataFrame(
        {"factor_a__label": [0.10, 0.20]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    turnover = pd.DataFrame(
        {"factor_a": [1.0, 0.5]},
        index=returns.index,
    )

    adjusted = apply_transaction_cost(returns, turnover, rate_bps=10)

    assert adjusted.loc[pd.Timestamp("2024-01-01"), "factor_a__label"] == (
        pytest.approx(0.099)
    )
    assert adjusted.loc[pd.Timestamp("2024-01-02"), "factor_a__label"] == (
        pytest.approx(0.1995)
    )


def test_build_quantile_portfolio_weights_matches_long_short_returns() -> None:
    factors, _ = _sample_factor_label_panels()

    weights = build_quantile_portfolio_weights(
        factors,
        factor_columns=["factor_a"],
        rebalance_dates=pd.DatetimeIndex([pd.Timestamp("2024-01-01")]),
        n_quantiles=5,
        portfolio="long_short",
    )

    first_date = weights.xs(pd.Timestamp("2024-01-01"), level="date")["factor_a"]
    assert first_date.loc["000001"] == pytest.approx(-1.0)
    assert first_date.loc["000005"] == pytest.approx(1.0)
    assert first_date.sum() == pytest.approx(0.0)


def test_run_cost_sensitivity_returns_adjusted_returns_and_summary() -> None:
    returns = pd.DataFrame(
        {"factor_a__label": [0.10, 0.20]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    turnover = pd.DataFrame({"factor_a": [1.0, 0.5]}, index=returns.index)

    long_short_adjusted, long_only_adjusted, summary = run_cost_sensitivity(
        returns,
        returns,
        turnover,
        turnover,
        rates_bps=[10],
    )

    assert "factor_a__label__cost_10bps" in long_short_adjusted.columns
    assert "factor_a__label__cost_10bps" in long_only_adjusted.columns
    assert summary.loc[0, "portfolio"] == "long_short"
    assert summary.loc[0, "cost_bps"] == 10


def test_compute_cost_analysis_writes_outputs(tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    factors, labels = _sample_factor_label_panels()
    factors.to_parquet(processed_dir / "factor_panel.parquet")
    labels.to_parquet(processed_dir / "label_panel.parquet")
    config_path = _write_config(tmp_path, processed_dir)

    paths = compute_cost_analysis(config_path=str(config_path))

    assert paths.keys() == {
        "long_short_turnover",
        "long_only_turnover",
        "cost_adjusted_long_short_returns",
        "cost_adjusted_long_only_returns",
        "cost_sensitivity_summary",
        "cost_sensitivity_figure",
    }
    assert all(path.exists() for path in paths.values())
    summary = pd.read_csv(paths["cost_sensitivity_summary"])
    assert set(summary["portfolio"]) == {"long_short", "long_only"}
    assert paths["cost_sensitivity_figure"].stat().st_size > 0


def _sample_factor_label_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    tickers = [f"{ticker:06d}" for ticker in range(1, 6)]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    factors = pd.DataFrame({"factor_a": [1.0, 2.0, 3.0, 4.0, 5.0] * 2}, index=index)
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


def _write_config(tmp_path: Path, processed_dir: Path) -> Path:
    config = {
        "data": {"processed_dir": str(processed_dir)},
        "features": {"factors": ["factor_a"]},
        "labels": {"horizons": [5], "use_excess_return": True},
        "backtest": {"n_quantiles": 5, "rebalance": "D"},
        "costs": {"rates_bps": [10, 20]},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path
