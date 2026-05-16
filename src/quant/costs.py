from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from quant.backtest import (
    LABEL_HORIZON_REBALANCE,
    assign_quantile_groups,
    compute_cumulative_returns,
    compute_long_only_returns,
    compute_long_short_returns,
    compute_quantile_returns,
    direction_for_pair,
    infer_factor_directions,
    load_factor_directions,
    select_rebalance_dates,
)
from quant.factors import FACTOR_COLUMNS
from quant.fetch import PANEL_INDEX
from quant.labels import LABEL_COLUMNS

DEFAULT_REPORTS_DIR = "reports"


def compute_turnover(weights: pd.DataFrame) -> pd.DataFrame:
    if list(weights.index.names) != PANEL_INDEX:
        raise ValueError("weights must be indexed by date and ticker.")

    dates = pd.Index(weights.index.get_level_values("date").unique()).sort_values()
    result = pd.DataFrame(index=dates)
    result.index.name = "date"
    for column in weights.columns:
        matrix = weights[column].dropna().unstack("ticker").fillna(0).sort_index()
        turnover = matrix.diff().abs().sum(axis=1)
        if not matrix.empty:
            turnover.iloc[0] = matrix.iloc[0].abs().sum()
        result[column] = turnover
    return result


def apply_transaction_cost(
    returns: pd.DataFrame,
    turnover: pd.DataFrame,
    rate_bps: float,
) -> pd.DataFrame:
    rate = rate_bps / 10_000
    adjusted = returns.copy()
    for column in adjusted.columns:
        turnover_column = _turnover_column_for_return(column, turnover)
        adjusted[column] = (
            adjusted[column] - turnover[turnover_column].reindex(adjusted.index) * rate
        )
    return adjusted


def run_cost_sensitivity(
    long_short_returns: pd.DataFrame,
    long_only_returns: pd.DataFrame,
    long_short_turnover: pd.DataFrame,
    long_only_turnover: pd.DataFrame,
    rates_bps: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    long_short_adjusted = _cost_adjusted_returns_by_rate(
        long_short_returns,
        long_short_turnover,
        rates_bps,
    )
    long_only_adjusted = _cost_adjusted_returns_by_rate(
        long_only_returns,
        long_only_turnover,
        rates_bps,
    )
    summary = pd.concat(
        [
            _summarize_costs(
                long_short_returns,
                long_short_turnover,
                long_short_adjusted,
                rates_bps,
                portfolio="long_short",
            ),
            _summarize_costs(
                long_only_returns,
                long_only_turnover,
                long_only_adjusted,
                rates_bps,
                portfolio="long_only",
            ),
        ],
        ignore_index=True,
    )
    return long_short_adjusted, long_only_adjusted, summary


def build_quantile_portfolio_weights(
    factor_panel: pd.DataFrame,
    factor_columns: Sequence[str],
    rebalance_dates: pd.Index,
    n_quantiles: int = 5,
    portfolio: str = "long_short",
    directions: dict[str, int] | None = None,
) -> pd.DataFrame:
    if portfolio not in {"long_short", "long_only"}:
        raise ValueError("portfolio must be 'long_short' or 'long_only'.")

    filtered = _filter_rebalance_dates(factor_panel, rebalance_dates)
    weights = pd.DataFrame(index=filtered.index)
    for factor in factor_columns:
        groups = assign_quantile_groups(filtered[factor], n_quantiles=n_quantiles)
        weights[factor] = _weights_from_groups(
            groups,
            n_quantiles=n_quantiles,
            portfolio=portfolio,
            direction=direction_for_pair(factor, directions),
        )
    return weights.sort_index()


def _build_return_portfolio_weights(
    factor_panel: pd.DataFrame,
    returns: pd.DataFrame,
    n_quantiles: int,
    portfolio: str,
    directions: dict[str, int],
) -> pd.DataFrame:
    frames = []
    for column in returns.columns:
        rebalance_dates = returns.index[returns[column].notna()]
        factor = _factor_from_pair(column)
        weights = build_quantile_portfolio_weights(
            factor_panel,
            [factor],
            rebalance_dates,
            n_quantiles=n_quantiles,
            portfolio=portfolio,
            directions={factor: direction_for_pair(column, directions)},
        )
        frames.append(weights.rename(columns={factor: column}))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


def compute_cost_analysis(config_path: str = "config.yaml") -> dict[str, Path]:
    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    work_dir = Path(config["data"]["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = _configured_reports_dir(config, config_file)
    figures_dir = reports_dir / "figures"
    n_quantiles = int(config.get("backtest", {}).get("n_quantiles", 5))
    rebalance = config.get("backtest", {}).get(
        "rebalance",
        LABEL_HORIZON_REBALANCE,
    )
    rates_bps = [float(rate) for rate in config.get("costs", {}).get("rates_bps", [])]
    if not rates_bps:
        raise ValueError("At least one transaction cost rate is required.")

    factor_panel = pd.read_parquet(work_dir / "factor_panel.parquet")
    label_panel = pd.read_parquet(work_dir / "label_panel.parquet")
    factor_columns = _configured_factor_columns(config, factor_panel)
    label_columns = _configured_label_columns(config, label_panel)
    directions = load_factor_directions(work_dir)
    if not directions:
        directions = infer_factor_directions(
            factor_panel,
            label_panel,
            factor_columns,
            label_columns,
        )

    quantile_returns = compute_quantile_returns(
        factor_panel,
        label_panel,
        factor_columns=factor_columns,
        label_columns=label_columns,
        n_quantiles=n_quantiles,
    )
    quantile_returns = select_rebalance_dates(quantile_returns, rebalance)
    long_short_returns = compute_long_short_returns(
        quantile_returns,
        n_quantiles=n_quantiles,
        directions=directions,
    )
    long_only_returns = compute_long_only_returns(
        quantile_returns,
        long_quantile=n_quantiles,
        directions=directions,
    )

    long_short_weights = _build_return_portfolio_weights(
        factor_panel,
        long_short_returns,
        n_quantiles=n_quantiles,
        portfolio="long_short",
        directions=directions,
    )
    long_only_weights = _build_return_portfolio_weights(
        factor_panel,
        long_only_returns,
        n_quantiles=n_quantiles,
        portfolio="long_only",
        directions=directions,
    )
    long_short_turnover = compute_turnover(long_short_weights)
    long_only_turnover = compute_turnover(long_only_weights)
    long_short_adjusted, long_only_adjusted, summary = run_cost_sensitivity(
        long_short_returns,
        long_only_returns,
        long_short_turnover,
        long_only_turnover,
        rates_bps,
    )

    long_short_turnover_path = work_dir / "long_short_turnover.parquet"
    long_only_turnover_path = work_dir / "long_only_turnover.parquet"
    long_short_adjusted_path = work_dir / "cost_adjusted_long_short_returns.parquet"
    long_only_adjusted_path = work_dir / "cost_adjusted_long_only_returns.parquet"
    summary_path = work_dir / "cost_sensitivity_summary.parquet"
    long_short_turnover.to_parquet(long_short_turnover_path)
    long_only_turnover.to_parquet(long_only_turnover_path)
    long_short_adjusted.to_parquet(long_short_adjusted_path)
    long_only_adjusted.to_parquet(long_only_adjusted_path)
    summary.to_parquet(summary_path)
    figure_path = plot_cost_sensitivity(
        long_short_adjusted,
        long_only_adjusted,
        figures_dir / "cost_sensitivity.png",
    )

    return {
        "long_short_turnover": long_short_turnover_path,
        "long_only_turnover": long_only_turnover_path,
        "cost_adjusted_long_short_returns": long_short_adjusted_path,
        "cost_adjusted_long_only_returns": long_only_adjusted_path,
        "cost_sensitivity_summary": summary_path,
        "cost_sensitivity_figure": figure_path,
    }


def plot_cost_sensitivity(
    long_short_adjusted: pd.DataFrame,
    long_only_adjusted: pd.DataFrame,
    output_path: str | Path,
    max_columns: int = 5,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    long_short_cumulative = compute_cumulative_returns(long_short_adjusted)
    long_only_cumulative = compute_cumulative_returns(long_only_adjusted)
    long_short_selected = _select_return_columns(long_short_cumulative, max_columns)
    long_only_selected = _select_return_columns(long_only_cumulative, max_columns)

    figure, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)
    if long_short_selected:
        long_short_cumulative[long_short_selected].plot(ax=axes[0], linewidth=1.1)
    axes[0].axhline(0, color="#111827", linewidth=1)
    axes[0].set_title("Cost-Adjusted Q5-Q1 Cumulative Returns")
    axes[0].set_ylabel("cumulative return")
    axes[0].legend(long_short_selected, fontsize=7, loc="best")

    if long_only_selected:
        long_only_cumulative[long_only_selected].plot(ax=axes[1], linewidth=1.1)
    axes[1].axhline(0, color="#111827", linewidth=1)
    axes[1].set_title("Cost-Adjusted Q5 Long-Only Cumulative Returns")
    axes[1].set_xlabel("date")
    axes[1].set_ylabel("cumulative return")
    axes[1].legend(long_only_selected, fontsize=7, loc="best")

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _cost_adjusted_returns_by_rate(
    returns: pd.DataFrame,
    turnover: pd.DataFrame,
    rates_bps: Sequence[float],
) -> pd.DataFrame:
    frames = []
    for rate in rates_bps:
        adjusted = apply_transaction_cost(returns, turnover, rate)
        rename_map = {
            column: f"{column}__cost_{_rate_label(rate)}bps" for column in adjusted
        }
        adjusted = adjusted.rename(
            columns=rename_map,
        )
        frames.append(adjusted)
    return pd.concat(frames, axis=1).sort_index()


def _summarize_costs(
    gross_returns: pd.DataFrame,
    turnover: pd.DataFrame,
    adjusted_returns: pd.DataFrame,
    rates_bps: Sequence[float],
    portfolio: str,
) -> pd.DataFrame:
    rows = []
    gross_cumulative = compute_cumulative_returns(gross_returns)
    adjusted_cumulative = compute_cumulative_returns(adjusted_returns)
    for rate in rates_bps:
        rate_label = _rate_label(rate)
        for column in gross_returns.columns:
            adjusted_column = f"{column}__cost_{rate_label}bps"
            turnover_column = _turnover_column_for_return(column, turnover)
            rows.append(
                {
                    "factor_label": column,
                    "portfolio": portfolio,
                    "cost_bps": rate,
                    "gross_mean": gross_returns[column].mean(),
                    "net_mean": adjusted_returns[adjusted_column].mean(),
                    "gross_cumulative_return": gross_cumulative[column].iloc[-1],
                    "net_cumulative_return": adjusted_cumulative[adjusted_column].iloc[
                        -1
                    ],
                    "average_turnover": turnover[turnover_column].mean(),
                    "n_periods": gross_returns[column].count(),
                }
            )
    return pd.DataFrame(rows)


def _weights_from_groups(
    groups: pd.Series,
    n_quantiles: int,
    portfolio: str,
    direction: int = 1,
) -> pd.Series:
    weights = pd.Series(0.0, index=groups.index)
    long_quantile = 1 if direction < 0 else n_quantiles
    short_quantile = n_quantiles if direction < 0 else 1
    for _, group in groups.groupby(level="date", sort=True):
        long_index = group[group == long_quantile].index
        short_index = group[group == short_quantile].index
        if len(long_index) > 0:
            weights.loc[long_index] = 1 / len(long_index)
        if portfolio == "long_short" and len(short_index) > 0:
            weights.loc[short_index] = -1 / len(short_index)
    return weights


def _filter_rebalance_dates(
    factor_panel: pd.DataFrame,
    rebalance_dates: pd.Index,
) -> pd.DataFrame:
    dates = factor_panel.index.get_level_values("date")
    return factor_panel[dates.isin(rebalance_dates)].sort_index()


def _turnover_column_for_return(column: str, turnover: pd.DataFrame) -> str:
    if column in turnover.columns:
        return column
    factor = _factor_from_pair(column)
    if factor in turnover.columns:
        return factor
    raise ValueError(f"Missing turnover column for {column}.")


def _select_return_columns(returns: pd.DataFrame, max_columns: int) -> list[str]:
    if returns.empty:
        return []
    final_returns = returns.iloc[-1].abs().sort_values(ascending=False)
    return [column for column in final_returns.index[:max_columns]]


def _factor_from_pair(column: str) -> str:
    return column.split("__", maxsplit=1)[0]


def _rate_label(rate: float) -> str:
    return str(int(rate)) if float(rate).is_integer() else str(rate).replace(".", "p")


def _configured_reports_dir(config: dict, config_path: Path) -> Path:
    reports_dir = Path(config.get("reports", {}).get("dir", DEFAULT_REPORTS_DIR))
    if reports_dir.is_absolute():
        return reports_dir
    return config_path.parent / reports_dir


def _configured_factor_columns(
    config: dict,
    factor_panel: pd.DataFrame,
) -> list[str]:
    configured = config.get("features", {}).get("factors", FACTOR_COLUMNS)
    return _require_columns(factor_panel, configured, "factor panel")


def _configured_label_columns(
    config: dict,
    label_panel: pd.DataFrame,
) -> list[str]:
    if "labels" not in config:
        return _require_columns(label_panel, LABEL_COLUMNS, "label panel")

    use_excess_return = config["labels"].get("use_excess_return", True)
    prefix = "fwd_excess_ret" if use_excess_return else "fwd_ret"
    configured = [
        f"{prefix}_{int(horizon)}d" for horizon in config["labels"]["horizons"]
    ]
    return _require_columns(label_panel, configured, "label panel")


def _require_columns(
    panel: pd.DataFrame,
    columns: Sequence[str],
    panel_name: str,
) -> list[str]:
    missing = [column for column in columns if column not in panel.columns]
    if missing:
        raise ValueError(f"Missing {panel_name} column(s): {', '.join(missing)}.")
    return list(columns)
