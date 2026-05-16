from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from quant.backtest import plot_cumulative_returns
from quant.costs import plot_cost_sensitivity
from quant.factors import FACTOR_COLUMNS

DEFAULT_REPORTS_DIR = "reports"
FINAL_REPORT_TEMPLATE = "final_report.md"


def generate_report(config_path: str = "config.yaml") -> dict[str, Path]:
    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    source_dir = Path(config["data"]["source_dir"])
    work_dir = Path(config["data"]["work_dir"])
    reports_dir = _configured_reports_dir(config, config_file)
    figures_dir = reports_dir / "figures"
    tables_dir = reports_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    context = _load_report_context(config, source_dir, work_dir)
    paths: dict[str, Path] = {}
    paths.update(_write_report_tables(context, tables_dir))
    paths.update(_write_report_figures(context, figures_dir))
    paths["final_report_template"] = _ensure_final_report_template(reports_dir)
    return paths


def _load_report_context(
    config: dict[str, Any],
    source_dir: Path,
    work_dir: Path,
) -> dict[str, Any]:
    benchmark_path = (
        source_dir / f"benchmark_{config['data']['benchmark']}_ohlcv.parquet"
    )
    if benchmark_path.exists():
        benchmark_panel = pd.read_parquet(benchmark_path)
    else:
        benchmark_panel = None

    return {
        "config": config,
        "clean_panel": pd.read_parquet(work_dir / "clean_panel.parquet"),
        "factor_panel": pd.read_parquet(work_dir / "factor_panel.parquet"),
        "label_panel": pd.read_parquet(work_dir / "label_panel.parquet"),
        "benchmark_panel": benchmark_panel,
        "ic_summary": pd.read_parquet(work_dir / "ic_summary.parquet"),
        "rolling_ic": pd.read_parquet(work_dir / "rolling_ic.parquet"),
        "backtest_summary": pd.read_parquet(work_dir / "backtest_summary.parquet"),
        "long_short_returns": pd.read_parquet(work_dir / "long_short_returns.parquet"),
        "long_only_returns": pd.read_parquet(work_dir / "long_only_returns.parquet"),
        "cost_summary": pd.read_parquet(work_dir / "cost_sensitivity_summary.parquet"),
        "cost_adjusted_long_short": pd.read_parquet(
            work_dir / "cost_adjusted_long_short_returns.parquet"
        ),
        "cost_adjusted_long_only": pd.read_parquet(
            work_dir / "cost_adjusted_long_only_returns.parquet"
        ),
        "bootstrap_summary": pd.read_parquet(work_dir / "bootstrap_ic_summary.parquet"),
    }


def _write_report_tables(
    context: dict[str, Any],
    tables_dir: Path,
) -> dict[str, Path]:
    config = context["config"]
    paths = {
        "data_table": tables_dir / "data.md",
        "factor_table": tables_dir / "factors.md",
        "label_table": tables_dir / "labels.md",
        "ic_table": tables_dir / "ic_summary.md",
        "backtest_table": tables_dir / "backtest_summary.md",
        "cost_table": tables_dir / "cost_sensitivity.md",
        "bootstrap_table": tables_dir / "bootstrap_ic.md",
    }

    _write_markdown(paths["data_table"], "数据概览", _data_table(context))
    _write_markdown(paths["factor_table"], "因子定义", _factor_table(config))
    _write_markdown(paths["label_table"], "标签构造", _label_table(config))
    _write_markdown(
        paths["ic_table"],
        "IC 汇总",
        _ic_table(context["ic_summary"]),
    )
    _write_markdown(
        paths["backtest_table"],
        "分组回测汇总",
        _backtest_table(context["backtest_summary"]),
    )
    _write_markdown(
        paths["cost_table"],
        "交易成本敏感性",
        _cost_table(context["cost_summary"]),
    )
    _write_markdown(
        paths["bootstrap_table"],
        "Bootstrap IC 置信区间",
        _bootstrap_table(context["bootstrap_summary"]),
    )
    return paths


def _write_report_figures(
    context: dict[str, Any],
    figures_dir: Path,
) -> dict[str, Path]:
    ic_summary = context["ic_summary"].set_index("factor_label")
    paths = {
        "rolling_ic_figure": figures_dir / "rolling_ic.png",
        "quantile_figure": figures_dir / "quantile_cumulative_returns.png",
        "cost_figure": figures_dir / "cost_sensitivity.png",
    }
    plot_rolling_ic(context["rolling_ic"], ic_summary, paths["rolling_ic_figure"])
    plot_cumulative_returns(
        context["long_short_returns"],
        context["long_only_returns"],
        paths["quantile_figure"],
    )
    plot_cost_sensitivity(
        context["cost_adjusted_long_short"],
        context["cost_adjusted_long_only"],
        paths["cost_figure"],
    )
    return paths


def plot_rolling_ic(
    rolling_ic: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: str | Path,
    max_columns: int = 10,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    selected = _select_rolling_ic_columns(rolling_ic, summary, max_columns=max_columns)
    figure, axis = plt.subplots(figsize=(14, 8))
    rolling_ic[selected].plot(ax=axis, linewidth=1.1)
    axis.axhline(0, color="#111827", linewidth=1)
    axis.set_title("Rolling IC")
    axis.set_xlabel("date")
    axis.set_ylabel("rolling IC")
    axis.legend(
        [_short_pair_label(column) for column in selected],
        fontsize=8,
        loc="best",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _ensure_final_report_template(reports_dir: Path) -> Path:
    path = reports_dir / FINAL_REPORT_TEMPLATE
    if path.exists():
        current = path.read_text(encoding="utf-8")
        if not _is_generated_english_template(current):
            return path
    path.write_text(_final_report_template(), encoding="utf-8")
    return path


def _is_generated_english_template(content: str) -> bool:
    normalized = content.replace("\r\n", "\n").strip()
    if normalized == _english_template().strip():
        return True
    return (
        normalized.startswith("# CSI500 Factor Research Report\n")
        and "Use `tables/data.md`." in normalized
        and "Write the final interpretation manually." in normalized
    )


def _final_report_template() -> str:
    return """# 中证 500 因子研究报告

## 1. 研究问题

在这里手动填写研究问题和核心结论。

## 2. 数据

参考 `tables/data.md`。

## 3. 因子定义

参考 `tables/factors.md`。

## 4. 标签构造

参考 `tables/labels.md`。

## 5. IC 分析

参考 `tables/ic_summary.md`。

![Rolling IC](figures/rolling_ic.png)

## 6. 分组回测

参考 `tables/backtest_summary.md`。

![分组累计收益](figures/quantile_cumulative_returns.png)

## 7. 交易成本敏感性

参考 `tables/cost_sensitivity.md`。

![交易成本敏感性](figures/cost_sensitivity.png)

## 8. 稳健性检验

参考 `tables/bootstrap_ic.md`。

## 9. 局限性

在这里手动填写最终解释。

## 10. 结论

在这里手动填写最终解释。
"""


def _english_template() -> str:
    return """# CSI500 Factor Research Report

## 1. Research Question

Write the research question and high-level conclusion here.

## 2. Data

Use `tables/data.md`.

## 3. Factor Definitions

Use `tables/factors.md`.

## 4. Label Construction

Use `tables/labels.md`.

## 5. IC Analysis

Use `tables/ic_summary.md`.

![Rolling IC](figures/rolling_ic.png)

## 6. Quantile Backtest

Use `tables/backtest_summary.md`.

![Quantile Cumulative Returns](figures/quantile_cumulative_returns.png)

## 7. Transaction Cost Sensitivity

Use `tables/cost_sensitivity.md`.

![Cost Sensitivity](figures/cost_sensitivity.png)

## 8. Robustness Checks

Use `tables/bootstrap_ic.md`.

## 9. Limitations

- Current CSI 500 constituents are used instead of historical membership.
- Public data quality and API availability can affect reproducibility.
- Transaction-cost modeling is simplified.
- Long-short results evaluate ranking power and are not directly executable.

## 10. Conclusion

Write the final interpretation manually.
"""


def _data_table(context: dict[str, Any]) -> str:
    config = context["config"]
    clean_panel = context["clean_panel"]
    benchmark_panel = context["benchmark_panel"]
    dates = _panel_values(clean_panel, "date")
    tickers = _panel_values(clean_panel, "ticker")
    rows = [
        ["股票池", str(config["data"]["universe"])],
        ["Benchmark", str(config["data"]["benchmark"])],
        ["开始日期", str(dates.min().date())],
        ["结束日期", str(dates.max().date())],
        ["股票面板行数", str(len(clean_panel))],
        ["股票数量", str(tickers.nunique())],
    ]
    if benchmark_panel is not None:
        benchmark_dates = _panel_values(benchmark_panel, "date")
        rows.extend(
            [
                ["Benchmark 行数", str(len(benchmark_panel))],
                ["Benchmark 开始日期", str(benchmark_dates.min().date())],
                ["Benchmark 结束日期", str(benchmark_dates.max().date())],
            ]
        )
    return _markdown_table(["项目", "值"], rows)


def _panel_values(panel: pd.DataFrame, name: str) -> pd.Index:
    if name in panel.index.names:
        return panel.index.get_level_values(name)
    if name in panel.columns:
        return pd.Index(panel[name])
    raise ValueError(f"Panel is missing required field: {name}.")


def _factor_table(config: dict[str, Any]) -> str:
    configured = sorted(config.get("features", {}).get("factors", FACTOR_COLUMNS))
    rows = [[factor, _factor_description(factor)] for factor in configured]
    return _markdown_table(["因子", "定义"], rows)


def _label_table(config: dict[str, Any]) -> str:
    horizons = config.get("labels", {}).get("horizons", [5, 20])
    use_excess = config.get("labels", {}).get("use_excess_return", True)
    rows = []
    for horizon in horizons:
        label = f"{'fwd_excess_ret' if use_excess else 'fwd_ret'}_{int(horizon)}d"
        construction = (
            "个股 forward return 减 Benchmark forward return"
            if use_excess
            else "个股 forward return"
        )
        rows.append([label, construction])
    rows.sort(key=lambda row: row[0])
    return _markdown_table(["标签", "构造方式"], rows)


def _ic_table(ic_summary: pd.DataFrame) -> str:
    table = ic_summary.sort_values("factor_label")
    return _dataframe_table(
        table,
        [
            "factor_label",
            "ic_mean",
            "ic_ir",
            "rank_ic_mean",
            "rank_ic_ir",
            "rank_ic_positive_rate",
        ],
        header_map={
            "factor_label": "因子-标签",
            "ic_mean": "IC 均值",
            "ic_ir": "ICIR",
            "rank_ic_mean": "Rank IC 均值",
            "rank_ic_ir": "Rank ICIR",
            "rank_ic_positive_rate": "Rank IC 为正比例",
        },
    )


def _backtest_table(backtest_summary: pd.DataFrame) -> str:
    table = backtest_summary.sort_values("factor_label")
    return _dataframe_table(
        table,
        [
            "factor_label",
            "long_short_mean",
            "long_short_hit_rate",
            "long_short_cumulative_return",
            "long_only_mean",
            "long_only_hit_rate",
            "long_only_cumulative_return",
        ],
        header_map={
            "factor_label": "因子-标签",
            "long_short_mean": "Q5-Q1 平均收益",
            "long_short_hit_rate": "Q5-Q1 胜率",
            "long_short_cumulative_return": "Q5-Q1 累计收益",
            "long_only_mean": "Q5 平均收益",
            "long_only_hit_rate": "Q5 胜率",
            "long_only_cumulative_return": "Q5 累计收益",
        },
    )


def _cost_table(cost_summary: pd.DataFrame) -> str:
    grouped = (
        cost_summary.groupby(["portfolio", "cost_bps"], as_index=False)
        .agg(
            net_mean=("net_mean", "mean"),
            net_cumulative_return=("net_cumulative_return", "mean"),
            average_turnover=("average_turnover", "mean"),
        )
        .sort_values(["portfolio", "cost_bps"])
    )
    return _dataframe_table(
        grouped,
        [
            "portfolio",
            "cost_bps",
            "net_mean",
            "net_cumulative_return",
            "average_turnover",
        ],
        header_map={
            "portfolio": "组合",
            "cost_bps": "成本 bps",
            "net_mean": "扣成本后平均收益",
            "net_cumulative_return": "扣成本后累计收益",
            "average_turnover": "平均换手",
        },
    )


def _bootstrap_table(bootstrap_summary: pd.DataFrame) -> str:
    table = bootstrap_summary.sort_values(["factor_label", "metric"])
    count_column = "n_obs" if "n_obs" in bootstrap_summary.columns else "n_observations"
    if {"factor", "label"}.issubset(bootstrap_summary.columns):
        columns = ["factor", "label", "metric", "mean"]
    else:
        columns = ["factor_label", "metric", "mean"]
    if "bootstrap_std" in bootstrap_summary.columns:
        columns.append("bootstrap_std")
    columns.extend(["ci_lower", "ci_upper"])
    if "block_length" in bootstrap_summary.columns:
        columns.append("block_length")
    columns.append(count_column)
    return _dataframe_table(
        table,
        columns,
        header_map={
            "factor_label": "因子-标签",
            "metric": "指标",
            "mean": "均值",
            "ci_lower": "CI 下界",
            "ci_upper": "CI 上界",
            "n_observations": "样本数",
        },
    )


def _write_markdown(path: Path, title: str, table: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{table}\n", encoding="utf-8")


def _dataframe_table(
    frame: pd.DataFrame,
    columns: list[str],
    header_map: dict[str, str] | None = None,
) -> str:
    rows = []
    for _, row in frame[columns].iterrows():
        rows.append([_format_value(row[column]) for column in columns])
    if header_map:
        headers = [header_map.get(column, column) for column in columns]
    else:
        headers = columns
    return _markdown_table(headers, rows)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _select_rolling_ic_columns(
    rolling_ic: pd.DataFrame,
    summary: pd.DataFrame,
    max_columns: int,
) -> list[str]:
    if rolling_ic.empty:
        return []
    return sorted(rolling_ic.columns)[:max_columns]


def _short_pair_label(label: str) -> str:
    return label.replace("__fwd_excess_ret_", " -> ").replace("__fwd_ret_", " -> ")


def _factor_description(factor: str) -> str:
    descriptions = {
        "reversal_5": "-ret_5d",
        "momentum_20": "过去 20 个交易日收益率",
        "low_volatility_20": "过去 20 个交易日收益波动率的相反数",
        "turnover_change_20": "当前换手率 / 过去 20 日平均换手率 - 1",
        "liquidity_20": "过去 20 日平均成交额取 log",
    }
    return descriptions.get(factor, "配置中的因子")


def _configured_reports_dir(config: dict[str, Any], config_path: Path) -> Path:
    reports_dir = Path(config.get("reports", {}).get("dir", DEFAULT_REPORTS_DIR))
    if reports_dir.is_absolute():
        return reports_dir
    return config_path.parent / reports_dir
