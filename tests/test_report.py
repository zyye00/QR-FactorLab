from pathlib import Path

import pandas as pd
import yaml

from quant.report import _ic_table, _select_rolling_ic_columns, generate_report


def test_generate_report_writes_tables_figures_and_template(tmp_path) -> None:
    source_dir = tmp_path / "source"
    work_dir = tmp_path / "work"
    source_dir.mkdir()
    work_dir.mkdir()
    _write_report_inputs(source_dir, work_dir)
    config_path = _write_config(tmp_path, source_dir, work_dir)

    paths = generate_report(config_path=str(config_path))

    assert paths["data_table"] == tmp_path / "reports" / "tables" / "data.md"
    assert paths["rolling_ic_figure"] == (
        tmp_path / "reports" / "figures" / "rolling_ic.png"
    )
    assert paths["final_report_template"] == tmp_path / "reports" / "final_report.md"
    assert all(path.exists() for path in paths.values())
    assert "factor_a__fwd_excess_ret_5d" in paths["ic_table"].read_text(
        encoding="utf-8"
    )
    data_table = paths["data_table"].read_text(encoding="utf-8")
    assert "# 数据概览" in data_table
    assert "股票池" in data_table
    template = paths["final_report_template"].read_text(encoding="utf-8")
    assert "# 中证 500 因子研究报告" in template
    assert "参考 `tables/data.md`" in template


def test_report_outputs_are_sorted_by_name() -> None:
    summary_for_plot = pd.DataFrame(
        {
            "ic_mean": [0.1, 0.9, 0.2],
        },
        index=["factor_c__label", "factor_a__label", "factor_b__label"],
    )
    rolling_ic = pd.DataFrame(
        {
            "factor_c__label": [0.1],
            "factor_a__label": [0.2],
            "factor_b__label": [0.3],
        }
    )
    summary_for_table = pd.DataFrame(
        {
            "factor_label": [
                "factor_c__label",
                "factor_a__label",
                "factor_b__label",
            ],
            "ic_mean": [0.1, 0.9, 0.2],
            "ic_ir": [1.0, 9.0, 2.0],
            "rank_ic_mean": [0.1, 0.9, 0.2],
            "rank_ic_ir": [1.0, 9.0, 2.0],
            "rank_ic_positive_rate": [0.4, 0.6, 0.5],
        }
    )

    selected = _select_rolling_ic_columns(rolling_ic, summary_for_plot, max_columns=10)
    table = _ic_table(summary_for_table)

    assert selected == ["factor_a__label", "factor_b__label", "factor_c__label"]
    assert table.index("factor_a__label") < table.index("factor_b__label")
    assert table.index("factor_b__label") < table.index("factor_c__label")


def test_generate_report_does_not_overwrite_existing_template(tmp_path) -> None:
    source_dir = tmp_path / "source"
    work_dir = tmp_path / "work"
    source_dir.mkdir()
    work_dir.mkdir()
    _write_report_inputs(source_dir, work_dir)
    config_path = _write_config(tmp_path, source_dir, work_dir)
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    template_path = reports_dir / "final_report.md"
    template_path.write_text("manual notes", encoding="utf-8")

    paths = generate_report(config_path=str(config_path))

    assert paths["final_report_template"] == template_path
    assert template_path.read_text(encoding="utf-8") == "manual notes"


def _write_report_inputs(source_dir: Path, work_dir: Path) -> None:
    index = pd.MultiIndex.from_product(
        [
            pd.to_datetime(["2024-01-01", "2024-01-02"]),
            ["000001", "000002"],
        ],
        names=["date", "ticker"],
    )
    clean_panel = pd.DataFrame(
        {
            "open": [1.0, 2.0, 1.1, 2.1],
            "high": [1.0, 2.0, 1.1, 2.1],
            "low": [1.0, 2.0, 1.1, 2.1],
            "close": [1.0, 2.0, 1.1, 2.1],
            "volume": [100, 100, 100, 100],
            "amount": [1000, 1000, 1000, 1000],
            "turnover": [0.1, 0.1, 0.1, 0.1],
            "ret_1d": [0.0, 0.0, 0.1, 0.05],
            "ret_5d": [0.0, 0.0, 0.1, 0.05],
            "ret_20d": [0.0, 0.0, 0.1, 0.05],
        },
        index=index,
    )
    clean_panel.to_parquet(work_dir / "clean_panel.parquet")
    pd.DataFrame({"factor_a": [1.0, 2.0, 1.5, 2.5]}, index=index).to_parquet(
        work_dir / "factor_panel.parquet"
    )
    pd.DataFrame(
        {"fwd_excess_ret_5d": [0.01, 0.02, 0.03, 0.04]},
        index=index,
    ).to_parquet(
        work_dir / "label_panel.parquet",
    )
    clean_panel.iloc[:2, :7].to_parquet(source_dir / "benchmark_000905_ohlcv.parquet")
    pd.DataFrame(
        {
            "factor_label": ["factor_a__fwd_excess_ret_5d"],
            "ic_mean": [0.1],
            "ic_ir": [1.2],
            "rank_ic_mean": [0.2],
            "rank_ic_ir": [1.5],
            "rank_ic_positive_rate": [0.6],
        }
    ).to_parquet(work_dir / "ic_summary.parquet")
    pd.DataFrame(
        {"factor_a__fwd_excess_ret_5d": [0.1, 0.2]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    ).to_parquet(work_dir / "rolling_ic.parquet")
    pd.DataFrame(
        {
            "factor_label": ["factor_a__fwd_excess_ret_5d"],
            "long_short_mean": [0.01],
            "long_short_hit_rate": [0.5],
            "long_short_cumulative_return": [0.1],
            "long_only_mean": [0.02],
            "long_only_hit_rate": [0.6],
            "long_only_cumulative_return": [0.2],
        }
    ).to_parquet(work_dir / "backtest_summary.parquet")
    pd.DataFrame(
        {"factor_a__fwd_excess_ret_5d": [0.01, 0.02]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    ).to_parquet(work_dir / "long_short_returns.parquet")
    pd.DataFrame(
        {"factor_a__fwd_excess_ret_5d": [0.02, 0.03]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    ).to_parquet(work_dir / "long_only_returns.parquet")
    pd.DataFrame(
        {
            "factor_label": ["factor_a__fwd_excess_ret_5d"],
            "portfolio": ["long_short"],
            "cost_bps": [10],
            "net_mean": [0.01],
            "net_cumulative_return": [0.08],
            "average_turnover": [0.5],
        }
    ).to_parquet(work_dir / "cost_sensitivity_summary.parquet")
    pd.DataFrame(
        {"factor_a__fwd_excess_ret_5d__cost_10bps": [0.01, 0.02]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    ).to_parquet(work_dir / "cost_adjusted_long_short_returns.parquet")
    pd.DataFrame(
        {"factor_a__fwd_excess_ret_5d__cost_10bps": [0.02, 0.03]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    ).to_parquet(work_dir / "cost_adjusted_long_only_returns.parquet")
    pd.DataFrame(
        {
            "factor_label": ["factor_a__fwd_excess_ret_5d"],
            "metric": ["ic"],
            "mean": [0.1],
            "ci_lower": [0.01],
            "ci_upper": [0.2],
            "n_observations": [2],
        }
    ).to_parquet(work_dir / "bootstrap_ic_summary.parquet")


def _write_config(tmp_path: Path, source_dir: Path, work_dir: Path) -> Path:
    config = {
        "data": {
            "source_dir": str(source_dir),
            "work_dir": str(work_dir),
            "universe": "CSI500",
            "benchmark": "000905",
        },
        "features": {"factors": ["factor_a"]},
        "labels": {"horizons": [5], "use_excess_return": True},
        "backtest": {"rebalance": "W"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path
