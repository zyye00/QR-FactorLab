from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from quant.metrics import (
    compute_ic,
    compute_ic_analysis,
    compute_icir,
    compute_rank_ic,
    compute_rolling_ic,
    summarize_ic,
)


def test_compute_ic_and_rank_ic_by_cross_sectional_date() -> None:
    factors, labels = _sample_factor_label_panels()

    ic = compute_ic(factors, labels)
    rank_ic = compute_rank_ic(factors, labels)

    positive_pair = "factor_a__fwd_excess_ret_5d"
    inverse_pair = "factor_b__fwd_excess_ret_5d"
    constant_pair = "constant__fwd_excess_ret_5d"

    assert ic.loc[pd.Timestamp("2024-01-01"), positive_pair] == pytest.approx(1.0)
    assert rank_ic.loc[pd.Timestamp("2024-01-01"), positive_pair] == pytest.approx(1.0)
    assert ic.loc[pd.Timestamp("2024-01-01"), inverse_pair] == pytest.approx(-1.0)
    assert np.isnan(ic.loc[pd.Timestamp("2024-01-01"), constant_pair])


def test_summarize_ic_computes_mean_icir_and_rank_ic() -> None:
    ic_panel = pd.DataFrame(
        {"factor_a__label": [1.0, -1.0, np.nan]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    rank_ic_panel = pd.DataFrame(
        {"factor_a__label": [0.5, 0.5, np.nan]},
        index=ic_panel.index,
    )

    summary = summarize_ic(ic_panel, rank_ic_panel)

    row = summary.loc["factor_a__label"]
    assert row["ic_mean"] == pytest.approx(0.0)
    assert row["ic_std"] == pytest.approx(np.sqrt(2.0))
    assert row["ic_ir"] == pytest.approx(0.0)
    assert row["ic_positive_rate"] == pytest.approx(0.5)
    assert row["ic_n_days"] == 2
    assert row["rank_ic_mean"] == pytest.approx(0.5)
    assert np.isnan(row["rank_ic_ir"])


def test_compute_icir_returns_nan_for_constant_or_empty_series() -> None:
    assert np.isnan(compute_icir(pd.Series([0.2, 0.2])))
    assert np.isnan(compute_icir(pd.Series([np.nan])))


def test_compute_rolling_ic_uses_full_window_by_default() -> None:
    ic_panel = pd.DataFrame(
        {"factor_a__label": [0.1, 0.2, 0.3]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )

    rolling = compute_rolling_ic(ic_panel, window=2)

    assert np.isnan(rolling.loc[pd.Timestamp("2024-01-01"), "factor_a__label"])
    assert rolling.loc[pd.Timestamp("2024-01-02"), "factor_a__label"] == pytest.approx(
        0.15
    )
    assert rolling.loc[pd.Timestamp("2024-01-03"), "factor_a__label"] == pytest.approx(
        0.25
    )


def test_compute_ic_analysis_writes_outputs(tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    factors, labels = _sample_factor_label_panels()
    factors[["factor_a"]].to_parquet(processed_dir / "factor_panel.parquet")
    labels.to_parquet(processed_dir / "label_panel.parquet")
    config_path = _write_config(tmp_path, processed_dir)

    paths = compute_ic_analysis(config_path=str(config_path))

    assert paths.keys() == {
        "ic_panel",
        "rank_ic_panel",
        "rolling_ic",
        "ic_summary",
    }
    assert all(path.exists() for path in paths.values())
    summary = pd.read_csv(paths["ic_summary"])
    assert summary.loc[0, "factor_label"] == "factor_a__fwd_excess_ret_5d"
    assert summary.loc[0, "ic_mean"] == pytest.approx(0.0)
    assert not (tmp_path / "reports" / "ic_summary.md").exists()
    assert not (tmp_path / "reports" / "figures" / "rolling_ic.png").exists()


def _sample_factor_label_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.MultiIndex.from_product(
        [
            pd.to_datetime(["2024-01-01", "2024-01-02"]),
            ["000001", "000002", "000003"],
        ],
        names=["date", "ticker"],
    )
    factors = pd.DataFrame(
        {
            "factor_a": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "factor_b": [3.0, 2.0, 1.0, 1.0, 2.0, 3.0],
            "constant": [1.0] * 6,
        },
        index=index,
    )
    labels = pd.DataFrame(
        {
            "fwd_excess_ret_5d": [2.0, 4.0, 6.0, 6.0, 4.0, 2.0],
        },
        index=index,
    )
    return factors, labels


def _write_config(tmp_path: Path, processed_dir: Path) -> Path:
    config = {
        "data": {"processed_dir": str(processed_dir)},
        "features": {"factors": ["factor_a"]},
        "labels": {"horizons": [5], "use_excess_return": True},
        "metrics": {"rolling_window": 2},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path
