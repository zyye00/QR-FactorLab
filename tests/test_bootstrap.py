from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from quant import bootstrap as bootstrap_module
from quant.bootstrap import (
    BootstrapSampler,
    CircularBlock,
    bootstrap_ic_summary,
    bootstrap_mean_ci,
    bootstrap_mean_result,
    compute_bootstrap_ic,
    create_bootstrap_sampler,
)


def test_bootstrap_mean_ci_handles_constant_values() -> None:
    mean, lower, upper = bootstrap_mean_ci(
        [0.2, 0.2, 0.2],
        n_samples=100,
        confidence_level=0.95,
        random_seed=42,
    )

    assert mean == pytest.approx(0.2)
    assert lower == pytest.approx(0.2)
    assert upper == pytest.approx(0.2)


def test_bootstrap_mean_ci_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        bootstrap_mean_ci([1.0], n_samples=0)
    with pytest.raises(ValueError, match="confidence_level"):
        bootstrap_mean_ci([1.0], confidence_level=1.0)


def test_bootstrap_mean_result_uses_registered_sampler_class(monkeypatch) -> None:
    monkeypatch.setitem(
        bootstrap_module.BOOTSTRAP_SAMPLERS,
        ConstantBootstrapSampler.method,
        ConstantBootstrapSampler,
    )
    sampler = create_bootstrap_sampler(
        {
            "method": ConstantBootstrapSampler.method,
            "n_bootstrap": 20,
            "confidence_level": 0.90,
        }
    )

    result = bootstrap_mean_result(
        [0.1, 0.2, 0.3],
        sampler=sampler,
    )

    assert result.method == "constant"
    assert result.mean == pytest.approx(0.2)
    assert result.ci_lower == pytest.approx(0.2)
    assert result.ci_upper == pytest.approx(0.2)


def test_bootstrap_ic_summary_returns_ic_and_rank_ic_rows() -> None:
    ic_panel = pd.DataFrame({"factor_a__label": [0.1, 0.2, 0.3]})
    rank_ic_panel = pd.DataFrame({"factor_a__label": [0.2, 0.2, 0.2]})
    sampler = create_bootstrap_sampler(
        {
            "n_bootstrap": 100,
            "confidence_level": 0.90,
            "random_seed": 42,
        }
    )

    summary = bootstrap_ic_summary(
        ic_panel,
        rank_ic_panel,
        sampler=sampler,
    )

    assert summary["metric"].tolist() == ["ic", "rank_ic"]
    assert set(summary["method"]) == {"iid"}
    assert summary.loc[summary["metric"] == "ic", "mean"].iloc[0] == pytest.approx(0.2)
    rank_row = summary.loc[summary["metric"] == "rank_ic"].iloc[0]
    assert rank_row["ci_lower"] == pytest.approx(0.2)
    assert rank_row["ci_upper"] == pytest.approx(0.2)


def test_bootstrap_mean_ci_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="non-NaN"):
        bootstrap_mean_ci([np.nan])


def test_create_bootstrap_sampler_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown bootstrap method"):
        create_bootstrap_sampler({"method": "missing"})


def test_circular_block_sampler_uses_inferred_block_length() -> None:
    values = pd.Series(np.linspace(0.01, 0.12, 12))
    sampler = CircularBlock(
        n_bootstrap=100,
        confidence_level=0.90,
        random_seed=42,
        block_length_multiplier=2,
    )

    result = bootstrap_mean_result(
        values,
        sampler=sampler,
        factor_label="factor_a__fwd_excess_ret_5d",
    )

    assert result.method == "circular_block"
    assert result.block_length == 10
    assert result.n_bootstrap == 100
    assert result.n_obs == 12
    assert result.bootstrap_std >= 0
    assert result.ci_lower <= result.ci_upper


def test_circular_block_sampler_is_reproducible() -> None:
    values = pd.Series(np.linspace(0.01, 0.2, 20))
    sampler = CircularBlock(
        n_bootstrap=100,
        random_seed=42,
        block_length=4,
    )

    first = bootstrap_mean_result(values, sampler=sampler)
    second = bootstrap_mean_result(values, sampler=sampler)

    assert first.bootstrap_mean == pytest.approx(second.bootstrap_mean)
    assert first.bootstrap_std == pytest.approx(second.bootstrap_std)
    assert first.ci_lower == pytest.approx(second.ci_lower)
    assert first.ci_upper == pytest.approx(second.ci_upper)


def test_circular_block_sampler_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="block_length"):
        bootstrap_mean_result(
            [0.1, 0.2, 0.3],
            sampler=CircularBlock(block_length=4),
        )
    with pytest.raises(ValueError, match="non-NaN"):
        bootstrap_mean_result(
            [np.nan],
            sampler=CircularBlock(block_length=1),
        )


def test_compute_bootstrap_ic_writes_summary(tmp_path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    pd.DataFrame(
        {"factor_a__fwd_excess_ret_5d": np.linspace(0.01, 0.12, 12)}
    ).to_parquet(work_dir / "ic_panel.parquet")
    pd.DataFrame({"factor_a__fwd_excess_ret_5d": np.full(12, 0.2)}).to_parquet(
        work_dir / "rank_ic_panel.parquet"
    )
    config_path = _write_config(tmp_path, work_dir)

    paths = compute_bootstrap_ic(config_path=str(config_path))

    assert paths.keys() == {"bootstrap_ic_summary"}
    summary = pd.read_parquet(paths["bootstrap_ic_summary"])
    assert len(summary) == 2
    assert set(summary["metric"]) == {"ic", "rank_ic"}
    assert set(summary["method"]) == {"circular_block"}
    assert set(summary["factor"]) == {"factor_a"}
    assert set(summary["label"]) == {"fwd_excess_ret_5d"}
    assert set(summary["block_length"]) == {10}
    assert "bootstrap_std" in summary.columns


def _write_config(tmp_path: Path, work_dir: Path) -> Path:
    config = {
        "data": {"work_dir": str(work_dir)},
        "bootstrap": {
            "method": "circular_block",
            "n_bootstrap": 100,
            "confidence_level": 0.90,
            "block_length_multiplier": 2,
            "random_seed": 42,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


class ConstantBootstrapSampler(BootstrapSampler):
    method = "constant"

    def sample_means(
        self,
        values: np.ndarray,
        factor_label: str | None = None,
    ) -> np.ndarray:
        return np.full(self.n_bootstrap, values.mean())
