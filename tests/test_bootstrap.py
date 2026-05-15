from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from quant import bootstrap as bootstrap_module
from quant.bootstrap import (
    BootstrapSampler,
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
            "n_samples": 20,
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
            "n_samples": 100,
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
    assert summary.loc[summary["metric"] == "ic", "mean"].iloc[0] == pytest.approx(
        0.2
    )
    rank_row = summary.loc[summary["metric"] == "rank_ic"].iloc[0]
    assert rank_row["ci_lower"] == pytest.approx(0.2)
    assert rank_row["ci_upper"] == pytest.approx(0.2)


def test_bootstrap_mean_ci_returns_nan_for_empty_values() -> None:
    mean, lower, upper = bootstrap_mean_ci([np.nan])

    assert np.isnan(mean)
    assert np.isnan(lower)
    assert np.isnan(upper)


def test_create_bootstrap_sampler_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown bootstrap method"):
        create_bootstrap_sampler({"method": "missing"})


def test_compute_bootstrap_ic_writes_summary(tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    pd.DataFrame({"factor_a__label": [0.1, 0.2, 0.3]}).to_parquet(
        processed_dir / "ic_panel.parquet"
    )
    pd.DataFrame({"factor_a__label": [0.2, 0.2, 0.2]}).to_parquet(
        processed_dir / "rank_ic_panel.parquet"
    )
    config_path = _write_config(tmp_path, processed_dir)

    paths = compute_bootstrap_ic(config_path=str(config_path))

    assert paths.keys() == {"bootstrap_ic_summary"}
    summary = pd.read_csv(paths["bootstrap_ic_summary"])
    assert len(summary) == 2
    assert set(summary["metric"]) == {"ic", "rank_ic"}


def _write_config(tmp_path: Path, processed_dir: Path) -> Path:
    config = {
        "data": {"processed_dir": str(processed_dir)},
        "bootstrap": {
            "n_samples": 100,
            "confidence_level": 0.90,
            "random_seed": 42,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


class ConstantBootstrapSampler(BootstrapSampler):
    method = "constant"

    def sample_means(self, values: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
        return np.full(self.n_samples, values.mean()), {"custom_parameter": 1}
