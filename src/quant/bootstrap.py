from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar, Self

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class BootstrapSampler(ABC):
    method: ClassVar[str]

    n_bootstrap: int = 1000
    confidence_level: float = 0.95
    random_seed: int | None = None

    def validate(self) -> None:
        if self.n_bootstrap <= 0:
            raise ValueError("n_samples/n_bootstrap must be positive.")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1.")

    def with_seed_offset(self, offset: int) -> Self:
        if self.random_seed is None:
            return self
        return replace(self, random_seed=self.random_seed + offset)

    @abstractmethod
    def sample_means(self, values: np.ndarray, **_) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class Iid(BootstrapSampler):
    method: ClassVar[str] = "iid"

    def sample_means(self, values: np.ndarray, **_) -> np.ndarray:
        rng = np.random.default_rng(self.random_seed)
        samples = rng.choice(values, (self.n_bootstrap, len(values)), replace=True)
        return samples.mean(axis=1)


@dataclass(frozen=True)
class CircularBlock(BootstrapSampler):
    method: ClassVar[str] = "circular_block"

    block_length_multiplier: int = 2
    default_horizon: int = 20
    block_length: int | None = None

    def validate(self) -> None:
        super().validate()
        if self.block_length_multiplier <= 0:
            raise ValueError("block_length_multiplier must be positive.")
        if self.default_horizon <= 0:
            raise ValueError("default_horizon must be positive.")
        if self.block_length is not None and self.block_length <= 0:
            raise ValueError("block_length must be positive.")

    def sample_means(
        self, values: np.ndarray, factor_label: str | None = None, **_
    ) -> np.ndarray:
        block_length = self._block_length_for(factor_label)
        n_obs = len(values)
        if block_length > n_obs:
            raise ValueError("block_length must be less than or equal to n_obs.")

        rng = np.random.default_rng(self.random_seed)
        n_blocks = (n_obs + block_length - 1) // block_length
        starts = rng.integers(0, n_obs, size=(self.n_bootstrap, n_blocks))
        offsets = np.arange(block_length)
        indices = (starts[..., None] + offsets) % n_obs
        samples = values[indices.reshape(self.n_bootstrap, -1)[:, :n_obs]]
        return samples.mean(axis=1)

    def _block_length_for(self, factor_label: str | None) -> int:
        if self.block_length is not None:
            return self.block_length
        match = re.search(r"(?<!\d)(\d+)d\b", factor_label or "")
        horizon = int(match.group(1)) if match else self.default_horizon
        return self.block_length_multiplier * horizon


@dataclass(frozen=True)
class BootstrapMeanResult:
    mean: float
    bootstrap_mean: float
    bootstrap_std: float
    ci_lower: float
    ci_upper: float
    method: str
    confidence_level: float
    n_bootstrap: int
    n_obs: int
    block_length: int | None = None


DEFAULT_BOOTSTRAP_METHOD = Iid.method
BOOTSTRAP_SAMPLERS: dict[str, type[BootstrapSampler]] = {
    Iid.method: Iid,
    CircularBlock.method: CircularBlock,
}


def create_bootstrap_sampler(
    config: Mapping[str, object] | None = None,
) -> BootstrapSampler:
    config = dict(config or {})
    method = config.pop("method", DEFAULT_BOOTSTRAP_METHOD)
    sampler_cls = _resolve_bootstrap_sampler(method)
    return sampler_cls(**config)


def bootstrap_mean_ci(
    values: Sequence[float] | pd.Series,
    n_samples: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int | None = None,
    method: str = DEFAULT_BOOTSTRAP_METHOD,
) -> tuple[float, float, float]:
    result = bootstrap_mean_result(
        values,
        sampler=create_bootstrap_sampler(
            {
                "method": method,
                "n_bootstrap": n_samples,
                "confidence_level": confidence_level,
                "random_seed": random_seed,
            }
        ),
    )
    return result.mean, result.ci_lower, result.ci_upper


def bootstrap_mean_result(
    values: Sequence[float] | pd.Series,
    sampler: BootstrapSampler | None = None,
    factor_label: str | None = None,
) -> BootstrapMeanResult:
    sampler = sampler or Iid()
    clean = pd.Series(values, dtype="float64").dropna().to_numpy()
    if len(clean) == 0:
        raise ValueError("values must contain at least one non-NaN observation.")

    sampler.validate()
    bootstrap_means = sampler.sample_means(clean, factor_label=factor_label)
    if len(bootstrap_means) == 0:
        raise ValueError(f"{sampler.method} returned no bootstrap samples.")
    alpha = 1 - sampler.confidence_level
    lower = np.quantile(bootstrap_means, alpha / 2)
    upper = np.quantile(bootstrap_means, 1 - alpha / 2)
    ddof = 1 if len(bootstrap_means) > 1 else 0
    return BootstrapMeanResult(
        mean=float(clean.mean()),
        bootstrap_mean=float(bootstrap_means.mean()),
        bootstrap_std=float(bootstrap_means.std(ddof=ddof)),
        ci_lower=float(lower),
        ci_upper=float(upper),
        method=sampler.method,
        confidence_level=sampler.confidence_level,
        n_bootstrap=sampler.n_bootstrap,
        n_obs=len(clean),
        block_length=_block_length_for_result(sampler, factor_label),
    )


def _resolve_bootstrap_sampler(method: str) -> type[BootstrapSampler]:
    try:
        return BOOTSTRAP_SAMPLERS[method]
    except KeyError as error:
        available = ", ".join(sorted(BOOTSTRAP_SAMPLERS))
        raise ValueError(
            f"Unknown bootstrap method '{method}'. Available methods: {available}."
        ) from error


def bootstrap_ic_summary(
    ic_panel: pd.DataFrame,
    rank_ic_panel: pd.DataFrame | None = None,
    sampler: BootstrapSampler | None = None,
) -> pd.DataFrame:
    sampler = sampler or Iid()
    rows = _bootstrap_rows(ic_panel, metric="ic", sampler=sampler)
    if rank_ic_panel is not None:
        rows.extend(_bootstrap_rows(rank_ic_panel, metric="rank_ic", sampler=sampler))
    return (
        pd.DataFrame(rows)
        .sort_values(["factor_label", "metric"])
        .reset_index(drop=True)
    )


def compute_bootstrap_ic(config_path: str = "config.yaml") -> dict[str, Path]:
    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    work_dir = Path(config["data"]["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    ic_panel = pd.read_parquet(work_dir / "ic_panel.parquet")
    rank_ic_panel = pd.read_parquet(work_dir / "rank_ic_panel.parquet")
    bootstrap_config = config.get("bootstrap", {})
    sampler = create_bootstrap_sampler(bootstrap_config)
    summary = bootstrap_ic_summary(ic_panel, rank_ic_panel, sampler=sampler)

    summary_path = work_dir / "bootstrap_ic_summary.parquet"
    summary.to_parquet(summary_path)
    return {"bootstrap_ic_summary": summary_path}


def _bootstrap_rows(
    panel: pd.DataFrame, metric: str, sampler: BootstrapSampler
) -> list[dict[str, float | int | str | None]]:
    rows = []
    for offset, column in enumerate(panel.columns):
        factor, label = _split_factor_label(str(column))
        result = bootstrap_mean_result(
            panel[column],
            sampler=sampler.with_seed_offset(offset),
            factor_label=str(column),
        )
        row: dict[str, float | int | str | None] = {
            "factor_label": column,
            "factor": factor,
            "label": label,
            "metric": metric,
            "mean": result.mean,
            "bootstrap_mean": result.bootstrap_mean,
            "bootstrap_std": result.bootstrap_std,
            "ci_lower": result.ci_lower,
            "ci_upper": result.ci_upper,
            "confidence_level": result.confidence_level,
            "n_bootstrap": result.n_bootstrap,
            "n_obs": result.n_obs,
            "n_observations": result.n_obs,
            "method": result.method,
        }
        if result.block_length is not None:
            row["block_length"] = result.block_length
        rows.append(row)
    return rows


def _block_length_for_result(
    sampler: BootstrapSampler, factor_label: str | None
) -> int | None:
    if isinstance(sampler, CircularBlock):
        return sampler._block_length_for(factor_label)
    return None


def _split_factor_label(factor_label: str) -> tuple[str, str]:
    if "__" not in factor_label:
        return factor_label, ""
    factor, label = factor_label.split("__", maxsplit=1)
    return factor, label
