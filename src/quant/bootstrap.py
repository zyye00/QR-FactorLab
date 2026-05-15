from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import ClassVar, Self

import numpy as np
import pandas as pd
import yaml

BootstrapMetadata = dict[str, float | int | str | None]


@dataclass(frozen=True)
class BootstrapSampler(ABC):
    method: ClassVar[str]

    n_samples: int = 1000
    confidence_level: float = 0.95
    random_seed: int | None = None

    def validate(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1.")

    def with_seed_offset(self, offset: int) -> Self:
        if self.random_seed is None:
            return self
        return replace(self, random_seed=self.random_seed + offset)

    @abstractmethod
    def sample_means(self, values: np.ndarray) -> tuple[np.ndarray, BootstrapMetadata]:
        raise NotImplementedError


@dataclass(frozen=True)
class IidBootstrapSampler(BootstrapSampler):
    method: ClassVar[str] = "iid"

    def sample_means(self, values: np.ndarray) -> tuple[np.ndarray, BootstrapMetadata]:
        rng = np.random.default_rng(self.random_seed)
        samples = rng.choice(
            values,
            size=(self.n_samples, len(values)),
            replace=True,
        )
        return samples.mean(axis=1), {}


@dataclass(frozen=True)
class BootstrapMeanResult:
    mean: float
    ci_lower: float
    ci_upper: float
    method: str
    confidence_level: float
    n_samples: int
    n_observations: int
    metadata: BootstrapMetadata = field(default_factory=dict)


DEFAULT_BOOTSTRAP_METHOD = IidBootstrapSampler.method
BOOTSTRAP_SAMPLERS: dict[str, type[BootstrapSampler]] = {
    IidBootstrapSampler.method: IidBootstrapSampler,
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
                "n_samples": n_samples,
                "confidence_level": confidence_level,
                "random_seed": random_seed,
            }
        ),
    )
    return result.mean, result.ci_lower, result.ci_upper


def bootstrap_mean_result(
    values: Sequence[float] | pd.Series,
    sampler: BootstrapSampler | None = None,
) -> BootstrapMeanResult:
    sampler = sampler or IidBootstrapSampler()
    clean = pd.Series(values, dtype="float64").dropna().to_numpy()
    if len(clean) == 0:
        return BootstrapMeanResult(
            mean=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            method=sampler.method,
            confidence_level=sampler.confidence_level,
            n_samples=sampler.n_samples,
            n_observations=0,
        )

    sampler.validate()
    bootstrap_means, metadata = sampler.sample_means(clean)
    if len(bootstrap_means) == 0:
        raise ValueError(f"{sampler.method} returned no bootstrap samples.")
    alpha = 1 - sampler.confidence_level
    lower = np.quantile(bootstrap_means, alpha / 2)
    upper = np.quantile(bootstrap_means, 1 - alpha / 2)
    return BootstrapMeanResult(
        mean=float(clean.mean()),
        ci_lower=float(lower),
        ci_upper=float(upper),
        method=sampler.method,
        confidence_level=sampler.confidence_level,
        n_samples=sampler.n_samples,
        n_observations=len(clean),
        metadata=dict(metadata),
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
    sampler = sampler or IidBootstrapSampler()
    rows = _bootstrap_rows(
        ic_panel,
        metric="ic",
        sampler=sampler,
    )
    if rank_ic_panel is not None:
        rows.extend(
            _bootstrap_rows(
                rank_ic_panel,
                metric="rank_ic",
                sampler=sampler,
            )
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["factor_label", "metric"])
        .reset_index(drop=True)
    )


def compute_bootstrap_ic(config_path: str = "config.yaml") -> dict[str, Path]:
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    processed_dir = Path(config["data"]["processed_dir"])
    ic_panel = pd.read_parquet(processed_dir / "ic_panel.parquet")
    rank_ic_panel = pd.read_parquet(processed_dir / "rank_ic_panel.parquet")
    bootstrap_config = config.get("bootstrap", {})
    sampler = create_bootstrap_sampler(bootstrap_config)
    summary = bootstrap_ic_summary(
        ic_panel,
        rank_ic_panel,
        sampler=sampler,
    )

    summary_path = processed_dir / "bootstrap_ic_summary.csv"
    summary.to_csv(summary_path, index=False)
    return {"bootstrap_ic_summary": summary_path}


def _bootstrap_rows(
    panel: pd.DataFrame,
    metric: str,
    sampler: BootstrapSampler,
) -> list[dict[str, float | int | str | None]]:
    rows = []
    for offset, column in enumerate(panel.columns):
        result = bootstrap_mean_result(
            panel[column],
            sampler=sampler.with_seed_offset(offset),
        )
        row: dict[str, float | int | str | None] = {
            "factor_label": column,
            "metric": metric,
            "mean": result.mean,
            "ci_lower": result.ci_lower,
            "ci_upper": result.ci_upper,
            "confidence_level": result.confidence_level,
            "n_samples": result.n_samples,
            "n_observations": result.n_observations,
            "method": result.method,
        }
        row.update(result.metadata)
        rows.append(row)
    return rows
