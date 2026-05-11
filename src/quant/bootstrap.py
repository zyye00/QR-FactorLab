from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def bootstrap_mean_ci(
    values: Sequence[float] | pd.Series,
    n_samples: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int | None = None,
) -> tuple[float, float, float]:
    clean = pd.Series(values, dtype="float64").dropna().to_numpy()
    if len(clean) == 0:
        return np.nan, np.nan, np.nan
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    rng = np.random.default_rng(random_seed)
    samples = rng.choice(clean, size=(n_samples, len(clean)), replace=True)
    means = samples.mean(axis=1)
    alpha = 1 - confidence_level
    lower = np.quantile(means, alpha / 2)
    upper = np.quantile(means, 1 - alpha / 2)
    return clean.mean(), lower, upper


def bootstrap_ic_summary(
    ic_panel: pd.DataFrame,
    rank_ic_panel: pd.DataFrame | None = None,
    n_samples: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int | None = None,
) -> pd.DataFrame:
    rows = _bootstrap_rows(
        ic_panel,
        metric="ic",
        n_samples=n_samples,
        confidence_level=confidence_level,
        random_seed=random_seed,
    )
    if rank_ic_panel is not None:
        rows.extend(
            _bootstrap_rows(
                rank_ic_panel,
                metric="rank_ic",
                n_samples=n_samples,
                confidence_level=confidence_level,
                random_seed=random_seed,
            )
        )
    return pd.DataFrame(rows).sort_values(["factor_label", "metric"]).reset_index(
        drop=True
    )


def compute_bootstrap_ic(config_path: str = "config.yaml") -> dict[str, Path]:
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    processed_dir = Path(config["data"]["processed_dir"])
    ic_panel = pd.read_parquet(processed_dir / "ic_panel.parquet")
    rank_ic_panel = pd.read_parquet(processed_dir / "rank_ic_panel.parquet")
    bootstrap_config = config.get("bootstrap", {})
    summary = bootstrap_ic_summary(
        ic_panel,
        rank_ic_panel,
        n_samples=int(bootstrap_config.get("n_samples", 1000)),
        confidence_level=float(bootstrap_config.get("confidence_level", 0.95)),
        random_seed=bootstrap_config.get("random_seed"),
    )

    summary_path = processed_dir / "bootstrap_ic_summary.csv"
    summary.to_csv(summary_path, index=False)
    return {"bootstrap_ic_summary": summary_path}


def _bootstrap_rows(
    panel: pd.DataFrame,
    metric: str,
    n_samples: int,
    confidence_level: float,
    random_seed: int | None,
) -> list[dict[str, float | int | str]]:
    rows = []
    for offset, column in enumerate(panel.columns):
        mean, lower, upper = bootstrap_mean_ci(
            panel[column],
            n_samples=n_samples,
            confidence_level=confidence_level,
            random_seed=None if random_seed is None else random_seed + offset,
        )
        rows.append(
            {
                "factor_label": column,
                "metric": metric,
                "mean": mean,
                "ci_lower": lower,
                "ci_upper": upper,
                "confidence_level": confidence_level,
                "n_samples": n_samples,
                "n_observations": panel[column].count(),
            }
        )
    return rows
