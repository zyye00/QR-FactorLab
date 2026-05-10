from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FACTOR_PATH = Path("data/processed/factor_panel.parquet")
FIGURE_PATH = Path("reports/figures/factor_overview.png")


def main() -> None:
    factors = pd.read_parquet(FACTOR_PATH)

    print(f"file: {FACTOR_PATH}")
    print(f"shape: {factors.shape}")
    print(f"index: {factors.index.names}")
    print(f"columns: {list(factors.columns)}")
    print(f"tickers: {factors.index.get_level_values('ticker').nunique()}")
    print(
        "date range: "
        f"{factors.index.get_level_values('date').min()} -> "
        f"{factors.index.get_level_values('date').max()}"
    )
    print(f"duplicate index rows: {factors.index.duplicated().sum()}")

    print("\nmissing values by column:")
    print(factors.isna().sum().to_string())

    print("\nhead:")
    print(factors.head(10).to_string())

    plot_factor_overview(factors)
    print(f"\nfigure saved: {FIGURE_PATH}")


def plot_factor_overview(factors: pd.DataFrame) -> None:
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    coverage = factors.notna().groupby(level="date").sum()
    coverage = coverage[(coverage > 0).all(axis=1)]
    latest_date = factors.dropna(how="all").index.get_level_values("date").max()
    latest_slice = factors.xs(latest_date, level="date")
    correlations = factors.corr()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    coverage.plot(ax=axes[0, 0], linewidth=1)
    axes[0, 0].set_title("Factor coverage over time")
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel("Non-null ticker count")
    axes[0, 0].legend(fontsize=8)

    latest_slice.plot.hist(ax=axes[0, 1], bins=40, alpha=0.45)
    axes[0, 1].set_title(f"Latest cross-section distributions: {latest_date.date()}")
    axes[0, 1].set_xlabel("Z-score")
    axes[0, 1].set_ylabel("Count")

    axes[1, 0].axis("off")
    summary = factors.describe().round(3).T
    table = axes[1, 0].table(
        cellText=summary[["mean", "std", "min", "max"]].values,
        colLabels=["mean", "std", "min", "max"],
        rowLabels=summary.index,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    axes[1, 0].set_title("Overall summary")

    image = axes[1, 1].imshow(correlations, vmin=-1, vmax=1, cmap="coolwarm")
    axes[1, 1].set_title("Factor correlation")
    axes[1, 1].set_xticks(range(len(correlations.columns)), correlations.columns)
    axes[1, 1].set_yticks(range(len(correlations.index)), correlations.index)
    axes[1, 1].tick_params(axis="x", rotation=45)
    fig.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
