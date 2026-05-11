# CSI500 Factor Lab

A small, reproducible research project for CSI 500 cross-sectional factor
analysis. The current implementation can download the current CSI 500 universe,
stock OHLCV data, and benchmark OHLCV data to parquet files.

## Scope

This project will gradually build:

- CSI 500 universe and OHLCV data ingestion
- basic return and forward-return labels
- simple cross-sectional factors
- IC / Rank IC evaluation
- quantile portfolio backtests
- transaction-cost and robustness checks
- a final Markdown research report

## Project Layout

```text
config.yaml
pyproject.toml
src/quant/
tests/
```

## Data Download

Install the project in editable mode first:

```bash
python -m pip install -e ".[dev]"
```

Then run the installed CLI:

```bash
quant download-data
```

Outputs:

```text
data/raw/csi500_universe.parquet
data/processed/stock_panel.parquet
data/processed/benchmark_000905.parquet
```

## Preprocess

After downloading data, build the cleaned panel with basic returns:

```bash
quant preprocess-data
```

Output:

```text
data/processed/clean_panel.parquet
```

The cleaned panel removes duplicate `date` + `ticker` rows, rows with missing
or non-positive `close`, and rows with `volume <= 0`. It adds `ret_1d`,
`ret_5d`, and `ret_20d`.

## Factors

After preprocessing data, compute the standardized factor panel:

```bash
quant compute-factors
```

Output:

```text
data/processed/factor_panel.parquet
```

The factor panel contains `reversal_5`, `momentum_20`, `low_volatility_20`,
`turnover_change_20`, and `liquidity_20`. Each factor is winsorized and z-scored
cross-sectionally by date.

## Labels

After preprocessing data, compute forward excess-return labels:

```bash
quant compute-labels
```

Output:

```text
data/processed/label_panel.parquet
```

The label panel contains `fwd_excess_ret_5d` and `fwd_excess_ret_20d`, aligned to
the same `date` + `ticker` row as the factor observation. Tail rows without a
full forward horizon are kept as `NaN` so downstream evaluation can drop them
explicitly.

## IC Analysis

After computing factors and labels, evaluate factor predictive power with IC and
Rank IC:

```bash
quant compute-ic
```

Outputs:

```text
data/processed/ic_panel.parquet
data/processed/rank_ic_panel.parquet
data/processed/rolling_ic.parquet
data/processed/ic_summary.csv
reports/ic_summary.md
reports/figures/rolling_ic.png
```

IC is the daily cross-sectional Pearson correlation between a factor and a
forward-return label. Rank IC uses Spearman correlation on the same aligned
factor-label pairs.

## Quantile Backtest

After computing factors and labels, run the simple quantile backtest:

```bash
quant run-backtest
```

Outputs:

```text
data/processed/quantile_returns.parquet
data/processed/long_short_returns.parquet
data/processed/long_only_returns.parquet
data/processed/backtest_summary.csv
reports/figures/quantile_cumulative_returns.png
```

Each factor is split into Q1-Q5 by date. The backtest reports average forward
label returns for each quantile, Q5-Q1 long-short returns, and Q5 long-only
returns.

## Transaction Costs And Bootstrap

After running the quantile backtest, evaluate transaction-cost sensitivity:

```bash
quant analyze-costs
```

Outputs:

```text
data/processed/long_short_turnover.parquet
data/processed/long_only_turnover.parquet
data/processed/cost_adjusted_long_short_returns.parquet
data/processed/cost_adjusted_long_only_returns.parquet
data/processed/cost_sensitivity_summary.csv
reports/figures/cost_sensitivity.png
```

Bootstrap IC mean confidence intervals:

```bash
quant bootstrap-ic
```

Output:

```text
data/processed/bootstrap_ic_summary.csv
```

## Checks

```bash
ruff check .
pytest
```

## CI

GitHub Actions runs the same checks on pushes and pull requests to `main`:

```bash
ruff check .
pytest
```

The CI does not download market data, so external data-source outages do not
block code checks.

## Notes

This project is intended for research and educational purposes only. It does
not constitute investment advice.

The first-stage implementation uses current CSI 500 constituents instead of
historical index membership. Therefore, the backtest may suffer from
survivorship bias.

The transaction cost model is simplified and does not fully capture market
impact, limit-up/limit-down constraints, or intraday liquidity.

Long-short portfolio results are used to evaluate factor ranking power. They do
not directly represent an executable A-share trading strategy due to
short-selling constraints.
