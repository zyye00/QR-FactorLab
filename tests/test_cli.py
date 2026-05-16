from pathlib import Path

from quant import cli


def test_download_data_cli_calls_downloader(monkeypatch, capsys) -> None:
    calls = {}

    def fake_download_data(config_path: str) -> dict[str, Path]:
        calls["config_path"] = config_path
        return {"stock_ohlcv": Path("data/source/stock_ohlcv.parquet")}

    monkeypatch.setattr(cli, "download_data", fake_download_data)

    cli.main(["download-data", "--config", "custom.yaml"])

    assert calls == {"config_path": "custom.yaml"}
    output = capsys.readouterr().out
    assert "stock_ohlcv:" in output
    assert "stock_ohlcv.parquet" in output


def test_preprocess_data_cli_calls_preprocessor(monkeypatch, capsys) -> None:
    calls = {}

    def fake_preprocess_data(config_path: str) -> Path:
        calls["config_path"] = config_path
        return Path("data/work/clean_panel.parquet")

    monkeypatch.setattr(cli, "preprocess_data", fake_preprocess_data)

    cli.main(["preprocess-data", "--config", "custom.yaml"])

    assert calls == {"config_path": "custom.yaml"}
    output = capsys.readouterr().out
    assert "clean_panel:" in output
    assert "clean_panel.parquet" in output


def test_compute_factors_cli_calls_factor_builder(monkeypatch, capsys) -> None:
    calls = {}

    def fake_compute_factors(config_path: str) -> Path:
        calls["config_path"] = config_path
        return Path("data/work/factor_panel.parquet")

    monkeypatch.setattr(cli, "compute_factors", fake_compute_factors)

    cli.main(["compute-factors", "--config", "custom.yaml"])

    assert calls == {"config_path": "custom.yaml"}
    output = capsys.readouterr().out
    assert "factor_panel:" in output
    assert "factor_panel.parquet" in output


def test_compute_labels_cli_calls_label_builder(monkeypatch, capsys) -> None:
    calls = {}

    def fake_compute_labels(config_path: str) -> Path:
        calls["config_path"] = config_path
        return Path("data/work/label_panel.parquet")

    monkeypatch.setattr(cli, "compute_labels", fake_compute_labels)

    cli.main(["compute-labels", "--config", "custom.yaml"])

    assert calls == {"config_path": "custom.yaml"}
    output = capsys.readouterr().out
    assert "label_panel:" in output
    assert "label_panel.parquet" in output


def test_compute_ic_cli_calls_metric_builder(monkeypatch, capsys) -> None:
    calls = {}

    def fake_compute_ic_analysis(config_path: str) -> dict[str, Path]:
        calls["config_path"] = config_path
        return {"ic_panel": Path("data/work/ic_panel.parquet")}

    monkeypatch.setattr(cli, "compute_ic_analysis", fake_compute_ic_analysis)

    cli.main(["compute-ic", "--config", "custom.yaml"])

    assert calls == {"config_path": "custom.yaml"}
    output = capsys.readouterr().out
    assert "ic_panel:" in output
    assert "ic_panel.parquet" in output


def test_run_backtest_cli_calls_backtest_builder(monkeypatch, capsys) -> None:
    calls = {}

    def fake_compute_quantile_backtest(config_path: str) -> dict[str, Path]:
        calls["config_path"] = config_path
        return {"quantile_returns": Path("data/work/quantile_returns.parquet")}

    monkeypatch.setattr(
        cli,
        "compute_quantile_backtest",
        fake_compute_quantile_backtest,
    )

    cli.main(["run-backtest", "--config", "custom.yaml"])

    assert calls == {"config_path": "custom.yaml"}
    output = capsys.readouterr().out
    assert "quantile_returns:" in output
    assert "quantile_returns.parquet" in output


def test_analyze_costs_cli_calls_cost_builder(monkeypatch, capsys) -> None:
    calls = {}

    def fake_compute_cost_analysis(config_path: str) -> dict[str, Path]:
        calls["config_path"] = config_path
        return {"cost_sensitivity_summary": Path("data/work/costs.parquet")}

    monkeypatch.setattr(cli, "compute_cost_analysis", fake_compute_cost_analysis)

    cli.main(["analyze-costs", "--config", "custom.yaml"])

    assert calls == {"config_path": "custom.yaml"}
    output = capsys.readouterr().out
    assert "cost_sensitivity_summary:" in output
    assert "costs.parquet" in output


def test_bootstrap_ic_cli_calls_bootstrap_builder(monkeypatch, capsys) -> None:
    calls = {}

    def fake_compute_bootstrap_ic(config_path: str) -> dict[str, Path]:
        calls["config_path"] = config_path
        return {"bootstrap_ic_summary": Path("data/work/bootstrap_ic.parquet")}

    monkeypatch.setattr(cli, "compute_bootstrap_ic", fake_compute_bootstrap_ic)

    cli.main(["bootstrap-ic", "--config", "custom.yaml"])

    assert calls == {"config_path": "custom.yaml"}
    output = capsys.readouterr().out
    assert "bootstrap_ic_summary:" in output
    assert "bootstrap_ic.parquet" in output


def test_generate_report_cli_calls_report_builder(monkeypatch, capsys) -> None:
    calls = {}

    def fake_generate_report(config_path: str) -> dict[str, Path]:
        calls["config_path"] = config_path
        return {"final_report_template": Path("reports/final_report.md")}

    monkeypatch.setattr(cli, "generate_report", fake_generate_report)

    cli.main(["generate-report", "--config", "custom.yaml"])

    assert calls == {"config_path": "custom.yaml"}
    output = capsys.readouterr().out
    assert "final_report_template:" in output
    assert "final_report.md" in output
