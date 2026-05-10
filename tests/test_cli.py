from pathlib import Path

from quant import cli


def test_download_data_cli_calls_downloader(monkeypatch, capsys) -> None:
    calls = {}

    def fake_download_data(config_path: str, adjust: str) -> dict[str, Path]:
        calls["config_path"] = config_path
        calls["adjust"] = adjust
        return {"stock_panel": Path("data/processed/stock_panel.parquet")}

    monkeypatch.setattr(cli, "download_data", fake_download_data)

    cli.main(["download-data", "--config", "custom.yaml", "--adjust", ""])

    assert calls == {"config_path": "custom.yaml", "adjust": ""}
    output = capsys.readouterr().out
    assert "stock_panel:" in output
    assert "stock_panel.parquet" in output


def test_preprocess_data_cli_calls_preprocessor(monkeypatch, capsys) -> None:
    calls = {}

    def fake_preprocess_data(config_path: str) -> Path:
        calls["config_path"] = config_path
        return Path("data/processed/clean_panel.parquet")

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
        return Path("data/processed/factor_panel.parquet")

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
        return Path("data/processed/label_panel.parquet")

    monkeypatch.setattr(cli, "compute_labels", fake_compute_labels)

    cli.main(["compute-labels", "--config", "custom.yaml"])

    assert calls == {"config_path": "custom.yaml"}
    output = capsys.readouterr().out
    assert "label_panel:" in output
    assert "label_panel.parquet" in output
