import pandas as pd
import pytest

from data.validation import DataValidationError, validate_portfolio_data

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _make_clean_df(n_days: int = 100, start_price: float = 100.0) -> pd.DataFrame:
    dates = pd.bdate_range(end=pd.Timestamp.utcnow().tz_localize(None), periods=n_days)
    prices = start_price + pd.Series(range(n_days)).to_numpy() * 0.1
    return pd.DataFrame(
        {
            "Open": prices,
            "High": prices * 1.01,
            "Low": prices * 0.99,
            "Close": prices,
            "Volume": [1_000_000] * n_days,
        },
        index=dates,
    )


def test_missing_data_raises_validation_error():
    df = _make_clean_df(n_days=100)
    sparse = df.iloc[::3]  # keep ~33% of rows -> missing ratio far above 5%

    with pytest.raises(DataValidationError) as exc_info:
        validate_portfolio_data("test_portfolio", {"FAKE.NS": sparse})

    assert "FAKE.NS" in str(exc_info.value)
    failed = [r for r in exc_info.value.report["results"] if not r["passed"]]
    assert failed and failed[0]["missing_ratio"] > 0.05


def test_stale_data_raises_validation_error():
    df = _make_clean_df(n_days=50)
    stale_index = df.index - pd.Timedelta(days=10)
    df.index = stale_index

    with pytest.raises(DataValidationError) as exc_info:
        validate_portfolio_data("test_portfolio", {"FAKE.NS": df}, check_freshness=True)

    failed = [r for r in exc_info.value.report["results"] if not r["passed"]]
    assert failed and failed[0]["staleness_days"] > 2


def test_negative_prices_raises_validation_error():
    df = _make_clean_df(n_days=50)
    df.loc[df.index[5], "Close"] = -1.0

    with pytest.raises(DataValidationError) as exc_info:
        validate_portfolio_data("test_portfolio", {"FAKE.NS": df}, check_freshness=False)

    failed = [r for r in exc_info.value.report["results"] if not r["passed"]]
    assert failed and failed[0]["has_non_positive_price"] is True


def test_clean_data_passes_validation(tmp_path, monkeypatch):
    import data.validation as validation_module

    monkeypatch.setattr(validation_module, "VALIDATED_ROOT", tmp_path)
    df = _make_clean_df(n_days=50)

    report = validate_portfolio_data("test_portfolio", {"CLEAN.NS": df}, check_freshness=False)

    assert report["tickers_failed"] == 0
    assert (tmp_path / "test_portfolio" / "validation_report.json").exists()
