import numpy as np
import pandas as pd
import pytest

import feature_store.store as store
from data.features import build_feature_tensor

PORTFOLIO_NAME = "test_portfolio"
TICKERS = ["AAA.NS", "BBB.NS"]


def _make_raw_data(n_days: int = 60) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range(end=pd.Timestamp.utcnow().tz_localize(None), periods=n_days)
    data = {}
    rng = np.random.default_rng(0)
    for i, ticker in enumerate(TICKERS):
        prices = 100 + i * 10 + np.cumsum(rng.normal(scale=1.0, size=n_days))
        data[ticker] = pd.DataFrame(
            {
                "Open": prices,
                "High": prices * 1.01,
                "Low": prices * 0.99,
                "Close": prices,
                "Volume": rng.integers(1_000_000, 2_000_000, size=n_days),
            },
            index=dates,
        )
    return data


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORE_ROOT", tmp_path)
    return tmp_path


def test_compute_and_store_creates_first_version():
    raw_data = _make_raw_data()
    metadata = store.compute_and_store(PORTFOLIO_NAME, raw_data)

    assert metadata["version"] == "v1"
    assert metadata["tickers"] == TICKERS
    assert metadata["n_tickers"] == len(TICKERS)


def test_get_latest_for_training_reconstructs_same_tensor():
    raw_data = _make_raw_data()
    expected_tensor, expected_dates, expected_tickers = build_feature_tensor(PORTFOLIO_NAME, raw_data, save=False)

    store.compute_and_store(PORTFOLIO_NAME, raw_data)
    tensor, dates, tickers, metadata = store.get_latest_for_training(PORTFOLIO_NAME)

    assert tensor.shape == expected_tensor.shape
    assert tickers == expected_tickers
    np.testing.assert_allclose(tensor, expected_tensor, atol=1e-3)
    assert len(dates) == len(expected_dates)


def test_get_latest_observation_matches_last_row_of_tensor():
    raw_data = _make_raw_data()
    store.compute_and_store(PORTFOLIO_NAME, raw_data)

    tensor, _, _, _ = store.get_latest_for_training(PORTFOLIO_NAME)
    obs_row, tickers, metadata = store.get_latest_observation(PORTFOLIO_NAME)

    assert obs_row.shape == (tensor.shape[1] * tensor.shape[2],)
    np.testing.assert_allclose(obs_row, tensor[-1].flatten(), atol=1e-3)
    assert tickers == TICKERS
    assert "end_date" in metadata


def test_list_versions_grows_with_each_compute_and_store():
    raw_data = _make_raw_data()
    store.compute_and_store(PORTFOLIO_NAME, raw_data)
    store.compute_and_store(PORTFOLIO_NAME, raw_data)

    versions = store.list_versions(PORTFOLIO_NAME)
    assert list(versions["version"]) == ["v2", "v1"]


def test_get_latest_for_training_raises_when_no_versions_exist():
    with pytest.raises(LookupError):
        store.get_latest_for_training("never_computed_portfolio")


def test_log_and_read_predictions_round_trip():
    store.log_prediction(PORTFOLIO_NAME, {"AAA.NS": 0.6, "BBB.NS": 0.4}, "2026-06-19T00:00:00+00:00")
    store.log_prediction(PORTFOLIO_NAME, {"AAA.NS": 0.5, "BBB.NS": 0.5}, "2026-06-20T00:00:00+00:00")

    predictions = store.read_predictions(PORTFOLIO_NAME)

    assert len(predictions) == 2
    assert predictions[-1]["allocation"] == {"AAA.NS": 0.5, "BBB.NS": 0.5}
