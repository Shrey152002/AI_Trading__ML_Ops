import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import inference.service as inference_service

PORTFOLIO_ID = "banking"
TICKERS = ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS"]
N_FEATURES = 8
LOOKBACK = 60


class MockAgent:
    def predict(self, observation, deterministic: bool = True):
        return np.array([1.0, 0.5, 0.2, 0.1, 0.0], dtype=np.float32)


@pytest.fixture
def client(monkeypatch):
    metadata = {
        "model_name": f"autoportfolio-{PORTFOLIO_ID}",
        "version": 3,
        "algorithm": "PPO",
        "stage": "Production",
        "run_id": "fake-run-id",
        "tags": {"sharpe": "1.8", "registered_at": "2026-06-01T00:00:00+00:00"},
    }
    feature_metadata = {
        "version": "v1",
        "tickers": TICKERS,
        "feature_names": [f"f{i}" for i in range(N_FEATURES)],
        "end_date": "2026-06-19T00:00:00+00:00",
    }

    rng = np.random.default_rng(0)
    obs_row = rng.normal(size=len(TICKERS) * N_FEATURES).astype(np.float32)
    close_prices = 100 + np.cumsum(
        rng.normal(scale=0.5, size=(LOOKBACK + 1, len(TICKERS))), axis=0
    ).astype(np.float32)

    monkeypatch.setattr(inference_service, "get_production_model", lambda portfolio_id: (MockAgent(), metadata))
    monkeypatch.setattr(
        inference_service, "get_latest_observation", lambda portfolio_id: (obs_row, TICKERS, feature_metadata)
    )
    monkeypatch.setattr(
        inference_service, "_load_recent_close_prices", lambda portfolio_id, tickers, lookback: close_prices
    )
    monkeypatch.setattr(
        inference_service, "compute_live_returns", lambda portfolio_id, agent, window=10: np.full(window, 0.001)
    )
    monkeypatch.setattr(inference_service, "log_prediction", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        inference_service,
        "read_predictions",
        lambda portfolio_id, limit=30: [{"date": "2026-06-19T00:00:00+00:00", "allocation": {t: 0.2 for t in TICKERS}, "realized_return": None}],
    )

    return TestClient(api_main.app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert PORTFOLIO_ID in body["portfolios"]


def test_recommendation_endpoint_returns_valid_allocation(client):
    payload = {
        "portfolio_id": PORTFOLIO_ID,
        "current_holdings": {t: 0.2 for t in TICKERS},
        "risk_appetite": "moderate",
        "capital": 500000,
        "sector_constraints": [],
    }
    response = client.post("/portfolio/recommendation", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert set(body["recommended_allocation"].keys()) == set(TICKERS)
    assert abs(sum(body["recommended_allocation"].values()) - 1.0) < 1e-4
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["model_version"] == "autoportfolio-banking:v3"


def test_recommendation_endpoint_unknown_portfolio_returns_404(client):
    payload = {
        "portfolio_id": "does_not_exist",
        "current_holdings": {},
        "risk_appetite": "moderate",
        "capital": 100000,
        "sector_constraints": [],
    }
    response = client.post("/portfolio/recommendation", json=payload)
    assert response.status_code == 404


def test_status_endpoint_returns_sharpe_and_drift(client):
    response = client.get(f"/portfolio/{PORTFOLIO_ID}/status")
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "autoportfolio-banking:v3"
    assert body["algorithm"] == "PPO"
    assert body["sharpe"] == 1.8


def test_history_endpoint_returns_logged_predictions(client):
    response = client.get(f"/portfolio/{PORTFOLIO_ID}/history")
    assert response.status_code == 200
    body = response.json()
    assert len(body["history"]) == 1
    assert set(body["history"][0]["allocation"].keys()) == set(TICKERS)
