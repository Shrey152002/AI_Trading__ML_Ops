"""Online inference: recommendation, status, and history. Depends only on the model
registry and the feature store — never on training, scheduler, or raw ingestion/validation
modules — so the serving path can't accidentally couple to training-time concerns.
"""
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import get_portfolio
from data.ingestion import load_cached_portfolio_data
from drift.detector import compute_live_returns
from envs.portfolio_env import softmax
from feature_store.store import get_latest_observation, log_prediction, read_predictions
from monitoring.metrics import (
    data_freshness_hours as data_freshness_hours_metric,
    model_drift_score,
    portfolio_daily_return,
    portfolio_sharpe_ratio,
)
from registry.model_registry import get_production_model
from training.benchmark import compute_metrics

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
RETURN_LOOKBACK_DAYS = 60
DRIFT_WINDOW_DAYS = 10


def _load_recent_close_prices(portfolio_name: str, tickers: list[str], lookback: int) -> np.ndarray:
    """Close prices are raw market data, read straight from the ingestion cache."""
    cached = load_cached_portfolio_data(portfolio_name, tickers)
    close_df = pd.DataFrame({t: cached[t]["Close"] for t in tickers}).sort_index()
    close_df = close_df.tail(lookback + 1)
    return np.nan_to_num(close_df.to_numpy(dtype=np.float32), nan=0.0)


def _confidence_and_drift_ratio(portfolio_id: str, agent, benchmark_sharpe: float) -> float:
    """Computes the live-Sharpe/benchmark-Sharpe ratio and, as a side effect, refreshes the
    Prometheus gauges for this portfolio — every recommendation/status call keeps them current,
    not just the nightly drift check.
    """
    try:
        live_returns = compute_live_returns(portfolio_id, agent, window=DRIFT_WINDOW_DAYS)
        live_sharpe = compute_metrics(live_returns)["sharpe"]
    except ValueError:
        live_returns = np.array([])
        live_sharpe = benchmark_sharpe

    drift_ratio = float(live_sharpe / benchmark_sharpe) if benchmark_sharpe > 0 else 0.0

    portfolio_sharpe_ratio.labels(portfolio=portfolio_id).set(benchmark_sharpe)
    model_drift_score.labels(portfolio=portfolio_id).set(drift_ratio)
    if len(live_returns) > 0:
        portfolio_daily_return.labels(portfolio=portfolio_id).set(float(live_returns[-1]))

    return drift_ratio


def _build_explanation(recommended: dict, current: dict) -> str:
    deltas = {t: recommended.get(t, 0.0) - current.get(t, 0.0) for t in set(recommended) | set(current)}
    overweight = sorted(deltas.items(), key=lambda kv: kv[1], reverse=True)[:3]
    underweight = sorted(deltas.items(), key=lambda kv: kv[1])[:3]

    over_text = ", ".join(f"{t} (+{d:.1%})" for t, d in overweight if d > 0)
    under_text = ", ".join(f"{t} ({d:.1%})" for t, d in underweight if d < 0)

    parts = []
    if over_text:
        parts.append(f"Increasing exposure to {over_text} based on favorable momentum/risk-adjusted signals.")
    if under_text:
        parts.append(f"Reducing exposure to {under_text} due to weaker risk-adjusted outlook.")
    if not parts:
        parts.append("Recommended allocation closely matches current holdings; no major rebalancing needed.")
    return " ".join(parts)


def get_recommendation(portfolio_id: str, current_holdings: dict) -> dict:
    get_portfolio(portfolio_id)
    agent, metadata = get_production_model(portfolio_id)

    obs_row, tickers, _ = get_latest_observation(portfolio_id)
    action = agent.predict(obs_row, deterministic=True)
    weights = softmax(np.asarray(action, dtype=np.float32).flatten())
    recommended_allocation = {t: float(w) for t, w in zip(tickers, weights)}

    close_prices = _load_recent_close_prices(portfolio_id, tickers, lookback=RETURN_LOOKBACK_DAYS)
    with np.errstate(divide="ignore", invalid="ignore"):
        recent_returns = np.nan_to_num((close_prices[1:] / close_prices[:-1]) - 1.0, nan=0.0, posinf=0.0, neginf=0.0)
    portfolio_returns = recent_returns @ weights
    expected_return = float(np.mean(portfolio_returns) * TRADING_DAYS_PER_YEAR)
    expected_volatility = float(np.std(portfolio_returns) * np.sqrt(TRADING_DAYS_PER_YEAR))

    benchmark_sharpe = float(metadata["tags"].get("sharpe", 0.0) or 0.0)
    confidence = float(np.clip(_confidence_and_drift_ratio(portfolio_id, agent, benchmark_sharpe), 0.0, 1.0))

    explanation = _build_explanation(recommended_allocation, current_holdings)
    timestamp = datetime.now(timezone.utc).isoformat()

    log_prediction(portfolio_id, recommended_allocation, timestamp)

    return {
        "recommended_allocation": recommended_allocation,
        "expected_return": expected_return,
        "expected_volatility": expected_volatility,
        "confidence": confidence,
        "explanation": explanation,
        "model_version": f"{metadata['model_name']}:v{metadata['version']}",
        "timestamp": timestamp,
    }


def get_status(portfolio_id: str) -> dict:
    get_portfolio(portfolio_id)
    agent, metadata = get_production_model(portfolio_id)

    benchmark_sharpe = float(metadata["tags"].get("sharpe", 0.0) or 0.0)
    drift_score = _confidence_and_drift_ratio(portfolio_id, agent, benchmark_sharpe)

    _, _, feature_metadata = get_latest_observation(portfolio_id)
    last_date = datetime.fromisoformat(feature_metadata["end_date"])
    if last_date.tzinfo is None:
        last_date = last_date.replace(tzinfo=timezone.utc)
    freshness_hours = (datetime.now(timezone.utc) - last_date).total_seconds() / 3600
    data_freshness_hours_metric.labels(portfolio=portfolio_id).set(freshness_hours)

    return {
        "portfolio_id": portfolio_id,
        "model_version": f"{metadata['model_name']}:v{metadata['version']}",
        "algorithm": metadata["algorithm"],
        "last_training_date": metadata["tags"].get("registered_at"),
        "sharpe": benchmark_sharpe,
        "drift_score": drift_score,
        "data_freshness_hours": freshness_hours,
    }


def get_history(portfolio_id: str, limit: int = 30) -> list[dict]:
    get_portfolio(portfolio_id)
    return read_predictions(portfolio_id, limit=limit)
