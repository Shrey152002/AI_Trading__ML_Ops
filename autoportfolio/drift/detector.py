"""Tracks live model performance against its registered benchmark and triggers retraining.

Reads features through `feature_store` rather than `training.env_utils` since drift
checking (and the inference it backs) is a serving-time concern, not a training-time one.
"""
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import get_portfolio
from data.ingestion import load_cached_portfolio_data
from envs.portfolio_env import softmax
from feature_store.store import get_latest_for_training
from monitoring.metrics import model_drift_score, portfolio_daily_return, portfolio_sharpe_ratio
from registry.model_registry import get_production_model
from training.benchmark import compute_metrics

logger = logging.getLogger(__name__)

DRIFT_LOG_PATH = Path(__file__).resolve().parent / "drift_log.csv"
DRIFT_THRESHOLD_RATIO = 0.7
ROLLING_WINDOW_DAYS = 10

CSV_FIELDS = ["timestamp", "portfolio", "metric", "live_value", "benchmark_value", "threshold", "action"]


def _append_drift_log(row: dict) -> None:
    DRIFT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = DRIFT_LOG_PATH.exists()
    with open(DRIFT_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _load_close_prices(portfolio_name: str, tickers: list[str], dates: np.ndarray) -> np.ndarray:
    """Close prices are raw market data, not engineered features, so they come straight
    from the ingestion cache rather than the feature store."""
    get_portfolio(portfolio_name)  # validates the portfolio exists
    cached = load_cached_portfolio_data(portfolio_name, tickers)
    close_df = pd.DataFrame({t: cached[t]["Close"] for t in tickers})
    close_df = close_df.reindex(pd.to_datetime(dates))
    return np.nan_to_num(close_df.to_numpy(dtype=np.float32), nan=0.0)


def compute_live_returns(portfolio_name: str, agent, window: int = ROLLING_WINDOW_DAYS) -> np.ndarray:
    """Replays the production model's predictions over the most recent `window` trading days."""
    tensor, dates, tickers, _ = get_latest_for_training(portfolio_name)
    close_prices = _load_close_prices(portfolio_name, tickers, dates)
    n_days = tensor.shape[0]
    if n_days < window + 2:
        raise ValueError(f"Not enough recent data ({n_days} days) to compute a {window}-day live Sharpe")

    start_idx = n_days - window - 1
    returns = []
    for idx in range(start_idx, n_days - 1):
        obs = tensor[idx].flatten().astype(np.float32)
        action = agent.predict(obs, deterministic=True)
        weights = softmax(np.asarray(action, dtype=np.float32).flatten())

        current_prices = close_prices[idx]
        next_prices = close_prices[idx + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            forward_returns = np.nan_to_num((next_prices / current_prices) - 1.0, nan=0.0, posinf=0.0, neginf=0.0)

        returns.append(float(np.dot(weights, forward_returns)))

    return np.array(returns, dtype=np.float64)


def check_drift(
    portfolio_name: str,
    threshold_ratio: float = DRIFT_THRESHOLD_RATIO,
    window: int = ROLLING_WINDOW_DAYS,
) -> bool:
    """Returns True if the production model's rolling live Sharpe has degraded below
    `threshold_ratio` of its registered benchmark Sharpe, meaning retraining is warranted.
    """
    agent, metadata = get_production_model(portfolio_name)
    benchmark_sharpe = float(metadata["tags"].get("sharpe", 0.0) or 0.0)

    live_returns = compute_live_returns(portfolio_name, agent, window=window)
    live_sharpe = compute_metrics(live_returns)["sharpe"]

    drift_threshold = benchmark_sharpe * threshold_ratio
    drift_triggered = live_sharpe < drift_threshold
    action = "retrain_triggered" if drift_triggered else "healthy"

    drift_ratio = (live_sharpe / benchmark_sharpe) if benchmark_sharpe > 0 else 0.0
    portfolio_sharpe_ratio.labels(portfolio=portfolio_name).set(benchmark_sharpe)
    model_drift_score.labels(portfolio=portfolio_name).set(drift_ratio)
    if len(live_returns) > 0:
        portfolio_daily_return.labels(portfolio=portfolio_name).set(float(live_returns[-1]))

    _append_drift_log(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "portfolio": portfolio_name,
            "metric": "rolling_sharpe",
            "live_value": round(live_sharpe, 4),
            "benchmark_value": round(benchmark_sharpe, 4),
            "threshold": round(drift_threshold, 4),
            "action": action,
        }
    )

    logger.info(
        "Drift check for %s: live_sharpe=%.4f benchmark_sharpe=%.4f threshold=%.4f -> %s",
        portfolio_name,
        live_sharpe,
        benchmark_sharpe,
        drift_threshold,
        action,
    )
    return drift_triggered
