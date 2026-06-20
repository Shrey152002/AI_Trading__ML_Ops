"""Evaluates trained agents on a held-out window and ranks them by risk-adjusted performance."""
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from envs.portfolio_env import PortfolioEnv

logger = logging.getLogger(__name__)

REPORTS_ROOT = Path(__file__).resolve().parents[1] / "reports"
TRADING_DAYS_PER_YEAR = 252


def run_episode(agent: BaseAgent, env: PortfolioEnv, deterministic: bool = True) -> np.ndarray:
    """Runs one full pass over an env's window and returns the array of daily portfolio returns."""
    obs, _ = env.reset()
    returns = []
    terminated = truncated = False
    while not (terminated or truncated):
        action = agent.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, info = env.step(action)
        if "portfolio_return" in info:
            returns.append(info["portfolio_return"])
    return np.array(returns, dtype=np.float64)


def compute_metrics(returns: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    """Computes annualized Sharpe, max drawdown, total return, Calmar ratio, and volatility."""
    if len(returns) == 0:
        return {"sharpe": 0.0, "max_drawdown": 0.0, "total_return": 0.0, "calmar": 0.0, "volatility": 0.0}

    mean_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe = float((mean_return / std_return) * np.sqrt(periods_per_year)) if std_return > 1e-12 else 0.0

    equity_curve = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = float(np.min(drawdown))

    total_return = float(equity_curve[-1] - 1.0)
    volatility = float(std_return * np.sqrt(periods_per_year))

    annualized_return = float((1.0 + total_return) ** (periods_per_year / len(returns)) - 1.0)
    calmar = float(annualized_return / abs(max_drawdown)) if max_drawdown < -1e-12 else 0.0

    return {
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_drawdown, 4),
        "total_return": round(total_return, 4),
        "calmar": round(calmar, 4),
        "volatility": round(volatility, 4),
    }


def benchmark_models(
    models: dict[str, BaseAgent],
    test_env: PortfolioEnv,
    portfolio_name: str,
    save: bool = True,
) -> pd.DataFrame:
    """Evaluates every trained model on the same held-out test_env and ranks by Sharpe."""
    rows = []
    for algo_name, agent in models.items():
        returns = run_episode(agent, test_env)
        metrics = compute_metrics(returns)
        rows.append({"algorithm": algo_name, **metrics})

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    if save:
        REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
        out_path = REPORTS_ROOT / "benchmark_results.json"
        payload = {
            "portfolio": portfolio_name,
            "results": df.to_dict(orient="records"),
            "winner": df.iloc[0]["algorithm"] if not df.empty else None,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Wrote benchmark results for %s -> %s", portfolio_name, out_path)

    return df
