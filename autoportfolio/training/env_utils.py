"""Shared helpers for building train/val/test PortfolioEnv instances from feature_store data."""
import numpy as np
import pandas as pd

from config import get_portfolio
from data.ingestion import load_cached_portfolio_data
from envs.portfolio_env import PortfolioEnv, compute_split_indices
from feature_store.store import compute_and_store, get_latest_for_training


def load_aligned_features_and_prices(portfolio_name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Loads the feature_store tensor and the aligned Close-price matrix for a portfolio.

    Close prices are raw market data, not engineered features, so they're read directly
    from the ingestion cache rather than through the feature store.
    """
    portfolio = get_portfolio(portfolio_name)
    cached = load_cached_portfolio_data(portfolio_name, portfolio["tickers"])
    if not cached:
        raise FileNotFoundError(
            f"No cached raw data found for portfolio '{portfolio_name}'. Run ingestion first."
        )

    try:
        tensor, dates, tickers, _ = get_latest_for_training(portfolio_name)
    except LookupError:
        compute_and_store(portfolio_name, cached)
        tensor, dates, tickers, _ = get_latest_for_training(portfolio_name)

    close_df = pd.DataFrame({t: cached[t]["Close"] for t in tickers})
    close_df = close_df.reindex(pd.to_datetime(dates))
    close_prices = close_df.to_numpy(dtype=np.float32)
    close_prices = np.nan_to_num(close_prices, nan=0.0)

    return tensor, close_prices, tickers


def build_envs(
    portfolio_name: str,
    episode_length: int = 252,
    val_days: int = 120,
    test_days: int = 60,
    seed: int = 42,
) -> dict[str, PortfolioEnv]:
    tensor, close_prices, tickers = load_aligned_features_and_prices(portfolio_name)
    splits = compute_split_indices(tensor.shape[0], val_days=val_days, test_days=test_days)

    envs = {}
    for split_name, window in splits.items():
        envs[split_name] = PortfolioEnv(
            features=tensor,
            close_prices=close_prices,
            tickers=tickers,
            window=window,
            episode_length=episode_length if split_name == "train" else (window[1] - window[0] - 1),
            seed=seed,
        )
    return envs
