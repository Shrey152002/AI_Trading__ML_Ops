"""Custom Gymnasium environment for continuous portfolio weight allocation."""
import logging
from collections import deque
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)

EPS = 1e-8


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / (np.sum(e) + EPS)


def compute_split_indices(n_days: int, val_days: int = 120, test_days: int = 60) -> dict:
    """Splits a time series of length n_days into chronological train/val/test index ranges."""
    if n_days <= val_days + test_days + 10:
        raise ValueError(
            f"Not enough days ({n_days}) to carve out val={val_days} + test={test_days} windows"
        )
    test_start = n_days - test_days
    val_start = test_start - val_days
    return {
        "train": (0, val_start),
        "val": (val_start, test_start),
        "test": (test_start, n_days),
    }


class PortfolioEnv(gym.Env):
    """Allocates capital across tickers each trading day to maximize risk-adjusted return.

    Observation: flattened (n_tickers * n_features) feature vector for the current day.
    Action: raw logits in [0, 1]^n_tickers, normalized internally via softmax into weights
            that sum to 1.
    Reward: daily Sharpe contribution (portfolio_return / rolling_std) minus a turnover
            penalty of 0.001 * sum(|weight_change|).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        close_prices: np.ndarray,
        tickers: list[str],
        window: tuple[int, int],
        episode_length: int = 252,
        turnover_penalty: float = 0.001,
        rolling_std_window: int = 20,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if features.shape[0] != close_prices.shape[0]:
            raise ValueError("features and close_prices must have the same number of days")

        self.features = features
        self.close_prices = close_prices
        self.tickers = tickers
        self.n_days, self.n_tickers, self.n_features = features.shape
        self.window = window
        self.episode_length = episode_length
        self.turnover_penalty = turnover_penalty
        self.rolling_std_window = rolling_std_window

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_tickers * self.n_features,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_tickers,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self.current_idx = window[0]
        self.steps_taken = 0
        self.weights = np.ones(self.n_tickers, dtype=np.float32) / self.n_tickers
        self.return_history: deque = deque(maxlen=rolling_std_window)
        self.portfolio_value_history: list[float] = []

    def _get_obs(self) -> np.ndarray:
        return self.features[self.current_idx].flatten().astype(np.float32)

    def _get_info(self) -> dict:
        return {
            "date_idx": self.current_idx,
            "weights": self.weights.copy(),
            "steps_taken": self.steps_taken,
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        start, end = self.window
        latest_valid_start = max(start, end - self.episode_length - 1)
        if latest_valid_start <= start:
            self.current_idx = start
        else:
            self.current_idx = int(self._rng.integers(start, latest_valid_start))

        self.steps_taken = 0
        self.weights = np.ones(self.n_tickers, dtype=np.float32) / self.n_tickers
        self.return_history.clear()
        self.portfolio_value_history = [1.0]

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).flatten()
        new_weights = softmax(action)

        next_idx = self.current_idx + 1
        window_end = self.window[1]
        terminated = False
        truncated = False

        if next_idx >= window_end or next_idx >= self.n_days:
            terminated = True
            reward = 0.0
            obs = self._get_obs()
            self.weights = new_weights
            return obs, reward, terminated, truncated, self._get_info()

        current_prices = self.close_prices[self.current_idx]
        next_prices = self.close_prices[next_idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            forward_returns = np.nan_to_num((next_prices / current_prices) - 1.0, nan=0.0, posinf=0.0, neginf=0.0)

        portfolio_return = float(np.dot(new_weights, forward_returns))
        turnover = float(np.sum(np.abs(new_weights - self.weights)))

        self.return_history.append(portfolio_return)
        rolling_std = float(np.std(self.return_history)) if len(self.return_history) >= 2 else 1.0
        rolling_std = max(rolling_std, EPS)

        reward = (portfolio_return / rolling_std) - (self.turnover_penalty * turnover)

        self.weights = new_weights
        self.current_idx = next_idx
        self.steps_taken += 1
        self.portfolio_value_history.append(self.portfolio_value_history[-1] * (1.0 + portfolio_return))

        if self.steps_taken >= self.episode_length:
            truncated = True

        obs = self._get_obs()
        info = self._get_info()
        info["portfolio_return"] = portfolio_return
        info["rolling_std"] = rolling_std
        info["turnover"] = turnover

        return obs, reward, terminated, truncated, info

    def render(self):
        pass


def make_env(
    features: np.ndarray,
    close_prices: np.ndarray,
    tickers: list[str],
    split: str,
    val_days: int = 120,
    test_days: int = 60,
    episode_length: int = 252,
    seed: Optional[int] = None,
) -> PortfolioEnv:
    """Factory that builds a PortfolioEnv scoped to the train/val/test date window."""
    splits = compute_split_indices(features.shape[0], val_days=val_days, test_days=test_days)
    if split not in splits:
        raise ValueError(f"split must be one of {list(splits.keys())}, got '{split}'")
    return PortfolioEnv(
        features=features,
        close_prices=close_prices,
        tickers=tickers,
        window=splits[split],
        episode_length=episode_length,
        seed=seed,
    )
