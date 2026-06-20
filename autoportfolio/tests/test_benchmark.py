import numpy as np

from envs.portfolio_env import PortfolioEnv
from training.benchmark import benchmark_models, compute_metrics

N_DAYS = 80
N_TICKERS = 3
N_FEATURES = 5


class FixedWeightAgent:
    """Stub agent: always predicts a fixed logit vector, used to control Sharpe deterministically."""

    def __init__(self, logits: np.ndarray):
        self.logits = logits

    def predict(self, observation, deterministic: bool = True):
        return self.logits


def _make_env() -> PortfolioEnv:
    rng = np.random.default_rng(1)
    features = rng.normal(size=(N_DAYS, N_TICKERS, N_FEATURES)).astype(np.float32)

    # Ticker 0 trends up steadily (good Sharpe), ticker 1 is flat/noisy (poor Sharpe).
    trend = np.linspace(0, 20, N_DAYS)
    noise = rng.normal(scale=2.0, size=(N_DAYS, N_TICKERS))
    prices = np.zeros((N_DAYS, N_TICKERS), dtype=np.float32)
    prices[:, 0] = 100 + trend + noise[:, 0] * 0.1
    prices[:, 1] = 100 + noise[:, 1]
    prices[:, 2] = 100 + noise[:, 2]

    tickers = [f"T{i}" for i in range(N_TICKERS)]
    return PortfolioEnv(
        features=features,
        close_prices=prices,
        tickers=tickers,
        window=(0, N_DAYS),
        episode_length=N_DAYS - 1,
        seed=0,
    )


def test_compute_metrics_higher_return_lower_vol_gives_higher_sharpe():
    good_returns = np.full(100, 0.002) + np.random.default_rng(0).normal(0, 0.0005, 100)
    bad_returns = np.random.default_rng(1).normal(0, 0.02, 100)

    good_metrics = compute_metrics(good_returns)
    bad_metrics = compute_metrics(bad_returns)

    assert good_metrics["sharpe"] > bad_metrics["sharpe"]


def test_compute_metrics_handles_empty_returns():
    metrics = compute_metrics(np.array([]))
    assert metrics["sharpe"] == 0.0
    assert metrics["max_drawdown"] == 0.0


def test_benchmark_models_ranks_by_sharpe_descending(tmp_path, monkeypatch):
    import training.benchmark as benchmark_module

    monkeypatch.setattr(benchmark_module, "REPORTS_ROOT", tmp_path)

    env = _make_env()
    good_agent = FixedWeightAgent(np.array([10.0, -10.0, -10.0], dtype=np.float32))  # all-in on trending ticker
    bad_agent = FixedWeightAgent(np.array([-10.0, 10.0, 10.0], dtype=np.float32))  # all-in on noisy tickers

    models = {"GOOD": good_agent, "BAD": bad_agent}
    ranking = benchmark_models(models, env, "test_portfolio")

    assert ranking.iloc[0]["algorithm"] == "GOOD"
    assert ranking.iloc[0]["sharpe"] > ranking.iloc[-1]["sharpe"]
    assert list(ranking["rank"]) == [1, 2]
    assert (tmp_path / "benchmark_results.json").exists()
