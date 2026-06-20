import numpy as np

from envs.portfolio_env import PortfolioEnv, compute_split_indices

N_DAYS = 300
N_TICKERS = 4
N_FEATURES = 8


def _make_env(episode_length: int = 50) -> PortfolioEnv:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(N_DAYS, N_TICKERS, N_FEATURES)).astype(np.float32)
    prices = 100.0 + np.cumsum(rng.normal(scale=0.5, size=(N_DAYS, N_TICKERS)), axis=0).astype(np.float32)
    tickers = [f"T{i}" for i in range(N_TICKERS)]
    return PortfolioEnv(
        features=features,
        close_prices=prices,
        tickers=tickers,
        window=(0, N_DAYS),
        episode_length=episode_length,
        seed=0,
    )


def test_reset_returns_correct_observation_shape():
    env = _make_env()
    obs, info = env.reset()

    assert obs.shape == (N_TICKERS * N_FEATURES,)
    assert "weights" in info


def test_action_space_and_observation_space_shapes():
    env = _make_env()
    assert env.action_space.shape == (N_TICKERS,)
    assert env.observation_space.shape == (N_TICKERS * N_FEATURES,)


def test_weights_always_sum_to_one_after_step():
    env = _make_env()
    env.reset()

    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isclose(np.sum(info["weights"]), 1.0, atol=1e-5)
        assert np.all(info["weights"] >= 0.0)
        if terminated or truncated:
            break


def test_episode_truncates_at_episode_length():
    episode_length = 10
    env = _make_env(episode_length=episode_length)
    env.reset()

    steps = 0
    truncated = terminated = False
    while not (terminated or truncated) and steps < episode_length + 5:
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        steps += 1

    assert truncated
    assert steps == episode_length


def test_compute_split_indices_partitions_chronologically():
    splits = compute_split_indices(n_days=300, val_days=50, test_days=30)

    assert splits["train"][1] == splits["val"][0]
    assert splits["val"][1] == splits["test"][0]
    assert splits["test"][1] == 300
