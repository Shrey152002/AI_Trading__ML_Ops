"""Orchestrates training of all four agents for a portfolio and selects the winner."""
import logging
import os
import tempfile
import time
from pathlib import Path

import mlflow

from agents import AGENT_REGISTRY
from training.benchmark import benchmark_models, compute_metrics, run_episode
from training.env_utils import build_envs
from training.hyperopt import run_hyperopt

logger = logging.getLogger(__name__)

MLFLOW_EXPERIMENT_PREFIX = "autoportfolio"


def _set_experiment(portfolio_name: str) -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"{MLFLOW_EXPERIMENT_PREFIX}-{portfolio_name}")


def train_portfolio(
    portfolio_name: str,
    total_timesteps: int = 100_000,
    n_trials: int = 20,
    episode_length: int = 252,
    val_days: int = 120,
    test_days: int = 60,
) -> dict:
    """Trains PPO, A2C, SAC, and DDPG for a portfolio, benchmarks them, and returns the winner.

    Each algorithm gets its own MLflow run tagged with portfolio + algorithm, containing
    the best hyperparameters found by Optuna, training/eval metrics, training time, and
    the saved model artifact.
    """
    _set_experiment(portfolio_name)
    envs = build_envs(
        portfolio_name, episode_length=episode_length, val_days=val_days, test_days=test_days
    )
    train_env, val_env, test_env = envs["train"], envs["val"], envs["test"]

    trained_agents = {}
    run_results = []

    for algo_name, agent_class in AGENT_REGISTRY.items():
        with mlflow.start_run(run_name=f"{portfolio_name}-{algo_name}") as run:
            mlflow.set_tags({"portfolio": portfolio_name, "algorithm": algo_name})
            logger.info("Starting hyperopt for %s on portfolio %s", algo_name, portfolio_name)

            best_params, best_val_sharpe = run_hyperopt(
                algo_name, agent_class, train_env, val_env, n_trials=n_trials
            )
            mlflow.log_params({f"hp_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("val_sharpe", best_val_sharpe)

            agent = agent_class(hyperparams=best_params)
            start_time = time.time()
            agent.train(train_env, total_timesteps=total_timesteps, mlflow_run_id=run.info.run_id)
            training_time = time.time() - start_time

            test_returns = run_episode(agent, test_env)
            test_metrics = compute_metrics(test_returns)
            mlflow.log_metrics(test_metrics)
            mlflow.log_metric("training_time_seconds", training_time)

            with tempfile.TemporaryDirectory() as tmp_dir:
                model_path = Path(tmp_dir) / f"{algo_name.lower()}_model.zip"
                agent.save(str(model_path))
                mlflow.log_artifact(str(model_path), artifact_path="model")

            trained_agents[algo_name] = agent
            run_results.append(
                {
                    "algorithm": algo_name,
                    "run_id": run.info.run_id,
                    "best_params": best_params,
                    "training_time_seconds": training_time,
                    **test_metrics,
                }
            )
            logger.info("Finished %s for %s: %s", algo_name, portfolio_name, test_metrics)

    ranking = benchmark_models(trained_agents, test_env, portfolio_name)
    winner_algo = ranking.iloc[0]["algorithm"]
    winner_result = next(r for r in run_results if r["algorithm"] == winner_algo)

    logger.info("Winner for %s: %s (Sharpe=%.4f)", portfolio_name, winner_algo, winner_result["sharpe"])

    return {
        "portfolio": portfolio_name,
        "winner_algorithm": winner_algo,
        "winner_run_id": winner_result["run_id"],
        "winner_metrics": {k: winner_result[k] for k in ("sharpe", "max_drawdown", "total_return", "calmar", "volatility")},
        "all_results": run_results,
        "ranking": ranking.to_dict(orient="records"),
    }


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    name = sys.argv[1] if len(sys.argv) > 1 else "banking"
    result = train_portfolio(name, total_timesteps=20_000, n_trials=5)
    print(result)
