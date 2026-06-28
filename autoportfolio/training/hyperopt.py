"""Optuna-based hyperparameter search for each RL algorithm, optimizing validation Sharpe."""
import logging
from typing import Callable, Optional, Type

import optuna
from optuna.trial import Trial

from agents.base_agent import BaseAgent
from envs.portfolio_env import PortfolioEnv
from training.benchmark import compute_metrics, run_episode

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest_ppo(trial: Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_int("n_steps", 128, 2048, step=128),
        "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.1),
    }


def _suggest_a2c(trial: Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_int("n_steps", 5, 256, step=5),
        "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.1),
    }


def _suggest_sac(trial: Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "buffer_size": trial.suggest_int("buffer_size", 10000, 100000, step=10000),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
    }


def _suggest_ddpg(trial: Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "buffer_size": trial.suggest_int("buffer_size", 10000, 100000, step=10000),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "tau": trial.suggest_float("tau", 0.001, 0.1, log=True),
    }


SEARCH_SPACES = {
    "PPO": _suggest_ppo,
    "A2C": _suggest_a2c,
    "SAC": _suggest_sac,
    "DDPG": _suggest_ddpg,
}


def run_hyperopt(
    algo_name: str,
    agent_class: Type[BaseAgent],
    train_env: PortfolioEnv,
    val_env: PortfolioEnv,
    n_trials: int = 20,
    search_timesteps: int = 5000,
    on_trial_done: Optional[Callable[[int], None]] = None,
) -> tuple[dict, float]:
    """Runs an Optuna study maximizing validation-window Sharpe for one algorithm.

    Each trial trains a fresh agent for `search_timesteps` (a reduced budget for search
    speed) and scores it on val_env. Returns the best hyperparameter dict and its Sharpe.
    `on_trial_done`, if given, is called with the 1-based count of trials completed so far —
    used to report live progress while the search is running.
    """
    if algo_name not in SEARCH_SPACES:
        raise ValueError(f"No search space defined for algorithm '{algo_name}'")
    suggest_fn = SEARCH_SPACES[algo_name]

    def objective(trial: Trial) -> float:
        params = suggest_fn(trial)
        agent = agent_class(hyperparams=params)
        try:
            agent.train(train_env, total_timesteps=search_timesteps)
            returns = run_episode(agent, val_env)
            metrics = compute_metrics(returns)
            return metrics["sharpe"]
        except Exception as exc:  # noqa: BLE001 - bad hyperparams can raise from SB3/torch
            logger.warning("Trial failed for %s with params %s: %s", algo_name, params, exc)
            return -10.0

    def trial_callback(_study: optuna.Study, trial: Trial) -> None:
        if on_trial_done is not None:
            on_trial_done(trial.number + 1)

    study = optuna.create_study(direction="maximize", study_name=f"{algo_name}_hyperopt")
    study.optimize(
        objective, n_trials=n_trials, show_progress_bar=False, callbacks=[trial_callback]
    )

    logger.info(
        "Hyperopt for %s finished: best_sharpe=%.4f, best_params=%s",
        algo_name,
        study.best_value,
        study.best_params,
    )
    return study.best_params, study.best_value
