"""Orchestrates training of all four agents for a portfolio and selects the winner.

Each algorithm trains in its own worker process (via ProcessPoolExecutor) since the 4
algorithms are fully independent — same train/val/test windows, no shared model state — and
training is CPU-bound enough that real OS-level parallelism (not threads) is needed to get a
wall-clock speedup. Workers report progress and forward their logs back to the parent process
through multiprocessing primitives created by the orchestrator (train_portfolio).
"""
import logging
import logging.handlers
import os
import shutil
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import Manager
from pathlib import Path
from typing import Optional

import mlflow

from agents import AGENT_REGISTRY
from monitoring.metrics import training_runs_total
from training.benchmark import benchmark_models, compute_metrics, run_episode
from training.env_utils import build_envs
from training.hyperopt import run_hyperopt
from training.progress import (
    DEFAULT_SEARCH_TIMESTEPS,
    ProgressStepCallback,
    init_progress_state,
    make_trial_progress_hook,
    mark_finished,
    mark_phase,
)

logger = logging.getLogger(__name__)

MLFLOW_EXPERIMENT_PREFIX = "autoportfolio"


def _set_experiment(portfolio_name: str) -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"{MLFLOW_EXPERIMENT_PREFIX}-{portfolio_name}")


def _configure_worker_logging(log_queue) -> None:
    """Forwards this worker process's log records to the parent via a Manager queue.

    Each process has its own logging module state, so a handler attached in the parent
    (e.g. monitoring.log_buffer's buffer handler) never sees anything emitted in a worker —
    this routes records back so the live-log dashboard keeps working with parallel training.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(logging.handlers.QueueHandler(log_queue))


def _start_log_listener(log_queue) -> logging.handlers.QueueListener:
    """Runs in the parent process: re-emits records pulled off the queue through the normal
    logger hierarchy, so handlers already attached there (the dashboard's buffer) see them."""

    def _handle(record: logging.LogRecord) -> None:
        logging.getLogger(record.name).handle(record)

    listener = logging.handlers.QueueListener(log_queue, respect_handler_level=False)
    listener.handle = _handle  # type: ignore[method-assign]
    listener.start()
    return listener


def _train_one_algorithm(
    portfolio_name: str,
    algo_name: str,
    total_timesteps: int,
    n_trials: int,
    episode_length: int,
    val_days: int,
    test_days: int,
    search_timesteps: int,
    scratch_dir: str,
    shared_state,
    log_queue,
) -> dict:
    """Runs in a worker process: hyperopt + final training + test-set evaluation for one
    algorithm. Returns a result dict plus the path to the saved model so the parent (which
    needs every algorithm's model in the same process to benchmark them against each other)
    can reload it without pickling a torch model across the process boundary."""
    _configure_worker_logging(log_queue)
    import torch

    torch.set_num_threads(1)  # avoid 4 processes each oversubscribing all CPU cores with BLAS/OMP threads

    worker_logger = logging.getLogger("training.trainer")
    _set_experiment(portfolio_name)

    envs = build_envs(
        portfolio_name, episode_length=episode_length, val_days=val_days, test_days=test_days
    )
    train_env, val_env, test_env = envs["train"], envs["val"], envs["test"]

    agent_class = AGENT_REGISTRY[algo_name]
    worker_logger.info("Starting hyperopt for %s on portfolio %s", algo_name, portfolio_name)
    mark_phase(shared_state, algo_name, phase="hyperopt")
    on_trial_done = make_trial_progress_hook(shared_state, algo_name, search_timesteps)

    with mlflow.start_run(run_name=f"{portfolio_name}-{algo_name}") as run:
        mlflow.set_tags({"portfolio": portfolio_name, "algorithm": algo_name})

        best_params, best_val_sharpe = run_hyperopt(
            algo_name,
            agent_class,
            train_env,
            val_env,
            n_trials=n_trials,
            search_timesteps=search_timesteps,
            on_trial_done=on_trial_done,
        )
        mlflow.log_params({f"hp_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("val_sharpe", best_val_sharpe)

        base_steps = n_trials * search_timesteps
        mark_phase(shared_state, algo_name, phase="final_train", steps_done=base_steps)

        agent = agent_class(hyperparams=best_params)
        progress_cb = ProgressStepCallback(shared_state, algo_name, base_steps=base_steps)
        start_time = time.time()
        agent.train(
            train_env,
            total_timesteps=total_timesteps,
            mlflow_run_id=run.info.run_id,
            extra_callback=progress_cb,
        )
        training_time = time.time() - start_time

        test_returns = run_episode(agent, test_env)
        test_metrics = compute_metrics(test_returns)
        mlflow.log_metrics(test_metrics)
        mlflow.log_metric("training_time_seconds", training_time)

        with tempfile.TemporaryDirectory() as tmp_dir:
            mlflow_artifact_path = Path(tmp_dir) / f"{algo_name.lower()}_model.zip"
            agent.save(str(mlflow_artifact_path))
            mlflow.log_artifact(str(mlflow_artifact_path), artifact_path="model")

        persisted_path = str(Path(scratch_dir) / f"{algo_name.lower()}_model.zip")
        agent.save(persisted_path)

        run_id = run.info.run_id

    mark_phase(
        shared_state,
        algo_name,
        phase="done",
        steps_done=base_steps + total_timesteps,
    )
    worker_logger.info("Finished %s for %s: %s", algo_name, portfolio_name, test_metrics)

    return {
        "algorithm": algo_name,
        "run_id": run_id,
        "best_params": best_params,
        "training_time_seconds": training_time,
        "model_path": persisted_path,
        **test_metrics,
    }


def train_portfolio(
    portfolio_name: str,
    total_timesteps: int = 100_000,
    n_trials: int = 20,
    episode_length: int = 252,
    val_days: int = 120,
    test_days: int = 60,
    algorithms: Optional[list[str]] = None,
    search_timesteps: int = DEFAULT_SEARCH_TIMESTEPS,
    shared_state=None,
) -> dict:
    """Trains the selected algorithms (default: PPO, A2C, SAC, and DDPG) for a portfolio in
    parallel worker processes, benchmarks them, and returns the winner.

    Each algorithm gets its own MLflow run tagged with portfolio + algorithm, containing
    the best hyperparameters found by Optuna, training/eval metrics, training time, and
    the saved model artifact.
    """
    algos = algorithms or list(AGENT_REGISTRY.keys())
    unknown = [a for a in algos if a not in AGENT_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown algorithm(s) {unknown}; choose from {list(AGENT_REGISTRY)}")

    _set_experiment(portfolio_name)

    manager = Manager()
    own_shared_state = shared_state is None
    if own_shared_state:
        shared_state = init_progress_state(
            manager, portfolio_name, algos, n_trials, total_timesteps, search_timesteps
        )
    log_queue = manager.Queue()
    listener = _start_log_listener(log_queue)

    scratch_dir = tempfile.mkdtemp(prefix=f"autoportfolio_train_{portfolio_name}_")
    run_results = []
    max_workers = min(len(algos), os.cpu_count() or len(algos))

    # Starting several worker processes at once means several copies of torch/pyarrow
    # import and allocate native memory within the same second — on a memory-constrained
    # machine that can trip a transient allocation failure that has nothing to do with the
    # algorithm itself. One retry absorbs that without falling back to fully serial training.
    MAX_ATTEMPTS = 2

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:

            def submit(algo: str):
                return pool.submit(
                    _train_one_algorithm,
                    portfolio_name,
                    algo,
                    total_timesteps,
                    n_trials,
                    episode_length,
                    val_days,
                    test_days,
                    search_timesteps,
                    scratch_dir,
                    shared_state,
                    log_queue,
                )

            attempts = {algo: 1 for algo in algos}
            pending = {submit(algo): algo for algo in algos}

            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    algo = pending.pop(future)
                    try:
                        run_results.append(future.result())
                        training_runs_total.labels(
                            portfolio=portfolio_name, algorithm=algo, outcome="completed"
                        ).inc()
                    except Exception as exc:
                        if attempts[algo] < MAX_ATTEMPTS:
                            attempts[algo] += 1
                            logger.warning(
                                "Training failed for %s on portfolio %s, retrying (attempt %d/%d): %s",
                                algo,
                                portfolio_name,
                                attempts[algo],
                                MAX_ATTEMPTS,
                                exc,
                            )
                            mark_phase(shared_state, algo, phase="pending", trial=0, steps_done=0)
                            pending[submit(algo)] = algo
                        else:
                            training_runs_total.labels(
                                portfolio=portfolio_name, algorithm=algo, outcome="failed"
                            ).inc()
                            mark_phase(shared_state, algo, phase="failed")
                            logger.exception(
                                "Training failed for %s on portfolio %s", algo, portfolio_name
                            )

        if not run_results:
            raise RuntimeError(f"All agents failed to train for portfolio '{portfolio_name}'")

        trained_agents = {}
        for result in run_results:
            agent = AGENT_REGISTRY[result["algorithm"]]()
            agent.load(result["model_path"])
            trained_agents[result["algorithm"]] = agent

        test_env = build_envs(
            portfolio_name, episode_length=episode_length, val_days=val_days, test_days=test_days
        )["test"]
        ranking = benchmark_models(trained_agents, test_env, portfolio_name)
        winner_algo = ranking.iloc[0]["algorithm"]
        winner_result = next(r for r in run_results if r["algorithm"] == winner_algo)

        logger.info(
            "Winner for %s: %s (Sharpe=%.4f)", portfolio_name, winner_algo, winner_result["sharpe"]
        )

        return {
            "portfolio": portfolio_name,
            "winner_algorithm": winner_algo,
            "winner_run_id": winner_result["run_id"],
            "winner_metrics": {
                k: winner_result[k]
                for k in ("sharpe", "max_drawdown", "total_return", "calmar", "volatility")
            },
            "all_results": run_results,
            "ranking": ranking.to_dict(orient="records"),
        }
    finally:
        mark_finished(shared_state)
        listener.stop()
        shutil.rmtree(scratch_dir, ignore_errors=True)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    name = sys.argv[1] if len(sys.argv) > 1 else "banking"
    result = train_portfolio(name, total_timesteps=20_000, n_trials=5)
    print(result)
