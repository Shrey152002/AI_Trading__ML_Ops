"""Cross-process progress tracking for parallelized portfolio training runs.

Training now runs each algorithm in its own worker process (see training.trainer). Workers
report timestep progress into a multiprocessing.Manager dict created by the parent, since
that's the only state that's actually safe to share across process boundaries without custom
IPC. The parent process (the API server, inside a BackgroundTask) keeps a reference to that
dict in monitoring.progress_state so an unrelated HTTP request can read live progress.
"""
import time
from typing import Callable, Optional

from stable_baselines3.common.callbacks import BaseCallback

# Each hyperopt trial trains for this many timesteps — must match training.hyperopt's default
# so progress weighting (steps_total) reflects the real amount of work being done.
DEFAULT_SEARCH_TIMESTEPS = 5000


class ProgressStepCallback(BaseCallback):
    """SB3 callback that periodically reports cumulative timesteps for one algorithm's final
    training phase into the shared progress dict, so long final-training runs (the slow part)
    show live movement rather than jumping only at phase boundaries."""

    def __init__(self, shared_state, algo_name: str, base_steps: int, report_freq: int = 500):
        super().__init__()
        self._shared_state = shared_state
        self._algo_name = algo_name
        self._base_steps = base_steps
        self._report_freq = report_freq

    def _on_step(self) -> bool:
        if self.n_calls % self._report_freq == 0:
            _update_algo(self._shared_state, self._algo_name, steps_done=self._base_steps + self.num_timesteps)
        return True


def _algo_key(algo_name: str, field: str) -> str:
    return f"algo:{algo_name}:{field}"


def init_progress_state(
    manager,
    portfolio_name: str,
    algorithms: list[str],
    n_trials: int,
    total_timesteps: int,
    search_timesteps: int = DEFAULT_SEARCH_TIMESTEPS,
):
    """Creates the shared dict a training run reports into. Flat keys, not nested dicts,
    since Manager.dict() proxies only synchronize top-level assignment — a nested dict
    stored as a value is just a local copy once read back out."""
    state = manager.dict()
    state["portfolio"] = portfolio_name
    state["algorithms"] = list(algorithms)
    state["started_at"] = time.time()
    state["finished_at"] = None
    for algo in algorithms:
        steps_total = n_trials * search_timesteps + total_timesteps
        state[_algo_key(algo, "phase")] = "pending"
        state[_algo_key(algo, "trial")] = 0
        state[_algo_key(algo, "n_trials")] = n_trials
        state[_algo_key(algo, "steps_done")] = 0
        state[_algo_key(algo, "steps_total")] = steps_total
    return state


def _update_algo(shared_state, algo_name: str, **fields) -> None:
    for field, value in fields.items():
        shared_state[_algo_key(algo_name, field)] = value


def mark_phase(shared_state, algo_name: str, phase: str, **extra_fields) -> None:
    _update_algo(shared_state, algo_name, phase=phase, **extra_fields)


def make_trial_progress_hook(shared_state, algo_name: str, search_timesteps: int) -> Callable[[int], None]:
    """Returns a callback for training.hyperopt to call after each completed trial."""

    def on_trial_done(trial_number: int) -> None:
        _update_algo(shared_state, algo_name, trial=trial_number, steps_done=trial_number * search_timesteps)

    return on_trial_done


def mark_finished(shared_state) -> None:
    shared_state["finished_at"] = time.time()


def compute_eta(shared_state) -> dict:
    """Aggregates per-algorithm progress into an overall fraction-done + ETA estimate.

    Uses simple linear extrapolation (elapsed / fraction_done) rather than per-algorithm
    throughput modeling — good enough for a live-updating estimate, and avoids pretending
    to know each algorithm's relative speed before any of them have run.
    """
    algorithms = list(shared_state["algorithms"])
    algos_out = {}
    steps_done_total = 0
    steps_total_total = 0
    for algo in algorithms:
        steps_done = shared_state[_algo_key(algo, "steps_done")]
        steps_total = shared_state[_algo_key(algo, "steps_total")]
        steps_done_total += steps_done
        steps_total_total += steps_total
        algos_out[algo] = {
            "phase": shared_state[_algo_key(algo, "phase")],
            "trial": shared_state[_algo_key(algo, "trial")],
            "n_trials": shared_state[_algo_key(algo, "n_trials")],
            "steps_done": steps_done,
            "steps_total": steps_total,
        }

    finished_at = shared_state["finished_at"]
    elapsed = (finished_at or time.time()) - shared_state["started_at"]
    fraction = steps_done_total / steps_total_total if steps_total_total else 0.0

    eta_seconds: Optional[float] = None
    if finished_at is None and fraction > 0.02:
        eta_seconds = max(0.0, elapsed / fraction - elapsed)

    return {
        "portfolio": shared_state["portfolio"],
        "active": finished_at is None,
        "fraction_done": min(fraction, 1.0),
        "elapsed_seconds": elapsed,
        "eta_seconds": eta_seconds,
        "algorithms": algos_out,
    }
