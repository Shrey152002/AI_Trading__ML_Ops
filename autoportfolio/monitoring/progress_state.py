"""Same-process registry of in-flight training runs, keyed by portfolio name.

A pipeline run is triggered by one HTTP request (POST /pipeline/run) but its progress is
polled by later, unrelated requests (GET /pipeline/progress/{id}). Both run in the same API
process — FastAPI's BackgroundTasks execute in a threadpool inside the same process — so a
plain dict guarded by a lock is enough to hand the live multiprocessing.Manager dict (owned by
training.trainer) from the run's background thread to a later request's handler thread.
"""
from threading import Lock
from typing import Optional

from training.progress import compute_eta

_active_runs: dict = {}
_lock = Lock()


def register_run(portfolio_name: str, shared_state) -> None:
    with _lock:
        _active_runs[portfolio_name] = shared_state


def get_progress(portfolio_name: str) -> Optional[dict]:
    with _lock:
        shared_state = _active_runs.get(portfolio_name)
    if shared_state is None:
        return None
    return compute_eta(shared_state)
