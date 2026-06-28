import time

import pytest

from training.progress import compute_eta, init_progress_state, mark_finished, mark_phase


class FakeManager:
    """A plain dict behaves identically to a Manager().dict() proxy for item access, so
    tests don't need to pay for spinning up a real multiprocessing manager process."""

    def dict(self):
        return {}


def test_init_progress_state_weights_steps_by_trials_and_total_timesteps():
    state = init_progress_state(
        FakeManager(), "banking", ["PPO", "A2C"], n_trials=10, total_timesteps=1000, search_timesteps=100
    )
    eta = compute_eta(state)

    assert eta["algorithms"]["PPO"]["steps_total"] == 10 * 100 + 1000
    assert eta["algorithms"]["A2C"]["steps_total"] == 10 * 100 + 1000
    assert eta["fraction_done"] == 0.0
    assert eta["active"] is True


def test_compute_eta_reports_fraction_done_from_progress():
    state = init_progress_state(
        FakeManager(), "banking", ["PPO"], n_trials=10, total_timesteps=1000, search_timesteps=100
    )
    mark_phase(state, "PPO", phase="final_train", steps_done=1000)  # halfway through 2000 total

    eta = compute_eta(state)

    assert eta["fraction_done"] == 0.5
    assert eta["algorithms"]["PPO"]["phase"] == "final_train"


def test_compute_eta_extrapolates_remaining_time_linearly():
    state = init_progress_state(
        FakeManager(), "banking", ["PPO"], n_trials=0, total_timesteps=100, search_timesteps=100
    )
    state["started_at"] = time.time() - 10  # pretend 10s have elapsed
    mark_phase(state, "PPO", phase="final_train", steps_done=50)  # 50% done in 10s

    eta = compute_eta(state)

    assert eta["eta_seconds"] is not None
    assert eta["eta_seconds"] == pytest.approx(10, abs=2)


def test_compute_eta_no_eta_when_run_finished():
    state = init_progress_state(
        FakeManager(), "banking", ["PPO"], n_trials=0, total_timesteps=100, search_timesteps=100
    )
    mark_phase(state, "PPO", phase="done", steps_done=100)
    mark_finished(state)

    eta = compute_eta(state)

    assert eta["active"] is False
    assert eta["eta_seconds"] is None
    assert eta["fraction_done"] == 1.0
