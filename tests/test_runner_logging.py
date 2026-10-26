"""Logging schema and the training loop: the CSV header is fixed, rows round-trip,
and ``done`` ignores time-limit truncation.
"""

import time

import numpy as np
import pytest

from common.logging import BASE_COLUMNS, RunLogger, read_run_csv
from common.runner import run_training


def test_logger_writes_fixed_header_and_rows(tmp_path):
    path = tmp_path / "run.csv"
    with RunLogger(str(path)) as logger:
        logger.log(step=10, eval_return=-1.5, wall_time_sec=0.01)
        logger.log(step=20, eval_return=-0.5, wall_time_sec=0.02)

    data = read_run_csv(str(path))
    assert tuple(data.keys()) == BASE_COLUMNS
    np.testing.assert_array_equal(data["step"], [10, 20])
    np.testing.assert_allclose(data["eval_return"], [-1.5, -0.5])


def test_logger_supports_optional_columns(tmp_path):
    path = tmp_path / "run.csv"
    with RunLogger(str(path), extra_columns=["q_estimate"]) as logger:
        logger.log(step=5, eval_return=-2.0, wall_time_sec=0.1, q_estimate=0.3)
    data = read_run_csv(str(path))
    assert "q_estimate" in data
    np.testing.assert_allclose(data["q_estimate"], [0.3])


def test_logger_rejects_non_finite(tmp_path):
    with RunLogger(str(tmp_path / "run.csv")) as logger:
        with pytest.raises(ValueError):
            logger.log(step=1, eval_return=float("nan"), wall_time_sec=0.0)


def test_read_run_csv_rejects_bad_header(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("t,score,time\n0,1.0,0.0\n")
    with pytest.raises(ValueError):
        read_run_csv(str(path))


class _CountingAgent:
    """Records the ``done`` flag it was handed, to prove truncation is ignored."""

    def __init__(self):
        self.dones = []

    def act(self, state, explore=True):
        return np.zeros(1)

    def update(self, state, action, reward, next_state, done):
        self.dones.append(done)


class _ScriptedEnv:
    """Replays a fixed (terminated, truncated) schedule.

    The call counter is monotonic across resets so the schedule is unaffected by
    the runner resetting the env after a terminal step.
    """

    def __init__(self, schedule):
        self.schedule = schedule
        self.calls = 0

    def reset(self, **kwargs):
        return np.zeros(1), {}

    def step(self, action):
        terminated, truncated = self.schedule[self.calls]
        self.calls += 1
        return np.zeros(1), -1.0, terminated, truncated, {}


def test_runner_done_ignores_truncation(tmp_path):
    # step 2 terminates -> done True; step 4 truncates (time limit) -> done False.
    schedule = [(False, False), (True, False), (False, False), (False, True)]
    env = _ScriptedEnv(schedule)
    agent = _CountingAgent()
    with RunLogger(str(tmp_path / "run.csv")) as logger:
        run_training(
            env,
            agent,
            logger,
            steps=4,
            eval_every=2,
            evaluate_fn=lambda a: 0.0,
            start_time=time.perf_counter(),
        )
    assert agent.dones == [False, True, False, False]


def test_runner_logs_on_schedule(tmp_path):
    schedule = [(False, False)] * 4
    env = _ScriptedEnv(schedule)
    agent = _CountingAgent()
    path = tmp_path / "run.csv"
    with RunLogger(str(path)) as logger:
        run_training(
            env, agent, logger, steps=4, eval_every=2, evaluate_fn=lambda a: 1.0
        )
    data = read_run_csv(str(path))
    np.testing.assert_array_equal(data["step"], [2, 4])
