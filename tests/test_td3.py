"""Seeded-reset check carried over from the LunarLander TD3 training test.

The original ``tests/test_td3.py`` is not in this tree (the DDPG/TD3 agent is
still a stub, so that module was never committed). The patch only added two
assertions to ``test_training_bootstraps_time_limit_truncations``: record every
``reset`` call, and require the first training reset to receive the run seed.
Those assertions are reproduced here against ``run_training``, which is the
training loop this repository actually has.
"""

import numpy as np

from common.logging import RunLogger
from common.runner import run_training


def test_training_bootstraps_time_limit_truncations(tmp_path):
    class Environment:
        def __init__(self):
            self.action_space = type("ActionSpace", (), {"seed": lambda self, seed: None})()
            self.calls = 0
            self.reset_calls = []

        def reset(self, **kwargs):
            self.reset_calls.append(kwargs)
            return np.zeros(3, dtype=np.float32), {}

        def step(self, action):
            self.calls += 1
            # First step continues. Second step terminates, so done becomes True.
            # A time-limit truncation would leave done False; that case is covered
            # by tests/test_runner_logging.py.
            terminated = self.calls == 2
            return np.zeros(3, dtype=np.float32), 0.0, terminated, False, {}

    class Agent:
        def __init__(self):
            self.dones = []

        def act(self, state, explore=True):
            return np.zeros(1, dtype=np.float32)

        def update(self, state, action, reward, next_state, done):
            self.dones.append(done)

    env = Environment()
    agent = Agent()
    with RunLogger(str(tmp_path / "run.csv")) as logger:
        run_training(
            env,
            agent,
            logger,
            steps=2,
            eval_every=2,
            seed=0,
            evaluate_fn=lambda _agent: 0.0,
        )
    assert agent.dones == [False, True]
    assert env.reset_calls[0] == {"seed": 0}
