"""run.py end to end: a short bandit run writes the CSV and the resolved YAML."""

import numpy as np
import pytest
import yaml

import run
from common.logging import read_run_csv


def _write_config(tmp_path, **extra):
    cfg = dict(agent="copdac_q", env="bandit", action_dim=4, actor_lr=0.1, sigma_beta=0.1, **extra)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def _main(tmp_path, config, *extra_args):
    args = ["--agent", "copdac_q", "--env", "bandit", "--seed", "3", "--steps", "400",
            "--config", config, *extra_args]
    return run.main(args, results_root=str(tmp_path / "results"))


def test_bandit_run_writes_csv_and_yaml(tmp_path):
    assert _main(tmp_path, _write_config(tmp_path)) == 0
    out = tmp_path / "results" / "bandit_m4" / "copdac_q"
    data = read_run_csv(str(out / "seed3.csv"))
    np.testing.assert_array_equal(data["step"], np.arange(8, 401, 8))  # eval every 2m
    assert data["eval_return"][-1] < data["eval_return"][0]
    resolved = yaml.safe_load((out / "seed3.yaml").read_text())
    assert resolved["seed"] == 3 and resolved["steps"] == 400 and resolved["eval_every"] == 8


def test_natural_gradient_runs_go_to_their_own_folder(tmp_path):
    _main(tmp_path, _write_config(tmp_path, natural_gradient=True))
    assert (tmp_path / "results" / "bandit_m4" / "copdac_q_natural" / "seed3.csv").exists()


def test_config_that_disagrees_with_cli_is_rejected(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(dict(agent="ddpg", env="bandit", action_dim=4)))
    with pytest.raises(ValueError):
        _main(tmp_path, str(cfg))


def test_gym_environment_is_not_reset_during_construction(monkeypatch):
    import gymnasium

    class ActionSpace:
        def __init__(self):
            self.seeds = []

        def seed(self, seed):
            self.seeds.append(seed)

    class Environment:
        def __init__(self):
            self.action_space = ActionSpace()
            self.observation_space = object()
            self.reset_calls = []

        def reset(self, **kwargs):
            self.reset_calls.append(kwargs)

    base_env = Environment()
    monkeypatch.setattr(gymnasium, "make", lambda _: base_env)

    env = run.make_env("lunarlander", 23, {})

    assert env.reset_calls == []
    assert base_env.action_space.seeds == [23]


def test_lunarlander_evaluation_averages_episodes(monkeypatch):
    import gymnasium

    class ActionSpace:
        low = np.array([-1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0], dtype=np.float32)
        dtype = np.float32

        def seed(self, seed):
            self.seed_value = seed

    class Environment:
        def __init__(self):
            self.action_space = ActionSpace()
            self.observation_space = object()
            self.steps = 0

        def reset(self, **kwargs):
            self.steps = 0
            return np.zeros(8, dtype=np.float32), {}

        def step(self, action):
            self.steps += 1
            done = self.steps >= 2
            return np.zeros(8, dtype=np.float32), 1.5, done, False, {}

    monkeypatch.setattr(gymnasium, "make", lambda _: Environment())

    class Agent:
        def act(self, state, explore=True):
            return np.zeros(2, dtype=np.float32)

    env = run.make_env("lunarlander", 0, {})
    score = run.make_evaluate_fn(
        "lunarlander", env, {"env": "lunarlander", "seed": 0, "eval_episodes": 2}
    )(Agent())
    assert score == pytest.approx(3.0)


def test_removed_environment_is_rejected_by_the_parser():
    assert run.ENV_CHOICES == ("bandit", "lunarlander")
    with pytest.raises(SystemExit):
        run.build_parser().parse_args(
            ["--agent", "copdac_q", "--env", "pendulum", "--seed", "0",
             "--steps", "10", "--config", "x"]
        )
