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


def test_pendulum_evaluation_runs_episodes(tmp_path):
    cfg = dict(agent="copdac_q", env="pendulum", seed=0, actor_lr=1e-4, critic_lr=1e-3,
               sigma_beta=1.0, centres_per_dim=5, eval_episodes=1)
    env = run.make_env("pendulum", 0, cfg)
    agent = run.make_agent("copdac_q", env, cfg)
    score = run.make_evaluate_fn("pendulum", env, cfg)(agent)
    assert np.isfinite(score) and -2000 < score <= 0  # one 200-step episode
