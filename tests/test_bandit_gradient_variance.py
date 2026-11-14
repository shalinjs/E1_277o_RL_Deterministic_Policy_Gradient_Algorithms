"""T2.3: 100 actor-gradient estimates at a fixed bandit parameter, both linear methods."""

import csv

import numpy as np
import pytest

from agents.copdac_q import CopdacQ
from agents.stochastic_ac import StochasticActorCritic
from envs.bandit import QuadraticBandit
from plots.bandit_gradient_variance import batch_actor_gradient, gradient_variances, write_variance_figure


def test_one_batch_restores_the_fixed_parameter():
    env = QuadraticBandit(4, seed=0)
    reward = lambda action: -env.cost(action)
    stochastic = StochasticActorCritic(4, actor_lr=1.0, seed=0, init_log_std=0.0, batch_size=8)
    copdac = CopdacQ(4, actor_lr=1.0, sigma_beta=1.0, seed=0, batch_size=8)
    y = stochastic.y
    for agent in (stochastic, copdac):
        gradient = batch_actor_gradient(agent, reward)
        assert gradient.shape == (4,)
        assert np.all(np.isfinite(gradient))
        np.testing.assert_array_equal(agent.theta, np.zeros(4))
        assert agent._batch_actions == []
    assert stochastic.y == y


def test_repeated_calls_with_the_same_seed_match():
    first = gradient_variances((10,), n_estimates=20)
    second = gradient_variances((10,), n_estimates=20)
    assert [row["trace_variance"] for row in first] == [row["trace_variance"] for row in second]


def test_variance_at_m10_and_m50_is_finite_and_copdac_is_lower():
    rows = gradient_variances((10, 50), n_estimates=100)
    by_key = {(row["action_dim"], row["agent"]): row for row in rows}
    for action_dim in (10, 50):
        stochastic = by_key[(action_dim, "stochastic_ac")]
        copdac = by_key[(action_dim, "copdac_q")]
        assert stochastic["n_estimates"] == 100 and copdac["n_estimates"] == 100
        assert stochastic["trace_variance"] > 0 and copdac["trace_variance"] > 0
        assert np.isfinite(stochastic["trace_variance"]) and np.isfinite(copdac["trace_variance"])
        # Same exploration std: the deterministic estimator is the lower-variance one.
        assert copdac["trace_variance"] < stochastic["trace_variance"]


def test_script_writes_csv_and_png(tmp_path):
    rows = gradient_variances((10,), n_estimates=10)
    csv_path = tmp_path / "bandit_gradient_variance.csv"
    png_path = tmp_path / "gradient_variance.png"
    write_variance_figure(rows, str(csv_path), str(png_path))
    assert png_path.stat().st_size > 0
    with csv_path.open() as handle:
        written = list(csv.DictReader(handle))
    assert {row["agent"] for row in written} == {"stochastic_ac", "copdac_q"}
    assert written[0]["action_dim"] == "10"


def test_rejects_a_single_estimate():
    with pytest.raises(ValueError):
        gradient_variances((10,), n_estimates=1)
