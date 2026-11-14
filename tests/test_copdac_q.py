"""T1.2: COPDAC-Q on the bandit (batch form of eqs. 19--22, identity Jacobian)."""

import numpy as np
import pytest

from agents.copdac_q import make_copdac_q
from envs.bandit import QuadraticBandit


def _run_bandit(agent, env, steps):
    obs, _ = env.reset()
    costs = []
    for t in range(1, steps + 1):
        a = agent.act(obs)
        nxt, r, term, _, _ = env.step(a)
        agent.update(obs, a, r, nxt, term)
        if t % agent.batch_size == 0:
            costs.append(env.cost(agent.act(obs, explore=False)))
    return costs


def _bandit_agent(m=10, **overrides):
    cfg = {**dict(actor_lr=0.1, sigma_beta=0.1, seed=0, action_dim=m), **overrides}
    env = QuadraticBandit(m, seed=0)
    return make_copdac_q("bandit", env, cfg), env


def test_bandit_cost_falls_and_mean_moves_to_four():
    agent, env = _bandit_agent()
    start = env.cost(agent.act(None, explore=False))
    costs = _run_bandit(agent, env, 2000)
    assert costs[-1] < 0.01 * start
    assert np.abs(agent.theta - 4.0).max() < 0.5


def test_explore_false_adds_no_noise():
    agent, _ = _bandit_agent()
    agent.theta = np.arange(10.0)
    np.testing.assert_array_equal(agent.act(None, explore=False), np.arange(10.0))
    assert not np.allclose(agent.act(None, explore=True), np.arange(10.0))


def test_bandit_only_updates_once_per_batch():
    agent, env = _bandit_agent(m=3)
    obs, _ = env.reset()
    for _ in range(agent.batch_size - 1):
        a = agent.act(obs)
        agent.update(obs, a, -env.cost(a), obs, False)
    np.testing.assert_array_equal(agent.theta, 0.0)
    a = agent.act(obs)
    agent.update(obs, a, -env.cost(a), obs, False)
    assert np.any(agent.theta != 0.0)


def test_bandit_critic_recovers_reward_gradient():
    # Reward is linear in a, so least squares must return its exact slope.
    agent, _ = _bandit_agent(m=4, actor_lr=0.0)
    slope = np.array([1.0, -2.0, 0.5, 3.0])
    for _ in range(agent.batch_size):
        a = agent.act(None)
        agent.update(None, a, float(slope @ a) + 7.0, None, False)
    np.testing.assert_allclose(agent.w, slope, atol=1e-8)
    np.testing.assert_allclose(agent.v, [7.0], atol=1e-8)


def test_natural_gradient_flag_matches_the_ordinary_bandit_step():
    plain, env = _bandit_agent(m=3, natural_gradient=False)
    natural, _ = _bandit_agent(m=3, natural_gradient=True)
    obs, _ = env.reset()
    for _ in range(plain.batch_size):
        a = plain.act(obs)
        reward = -env.cost(a)
        plain.update(obs, a, reward, obs, False)
        natural.update(obs, a, reward, obs, False)
    np.testing.assert_allclose(natural.theta, plain.theta)
    np.testing.assert_allclose(natural.w, plain.w)
    assert natural.natural_gradient and not plain.natural_gradient


def test_act_does_not_clip_a_large_mean():
    agent, _ = _bandit_agent()
    agent.theta = np.full(agent.action_dim, 100.0)
    np.testing.assert_array_equal(agent.act(None, explore=False), agent.theta)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_nonfinite_reward_is_loud():
    agent, _ = _bandit_agent(m=3)
    with pytest.raises(FloatingPointError):
        for _ in range(agent.batch_size):
            a = agent.act(None)
            agent.update(None, a, np.inf, None, False)


def test_factory_rejects_a_non_bandit_env():
    with pytest.raises(ValueError):
        make_copdac_q(
            "lunarlander", None,
            {"action_dim": 2, "actor_lr": 0.01, "sigma_beta": 0.1, "seed": 0},
        )


def test_save_load_roundtrip(tmp_path):
    agent, env = _bandit_agent()
    _run_bandit(agent, env, 200)
    path = str(tmp_path / "copdac.npz")
    agent.save(path)
    fresh, _ = _bandit_agent()
    fresh.load(path)
    for name in ("theta", "w", "v"):
        np.testing.assert_array_equal(getattr(fresh, name), getattr(agent, name))
