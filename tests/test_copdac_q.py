"""T1.2: COPDAC-Q on the bandit and the incremental (Pendulum) updates of eqs. 19--22."""

import numpy as np
import pytest

from agents.copdac_q import CopdacQ, make_copdac_q
from envs.bandit import QuadraticBandit
from features.rbf import feature_dim, pendulum_phi


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


def _linear_agent(natural):
    n = feature_dim(5)
    return CopdacQ(
        1, mode="linear", actor_lr=0.01, critic_lr=0.1, sigma_beta=0.5, seed=0,
        gamma=0.9, natural_gradient=natural,
        feature_fn=lambda s: pendulum_phi(s, centres_per_dim=5), feature_dim=n,
    )


def test_linear_update_matches_equations_19_to_22():
    agent = _linear_agent(natural=False)
    rng = np.random.default_rng(1)
    agent.theta = rng.normal(size=agent.theta.shape) * 0.1
    agent.w = rng.normal(size=agent.w.shape) * 0.1
    agent.v = rng.normal(size=agent.v.shape) * 0.1
    th, w, v = agent.theta.copy(), agent.w.copy(), agent.v.copy()

    s, s2 = np.array([0.6, 0.8, 1.0]), np.array([0.0, 1.0, -2.0])
    a, r = np.array([0.7]), -3.0
    phi, phi2 = pendulum_phi(s, 5), pendulum_phi(s2, 5)
    phi_sa = phi[:, None] * (a - phi @ th)
    delta = r + 0.9 * (phi2 @ v) - (np.sum(phi_sa * w) + phi @ v)

    agent.update(s, a, r, s2, False)
    np.testing.assert_allclose(agent.theta, th + 0.01 * phi[:, None] * (phi @ w))
    np.testing.assert_allclose(agent.w, w + 0.1 * delta * phi_sa)
    np.testing.assert_allclose(agent.v, v + 0.1 * delta * phi)


def test_natural_gradient_flag_changes_actor_only():
    plain, natural = _linear_agent(False), _linear_agent(True)
    w = np.random.default_rng(2).normal(size=plain.w.shape)
    plain.w, natural.w = w.copy(), w.copy()
    s = np.array([1.0, 0.0, 0.5])
    plain.update(s, np.array([0.3]), -1.0, s, False)
    natural.update(s, np.array([0.3]), -1.0, s, False)
    np.testing.assert_allclose(natural.theta, 0.01 * w)
    assert not np.allclose(plain.theta, natural.theta)
    np.testing.assert_allclose(plain.w, natural.w)
    np.testing.assert_allclose(plain.v, natural.v)


def test_done_drops_bootstrap():
    a1, a2 = _linear_agent(False), _linear_agent(False)
    a1.v[:] = a2.v[:] = 1.0
    s = np.array([1.0, 0.0, 0.0])
    a1.update(s, np.array([0.0]), 0.0, s, True)
    a2.update(s, np.array([0.0]), 0.0, s, False)
    assert not np.allclose(a1.v, a2.v)


def test_compatible_feature_stays_bounded_when_mean_leaves_range():
    # The agent does not clip, so (a - mu) is pure exploration noise even if mu is huge.
    agent = _linear_agent(False)
    agent.theta[:] = 100.0
    s = np.array([1.0, 0.0, 0.0])
    a = agent.act(s)
    assert np.isfinite(a).all()
    assert abs(a[0] - agent.mu(s)[0]) < 10 * agent.sigma_beta


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_divergence_is_loud():
    agent = _linear_agent(False)
    with pytest.raises(FloatingPointError):
        agent.update(np.array([1.0, 0.0, 0.0]), np.array([0.0]), np.inf, np.array([1.0, 0.0, 0.0]), False)


def test_save_load_roundtrip(tmp_path):
    agent, env = _bandit_agent()
    _run_bandit(agent, env, 200)
    path = str(tmp_path / "copdac.npz")
    agent.save(path)
    fresh, _ = _bandit_agent()
    fresh.load(path)
    for name in ("theta", "w", "v"):
        np.testing.assert_array_equal(getattr(fresh, name), getattr(agent, name))
