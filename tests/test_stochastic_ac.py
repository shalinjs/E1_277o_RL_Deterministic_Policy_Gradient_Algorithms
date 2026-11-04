"""T1.1: Stochastic Actor-Critic on the bandit, plus the Gaussian scores and the
incremental (Pendulum) TD updates from code.md.
"""

import numpy as np
import pytest

from agents.stochastic_ac import StochasticActorCritic, make_stochastic_ac
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
    cfg = {**dict(actor_lr=0.05, seed=0, action_dim=m), **overrides}
    env = QuadraticBandit(m, seed=0)
    return make_stochastic_ac("bandit", env, cfg), env


# -- T1.1 done-when --------------------------------------------------------------
def test_bandit_cost_falls_and_mean_moves_to_four():
    agent, env = _bandit_agent()
    start = env.cost(agent.act(None, explore=False))
    costs = _run_bandit(agent, env, 10000)
    assert costs[-1] < start                       # cost falls across the run
    assert costs[-1] < 0.01 * start                # and falls a lot
    assert np.abs(agent.theta - 4.0).max() < 0.5   # mean is near a_star = 4


def test_cost_decreases_monotonically_on_average():
    agent, env = _bandit_agent()
    costs = _run_bandit(agent, env, 10000)
    # First-quarter mean cost is far above the last-quarter mean cost.
    q = len(costs) // 4
    assert np.mean(costs[:q]) > np.mean(costs[-q:])


# -- policy / exploration --------------------------------------------------------
def test_explore_false_adds_no_noise():
    agent, _ = _bandit_agent()
    agent.theta = np.arange(10.0)
    np.testing.assert_array_equal(agent.act(None, explore=False), np.arange(10.0))
    assert not np.allclose(agent.act(None, explore=True), np.arange(10.0))


def test_initial_std_follows_init_log_std():
    agent, _ = _bandit_agent(init_log_std=np.log(0.3))
    assert agent.std == pytest.approx(0.3)
    assert agent.var == pytest.approx(0.09)


def test_log_std_is_clamped():
    agent, _ = _bandit_agent(init_log_std=99.0)
    assert agent.y == pytest.approx(agent.log_std_bounds[1])
    agent2, _ = _bandit_agent(init_log_std=-99.0)
    assert agent2.y == pytest.approx(agent2.log_std_bounds[0])


# -- Gaussian score functions ----------------------------------------------------
def _log_pi(a, mu, y, k):
    var = np.exp(2 * y)
    return -0.5 * np.sum((a - mu) ** 2) / var - k * y - 0.5 * k * np.log(2 * np.pi)


def test_score_matches_finite_difference():
    agent, _ = _bandit_agent(m=3, init_log_std=0.2)
    mu = np.array([0.1, -0.4, 0.7])
    a = np.array([0.5, 0.0, 1.0])
    grad_mu, grad_y = agent._grad_log_pi(a, mu)

    eps = 1e-6
    for i in range(3):
        da = np.zeros(3)
        da[i] = eps
        # d(log pi)/d(mu_i) = (a_i - mu_i)/var = grad_mu[i].
        fd = (_log_pi(a, mu + da, agent.y, 3) - _log_pi(a, mu - da, agent.y, 3)) / (2 * eps)
        assert grad_mu[i] == pytest.approx(fd, rel=1e-4)

    fd_y = (_log_pi(a, mu, agent.y + eps, 3) - _log_pi(a, mu, agent.y - eps, 3)) / (2 * eps)
    assert grad_y == pytest.approx(fd_y, rel=1e-4)


# -- batch critic ----------------------------------------------------------------
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


def test_critic_recovers_reward_linear_in_compatible_features():
    # If reward is exactly linear in [grad_mu, grad_y, 1], least squares recovers it.
    agent, _ = _bandit_agent(m=4, actor_lr=0.0)  # actor_lr 0 => theta, y frozen
    c_mu = np.array([1.0, -2.0, 0.5, 3.0])
    c_y, bias = -0.75, 7.0
    var = agent.var
    for _ in range(agent.batch_size):
        a = agent.act(None)
        grad_mu = (a - agent.theta) / var
        grad_y = float((a - agent.theta) @ (a - agent.theta)) / var - agent.action_dim
        reward = float(grad_mu @ c_mu) + c_y * grad_y + bias
        agent.update(None, a, reward, None, False)
    np.testing.assert_allclose(agent.w_mu, c_mu, atol=1e-7)
    assert agent.w_y == pytest.approx(c_y, abs=1e-7)
    assert agent.baseline == pytest.approx(bias, abs=1e-7)


# -- linear (Pendulum) mode ------------------------------------------------------
def _linear_agent(**overrides):
    n = feature_dim(5)
    cfg = dict(actor_lr=0.01, critic_lr=0.1, seed=0, gamma=0.9, init_log_std=0.0)
    cfg.update(overrides)
    return StochasticActorCritic(
        1, mode="linear",
        feature_fn=lambda s: pendulum_phi(s, centres_per_dim=5), feature_dim=n, **cfg,
    )


def test_linear_update_matches_code_md():
    agent = _linear_agent()
    rng = np.random.default_rng(1)
    agent.theta = rng.normal(size=agent.theta.shape) * 0.1
    agent.v = rng.normal(size=agent.v.shape) * 0.1
    th, v, y = agent.theta.copy(), agent.v.copy(), agent.y

    s, s2 = np.array([0.6, 0.8, 1.0]), np.array([0.0, 1.0, -2.0])
    a, r = np.array([0.7]), -3.0
    phi, phi2 = pendulum_phi(s, 5), pendulum_phi(s2, 5)
    mu = phi @ th
    var = np.exp(2 * y)
    grad_mu = (a - mu) / var
    grad_y = float((a - mu) @ (a - mu)) / var - 1
    delta = r + 0.9 * (phi2 @ v) - (phi @ v)

    agent.update(s, a, r, s2, False)
    np.testing.assert_allclose(agent.v, v + 0.1 * delta * phi)
    np.testing.assert_allclose(agent.theta, th + 0.01 * delta * np.outer(phi, grad_mu))
    assert agent.y == pytest.approx(y + 0.01 * delta * grad_y)


def test_linear_done_drops_bootstrap():
    a1, a2 = _linear_agent(), _linear_agent()
    a1.v[:] = a2.v[:] = 1.0
    s = np.array([1.0, 0.0, 0.0])
    a1.update(s, np.array([0.0]), 0.0, s, True)
    a2.update(s, np.array([0.0]), 0.0, s, False)
    assert not np.allclose(a1.v, a2.v)


def test_linear_explore_false_returns_mean():
    agent = _linear_agent()
    agent.theta = np.ones_like(agent.theta)
    s = np.array([1.0, 0.0, 0.5])
    expected = pendulum_phi(s, 5) @ agent.theta
    np.testing.assert_array_equal(agent.act(s, explore=False), expected)


# -- robustness / persistence ----------------------------------------------------
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_divergence_is_loud():
    agent = _linear_agent()
    with pytest.raises(FloatingPointError):
        agent.update(np.array([1.0, 0.0, 0.0]), np.array([0.0]), np.inf,
                     np.array([1.0, 0.0, 0.0]), False)


def test_runaway_but_finite_params_are_loud():
    # A huge-but-finite blow-up must raise, not log garbage (the 1/var pathology).
    agent, env = _bandit_agent(m=3)
    agent.theta = np.full(3, 1e9)  # finite but way past PARAM_ABS_MAX
    obs, _ = env.reset()
    with pytest.raises(FloatingPointError):
        for _ in range(agent.batch_size):
            a = agent.act(obs)
            agent.update(obs, a, -env.cost(a), obs, False)


def test_save_load_roundtrip_bandit(tmp_path):
    agent, env = _bandit_agent()
    _run_bandit(agent, env, 200)
    path = str(tmp_path / "sac_bandit.npz")
    agent.save(path)
    fresh, _ = _bandit_agent()
    fresh.load(path)
    np.testing.assert_array_equal(fresh.theta, agent.theta)
    assert fresh.y == pytest.approx(agent.y)


def test_save_load_roundtrip_linear(tmp_path):
    agent = _linear_agent()
    agent.update(np.array([0.3, 0.9, 1.0]), np.array([0.2]), -1.0,
                 np.array([0.1, 0.0, 0.5]), False)
    path = str(tmp_path / "sac_linear.npz")
    agent.save(path)
    fresh = _linear_agent()
    fresh.load(path)
    np.testing.assert_array_equal(fresh.theta, agent.theta)
    np.testing.assert_array_equal(fresh.v, agent.v)
    assert fresh.y == pytest.approx(agent.y)


def test_factory_builds_pendulum_linear_mode():
    import gymnasium

    from common.wrappers import ClipAction

    env = ClipAction(gymnasium.make("Pendulum-v1"))
    cfg = dict(actor_lr=0.001, critic_lr=0.01, seed=0, action_dim=1, centres_per_dim=5)
    agent = make_stochastic_ac("pendulum", env, cfg)
    assert agent.mode == "linear"
    assert agent.theta.shape == (feature_dim(5), 1)
    env.close()
