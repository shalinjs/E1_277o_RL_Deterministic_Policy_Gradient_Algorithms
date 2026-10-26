"""T0.3: the quadratic bandit is deterministic from its seed, and the cost matches
a hand calculation at ``a = 0`` and at ``a = a_star``.
"""

import numpy as np
import pytest

from envs.bandit import TARGET_VALUE, QuadraticBandit


def test_cost_matrix_is_deterministic_from_seed():
    a = QuadraticBandit(action_dim=10, seed=0)
    b = QuadraticBandit(action_dim=10, seed=0)
    np.testing.assert_array_equal(a.C, b.C)
    np.testing.assert_array_equal(a.a_star, b.a_star)


def test_different_seeds_give_different_matrices():
    a = QuadraticBandit(action_dim=10, seed=0)
    b = QuadraticBandit(action_dim=10, seed=1)
    assert not np.array_equal(a.C, b.C)


def test_cost_is_zero_at_optimal_action():
    env = QuadraticBandit(action_dim=10, seed=0)
    assert env.cost(env.optimal_action) == pytest.approx(0.0, abs=1e-9)


def test_cost_at_zero_matches_hand_formula():
    env = QuadraticBandit(action_dim=10, seed=0)
    # a = 0  =>  (0 - a_star)^T C (0 - a_star) = a_star^T C a_star
    expected = float(env.a_star @ env.C @ env.a_star)
    assert env.cost(np.zeros(env.action_dim)) == pytest.approx(expected, rel=1e-9)


def test_cost_matrix_is_symmetric_positive_definite():
    env = QuadraticBandit(action_dim=8, seed=3)
    np.testing.assert_allclose(env.C, env.C.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(env.C)
    assert eigvals.min() > 0.0


def test_eigenvalues_come_from_the_specified_set():
    env = QuadraticBandit(action_dim=12, seed=7)
    eigvals = np.sort(np.linalg.eigvalsh(env.C))
    for value in eigvals:
        assert np.isclose(value, 0.1) or np.isclose(value, 1.0)


def test_reward_is_negative_cost():
    env = QuadraticBandit(action_dim=4, seed=0)
    env.reset()
    action = np.array([1.0, 2.0, 3.0, 4.0])
    obs, reward, terminated, truncated, info = env.step(action)
    assert reward == pytest.approx(-env.cost(action))
    assert obs.shape == (1,) and obs[0] == 0.0
    assert truncated is False
    assert terminated is False


def test_termination_at_max_steps():
    env = QuadraticBandit(action_dim=2, seed=0, max_steps=3)
    env.reset()
    flags = [env.step(np.zeros(2))[2] for _ in range(3)]
    assert flags == [False, False, True]


def test_target_value_matches_paper():
    env = QuadraticBandit(action_dim=5, seed=0)
    assert np.all(env.a_star == TARGET_VALUE)
