"""T0.5: Pendulum-v1 resets and runs 100 steps under seed 0, actions outside
[-2, 2] are clipped by the wrapper (not hoped for), and LunarLanderContinuous-v3
either imports or is skipped with the recorded swig reason.
"""

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")

from common.wrappers import ClipAction  # noqa: E402


def _make(env_id):
    env = gymnasium.make(env_id)
    env.reset(seed=0)
    return ClipAction(env)


def test_pendulum_resets_and_runs_100_steps():
    env = _make("Pendulum-v1")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3,)
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
    env.close()


def test_wrapper_clips_out_of_range_torque():
    env = _make("Pendulum-v1")
    env.reset(seed=0)
    high = float(env.action_space.high[0])  # 2.0 for Pendulum-v1
    # A torque well outside the legal band must be clipped before stepping.
    obs, reward, *_ = env.step(np.array([100.0], dtype=np.float32))
    assert np.all(np.isfinite(obs)) and np.isfinite(reward)
    # The same action clipped to the bound gives the same reward.
    env.reset(seed=0)
    clipped_reward = env.step(np.array([high], dtype=np.float32))[1]
    env.reset(seed=0)
    unclipped_reward = env.step(np.array([100.0], dtype=np.float32))[1]
    assert clipped_reward == pytest.approx(unclipped_reward)
    env.close()


def test_lunarlander_imports_or_is_skipped():
    try:
        env = _make("LunarLanderContinuous-v3")
    except Exception as exc:  # Box2D/swig not installed
        pytest.skip(
            "LunarLanderContinuous-v3 unavailable (Box2D/swig). "
            f"Record in task.md T0.5; next: pip install 'gymnasium[box2d]' swig. {exc}"
        )
    obs, _ = env.reset(seed=0)
    assert obs.shape[0] == 8
    for _ in range(10):
        env.step(env.action_space.sample())
    env.close()
