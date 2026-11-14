"""T0.5: LunarLanderContinuous-v3 resets and runs 10 steps under seed 0, and
actions outside the legal box are clipped by the wrapper.
"""

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")

from common.wrappers import ClipAction  # noqa: E402


def _make():
    try:
        env = gymnasium.make("LunarLanderContinuous-v3")
    except Exception as exc:  # Box2D/swig not installed
        pytest.skip(
            "LunarLanderContinuous-v3 unavailable (Box2D/swig). "
            f"Record in task.md T0.5; next: pip install 'gymnasium[box2d]' after brew install swig. {exc}"
        )
    env.reset(seed=0)
    return ClipAction(env)


def test_lunarlander_resets_and_runs_10_steps():
    env = _make()
    obs, _ = env.reset(seed=0)
    assert obs.shape[0] == 8
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, _, _, _ = env.step(action)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
    env.close()


def test_wrapper_clips_out_of_range_action():
    env = _make()
    env.reset(seed=0)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    # An action well outside the legal box must be clipped before stepping.
    _, clipped_reward, *_ = env.step(high)
    env.reset(seed=0)
    _, unclipped_reward, *_ = env.step(np.full_like(high, 100.0))
    assert clipped_reward == pytest.approx(unclipped_reward)
    env.close()
