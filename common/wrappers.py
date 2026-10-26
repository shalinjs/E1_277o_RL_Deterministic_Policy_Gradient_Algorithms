"""A thin action-clipping wrapper, applied to every Gymnasium environment.

It clips the action to the environment's bounds before stepping and leaves the
reward untouched (code.md).
"""

from __future__ import annotations

import numpy as np


class ClipAction:
    """Clip actions to ``action_space`` bounds; pass everything else through."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        low = self.action_space.low
        high = self.action_space.high
        clipped = np.clip(np.asarray(action, dtype=self.action_space.dtype), low, high)
        return self.env.step(clipped)

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        # Only reached for attributes not set on the wrapper itself.
        return getattr(self.env, name)
