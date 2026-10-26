"""Radial-basis features for Pendulum (paper Section 5.2 uses tile/RBF features).

The Pendulum observation is ``(cos theta, sin theta, angular velocity)``. Each axis
is scaled to about [-1, 1]: cosine and sine are already in range, and angular
velocity is divided by its bound (8 rad/s for Pendulum-v1). A regular grid of
centres is laid in the scaled cube, the RBF width is the spacing between adjacent
centres, and a constant bias coordinate is appended.

Pure function: the output depends only on the inputs. The centre grid is memoised
because it is a deterministic function of ``centres_per_dim`` (no fitting).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

ANGULAR_VELOCITY_BOUND = 8.0  # Pendulum-v1 max speed
DEFAULT_CENTRES_PER_DIM = 6


@lru_cache(maxsize=8)
def _grid_and_width(centres_per_dim: int):
    if not 2 <= centres_per_dim <= 12:
        raise ValueError("centres_per_dim should be a small integer >= 2")
    axis = np.linspace(-1.0, 1.0, centres_per_dim)
    mesh = np.meshgrid(axis, axis, axis, indexing="ij")
    grid = np.stack([m.ravel() for m in mesh], axis=1)  # (k^3, 3)
    width = 2.0 / (centres_per_dim - 1)  # spacing between adjacent centres
    return grid, width


def feature_dim(centres_per_dim: int = DEFAULT_CENTRES_PER_DIM) -> int:
    """Length of the feature vector, including the bias coordinate."""
    return centres_per_dim**3 + 1


def pendulum_phi(
    state: np.ndarray, centres_per_dim: int = DEFAULT_CENTRES_PER_DIM
) -> np.ndarray:
    """Map a Pendulum observation to a fixed RBF vector with a trailing bias of 1."""
    s = np.asarray(state, dtype=np.float64).reshape(3)
    scaled = np.array([s[0], s[1], s[2] / ANGULAR_VELOCITY_BOUND], dtype=np.float64)
    grid, width = _grid_and_width(centres_per_dim)
    sq_dist = np.sum((grid - scaled) ** 2, axis=1)
    feats = np.exp(-sq_dist / (2.0 * width * width))
    return np.concatenate([feats, np.ones(1, dtype=np.float64)])


if __name__ == "__main__":  # manual check (task T0.4)
    phi = pendulum_phi(np.array([1.0, 0.0, 0.0]))
    print(f"feature_dim(6) = {feature_dim(6)}, produced = {phi.shape[0]}")
    print(f"bias coordinate = {phi[-1]}")
