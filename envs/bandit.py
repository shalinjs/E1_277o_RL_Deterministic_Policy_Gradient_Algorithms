"""Continuous quadratic bandit (paper Section 5.1).

No state. The agent chooses an action ``a`` in R^m and is punished by a quadratic
cost ``(a - a_star)^T C (a - a_star)``; the reward is the negative cost. The matrix
``C`` is fixed by the seed, so a seed fully determines the problem instance.

A dummy one-dimensional zero observation is exposed so the shared agent interface
does not have to branch between this environment and the Gymnasium ones.
"""

from __future__ import annotations

import numpy as np

try:  # the project depends on gymnasium; this matches code.md (spaces are Box)
    from gymnasium.spaces import Box
except ImportError as exc:  # pragma: no cover - gymnasium is a hard dependency
    raise ImportError(
        "gymnasium is required for envs.bandit (see requirements.txt)"
    ) from exc

TARGET_VALUE = 4.0  # a_star = [4, ..., 4], paper Section 5.1
DEFAULT_MAX_STEPS = 10_000


def _build_cost_matrix(action_dim: int, rng: np.random.Generator) -> np.ndarray:
    """C = Q diag(d) Q^T, with Q an orthogonal matrix and d_i drawn from {0.1, 1}.

    The sign of each column of Q is fixed so that the diagonal of R is positive,
    which makes the QR factor unique and therefore reproducible from the seed.
    """
    gaussian = rng.standard_normal((action_dim, action_dim))
    q, r = np.linalg.qr(gaussian)
    q = q * np.sign(np.diag(r))  # unique orthogonal factor
    d = rng.choice(np.array([0.1, 1.0]), size=action_dim)
    return (q * d) @ q.T  # == q @ diag(d) @ q.T, symmetric positive definite


class QuadraticBandit:
    """Gymnasium-like quadratic-cost bandit with a fixed, seed-determined ``C``."""

    def __init__(self, action_dim: int, seed: int, max_steps: int = DEFAULT_MAX_STEPS):
        if action_dim < 1:
            raise ValueError("action_dim must be >= 1")
        self.action_dim = int(action_dim)
        self.max_steps = int(max_steps)

        rng = np.random.default_rng(seed)
        self.C = _build_cost_matrix(self.action_dim, rng)
        self.a_star = np.full(self.action_dim, TARGET_VALUE, dtype=np.float64)

        # A single constant observation; agents must not learn from it (code.md).
        self.observation_space = Box(low=0.0, high=0.0, shape=(1,), dtype=np.float32)
        self.action_space = Box(
            low=-np.inf, high=np.inf, shape=(self.action_dim,), dtype=np.float32
        )
        self._t = 0

    # -- Gymnasium-like API --------------------------------------------------
    def reset(self, *, seed: int | None = None, options=None):
        """Reset the step counter. ``C`` is not rebuilt; it is fixed at construction."""
        self._t = 0
        return self._observation(), {}

    def step(self, action):
        cost = self.cost(action)
        reward = -cost
        self._t += 1
        terminated = self._t >= self.max_steps
        truncated = False
        return self._observation(), reward, terminated, truncated, {}

    # -- Helpers used by the runner and the tests ----------------------------
    def cost(self, action) -> float:
        """Quadratic cost of an action. Defined for every real vector (no clip)."""
        a = np.asarray(action, dtype=np.float64).reshape(self.action_dim)
        diff = a - self.a_star
        return float(diff @ self.C @ diff)

    @property
    def optimal_action(self) -> np.ndarray:
        return self.a_star.copy()

    @staticmethod
    def _observation() -> np.ndarray:
        return np.zeros(1, dtype=np.float32)


if __name__ == "__main__":  # a small manual check (task T0.3)
    env = QuadraticBandit(action_dim=10, seed=0)
    print(f"action_dim = {env.action_dim}")
    print(f"cost at a=0       : {env.cost(np.zeros(env.action_dim)):.6f}")
    print(f"cost at a=a_star  : {env.cost(env.optimal_action):.6f}")
