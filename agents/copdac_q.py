"""COPDAC-Q -- off-policy deterministic actor-critic with a compatible linear critic.

Silver et al. 2014, section 4.3, equations 19--22. Natural gradient (end of 4.3) is
the constructor flag ``natural_gradient``, not a separate agent.

Target policy   mu_theta(s) = theta^T phi(s)        (bandit: phi = 1, mu = theta)
Behaviour       a ~ Normal(mu_theta(s), sigma_beta^2 I)
Critic          Q(s, a) = (a - mu(s))^T (grad_theta mu(s))^T w + v^T phi(s)

With theta of shape (n, k) the compatible feature grad_theta mu(s) (a - mu(s)) is the
outer product phi(s) (a - mu(s))^T, so w has the same (n, k) shape as theta, and
grad_theta mu(s) grad_theta mu(s)^T w = phi(s) (phi(s)^T w).

Two modes:

- ``bandit``: the critic is refit by least squares every ``batch_size`` (= 2m)
  steps on the features (a - theta) plus a bias, regressed onto the observed
  rewards (reward = -cost, so w estimates the reward gradient and the actor
  ascends it). One actor step per batch (paper section 5.1). Since grad mu = I,
  the natural and ordinary updates coincide here.
- ``linear``: incremental TD updates on Pendulum with features ``phi(s)``
  (equations 19--22). The bootstrap uses mu(s'), whose advantage term is zero, so
  Q(s', mu(s')) = V(s').

No importance ratio appears anywhere: the deterministic gradient does not
integrate over actions (paper section 4.2).
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np

AGENT_NAME = "copdac_q"

FeatureFn = Callable[[np.ndarray], np.ndarray]


class CopdacQ:
    """COPDAC-Q agent implementing the shared ``act / update / save / load`` interface."""

    def __init__(
        self,
        action_dim: int,
        *,
        mode: str,
        actor_lr: float,
        sigma_beta: float,
        seed: int,
        critic_lr: float | None = None,
        gamma: float = 0.99,
        natural_gradient: bool = False,
        feature_fn: FeatureFn | None = None,
        feature_dim: int | None = None,
        batch_size: int | None = None,
    ):
        if mode not in ("bandit", "linear"):
            raise ValueError(f"mode must be 'bandit' or 'linear', got {mode!r}")
        self.mode = mode
        self.action_dim = int(action_dim)
        self.actor_lr = float(actor_lr)
        self.sigma_beta = float(sigma_beta)
        self.gamma = float(gamma)
        self.natural_gradient = bool(natural_gradient)
        self.rng = np.random.default_rng(seed)

        k = self.action_dim

        if mode == "bandit":
            self.batch_size = int(batch_size) if batch_size is not None else 2 * k
            if self.batch_size < k + 1:
                raise ValueError("batch_size must be >= action_dim + 1 for the least-squares fit")
            self.theta = np.zeros(k)
            self.w = np.zeros(k)
            self.v = np.zeros(1)  # intercept of the batch fit, i.e. V
            self._batch_actions: list[np.ndarray] = []
            self._batch_rewards: list[float] = []
        else:
            if feature_fn is None or feature_dim is None:
                raise ValueError("linear mode needs feature_fn and feature_dim")
            if critic_lr is None:
                raise ValueError("linear mode needs critic_lr")
            self.critic_lr = float(critic_lr)
            self.feature_fn = feature_fn
            n = int(feature_dim)
            self.theta = np.zeros((n, k))
            self.w = np.zeros((n, k))
            self.v = np.zeros(n)

    # -- policy ----------------------------------------------------------------
    def mu(self, state) -> np.ndarray:
        if self.mode == "bandit":
            return self.theta.copy()
        return self.feature_fn(state) @ self.theta

    def act(self, state, explore: bool = True) -> np.ndarray:
        a = self.mu(state)
        if explore:
            a = a + self.sigma_beta * self.rng.standard_normal(self.action_dim)
        # No clip here: the env wrapper clips. Clipping inside the agent makes
        # (a - mu) grow without bound once mu leaves the legal range, and the
        # compatible critic then diverges. The critic instead learns Q of the
        # clipped environment, which is flat beyond the bounds.
        return a

    # -- critic ----------------------------------------------------------------
    def q_value(self, state, action) -> float:
        """Compatible critic Q^w(s, a) = (a - mu(s))^T grad mu(s)^T w + V(s)."""
        action = np.asarray(action, dtype=np.float64).reshape(self.action_dim)
        if self.mode == "bandit":
            return float((action - self.theta) @ self.w + self.v[0])
        phi = self.feature_fn(state)
        advantage = (action - phi @ self.theta) @ (phi @ self.w)
        return float(advantage + phi @ self.v)

    # -- learning --------------------------------------------------------------
    def update(self, state, action, reward, next_state, done: bool) -> None:
        if self.mode == "bandit":
            self._update_bandit(action, reward)
        else:
            self._update_linear(state, action, reward, next_state, done)

    def _update_bandit(self, action, reward) -> None:
        self._batch_actions.append(np.asarray(action, dtype=np.float64).reshape(self.action_dim))
        self._batch_rewards.append(float(reward))
        if len(self._batch_actions) < self.batch_size:
            return

        actions = np.stack(self._batch_actions)
        rewards = np.asarray(self._batch_rewards)
        self._batch_actions.clear()
        self._batch_rewards.clear()

        # Least squares: r ~ (a - theta)^T w + v, bias column last.
        design = np.hstack([actions - self.theta, np.ones((len(rewards), 1))])
        coef, *_ = np.linalg.lstsq(design, rewards, rcond=None)
        self.w = coef[:-1]
        self.v = coef[-1:]

        # grad_theta mu = I, so ordinary and natural updates are both theta += lr * w.
        self.theta = self.theta + self.actor_lr * self.w
        self._check_finite()

    def _update_linear(self, state, action, reward, next_state, done) -> None:
        phi = self.feature_fn(state)
        action = np.asarray(action, dtype=np.float64).reshape(self.action_dim)
        centred = action - phi @ self.theta            # a - mu(s), shape (k,)
        phi_sa = np.outer(phi, centred)                # compatible feature, shape (n, k)

        q_sa = float(np.sum(phi_sa * self.w) + phi @ self.v)
        # Q(s', mu(s')) = V(s'): the advantage term vanishes at a = mu(s').
        q_next = 0.0 if done else float(self.feature_fn(next_state) @ self.v)
        delta = reward + self.gamma * q_next - q_sa

        # Actor first, using w_t (eq. 20 / natural gradient at end of section 4.3).
        if self.natural_gradient:
            actor_step = self.w
        else:
            actor_step = np.outer(phi, phi @ self.w)   # grad mu (grad mu^T w)
        self.theta = self.theta + self.actor_lr * actor_step
        self.w = self.w + self.critic_lr * delta * phi_sa   # eq. 21
        self.v = self.v + self.critic_lr * delta * phi      # eq. 22
        self._check_finite()

    def _check_finite(self) -> None:
        for name in ("theta", "w", "v"):
            if not np.all(np.isfinite(getattr(self, name))):
                raise FloatingPointError(
                    f"COPDAC-Q {name} became non-finite; lower critic_lr (spec.md section 7)"
                )

    # -- persistence -----------------------------------------------------------
    def save(self, path: str) -> None:
        np.savez(path, theta=self.theta, w=self.w, v=self.v)

    def load(self, path: str) -> None:
        if not os.path.exists(path) and os.path.exists(path + ".npz"):
            path = path + ".npz"  # np.savez appends .npz when it is missing
        data = np.load(path)
        self.theta, self.w, self.v = (data[k].copy() for k in ("theta", "w", "v"))


def make_copdac_q(env_name: str, env, config: dict) -> CopdacQ:
    """Build the agent from a resolved config (code.md "Config keys")."""
    common = dict(
        actor_lr=config["actor_lr"],
        sigma_beta=config["sigma_beta"],
        seed=config["seed"],
        gamma=config.get("gamma", 0.99),
        natural_gradient=config.get("natural_gradient", False),
    )
    if env_name == "bandit":
        m = int(config["action_dim"])
        return CopdacQ(m, mode="bandit", batch_size=config.get("batch_size", 2 * m), **common)

    from functools import partial

    from features.rbf import DEFAULT_CENTRES_PER_DIM, feature_dim, pendulum_phi

    if env_name != "pendulum":
        raise ValueError("COPDAC-Q runs on the bandit and Pendulum only")
    k = int(config.get("centres_per_dim", DEFAULT_CENTRES_PER_DIM))
    space = env.action_space
    return CopdacQ(
        int(np.prod(space.shape)),
        mode="linear",
        critic_lr=config["critic_lr"],
        feature_fn=partial(pendulum_phi, centres_per_dim=k),
        feature_dim=feature_dim(k),
        **common,
    )
