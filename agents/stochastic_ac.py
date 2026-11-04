"""Stochastic Actor-Critic (Gaussian) -- the 2014 on-policy baseline.

Silver et al. 2014, section 2.2 (stochastic policy gradient) with the compatible
critic of Sutton et al. 2000. This is the stochastic foil the paper compares the
deterministic methods against (section 5.1).

Never abbreviate this agent as "SAC"; that name means Soft Actor-Critic (2018),
which is out of scope.

Policy
------
Isotropic Gaussian with a mean parameter and one shared log-standard-deviation:

    pi(a | s) = Normal(mu_theta(s), var * I),   std = exp(y),   var = std^2

``y`` is a scalar ``log_std`` (``exp(y) * I`` in code.md), clamped to a wide finite
interval so ``exp`` cannot overflow (a numerical guard, not a change of algorithm).

Score functions (analytic, not finite differences):

    grad_mu  log pi = (a - mu) / var                 # shape (k,)
    grad_y   log pi = ||a - mu||^2 / var - k         # scalar, k = action_dim

Two modes
---------
- ``bandit``: compatible features ``[grad_mu, grad_y]`` plus a bias column of ones
  are least-squares fit to the observed rewards every ``batch_size`` (= 2m) steps
  (reward = -cost, so the actor ascends reward). The fitted bias is the state-value
  baseline; it drops out of the actor gradient because the score is zero-mean, so
  only the compatible part is used as the advantage. One actor step per batch
  (paper section 5.1).
- ``linear``: incremental TD on Pendulum with a linear value ``V(s) = v^T phi(s)``
  and RBF features (code.md "Stochastic Actor-Critic, Pendulum").

On-policy: there is no importance ratio and no behaviour width; exploration comes
from the policy's own ``std``.
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np

AGENT_NAME = "stochastic_ac"

# Numerical guard on log_std so exp(y) neither overflows nor underflows to 0.
LOG_STD_BOUNDS = (-20.0, 2.0)

# Runaway guard: the stochastic policy gradient has a 1/var factor, so a run can
# explode to huge-but-finite parameters (not caught by isfinite). Treat anything
# past this as a divergence and fail loudly rather than logging garbage.
PARAM_ABS_MAX = 1e8

FeatureFn = Callable[[np.ndarray], np.ndarray]


class StochasticActorCritic:
    """Gaussian actor-critic with the shared ``act / update / save / load`` interface."""

    def __init__(
        self,
        action_dim: int,
        *,
        mode: str,
        actor_lr: float,
        seed: int,
        critic_lr: float | None = None,
        gamma: float = 0.99,
        init_log_std: float = 0.0,
        log_std_bounds: tuple[float, float] = LOG_STD_BOUNDS,
        feature_fn: FeatureFn | None = None,
        feature_dim: int | None = None,
        batch_size: int | None = None,
    ):
        if mode not in ("bandit", "linear"):
            raise ValueError(f"mode must be 'bandit' or 'linear', got {mode!r}")
        self.mode = mode
        self.action_dim = int(action_dim)
        self.actor_lr = float(actor_lr)
        self.gamma = float(gamma)
        self.log_std_bounds = (float(log_std_bounds[0]), float(log_std_bounds[1]))
        self.y = self._clamp_y(float(init_log_std))  # scalar log_std
        self.rng = np.random.default_rng(seed)

        k = self.action_dim

        if mode == "bandit":
            self.batch_size = int(batch_size) if batch_size is not None else 2 * k
            # Compatible features are [grad_mu (k), grad_y (1)] plus a bias, so the
            # least-squares design has k + 2 columns; need at least that many rows.
            if self.batch_size < k + 2:
                raise ValueError("batch_size must be >= action_dim + 2 for the least-squares fit")
            self.theta = np.zeros(k)                 # policy mean
            self.w_mu = np.zeros(k)                  # critic weight on grad_mu
            self.w_y = 0.0                           # critic weight on grad_y
            self.baseline = 0.0                      # fitted bias = V
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
            self.theta = np.zeros((n, k))            # mu(s) = theta^T phi(s)
            self.v = np.zeros(n)                     # V(s) = v^T phi(s)

    # -- policy ----------------------------------------------------------------
    def _clamp_y(self, y: float) -> float:
        lo, hi = self.log_std_bounds
        return float(min(max(y, lo), hi))

    @property
    def std(self) -> float:
        return float(np.exp(self.y))

    @property
    def var(self) -> float:
        return float(np.exp(2.0 * self.y))

    def mu(self, state) -> np.ndarray:
        if self.mode == "bandit":
            return self.theta.copy()
        return self.feature_fn(state) @ self.theta

    def act(self, state, explore: bool = True) -> np.ndarray:
        a = self.mu(state)
        if explore:
            a = a + self.std * self.rng.standard_normal(self.action_dim)
        # No clip here: the env wrapper clips. (See code.md -- clipping in the agent
        # makes the compatible score blow up once mu leaves the legal range.)
        return a

    # -- score functions -------------------------------------------------------
    def _grad_log_pi(self, action: np.ndarray, mu: np.ndarray) -> tuple[np.ndarray, float]:
        """Return (grad_mu log pi, grad_y log pi) at ``action`` under mean ``mu``."""
        var = self.var
        centred = action - mu
        grad_mu = centred / var
        grad_y = float(centred @ centred) / var - self.action_dim
        return grad_mu, grad_y

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

        var = self.var
        mu = self.theta
        grad_mu = (actions - mu) / var                       # (N, k)
        grad_y = np.sum((actions - mu) ** 2, axis=1) / var - self.action_dim  # (N,)

        # Compatible critic: reward ~ [grad_mu, grad_y] . w + baseline.
        design = np.hstack([grad_mu, grad_y[:, None], np.ones((len(rewards), 1))])
        coef, *_ = np.linalg.lstsq(design, rewards, rcond=None)
        self.w_mu = coef[:self.action_dim]
        self.w_y = float(coef[self.action_dim])
        self.baseline = float(coef[-1])

        # Advantage uses the compatible part only; the zero-mean score makes the
        # baseline drop out of the expected gradient (pure variance reduction).
        advantage = grad_mu @ self.w_mu + grad_y * self.w_y   # (N,)
        grad_theta = (grad_mu * advantage[:, None]).mean(axis=0)
        grad_y_step = float((grad_y * advantage).mean())

        self.theta = self.theta + self.actor_lr * grad_theta
        self.y = self._clamp_y(self.y + self.actor_lr * grad_y_step)
        self._check_finite()

    def _update_linear(self, state, action, reward, next_state, done) -> None:
        phi = self.feature_fn(state)
        action = np.asarray(action, dtype=np.float64).reshape(self.action_dim)
        mu = phi @ self.theta                                 # (k,)
        grad_mu, grad_y = self._grad_log_pi(action, mu)

        value = float(phi @ self.v)
        next_value = 0.0 if done else float(self.feature_fn(next_state) @ self.v)
        delta = reward + self.gamma * next_value - value

        self.v = self.v + self.critic_lr * delta * phi                 # critic
        self.theta = self.theta + self.actor_lr * delta * np.outer(phi, grad_mu)
        self.y = self._clamp_y(self.y + self.actor_lr * delta * grad_y)
        self._check_finite()

    def _check_finite(self) -> None:
        arrays = {"theta": self.theta, "y": self.y}
        arrays["v" if self.mode == "linear" else "w_mu"] = (
            self.v if self.mode == "linear" else self.w_mu
        )
        for name, value in arrays.items():
            if not np.all(np.isfinite(value)):
                raise FloatingPointError(
                    f"Stochastic AC {name} became non-finite; lower actor_lr/critic_lr"
                )
            if np.max(np.abs(value)) > PARAM_ABS_MAX:
                raise FloatingPointError(
                    f"Stochastic AC {name} exceeded {PARAM_ABS_MAX:g} (diverging); "
                    "lower actor_lr (the 1/var score term is amplifying the step)"
                )

    # -- persistence -----------------------------------------------------------
    def save(self, path: str) -> None:
        extra = {"v": self.v} if self.mode == "linear" else {}
        np.savez(path, theta=self.theta, log_std=np.array([self.y]), **extra)

    def load(self, path: str) -> None:
        if not os.path.exists(path) and os.path.exists(path + ".npz"):
            path = path + ".npz"  # np.savez appends .npz when it is missing
        data = np.load(path)
        self.theta = data["theta"].copy()
        self.y = self._clamp_y(float(data["log_std"][0]))
        if self.mode == "linear":
            self.v = data["v"].copy()


def make_stochastic_ac(env_name: str, env, config: dict) -> StochasticActorCritic:
    """Build the agent from a resolved config (code.md "Config keys")."""
    common = dict(
        actor_lr=config["actor_lr"],
        seed=config["seed"],
        gamma=config.get("gamma", 0.99),
        init_log_std=config.get("init_log_std", 0.0),
    )
    if env_name == "bandit":
        m = int(config["action_dim"])
        return StochasticActorCritic(
            m, mode="bandit", batch_size=config.get("batch_size", 2 * m), **common
        )

    from functools import partial

    from features.rbf import DEFAULT_CENTRES_PER_DIM, feature_dim, pendulum_phi

    if env_name != "pendulum":
        raise ValueError("Stochastic Actor-Critic runs on the bandit and Pendulum only")
    k = int(config.get("centres_per_dim", DEFAULT_CENTRES_PER_DIM))
    space = env.action_space
    return StochasticActorCritic(
        int(np.prod(space.shape)),
        mode="linear",
        critic_lr=config["critic_lr"],
        feature_fn=partial(pendulum_phi, centres_per_dim=k),
        feature_dim=feature_dim(k),
        **common,
    )
