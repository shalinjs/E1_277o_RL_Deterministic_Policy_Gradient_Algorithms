"""COPDAC-Q -- off-policy deterministic actor-critic with a compatible linear critic.

Silver et al. 2014, section 4.3, equations 19--22, in the bandit batch form of
section 5.1. Natural gradient (end of 4.3) is the constructor flag
``natural_gradient``, not a separate agent. The bandit is the only environment.

Target policy   mu = theta
Behaviour       a ~ Normal(theta, sigma_beta^2 I)
Critic          Q(a) = (a - theta)^T w + v

The critic is refit by least squares every ``batch_size`` (= 2m) steps on the
features (a - theta) plus a bias, regressed onto the observed rewards
(reward = -cost, so w estimates the reward gradient and the actor ascends it).
One actor step per batch. Since grad mu = I, the natural and ordinary updates
coincide; the flag still selects the ``copdac_q_natural`` result folder.

No importance ratio appears anywhere: the deterministic gradient does not
integrate over actions (paper section 4.2).
"""

from __future__ import annotations

import os

import numpy as np

AGENT_NAME = "copdac_q"


class CopdacQ:
    """COPDAC-Q agent implementing the shared ``act / update / save / load`` interface."""

    def __init__(
        self,
        action_dim: int,
        *,
        actor_lr: float,
        sigma_beta: float,
        seed: int,
        gamma: float = 0.99,
        natural_gradient: bool = False,
        batch_size: int | None = None,
    ):
        self.action_dim = int(action_dim)
        self.actor_lr = float(actor_lr)
        self.sigma_beta = float(sigma_beta)
        self.gamma = float(gamma)
        self.natural_gradient = bool(natural_gradient)
        self.rng = np.random.default_rng(seed)

        k = self.action_dim
        self.batch_size = int(batch_size) if batch_size is not None else 2 * k
        if self.batch_size < k + 1:
            raise ValueError("batch_size must be >= action_dim + 1 for the least-squares fit")
        self.theta = np.zeros(k)
        self.w = np.zeros(k)
        self.v = np.zeros(1)  # intercept of the batch fit, i.e. V
        self._batch_actions: list[np.ndarray] = []
        self._batch_rewards: list[float] = []

    # -- policy ----------------------------------------------------------------
    def mu(self, state) -> np.ndarray:
        return self.theta.copy()

    def act(self, state, explore: bool = True) -> np.ndarray:
        a = self.mu(state)
        if explore:
            a = a + self.sigma_beta * self.rng.standard_normal(self.action_dim)
        # No clip here. The bandit action is unbounded. The LunarLander wrapper
        # clips, and this agent does not run there.
        return a

    # -- critic ----------------------------------------------------------------
    def q_value(self, state, action) -> float:
        """Compatible critic Q^w(a) = (a - theta)^T w + v."""
        del state
        action = np.asarray(action, dtype=np.float64).reshape(self.action_dim)
        return float((action - self.theta) @ self.w + self.v[0])

    # -- learning --------------------------------------------------------------
    def update(self, state, action, reward, next_state, done: bool) -> None:
        del state, next_state, done
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

    def _check_finite(self) -> None:
        for name in ("theta", "w", "v"):
            if not np.all(np.isfinite(getattr(self, name))):
                raise FloatingPointError(
                    f"COPDAC-Q {name} became non-finite; lower the step size (spec.md section 7)"
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
    del env  # the bandit is built from action_dim; the registry still passes the env
    if env_name != "bandit":
        raise ValueError(f"COPDAC-Q runs on the bandit only, got {env_name!r}")
    m = int(config["action_dim"])
    return CopdacQ(
        m,
        actor_lr=config["actor_lr"],
        sigma_beta=config["sigma_beta"],
        seed=config["seed"],
        gamma=config.get("gamma", 0.99),
        natural_gradient=config.get("natural_gradient", False),
        batch_size=config.get("batch_size", 2 * m),
    )
