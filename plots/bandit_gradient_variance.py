"""Variance of the bandit actor gradient at one fixed parameter (task T2.3).

The parameter is held at the training start, ``theta = 0``. Each estimate is the
actor step one batch of ``2m`` actions would take, divided by the learning rate,
and the parameter is restored afterwards. Both agents explore with the same
standard deviation (default 1, the stochastic policy's initial std) so the gap
is the estimator, not a different noise scale.

Stochastic Actor-Critic's direction is the batch policy gradient from its
compatible least-squares critic. COPDAC-Q's direction is the fitted ``w``
(identity Jacobian). The scalar in the figure is the trace of the empirical
covariance of these vectors across ``n_estimates`` independent batches: the
total variance of the vector estimator on the seed-0 bandit.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig_dpg"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from agents.copdac_q import CopdacQ  # noqa: E402
from agents.stochastic_ac import StochasticActorCritic  # noqa: E402
from envs.bandit import QuadraticBandit  # noqa: E402

DEFAULT_DIMS = (10, 25, 50)
DEFAULT_N = 100
DEFAULT_STD = 1.0
CSV_FIELDS = (
    "action_dim",
    "agent",
    "n_estimates",
    "batch_size",
    "exploration_std",
    "bandit_seed",
    "mean_coord_variance",
    "trace_variance",
    "mean_grad_norm",
)


def batch_actor_gradient(agent, reward_fn) -> np.ndarray:
    """Return one batch actor gradient and restore ``theta`` (and ``y``, if any)."""
    agent._batch_actions.clear()
    agent._batch_rewards.clear()
    theta = agent.theta.copy()
    y = getattr(agent, "y", None)
    for _ in range(agent.batch_size):
        action = agent.act(None, explore=True)
        agent.update(None, action, float(reward_fn(action)), None, False)
    gradient = (agent.theta - theta) / agent.actor_lr
    agent.theta = theta
    if y is not None:
        agent.y = y
    return gradient


def _summarise(agent_name: str, gradients: np.ndarray, *, action_dim: int, exploration_std: float, bandit_seed: int) -> dict:
    centered = gradients - gradients.mean(axis=0)
    trace = float(np.mean(np.sum(centered ** 2, axis=1)))
    return {
        "action_dim": action_dim,
        "agent": agent_name,
        "n_estimates": int(gradients.shape[0]),
        "batch_size": int(action_dim * 2),
        "exploration_std": float(exploration_std),
        "bandit_seed": int(bandit_seed),
        "mean_coord_variance": trace / action_dim,
        "trace_variance": trace,
        "mean_grad_norm": float(np.linalg.norm(gradients.mean(axis=0))),
    }


def gradient_variances(
    action_dims=DEFAULT_DIMS,
    *,
    n_estimates: int = DEFAULT_N,
    exploration_std: float = DEFAULT_STD,
    bandit_seed: int = 0,
    agent_seed: int = 0,
) -> list[dict]:
    """100 (or ``n_estimates``) batch gradients per agent, at ``theta = 0``."""
    if n_estimates < 2:
        raise ValueError("n_estimates must be >= 2 to compute a variance")
    if exploration_std <= 0:
        raise ValueError("exploration_std must be positive")
    rows = []
    for action_dim in action_dims:
        action_dim = int(action_dim)
        if action_dim < 1:
            raise ValueError("action_dim must be >= 1")
        env = QuadraticBandit(action_dim, seed=bandit_seed)
        reward_fn = lambda action, env=env: -env.cost(action)
        agents = (
            (
                "stochastic_ac",
                StochasticActorCritic(
                    action_dim,
                    actor_lr=1.0,
                    seed=agent_seed,
                    init_log_std=float(np.log(exploration_std)),
                    batch_size=2 * action_dim,
                ),
            ),
            (
                "copdac_q",
                CopdacQ(
                    action_dim,
                    actor_lr=1.0,
                    sigma_beta=exploration_std,
                    seed=agent_seed,
                    batch_size=2 * action_dim,
                ),
            ),
        )
        for name, agent in agents:
            gradients = np.stack(
                [batch_actor_gradient(agent, reward_fn) for _ in range(n_estimates)]
            )
            if not np.all(np.isfinite(gradients)):
                raise FloatingPointError(f"{name} gradient at m={action_dim} was non-finite")
            if not np.array_equal(agent.theta, np.zeros(action_dim)):
                raise RuntimeError(f"{name} moved theta; the parameter must stay fixed")
            rows.append(
                _summarise(
                    name,
                    gradients,
                    action_dim=action_dim,
                    exploration_std=exploration_std,
                    bandit_seed=bandit_seed,
                )
            )
    return rows


def write_variance_figure(rows: list[dict], csv_path: str, png_path: str) -> None:
    """Write the summary CSV and the variance-versus-dimension PNG."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in CSV_FIELDS})

    colours = {"stochastic_ac": "C2", "copdac_q": "C0"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for agent in ("stochastic_ac", "copdac_q"):
        series = [row for row in rows if row["agent"] == agent]
        ax.plot(
            [row["action_dim"] for row in series],
            [row["trace_variance"] for row in series],
            marker="o",
            color=colours[agent],
            label=f"{agent} (n={series[0]['n_estimates']})",
        )
    ax.set_xlabel("action dimension m")
    ax.set_ylabel("variance of the θ-gradient (trace)")
    ax.set_title("100 batch gradients at θ = 0, exploration std 1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bandit gradient-variance figure (T2.3).")
    parser.add_argument("--dims", default="10,25,50", help="comma-separated action dimensions")
    parser.add_argument("--n-estimates", type=int, default=DEFAULT_N)
    parser.add_argument("--exploration-std", type=float, default=DEFAULT_STD)
    parser.add_argument("--csv", default="results/bandit_gradient_variance/bandit_gradient_variance.csv")
    parser.add_argument("--png", default="plots/out/bandit_gradient_variance.png")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    dims = tuple(int(part) for part in args.dims.split(",") if part)
    rows = gradient_variances(
        dims,
        n_estimates=args.n_estimates,
        exploration_std=args.exploration_std,
    )
    write_variance_figure(rows, args.csv, args.png)
    for row in rows:
        print(
            f"m={row['action_dim']} {row['agent']}: "
            f"trace={row['trace_variance']:.6g} "
            f"mean_coord={row['mean_coord_variance']:.6g} "
            f"||mean||={row['mean_grad_norm']:.6g}"
        )
    print(f"wrote {args.csv}")
    print(f"wrote {args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
