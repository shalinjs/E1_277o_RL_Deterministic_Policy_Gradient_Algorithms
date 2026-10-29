"""Learning-curve plots and the results table, generated only by this script.

Reads ``results/<env>/<agent>/seed*.csv``, aggregates seeds into a mean line and a
standard-deviation band, and writes one PNG per environment. A missing seed is a
gap, never a zero. The y-axis is cost for the bandit (lower is better) and return
for the Gymnasium environments (higher is better).
"""

from __future__ import annotations

import argparse
import glob
import os
import tempfile

import numpy as np

# Matplotlib needs a writable config dir and a non-interactive backend.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig_dpg"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common.logging import read_run_csv  # noqa: E402

# Environments where a lower score is better.
LOWER_IS_BETTER_PREFIX = "bandit"  # bandit, bandit_m10, bandit_m25, ...



def aggregate(csv_paths):
    """Return (steps, mean, std) aligned on the common length across seeds.

    Seeds are truncated to the shortest available length rather than zero-filled.
    """
    if not csv_paths:
        raise ValueError("no CSV paths given to aggregate")
    series = []
    steps_ref = None
    for path in csv_paths:
        data = read_run_csv(path)
        series.append(data["eval_return"])
        if steps_ref is None or len(data["step"]) < len(steps_ref):
            steps_ref = data["step"]
    length = min(len(s) for s in series)
    stacked = np.vstack([s[:length] for s in series])
    return steps_ref[:length], stacked.mean(axis=0), stacked.std(axis=0)


def plot_runs(csv_paths, out_path, *, label="run", ylabel="eval_return", title=None):
    """Aggregate the given CSVs and save a single mean line with a std band."""
    steps, mean, std = aggregate(csv_paths)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, mean, label=label)
    ax.fill_between(steps, mean - std, mean + std, alpha=0.25)
    ax.set_xlabel("environment steps")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _agent_dirs(results_dir, env):
    env_dir = os.path.join(results_dir, env)
    if not os.path.isdir(env_dir):
        return []
    return sorted(
        d for d in os.listdir(env_dir) if os.path.isdir(os.path.join(env_dir, d))
    )


def make_plots(results_dir, out_dir):
    """Walk the results tree and write one learning-curve PNG per environment."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    envs = sorted(
        d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))
    ) if os.path.isdir(results_dir) else []

    for env in envs:
        ylabel = "cost (lower is better)" if env.startswith(LOWER_IS_BETTER_PREFIX) else "return (higher is better)"
        fig, ax = plt.subplots(figsize=(7, 4.5))
        drew_any = False
        for agent in _agent_dirs(results_dir, env):
            csvs = sorted(glob.glob(os.path.join(results_dir, env, agent, "seed*.csv")))
            if not csvs:
                continue
            steps, mean, std = aggregate(csvs)
            ax.plot(steps, mean, label=f"{agent} (n={len(csvs)})")
            ax.fill_between(steps, mean - std, mean + std, alpha=0.2)
            drew_any = True
        if not drew_any:
            plt.close(fig)
            continue
        ax.set_xlabel("environment steps")
        ax.set_ylabel(ylabel)
        ax.set_title(env)
        ax.legend()
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{env}_learning_curve.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        written.append(out_path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate DPG learning-curve plots.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="plots/out")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    written = make_plots(args.results_dir, args.out_dir)
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
