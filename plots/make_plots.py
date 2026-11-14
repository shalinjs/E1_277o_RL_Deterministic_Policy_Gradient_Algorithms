"""Learning-curve plots and the results table, generated only by this script.

Reads ``results/<env>/<agent>/seed<k>.csv``, aggregates seeds into a mean line and a
standard-deviation band, and writes one PNG per environment plus ``results_table.md``.
A missing seed is a gap, never a zero. The y-axis is cost for the bandit (lower is
better) and return for LunarLander (higher is better).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile

import numpy as np

# Allow both `python -m plots.make_plots` and `python plots/make_plots.py` by
# putting the repository root on the path before importing the project packages.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

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


LOG_BAND_FLOOR = 1e-6  # lower edge of the std band on a log axis (cost >= 0)
TABLE_NAME = "results_table.md"
# spec.md section 3: steps until the LunarLander mean return exceeds this value.
RETURN_THRESHOLD = 200.0


def _seed_id(path: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    return int(stem[4:])


def seed_csvs(results_dir, env, agent):
    """Seed logs only. ``seed0_episodes.csv`` is not a learning curve."""
    paths = [
        path
        for path in glob.glob(os.path.join(results_dir, env, agent, "seed*.csv"))
        if os.path.splitext(os.path.basename(path))[0][4:].isdigit()
    ]
    return sorted(paths, key=_seed_id)


def _axis_is_log(env: str, log_y: bool) -> bool:
    """Log scale is for bandit cost. LunarLander return crosses zero and stays linear."""
    return bool(log_y) and env.startswith(LOWER_IS_BETTER_PREFIX)


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def summarize_runs(env, agent, csvs):
    """One table row. Std is the same population std drawn on the learning curve."""
    steps, mean, std = aggregate(csvs)
    last = len(steps) - 1
    walls = []
    for path in csvs:
        data = read_run_csv(path)
        walls.append(float(data["wall_time_sec"][last]))
    walls = np.asarray(walls, dtype=float)
    if env.startswith(LOWER_IS_BETTER_PREFIX):
        threshold = "not defined"
    else:
        hits = np.flatnonzero(mean > RETURN_THRESHOLD)
        threshold = str(int(steps[hits[0]])) if hits.size else "never"
    return {
        "environment": env,
        "agent": agent,
        "seeds": ",".join(str(_seed_id(path)) for path in csvs),
        "n": str(len(csvs)),
        "final_mean": _fmt(float(mean[last])),
        "final_std": _fmt(float(std[last])),
        "wall_mean_sec": _fmt(float(walls.mean())),
        "wall_std_sec": _fmt(float(walls.std())),
        "threshold_step": threshold,
    }


def write_results_table(rows, out_dir) -> str:
    """Markdown table. Numbers come from the CSVs; a missing seed is left out."""
    path = os.path.join(out_dir, TABLE_NAME)
    header = (
        "| environment | agent | seeds | n | final mean | final std "
        "| wall-clock mean (s) | wall-clock std (s) | threshold step |"
    )
    rule = "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    lines = [
        "# Results",
        "",
        "Generated by `plots/make_plots.py` from `results/<env>/<agent>/seed<k>.csv`.",
        "Final mean and both standard deviations are taken across the seeds that exist",
        f"(population std, the same one drawn as the curve's band). "
        f"Threshold step is the first logged step at which the mean return exceeds "
        f"{RETURN_THRESHOLD:g}. Bandit rows have no threshold. "
        "A seed with no CSV is omitted, not filled with zero.",
        "",
        header,
        rule,
    ]
    for row in rows:
        lines.append(
            "| {environment} | {agent} | {seeds} | {n} | {final_mean} | {final_std} "
            "| {wall_mean_sec} | {wall_std_sec} | {threshold_step} |".format(**row)
        )
    if not rows:
        lines.append("| | | | | | | | | |")
    lines.append("")
    with open(path, "w") as handle:
        handle.write("\n".join(lines))
    return path


def make_plots(results_dir, out_dir, log_y=False):
    """Walk the results tree. Write one PNG per environment and one results table.

    ``log_y`` draws bandit cost on a log scale. LunarLander stays linear, because
    its return is negative for much of training. The std band's lower edge on a
    log axis is clamped to a small positive floor so the band stays drawable.
    """
    os.makedirs(out_dir, exist_ok=True)
    written = []
    rows = []
    envs = sorted(
        d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))
    ) if os.path.isdir(results_dir) else []

    for env in envs:
        use_log = _axis_is_log(env, log_y)
        ylabel = "cost (lower is better)" if env.startswith(LOWER_IS_BETTER_PREFIX) else "return (higher is better)"
        fig, ax = plt.subplots(figsize=(7, 4.5))
        drew_any = False
        for agent in _agent_dirs(results_dir, env):
            csvs = seed_csvs(results_dir, env, agent)
            if not csvs:
                continue
            steps, mean, std = aggregate(csvs)
            rows.append(summarize_runs(env, agent, csvs))
            lower = np.clip(mean - std, LOG_BAND_FLOOR, None) if use_log else mean - std
            ax.plot(steps, mean, label=f"{agent} (n={len(csvs)})")
            ax.fill_between(steps, lower, mean + std, alpha=0.2)
            drew_any = True
        if not drew_any:
            plt.close(fig)
            continue
        if use_log:
            ax.set_yscale("log")
            ylabel += ", log scale"
        ax.set_xlabel("environment steps")
        ax.set_ylabel(ylabel)
        ax.set_title(env)
        ax.legend()
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{env}_learning_curve.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        written.append(out_path)
    written.append(write_results_table(rows, out_dir))
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate DPG learning-curve plots.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="plots/out")
    parser.add_argument("--log-y", action="store_true", help="log-scale y-axis (good for the bandit)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    written = make_plots(args.results_dir, args.out_dir, log_y=args.log_y)
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
