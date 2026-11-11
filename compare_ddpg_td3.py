"""Compare the completed DDPG and TD3 LunarLander evaluation runs."""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig_ddpg_td3"))
import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT_DIR = (
    PROJECT_ROOT / "results" / "lunarlander_ddpg_td3_comparison_3seeds_200k"
)
REQUIRED_COLUMNS = {"step", "eval_return"}
SEEDS = (0, 1, 2)
INPUT_FILES = {
    ("DDPG", 0): Path("results/lunarlander/ddpg/seed0.csv"),
    ("DDPG", 1): Path("results/lunarlander/ddpg/seed1.csv"),
    ("DDPG", 2): Path("results/lunarlander/ddpg/seed2.csv"),
    ("TD3", 0): Path("results/lunarlander/td3/comparison_seed0.csv"),
    ("TD3", 1): Path("results/lunarlander/td3/comparison_seed1.csv"),
    # This explicit mapping intentionally excludes the interrupted seed-2 run.
    ("TD3", 2): Path("results/lunarlander/td3/comparison_seed2_rerun.csv"),
}
ALGORITHM_COLORS = {"DDPG": "#176B87", "TD3": "#D5683A"}
CONFIG_FIELDS = (
    "environment_name",
    "steps",
    "eval_every",
    "eval_episodes",
    "actor_lr",
    "critic_lr",
    "gamma",
    "tau",
    "replay_capacity",
    "batch_size",
    "hidden_dims",
    "exploration_std",
    "warmup_steps",
    "target_noise_std",
    "target_noise_clip",
    "policy_delay",
)


@dataclass
class Run:
    algorithm: str
    seed: int
    path: Path
    frame: pd.DataFrame
    config: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze and plot the completed LunarLander DDPG vs TD3 runs."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help="Experiment directory containing results/lunarlander (default: project experiment).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <experiment-dir>/comparison_plots).",
    )
    parser.add_argument(
        "--final-step", type=int, default=200_000, help="Expected final checkpoint."
    )
    parser.add_argument(
        "--eval-every", type=int, default=5_000, help="Expected evaluation interval."
    )
    return parser


def load_runs(experiment_dir: Path) -> list[Run]:
    runs = []
    errors = []
    for (algorithm, seed), relative_path in INPUT_FILES.items():
        path = experiment_dir / relative_path
        if not path.is_file():
            errors.append(f"Missing {algorithm} seed {seed} CSV: {path}")
            continue
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
            errors.append(f"Could not read {algorithm} seed {seed} CSV {path}: {error}")
            continue

        config_path = path.with_suffix(".yaml")
        config: dict[str, Any] = {}
        if config_path.is_file():
            try:
                loaded_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                if loaded_config is not None and not isinstance(loaded_config, dict):
                    raise ValueError("YAML root must be a mapping")
                config = loaded_config or {}
            except (OSError, yaml.YAMLError, ValueError) as error:
                errors.append(f"Could not read config {config_path}: {error}")
                continue
        runs.append(Run(algorithm, seed, path, frame, config))

    if errors:
        raise ValueError("Input loading failed:\n- " + "\n- ".join(errors))
    return runs


def validate_runs(
    runs: list[Run], *, final_step: int, eval_every: int
) -> np.ndarray:
    if eval_every <= 0 or final_step <= 0 or final_step % eval_every:
        raise ValueError("final-step and eval-every must be positive, with final-step divisible by eval-every")

    errors = []
    expected_keys = {(algorithm, seed) for algorithm in ALGORITHM_COLORS for seed in SEEDS}
    actual_keys = {(run.algorithm, run.seed) for run in runs}
    missing = sorted(expected_keys - actual_keys)
    if missing:
        errors.append(f"Missing runs: {missing}")
    if actual_keys - expected_keys:
        errors.append(f"Unexpected runs: {sorted(actual_keys - expected_keys)}")

    expected_steps = np.arange(eval_every, final_step + 1, eval_every, dtype=np.int64)
    for run in runs:
        label = f"{run.algorithm} seed {run.seed} ({run.path})"
        missing_columns = REQUIRED_COLUMNS - set(run.frame.columns)
        if missing_columns:
            errors.append(f"{label}: missing required columns {sorted(missing_columns)}")
            continue
        if run.frame.empty:
            errors.append(f"{label}: CSV contains no evaluation rows")
            continue

        for column in ("step", "eval_return"):
            converted = pd.to_numeric(run.frame[column], errors="coerce")
            if converted.isna().any() or not np.isfinite(converted.to_numpy(dtype=float)).all():
                errors.append(f"{label}: column '{column}' contains missing, non-numeric, or non-finite values")
            else:
                run.frame[column] = converted
        if run.frame[["step", "eval_return"]].isna().any().any():
            continue

        steps = run.frame["step"].to_numpy(dtype=float)
        if not np.equal(steps, np.floor(steps)).all():
            errors.append(f"{label}: evaluation steps must be integers")
            continue
        steps = steps.astype(np.int64)
        if len(np.unique(steps)) != len(steps):
            errors.append(f"{label}: duplicate evaluation steps found")
        if not np.array_equal(steps, np.sort(steps)):
            errors.append(f"{label}: evaluation steps are not in ascending order")
        if not np.array_equal(steps, expected_steps):
            errors.append(
                f"{label}: expected {len(expected_steps)} checkpoints "
                f"({eval_every} through {final_step}), found {len(steps)}; "
                f"missing={np.setdiff1d(expected_steps, steps).tolist()}, "
                f"unexpected={np.setdiff1d(steps, expected_steps).tolist()}"
            )

        configured_seed = run.config.get("seed")
        if configured_seed is not None:
            try:
                parsed_seed = int(configured_seed)
            except (TypeError, ValueError):
                errors.append(f"{label}: config seed must be an integer, got {configured_seed!r}")
            else:
                if parsed_seed != run.seed:
                    errors.append(f"{label}: config seed is {configured_seed}, expected {run.seed}")
        configured_agent = run.config.get("agent")
        if configured_agent is not None and str(configured_agent).lower() != run.algorithm.lower():
            errors.append(f"{label}: config agent is {configured_agent}, expected {run.algorithm}")
        for key, expected in (("steps", final_step), ("eval_every", eval_every)):
            configured = run.config.get(key)
            if configured is not None:
                try:
                    parsed_value = int(configured)
                except (TypeError, ValueError):
                    errors.append(f"{label}: config {key} must be an integer, got {configured!r}")
                else:
                    if parsed_value != expected:
                        errors.append(f"{label}: config {key} is {configured}, expected {expected}")

    if errors:
        raise ValueError("Validation failed:\n- " + "\n- ".join(errors))

    reference_steps = runs[0].frame["step"].to_numpy(dtype=np.int64)
    for run in runs[1:]:
        if not np.array_equal(run.frame["step"].to_numpy(dtype=np.int64), reference_steps):
            raise ValueError(
                f"Evaluation steps are not aligned: {runs[0].path} and {run.path} differ"
            )
    return expected_steps


def calculate_statistics(runs: list[Run], *, late_checkpoint_count: int) -> pd.DataFrame:
    summary = []
    for run in runs:
        rewards = run.frame["eval_return"]
        summary.append(
            {
                "Algorithm": run.algorithm,
                "Seed": run.seed,
                "Final evaluation reward": float(rewards.iloc[-1]),
                "Mean reward over final five checkpoints": float(rewards.tail(late_checkpoint_count).mean()),
                "Maximum evaluation reward": float(rewards.max()),
                "Overall mean evaluation reward": float(rewards.mean()),
                "Standard deviation": float(rewards.std(ddof=1)),
            }
        )
    return pd.DataFrame(summary).sort_values(["Algorithm", "Seed"]).reset_index(drop=True)


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DCE3E7", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#34434B")
    ax.xaxis.label.set_color("#26343B")
    ax.yaxis.label.set_color("#26343B")


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_mean_learning_curves(runs: list[Run], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for algorithm, color in ALGORITHM_COLORS.items():
        selected = [run for run in runs if run.algorithm == algorithm]
        rewards = np.vstack([run.frame["eval_return"].to_numpy() for run in selected])
        steps = selected[0].frame["step"].to_numpy()
        mean = rewards.mean(axis=0)
        std = rewards.std(axis=0)
        ax.plot(steps, mean, color=color, linewidth=2.4, label=f"{algorithm} mean (n={len(selected)})")
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
    ax.set_title("DDPG vs TD3: Mean Learning Curves", loc="left", weight="bold", pad=12)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Evaluation episode reward")
    ax.ticklabel_format(axis="x", style="plain")
    ax.legend(frameon=False, loc="best")
    _style_axes(ax)
    _save_figure(fig, output_dir, "mean_learning_curves")


def plot_individual_seed_curves(runs: list[Run], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for run in runs:
        ax.plot(
            run.frame["step"],
            run.frame["eval_return"],
            color=ALGORITHM_COLORS[run.algorithm],
            linestyle="-" if run.algorithm == "DDPG" else "--",
            linewidth=1.65,
            alpha=0.88,
            label=f"{run.algorithm} seed {run.seed}",
        )
    ax.set_title("Individual Seed Learning Curves", loc="left", weight="bold", pad=12)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Evaluation episode reward")
    ax.ticklabel_format(axis="x", style="plain")
    ax.legend(frameon=False, ncol=2, loc="best")
    _style_axes(ax)
    _save_figure(fig, output_dir, "individual_seed_curves")


def _plot_seed_comparison(
    summary: pd.DataFrame,
    output_dir: Path,
    *,
    stem: str,
    title: str,
    value_column: str,
    y_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    algorithm_positions = {"DDPG": 0, "TD3": 1}
    for algorithm, position in algorithm_positions.items():
        values = summary.loc[summary["Algorithm"] == algorithm, value_column].to_numpy()
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        color = ALGORITHM_COLORS[algorithm]
        ax.bar(position, mean, width=0.48, color=color, alpha=0.18, edgecolor=color, linewidth=1.5)
        ax.errorbar(position, mean, yerr=std, fmt="D", color=color, capsize=6, markersize=6,
                    linewidth=1.6, label=f"{algorithm} mean +/- SD")
        offsets = np.linspace(-0.13, 0.13, len(values))
        seed_rows = summary.loc[summary["Algorithm"] == algorithm].sort_values("Seed")
        for offset, (_, row) in zip(offsets, seed_rows.iterrows()):
            x = position + offset
            y = float(row[value_column])
            ax.scatter(x, y, color=color, edgecolor="white", linewidth=0.8, s=60, zorder=3)
            ax.annotate(f"seed {int(row['Seed'])}", (x, y), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=8, color="#34434B")
    ax.set_xticks([0, 1], ["DDPG", "TD3"])
    ax.set_title(title, loc="left", weight="bold", pad=12)
    ax.set_ylabel(y_label)
    ax.legend(frameon=False, loc="best")
    _style_axes(ax)
    _save_figure(fig, output_dir, stem)


def plot_final_comparison(summary: pd.DataFrame, output_dir: Path) -> None:
    _plot_seed_comparison(
        summary,
        output_dir,
        stem="final_reward_comparison",
        title="Final Evaluation Reward at 200K Steps",
        value_column="Final evaluation reward",
        y_label="Evaluation episode reward at 200K steps",
    )


def plot_late_training_comparison(summary: pd.DataFrame, output_dir: Path) -> None:
    _plot_seed_comparison(
        summary,
        output_dir,
        stem="late_training_comparison",
        title="Late-Training Reward: Mean of Final Five Checkpoints",
        value_column="Mean reward over final five checkpoints",
        y_label="Mean evaluation episode reward (180K-200K steps)",
    )


def _format_config_value(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(str(part) for part in value) + "]"
    return str(value)


def _configuration_table(runs: list[Run]) -> str:
    rows = ["| Algorithm | Parameter | Value |", "|---|---|---|"]
    for algorithm in ALGORITHM_COLORS:
        selected = [run for run in runs if run.algorithm == algorithm]
        for field in CONFIG_FIELDS:
            values = [_format_config_value(run.config[field]) for run in selected if field in run.config]
            if not values:
                continue
            value = values[0] if len(set(values)) == 1 else "; ".join(
                f"seed {run.seed}: {_format_config_value(run.config[field])}"
                for run in selected if field in run.config
            )
            rows.append(f"| {algorithm} | `{field}` | {value} |")
    return "\n".join(rows) if len(rows) > 2 else "Training YAML metadata was unavailable."


def _summary_markdown(summary: pd.DataFrame) -> str:
    columns = list(summary.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, record in summary.iterrows():
        values = []
        for column, value in zip(columns, record):
            if column == "Seed":
                values.append(str(int(value)))
            elif isinstance(value, (int, float, np.integer, np.floating)):
                values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def generate_report(runs: list[Run], summary: pd.DataFrame, output_path: Path) -> None:
    final_means = summary.groupby("Algorithm")["Final evaluation reward"].mean()
    late_means = summary.groupby("Algorithm")["Mean reward over final five checkpoints"].mean()
    final_winner = str(final_means.idxmax())
    late_winner = str(late_means.idxmax())
    delta_final = float(final_means["TD3"] - final_means["DDPG"])
    delta_late = float(late_means["TD3"] - late_means["DDPG"])
    config_table = _configuration_table(runs)
    result_table = _summary_markdown(summary)
    report = f"""# DDPG vs TD3 on LunarLanderContinuous-v3

## Experimental setup

- Environment: `LunarLanderContinuous-v3`.
- Algorithms: DDPG and TD3, with seeds 0, 1, and 2 per algorithm.
- Training horizon: 200,000 environment steps; evaluations every 5,000 steps (40 checkpoints per run).
- Evaluation metric: mean episode reward from each run's evaluation CSV. Each recorded evaluation uses five episodes according to the saved run configurations.
- TD3 seed 2 uses the completed `comparison_seed2_rerun.csv`; the interrupted `comparison_seed2.csv` was excluded.
- The primary comparison uses evaluation CSVs for both algorithms. DDPG episode-level logs were not available.

## Training configuration

{config_table}

## Results summary

{result_table}

Algorithm-level final reward (mean +/- population SD across three seeds): DDPG {final_means['DDPG']:.2f} +/- {summary.loc[summary['Algorithm'] == 'DDPG', 'Final evaluation reward'].std(ddof=0):.2f}; TD3 {final_means['TD3']:.2f} +/- {summary.loc[summary['Algorithm'] == 'TD3', 'Final evaluation reward'].std(ddof=0):.2f}.

Algorithm-level late-training reward (mean +/- population SD across three seeds): DDPG {late_means['DDPG']:.2f} +/- {summary.loc[summary['Algorithm'] == 'DDPG', 'Mean reward over final five checkpoints'].std(ddof=0):.2f}; TD3 {late_means['TD3']:.2f} +/- {summary.loc[summary['Algorithm'] == 'TD3', 'Mean reward over final five checkpoints'].std(ddof=0):.2f}.

## Key findings

- {final_winner} has the higher mean final-checkpoint reward across these seeds (TD3 minus DDPG: {delta_final:+.2f}).
- {late_winner} has the higher mean reward over the final five checkpoints (TD3 minus DDPG: {delta_late:+.2f}).
- Per-seed differences and across-seed variability are shown directly in the comparison plots and summary table; the three seeds show the extent of run-to-run variation in this experiment.

## Limitations

- Three training seeds per algorithm provide limited evidence about performance across the full distribution of possible runs.
- Each evaluation checkpoint averages five episodes, so evaluation-episode variability remains and is not included in the cross-seed learning-curve band.
- No DDPG episode-level logs were available; this report compares checkpoint evaluation rewards only.
- Results cover one environment and a 200K-step horizon. They do not establish statistical significance or generalize to other tasks or training budgets.

## Interpretation of DDPG vs TD3

TD3's clipped double critics, delayed actor updates, and target-policy smoothing are intended to reduce value overestimation and improve stability relative to DDPG. In this experiment, the aggregate {final_winner} final reward and {late_winner} late-training reward describe the observed comparison, but the small seed count and the visible between-seed spread mean this should be treated as descriptive evidence, not a definitive algorithm ranking. Use the individual runs alongside the means when interpreting the outcome.
"""
    output_path.write_text(report, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    experiment_dir = args.experiment_dir.expanduser().resolve()
    output_dir = (args.output_dir or experiment_dir / "comparison_plots").expanduser().resolve()
    try:
        runs = load_runs(experiment_dir)
        validate_runs(runs, final_step=args.final_step, eval_every=args.eval_every)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    late_checkpoint_count = 5
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = calculate_statistics(runs, late_checkpoint_count=late_checkpoint_count)
    summary_path = output_dir / "comparison_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.6f")

    plot_mean_learning_curves(runs, output_dir)
    plot_individual_seed_curves(runs, output_dir)
    plot_final_comparison(summary, output_dir)
    plot_late_training_comparison(summary, output_dir)
    generate_report(runs, summary, output_dir / "comparison_report.md")

    print(f"Validated {len(runs)} runs with {args.final_step // args.eval_every} aligned checkpoints each.")
    print(f"Wrote comparison outputs to {output_dir}")
    for filename in (
        "mean_learning_curves.png", "mean_learning_curves.pdf",
        "individual_seed_curves.png", "individual_seed_curves.pdf",
        "final_reward_comparison.png", "final_reward_comparison.pdf",
        "late_training_comparison.png", "late_training_comparison.pdf",
        "comparison_summary.csv", "comparison_report.md",
    ):
        print(f"  {filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
