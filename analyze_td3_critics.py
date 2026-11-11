"""Analyse TD3 twin-critic disagreement from saved LunarLander checkpoints.

Run from the project root:
    .\.venv\Scripts\python.exe analyze_td3_critics.py

Uses saved replay-buffer transitions; no retraining is performed.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from agents.ddpg_td3 import Critic

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "results" / "lunarlander_ddpg_td3_comparison_3seeds_200k"
CHECKPOINT_DIR = BASE / "checkpoints" / "lunarlander" / "td3"
OUTPUT_DIR = BASE / "critic_analysis"

CHECKPOINTS = {
    "seed0": "comparison_seed0_final.pt",
    "seed1": "comparison_seed1_final.pt",
    "seed2_rerun": "comparison_seed2_rerun_final.pt",
}
SAMPLES_PER_SEED = 10_000
SAMPLE_SEED = 2026
BATCH_SIZE = 1024


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def analyse(label, filename, device):
    path = CHECKPOINT_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("critic2") is None:
        raise ValueError(f"{filename} has no second critic; expected a TD3 checkpoint.")

    state_dim = int(ckpt["state_dim"])
    action_dim = len(to_numpy(ckpt["action_low"]).reshape(-1))
    hidden_dims = ckpt["hyperparameters"]["hidden_dims"]

    critic1 = Critic(state_dim, action_dim, hidden_dims).to(device)
    critic2 = Critic(state_dim, action_dim, hidden_dims).to(device)
    critic1.load_state_dict(ckpt["critic1"])
    critic2.load_state_dict(ckpt["critic2"])
    critic1.eval()
    critic2.eval()

    replay = ckpt["training_state"]["replay_buffer"]
    size = int(replay["size"])
    states = to_numpy(replay["states"])
    actions = to_numpy(replay["actions"])
    if states.shape != (size, state_dim):
        raise ValueError(f"{label}: unexpected states shape {states.shape}")
    if actions.shape != (size, action_dim):
        raise ValueError(f"{label}: unexpected actions shape {actions.shape}")
    if size < 1:
        raise ValueError(f"{label}: replay buffer is empty")

    count = min(SAMPLES_PER_SEED, size)
    rng = np.random.default_rng(SAMPLE_SEED)
    indices = np.sort(rng.choice(size, size=count, replace=False))

    q1_parts, q2_parts = [], []
    with torch.inference_mode():
        for start in range(0, count, BATCH_SIZE):
            idx = indices[start:start + BATCH_SIZE]
            s = torch.as_tensor(states[idx], dtype=torch.float32, device=device)
            a = torch.as_tensor(actions[idx], dtype=torch.float32, device=device)
            q1_parts.append(critic1(s, a).squeeze(-1).cpu().numpy())
            q2_parts.append(critic2(s, a).squeeze(-1).cpu().numpy())

    q1, q2 = np.concatenate(q1_parts), np.concatenate(q2_parts)
    diff = q1 - q2
    abs_diff = np.abs(diff)
    summary = {
        "seed": label,
        "checkpoint": filename,
        "replay_size": size,
        "sample_count": count,
        "q1_mean": float(np.mean(q1)),
        "q2_mean": float(np.mean(q2)),
        "signed_q1_minus_q2_mean": float(np.mean(diff)),
        "absolute_disagreement_mean": float(np.mean(abs_diff)),
        "absolute_disagreement_median": float(np.median(abs_diff)),
        "absolute_disagreement_p90": float(np.percentile(abs_diff, 90)),
        "absolute_disagreement_p95": float(np.percentile(abs_diff, 95)),
        "absolute_disagreement_p99": float(np.percentile(abs_diff, 99)),
        "absolute_disagreement_max": float(np.max(abs_diff)),
        "pearson_q1_q2": float(np.corrcoef(q1, q2)[0, 1])
        if np.std(q1) > 0 and np.std(q2) > 0 else float("nan"),
    }
    rows = [
        {
            "seed": label,
            "replay_index": int(idx),
            "q1": float(q1[i]),
            "q2": float(q2[i]),
            "q1_minus_q2": float(diff[i]),
            "absolute_disagreement": float(abs_diff[i]),
        }
        for i, idx in enumerate(indices)
    ]
    return summary, rows, abs_diff


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    summaries, all_rows, plot_data = [], [], {}

    for label, filename in CHECKPOINTS.items():
        print(f"Analysing {label}: {filename}")
        summary, rows, disagreement = analyse(label, filename, device)
        summaries.append(summary)
        all_rows.extend(rows)
        plot_data[label] = disagreement
        print(
            f"  n={summary['sample_count']:,}; "
            f"mean |Q1-Q2|={summary['absolute_disagreement_mean']:.4f}; "
            f"median={summary['absolute_disagreement_median']:.4f}; "
            f"p95={summary['absolute_disagreement_p95']:.4f}"
        )

    write_csv(OUTPUT_DIR / "td3_critic_disagreement_summary.csv", summaries)
    write_csv(OUTPUT_DIR / "td3_critic_disagreement_samples.csv", all_rows)

    labels = list(plot_data)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot([plot_data[k] for k in labels], tick_labels=labels, showfliers=False)
    ax.set_title("TD3 Twin-Critic Disagreement on Replay Transitions")
    ax.set_xlabel("Training seed")
    ax.set_ylabel("|Q1(s,a) - Q2(s,a)|")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "td3_critic_disagreement_boxplot.png", dpi=200)
    fig.savefig(OUTPUT_DIR / "td3_critic_disagreement_boxplot.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, values in plot_data.items():
        ax.hist(values, bins=60, alpha=0.4, density=True, label=label)
    ax.set_title("Distribution of TD3 Twin-Critic Disagreement")
    ax.set_xlabel("|Q1(s,a) - Q2(s,a)|")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "td3_critic_disagreement_histogram.png", dpi=200)
    fig.savefig(OUTPUT_DIR / "td3_critic_disagreement_histogram.pdf")
    plt.close(fig)

    print(f"\nAnalysis complete. Outputs saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
