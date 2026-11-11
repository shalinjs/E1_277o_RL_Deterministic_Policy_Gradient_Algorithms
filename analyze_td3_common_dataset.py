"""Compare all three TD3 checkpoints on one common state-action dataset.

Run from the project root:
    .\.venv\Scripts\python.exe analyze_td3_common_dataset.py

The common dataset is constructed by sampling equal numbers of transitions
from each seed's saved replay buffer, pooling them, then evaluating every
checkpoint on exactly those same state-action pairs. No retraining or
environment interaction is performed.
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
OUTPUT_DIR = BASE / "common_dataset_critic_analysis"

CHECKPOINTS = {
    "seed0": "comparison_seed0_final.pt",
    "seed1": "comparison_seed1_final.pt",
    "seed2_rerun": "comparison_seed2_rerun_final.pt",
}

# 3,333 transitions from each seed = 9,999 common evaluation pairs.
SAMPLES_PER_SOURCE_SEED = 3333
DATASET_SEED = 731


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_checkpoint(label: str, filename: str):
    path = CHECKPOINT_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("critic2") is None:
        raise ValueError(f"{filename} does not contain critic2.")
    return checkpoint


def build_common_dataset(checkpoints: dict):
    rng = np.random.default_rng(DATASET_SEED)
    state_parts, action_parts, source_parts, replay_index_parts = [], [], [], []

    expected_state_dim = int(checkpoints["seed0"]["state_dim"])
    expected_action_dim = len(
        to_numpy(checkpoints["seed0"]["action_low"]).reshape(-1)
    )

    for label, checkpoint in checkpoints.items():
        replay = checkpoint["training_state"]["replay_buffer"]
        size = int(replay["size"])
        states = to_numpy(replay["states"])
        actions = to_numpy(replay["actions"])

        if states.shape != (size, expected_state_dim):
            raise ValueError(f"{label}: unexpected state shape {states.shape}")
        if actions.shape != (size, expected_action_dim):
            raise ValueError(f"{label}: unexpected action shape {actions.shape}")
        if size < SAMPLES_PER_SOURCE_SEED:
            raise ValueError(
                f"{label}: replay has {size} entries, fewer than "
                f"{SAMPLES_PER_SOURCE_SEED} required."
            )

        indices = rng.choice(size, size=SAMPLES_PER_SOURCE_SEED, replace=False)
        state_parts.append(states[indices])
        action_parts.append(actions[indices])
        source_parts.extend([label] * len(indices))
        replay_index_parts.extend(indices.tolist())

    common_states = np.concatenate(state_parts, axis=0).astype(np.float32)
    common_actions = np.concatenate(action_parts, axis=0).astype(np.float32)
    source_parts = np.asarray(source_parts)
    replay_index_parts = np.asarray(replay_index_parts, dtype=np.int64)

    # Shuffle the pooled dataset once. All models receive the same ordering.
    order = rng.permutation(len(common_states))
    return (
        common_states[order],
        common_actions[order],
        source_parts[order],
        replay_index_parts[order],
    )


def evaluate_critics(label: str, checkpoint: dict, states: np.ndarray,
                     actions: np.ndarray, batch_size: int = 1024):
    state_dim = int(checkpoint["state_dim"])
    action_dim = actions.shape[1]
    hidden_dims = checkpoint["hyperparameters"]["hidden_dims"]

    critic1 = Critic(state_dim, action_dim, hidden_dims)
    critic2 = Critic(state_dim, action_dim, hidden_dims)
    critic1.load_state_dict(checkpoint["critic1"])
    critic2.load_state_dict(checkpoint["critic2"])
    critic1.eval()
    critic2.eval()

    q1_parts, q2_parts = [], []
    with torch.inference_mode():
        for start in range(0, len(states), batch_size):
            s = torch.as_tensor(states[start:start + batch_size], dtype=torch.float32)
            a = torch.as_tensor(actions[start:start + batch_size], dtype=torch.float32)
            q1_parts.append(critic1(s, a).squeeze(-1).numpy())
            q2_parts.append(critic2(s, a).squeeze(-1).numpy())

    q1 = np.concatenate(q1_parts)
    q2 = np.concatenate(q2_parts)
    abs_diff = np.abs(q1 - q2)
    summary = {
        "model_seed": label,
        "common_dataset_size": int(len(states)),
        "q1_mean": float(np.mean(q1)),
        "q2_mean": float(np.mean(q2)),
        "q1_std": float(np.std(q1, ddof=1)),
        "q2_std": float(np.std(q2, ddof=1)),
        "signed_q1_minus_q2_mean": float(np.mean(q1 - q2)),
        "absolute_disagreement_mean": float(np.mean(abs_diff)),
        "absolute_disagreement_median": float(np.median(abs_diff)),
        "absolute_disagreement_p90": float(np.percentile(abs_diff, 90)),
        "absolute_disagreement_p95": float(np.percentile(abs_diff, 95)),
        "absolute_disagreement_p99": float(np.percentile(abs_diff, 99)),
        "absolute_disagreement_max": float(np.max(abs_diff)),
        "pearson_q1_q2": float(np.corrcoef(q1, q2)[0, 1])
        if np.std(q1) > 0 and np.std(q2) > 0 else float("nan"),
    }
    return summary, q1, q2, abs_diff


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading TD3 checkpoints...")
    checkpoints = {
        label: load_checkpoint(label, filename)
        for label, filename in CHECKPOINTS.items()
    }

    print("Building one pooled common dataset...")
    states, actions, source_seed, source_index = build_common_dataset(checkpoints)
    dataset_rows = [
        {
            "common_sample_id": i,
            "source_replay_seed": source_seed[i],
            "source_replay_index": int(source_index[i]),
        }
        for i in range(len(states))
    ]
    write_csv(OUTPUT_DIR / "common_state_action_dataset_index.csv", dataset_rows)

    summaries = []
    per_sample_rows = []
    disagreement_for_plot = {}

    for model_seed, checkpoint in checkpoints.items():
        print(f"Evaluating model {model_seed} on {len(states):,} identical pairs...")
        summary, q1, q2, disagreement = evaluate_critics(
            model_seed, checkpoint, states, actions
        )
        summaries.append(summary)
        disagreement_for_plot[model_seed] = disagreement
        for i in range(len(states)):
            per_sample_rows.append({
                "model_seed": model_seed,
                "common_sample_id": i,
                "source_replay_seed": source_seed[i],
                "q1": float(q1[i]),
                "q2": float(q2[i]),
                "q1_minus_q2": float(q1[i] - q2[i]),
                "absolute_disagreement": float(disagreement[i]),
            })
        print(
            f"  mean |Q1-Q2|={summary['absolute_disagreement_mean']:.4f}; "
            f"median={summary['absolute_disagreement_median']:.4f}; "
            f"p95={summary['absolute_disagreement_p95']:.4f}; "
            f"corr={summary['pearson_q1_q2']:.6f}"
        )

    write_csv(OUTPUT_DIR / "common_dataset_critic_summary.csv", summaries)
    write_csv(OUTPUT_DIR / "common_dataset_critic_predictions.csv", per_sample_rows)

    labels = list(disagreement_for_plot)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(
        [disagreement_for_plot[name] for name in labels],
        tick_labels=labels,
        showfliers=False,
    )
    ax.set_title("TD3 Critic Disagreement on a Common Dataset")
    ax.set_xlabel("TD3 model checkpoint")
    ax.set_ylabel("|Q1(s,a) - Q2(s,a)|")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "common_dataset_disagreement_boxplot.png", dpi=200)
    fig.savefig(OUTPUT_DIR / "common_dataset_disagreement_boxplot.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, values in disagreement_for_plot.items():
        ax.hist(values, bins=60, alpha=0.4, density=True, label=label)
    ax.set_title("TD3 Critic Disagreement on Identical State-Action Pairs")
    ax.set_xlabel("|Q1(s,a) - Q2(s,a)|")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "common_dataset_disagreement_histogram.png", dpi=200)
    fig.savefig(OUTPUT_DIR / "common_dataset_disagreement_histogram.pdf")
    plt.close(fig)

    print("\nCompleted. All three models were evaluated on the same dataset.")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
