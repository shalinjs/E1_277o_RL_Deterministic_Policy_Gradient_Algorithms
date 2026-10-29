"""Single entry point for every run (code.md "Runner and logs").

    python run.py --agent copdac_q --env bandit --seed 0 --steps 10000 --config configs/bandit_copdac_q_m10.yaml

Order: parse arguments, load YAML, set the seed, build the environment and agent,
run the loop, write ``results/<env_dir>/<agent_dir>/seed<k>.csv`` and the resolved
YAML next to it.

Result folders (spec.md section 4):

- ``env_dir`` is the env name, except the bandit, which is ``bandit_m<m>`` so the
  three action dimensions do not overwrite each other.
- ``agent_dir`` is the agent name, except ``copdac_q`` with ``natural_gradient:
  true``, which is ``copdac_q_natural`` so the two variants stay separate.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

ENV_CHOICES = ("bandit", "pendulum", "lunarlander")
AGENT_CHOICES = ("stochastic_ac", "offpac", "copdac_q", "ddpg", "td3")

GYM_ENV_IDS = {
    "pendulum": "Pendulum-v1",
    "lunarlander": "LunarLanderContinuous-v3",
}

DEFAULT_EVAL_EPISODES = 5
EVAL_SEED_OFFSET = 10_000  # evaluation episodes never share seeds with training


def build_parser() -> argparse.ArgumentParser:
    """The five arguments from task T0.2."""
    parser = argparse.ArgumentParser(
        prog="run.py", description="Train one DPG-family agent on one environment."
    )
    parser.add_argument("--agent", required=True, choices=AGENT_CHOICES)
    parser.add_argument("--env", required=True, choices=ENV_CHOICES)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--config", required=True, help="path to the run's YAML config")
    return parser


def load_config(path: str) -> dict:
    import yaml

    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return config


def resolve_config(args, config: dict) -> dict:
    """Command-line agent, env, seed and steps win over the YAML; mismatches are errors."""
    for key in ("agent", "env"):
        if key in config and config[key] != getattr(args, key):
            raise ValueError(
                f"--{key} {getattr(args, key)!r} disagrees with config {key}: {config[key]!r}"
            )
    resolved = dict(config)
    resolved.update(agent=args.agent, env=args.env, seed=args.seed, steps=args.steps)
    resolved.setdefault("gamma", 0.99)
    if "eval_every" not in resolved:
        resolved["eval_every"] = (
            2 * int(resolved["action_dim"]) if args.env == "bandit" else 5000
        )
    return resolved


def make_env(env_name: str, seed: int, config: dict):
    """Construct the environment. Gymnasium envs get the action-clipping wrapper."""
    if env_name == "bandit":
        from envs.bandit import QuadraticBandit

        action_dim = config.get("action_dim")
        if action_dim is None:
            raise ValueError("bandit config requires 'action_dim'")
        return QuadraticBandit(action_dim=action_dim, seed=seed)

    import gymnasium

    from common.wrappers import ClipAction

    env = gymnasium.make(GYM_ENV_IDS[env_name])
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return ClipAction(env)


def make_agent(agent_name: str, env, config: dict):
    """Agent registry. Each owner adds their builder here."""
    env_name = config["env"]
    if agent_name == "copdac_q":
        from agents.copdac_q import make_copdac_q

        return make_copdac_q(env_name, env, config)
    raise NotImplementedError(f"agent '{agent_name}' is not implemented yet")


def make_evaluate_fn(env_name: str, env, config: dict):
    """Score the policy with exploration off (code.md "Runner and logs", step 5)."""
    if env_name == "bandit":
        dummy_obs = np.zeros(1, dtype=np.float32)
        # The cost of the mean action, never of the exploratory action.
        return lambda agent: env.cost(agent.act(dummy_obs, explore=False))

    episodes = int(config.get("eval_episodes", DEFAULT_EVAL_EPISODES))
    eval_env = make_env(env_name, config["seed"] + EVAL_SEED_OFFSET, config)
    base_seed = config["seed"] + EVAL_SEED_OFFSET

    def evaluate(agent) -> float:
        returns = []
        for i in range(episodes):
            obs, _ = eval_env.reset(seed=base_seed + i)
            total, finished = 0.0, False
            while not finished:
                obs, reward, terminated, truncated, _ = eval_env.step(
                    agent.act(obs, explore=False)
                )
                total += float(reward)
                finished = terminated or truncated
            returns.append(total)
        return float(np.mean(returns))

    return evaluate


def results_paths(config: dict, results_root: str = "results") -> tuple[str, str]:
    """Return (csv_path, yaml_path) for this run."""
    env_dir = config["env"]
    if env_dir == "bandit":
        env_dir = f"bandit_m{int(config['action_dim'])}"
    agent_dir = config["agent"]
    if agent_dir == "copdac_q" and config.get("natural_gradient", False):
        agent_dir = "copdac_q_natural"
    stem = os.path.join(results_root, env_dir, agent_dir, f"seed{config['seed']}")
    return stem + ".csv", stem + ".yaml"


def main(argv=None, results_root: str = "results") -> int:
    args = build_parser().parse_args(argv)
    config = resolve_config(args, load_config(args.config))

    from common.seeding import set_seed

    set_seed(args.seed)

    env = make_env(args.env, args.seed, config)
    agent = make_agent(args.agent, env, config)
    evaluate_fn = make_evaluate_fn(args.env, env, config)

    import yaml

    from common.logging import RunLogger
    from common.runner import run_training

    csv_path, yaml_path = results_paths(config, results_root)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(yaml_path, "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    start = time.perf_counter()
    with RunLogger(csv_path) as logger:
        run_training(
            env,
            agent,
            logger,
            steps=args.steps,
            eval_every=int(config["eval_every"]),
            evaluate_fn=evaluate_fn,
            start_time=start,
        )
    print(f"wrote {csv_path} ({time.perf_counter() - start:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
