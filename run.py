"""Single entry point for every run (code.md "Runner and logs").

    python run.py --agent copdac_q --env bandit --seed 0 --steps 10000 --config configs/bandit_copdac.yaml

Order: parse arguments, load YAML, set the seed, build the environment and agent,
run the loop, write ``results/<env>/<agent>/seed<k>.csv`` and the resolved YAML.
The agents themselves are implemented from Phase 1 onward; in Phase 0 the factory
raises a clear NotImplementedError.
"""

from __future__ import annotations

import argparse
import os

ENV_CHOICES = ("bandit", "pendulum", "lunarlander")
AGENT_CHOICES = ("stochastic_ac", "offpac", "copdac_q", "ddpg", "td3")

GYM_ENV_IDS = {
    "pendulum": "Pendulum-v1",
    "lunarlander": "LunarLanderContinuous-v3",
}


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
    return ClipAction(env)


def make_agent(agent_name: str, env, config: dict):
    """Agent registry. Bodies land from Phase 1 (see task.md and code.md)."""
    raise NotImplementedError(
        f"agent '{agent_name}' is implemented in Phase 1; Phase 0 only wires the CLI"
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    from common.seeding import set_seed

    set_seed(args.seed)

    env = make_env(args.env, args.seed, config)
    agent = make_agent(args.agent, env, config)  # raises in Phase 0

    # Phase 1+ wires common.runner.run_training here and writes the CSV/YAML to
    # results/<env>/<agent>/seed<k>.*.
    del env, agent
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
