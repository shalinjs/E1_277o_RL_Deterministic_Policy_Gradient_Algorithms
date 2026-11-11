"""The generic training loop shared by every agent.

The loop is deliberately agent-agnostic: it calls ``act``, steps the environment,
calls ``update`` with ``done = terminated`` (time-limit truncation still
bootstraps), and periodically logs the result of an evaluation callback.
"""

from __future__ import annotations

import time
from typing import Callable

from .logging import RunLogger

Agent = object  # structural; see code.md Agent protocol
EvaluateFn = Callable[[Agent], float]


def run_training(
    env,
    agent,
    logger: RunLogger,
    *,
    steps: int,
    eval_every: int,
    seed: int,
    evaluate_fn: EvaluateFn,
    start_time: float | None = None,
    episode_logger=None,
) -> None:
    """Run training, optionally logging completed training episodes."""

    if eval_every < 1:
        raise ValueError("eval_every must be >= 1")

    if start_time is None:
        start_time = time.perf_counter()

    obs, _ = env.reset(seed=seed)

    episode_count = 0
    episode_return = 0.0
    episode_length = 0

    for t in range(1, steps + 1):
        action = agent.act(obs, explore=True)

        next_obs, reward, terminated, truncated, _ = env.step(action)

        # Preserve the existing bootstrap behaviour:
        # time-limit truncation does not set done=True.
        agent.update(
            obs,
            action,
            reward,
            next_obs,
            bool(terminated),
        )

        episode_return += float(reward)
        episode_length += 1

        # Log an episode whenever it terminates or is truncated.
        if terminated or truncated:
            episode_count += 1

            if episode_logger is not None:
                episode_logger.log(
                    episode=episode_count,
                    step=t,
                    reward=episode_return,
                    length=episode_length,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    final_step_reward=float(reward),
                )

            obs, _ = env.reset()
            episode_return = 0.0
            episode_length = 0

        else:
            obs = next_obs

        # Periodic policy evaluation.
        if t % eval_every == 0:
            logger.log(
                step=t,
                eval_return=float(evaluate_fn(agent)),
                wall_time_sec=time.perf_counter() - start_time,
            )
