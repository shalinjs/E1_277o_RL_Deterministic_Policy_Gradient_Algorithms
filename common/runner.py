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
    evaluate_fn: EvaluateFn,
    start_time: float | None = None,
) -> None:
    """Run ``steps`` environment steps, logging ``evaluate_fn`` every ``eval_every``."""
    if eval_every < 1:
        raise ValueError("eval_every must be >= 1")
    if start_time is None:
        start_time = time.perf_counter()

    obs, _ = env.reset()
    for t in range(1, steps + 1):
        action = agent.act(obs, explore=True)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        # done ignores time-limit truncation: a truncated transition still bootstraps.
        agent.update(obs, action, reward, next_obs, bool(terminated))
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs

        if t % eval_every == 0:
            logger.log(
                step=t,
                eval_return=float(evaluate_fn(agent)),
                wall_time_sec=time.perf_counter() - start_time,
            )
