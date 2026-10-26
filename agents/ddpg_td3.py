"""DDPG and TD3 -- one neural actor-critic class, TD3 as three flags.

Phase 1 (DDPG, task T1.3) and Phase 2 (TD3 flags, task T2.5). DDPG sets twin,
delayed actor updates, and target smoothing off; TD3 sets all three on. This
module imports cleanly in Phase 0; the body is implemented later per spec.md
1.5--1.6 and code.md.
"""

from __future__ import annotations

from dataclasses import dataclass

AGENT_NAME_DDPG = "ddpg"
AGENT_NAME_TD3 = "td3"


@dataclass(frozen=True)
class TD3Flags:
    """The three differences between DDPG and TD3 (code.md).

    DDPG sets all three off: ``policy_delay == 1`` means the actor updates on every
    critic step. TD3 turns twin critics and target smoothing on and delays the
    actor.
    """

    twin: bool = False
    policy_delay: int = 1
    target_smoothing: bool = False

    @classmethod
    def ddpg(cls) -> "TD3Flags":
        return cls(twin=False, policy_delay=1, target_smoothing=False)

    @classmethod
    def td3(cls, policy_delay: int = 2) -> "TD3Flags":
        return cls(twin=True, policy_delay=policy_delay, target_smoothing=True)
