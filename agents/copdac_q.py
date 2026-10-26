"""COPDAC-Q -- off-policy deterministic actor-critic with a compatible linear critic.

Phase 1 (task T1.2). Natural gradient is a constructor flag, not a separate agent.
This module imports cleanly in Phase 0; the body is implemented in Phase 1 per
spec.md 1.4 and code.md (equations 19--22).
"""

from __future__ import annotations

AGENT_NAME = "copdac_q"
