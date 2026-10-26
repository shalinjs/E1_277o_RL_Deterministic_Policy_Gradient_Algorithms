"""Single seeding entry point, called from run.py before the env is built."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch so a run is reproducible.

    Torch is imported lazily so the NumPy-only agents and tests do not pay for it.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # torch is only needed by DDPG/TD3
        return
    torch.manual_seed(seed)
