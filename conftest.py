"""Make the repository root importable so ``import envs``, ``import common`` work
under pytest regardless of the invocation directory.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
