"""CSV logging and reading for runs.

Every run writes one CSV whose header starts with the three fixed columns
``step,eval_return,wall_time_sec``. Optional per-agent columns may follow.
Nobody edits a CSV by hand; the runner is the only writer.
"""

from __future__ import annotations

import csv
import os
from typing import Sequence

import numpy as np

BASE_COLUMNS: tuple[str, ...] = ("step", "eval_return", "wall_time_sec")


class RunLogger:
    """Append-only CSV writer with the fixed base header plus optional columns."""

    def __init__(self, path: str, extra_columns: Sequence[str] = ()):
        self.path = path
        self.extra_columns = tuple(extra_columns)
        self.columns = BASE_COLUMNS + self.extra_columns
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.columns)
        self._file.flush()

    def log(self, step: int, eval_return: float, wall_time_sec: float, **extra) -> None:
        for value in (eval_return, wall_time_sec, *extra.values()):
            if not np.isfinite(value):
                raise ValueError(f"non-finite value written to {self.path}: {value}")
        row = [int(step), float(eval_return), float(wall_time_sec)]
        for name in self.extra_columns:
            row.append(float(extra[name]))
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_run_csv(path: str) -> dict[str, np.ndarray]:
    """Read a run CSV, validating the fixed header. Raises ValueError on a bad header."""
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path} is empty") from exc
        if tuple(header[: len(BASE_COLUMNS)]) != BASE_COLUMNS:
            raise ValueError(
                f"{path} header must start with {BASE_COLUMNS}, got {tuple(header)}"
            )
        rows = [row for row in reader if row]

    columns = {name: np.array([float(r[i]) for r in rows]) for i, name in enumerate(header)}
    columns["step"] = columns["step"].astype(int)
    return columns
