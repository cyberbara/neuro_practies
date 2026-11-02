"""Common utilities: timing, parameter counting, lightweight metrics."""
from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def stopwatch():
    t0 = time.perf_counter()
    yield lambda: time.perf_counter() - t0


def count_params_numpy(*arrays):
    return sum(int(a.size) for a in arrays)
