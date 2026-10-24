"""Timing utilities for measuring inference latency and throughput.

Single timings are unreliable (caching, CPU frequency scaling, lazy
initialisation), so :func:`benchmark` runs warm-up calls first and then
reports statistics over many repeats. The median and 95th percentile are
reported because a production line must keep up with the *slowest* frames,
not just the average one.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np


def benchmark(fn: Callable[[], object], repeats: int = 50, warmup: int = 5) -> dict:
    """Time repeated calls of a zero-argument function.

    Args:
        fn: The function to time (wrap arguments with a lambda).
        repeats: Number of timed calls.
        warmup: Number of untimed calls made first.

    Returns:
        Dictionary of latencies in milliseconds: ``mean``, ``std``,
        ``median``, ``p95``, ``min``, plus ``fps`` (1000 / median) and the
        raw ``samples`` array.

    Raises:
        ValueError: If ``repeats`` is less than 1 or ``warmup`` is negative.
    """
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be >= 1 and warmup >= 0")
    for _ in range(warmup):
        fn()
    samples = np.empty(repeats)
    for i in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples[i] = (time.perf_counter() - t0) * 1000.0
    median = float(np.median(samples))
    return {
        "mean": float(samples.mean()),
        "std": float(samples.std(ddof=1)) if repeats > 1 else 0.0,
        "median": median,
        "p95": float(np.percentile(samples, 95)),
        "min": float(samples.min()),
        "fps": 1000.0 / median if median > 0 else float("inf"),
        "samples": samples,
    }
