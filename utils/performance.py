"""
Performance Infrastructure
===========================
Auto-detects hardware, manages parallelism, provides timing decorators.
"""
import os
import time
import functools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# ── Hardware detection ─────────────────────────────────────────────────────
N_CORES = os.cpu_count() or 1
N_WORKERS = max(1, N_CORES - 1)  # Leave one core for OS

try:
    import numba
    HAS_NUMBA = True
    NUMBA_VERSION = numba.__version__
except ImportError:
    HAS_NUMBA = False
    NUMBA_VERSION = None

try:
    import pyarrow
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


def print_system_info():
    """Print detected hardware and library configuration."""
    print(f"  CPU cores: {N_CORES} (using {N_WORKERS} workers)")
    print(f"  Numba JIT: {'yes (v' + NUMBA_VERSION + ')' if HAS_NUMBA else 'no (falling back to NumPy)'}")
    print(f"  PyArrow:   {'yes' if HAS_PYARROW else 'no (parquet via pandas fallback)'}")
    try:
        import numpy as np
        print(f"  NumPy BLAS: {np.show_config.__module__}")
    except Exception:
        pass


# ── Timing decorator ───────────────────────────────────────────────────────
def timed(label=None):
    """Decorator that prints execution time."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = label or func.__name__
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            dt = time.perf_counter() - t0
            if dt < 1:
                print(f"  [{name}] {dt*1000:.0f}ms")
            elif dt < 60:
                print(f"  [{name}] {dt:.1f}s")
            else:
                print(f"  [{name}] {dt/60:.1f}min")
            return result
        return wrapper
    if callable(label):
        # Used without arguments: @timed
        func = label
        label = None
        return decorator(func)
    return decorator


# ── Parallel map ───────────────────────────────────────────────────────────
def parallel_map(func, items, n_workers=None, use_threads=False, desc=""):
    """
    Apply func to each item in parallel.
    Uses ProcessPool for CPU-bound, ThreadPool for IO-bound.
    Falls back to sequential if n_workers=1 or items too few.
    """
    n = len(items)
    n_workers = n_workers or min(N_WORKERS, n)

    if n_workers <= 1 or n <= 2:
        return [func(item) for item in items]

    Pool = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    results = [None] * n

    with Pool(max_workers=n_workers) as executor:
        futures = {executor.submit(func, item): i for i, item in enumerate(items)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"  Warning: parallel task {idx} failed: {e}")
                results[idx] = None
    return results


def chunked_apply(df, func, n_chunks=None):
    """Split DataFrame into chunks, apply func in parallel, concatenate."""
    import pandas as pd
    n_chunks = n_chunks or N_WORKERS
    chunks = [chunk for _, chunk in df.groupby(df.index // (len(df) // n_chunks + 1))]

    results = parallel_map(func, chunks)
    return pd.concat([r for r in results if r is not None])
