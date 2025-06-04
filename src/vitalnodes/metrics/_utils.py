# src/vitalnodes/metrics/_utils.py

from multiprocessing import Pool, cpu_count
from typing import Callable, Iterable, Any, Optional

def _chunked_pool_map(
    func: Callable[[Any], Any],
    iterable: Iterable[Any],
    parallel: bool,
    processes: Optional[int]
):
    """
    Utility to run `func` over `iterable` either serially or via Pool().

    Parameters
    ----------
    func
        A top‐level function that takes a single argument (one element from iterable).
    iterable
        An iterable of arguments to feed into `func`.
    parallel
        If True, use multiprocessing.Pool; if False, use a normal map().
    processes
        Number of worker processes (None → cpu_count() - 1).
    """
    if not parallel:
        return map(func, iterable)
    procs = processes or max(10, 1)
    with Pool(procs) as pool:
        return pool.map(func, iterable)
