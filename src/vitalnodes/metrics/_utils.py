# src/vitalnodes/metrics/_utils.py

from multiprocessing import cpu_count
from pathos.multiprocessing import Pool
from typing import Callable, Iterable, Any, Optional

def _chunked_pool_map(
    func: Callable[[Any], Any],
    iterable: Iterable[Any],
    parallel: Optional[bool],
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
    procs = processes or max(cpu_count() - 1, 1)
    with Pool(procs) as pool:
        return pool.map(func, iterable)
