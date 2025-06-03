"""ninl.py
================
NINL – Node-Influence based on Neighbour-Layer information
(Zhang et al., Symmetry 13 (2021)).

The metric adds up a node’s degree with the degrees of nodes that lie within
*L* hops (``L = ceil(avg_shortest_path_length)`` by default), then **recursively
propagates** that score through its 1-hop neighbourhood for a user-defined
number of *layers*.

Public API
----------

* :func:`ninl`       – returns the *layers*-deep score
* :func:`ninl_layer0` – convenience helper (just the base layer)

All functions work with nothing but ``G``; you *may* pass pre-computed
``paths``/``degree`` to avoid duplicate work.  Parallel execution is enabled
for graphs with ≥ 500 nodes unless you set ``parallel=False``.
"""
from __future__ import annotations

import logging
import math
from multiprocessing import Pool, cpu_count
from typing import Dict, Optional, Tuple

import networkx as nx

__all__ = ["ninl", "ninl_layer0"]

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------#
# Helpers                                                                    #
# ---------------------------------------------------------------------------#


def _all_pairs_paths(G: nx.Graph) -> dict[int, dict[int, int]]:
    _LOG.debug("Computing APSP lengths for NINL …")
    return dict(nx.all_pairs_shortest_path_length(G))


def _chunked_pool_map(func, iterable, parallel: bool, processes: int | None):
    if not parallel:
        return map(func, iterable)
    procs = processes or max(cpu_count() - 1, 1)
    with Pool(procs) as pool:
        return pool.map(func, iterable)


# ---------------------------------------------------------------------------#
# Main routine                                                               #
# ---------------------------------------------------------------------------#


# Identifying Influential Nodes in Complex Networks Based on Node Itself and Neighbor Layer Information 
# https://www.mdpi.com/2073-8994/13/9/1570

def ninl(
    G: nx.Graph,
    *,
    layers: int = 3,
    degree: Optional[dict[int, int]] = None,
    paths: Optional[dict[int, dict[int, int]]] = None,
    avg_shortest_path: Optional[float] = None,
    parallel: bool | None = None,
    processes: int | None = None,
) -> Dict[int, float]:
    """
    Compute the **NINL** influence score.

    Parameters
    ----------
    layers
        How many propagation layers (≥ 0).  The original paper uses 3.
    degree, paths, avg_shortest_path
        Optional caches to speed up repeated calls.
    """
    if layers < 0:
        raise ValueError("layers must be non-negative")

    degree = degree or dict(G.degree())
    paths = paths or _all_pairs_paths(G)

    if avg_shortest_path is None:
        avg_shortest_path = nx.average_shortest_path_length(G)
    L = math.ceil(avg_shortest_path)

    use_mp = parallel if parallel is not None else len(G) >= 500

    # ------------------------------------------------------------------ #
    # layer-0: degree + neighbours within L hops                          #
    # ------------------------------------------------------------------ #

    def _layer0(n: int) -> Tuple[int, float]:
        acc = sum(
            degree[v]
            for v, dist in paths[n].items()
            if v != n and dist <= L
        )
        return n, degree[n] + acc

    scores: Dict[int, float] = dict(
        _chunked_pool_map(_layer0, G.nodes(), use_mp, processes)
    )

    # ------------------------------------------------------------------ #
    # propagate 1-hop aggregation for layers≥1                            #
    # ------------------------------------------------------------------ #
    for _ in range(1, layers + 1):
        scores = {n: sum(scores[v] for v in G.neighbors(n)) for n in G.nodes()}

    return scores


def ninl_layer0(
    G: nx.Graph,
    **kwargs,
) -> Dict[int, float]:
    """Return only the base NINL layer (no 1-hop propagation)."""
    return ninl(G, layers=0, **kwargs)
