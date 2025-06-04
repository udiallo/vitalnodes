"""gli.py
================
GLI – Global-Local Influence measures.

* **GLI** (Chen & Mei, 2020)    – original degree + i-kshell mass
* **GLI-new** (Liu et al., 2022) – adds Jaccard similarity weighting
"""
from __future__ import annotations

import logging
from multiprocessing import Pool, cpu_count
from typing import Dict, Optional, Tuple, List
import math
import networkx as nx

from vitalnodes.metrics._core import i_kshell
from vitalnodes.metrics._utils import _chunked_pool_map

__all__ = ["gli", "gli_new"]

_LOG = logging.getLogger(__name__)



# ---------------------------------------------------------------------------#
# Worker functions                                                           #
# ---------------------------------------------------------------------------#


def _gli_worker(args: Tuple[int, Dict[int, int], Dict[int, int], dict[int, dict[int, int]], int, float]) -> Tuple[int, float]:
    n1, degree, i_ks, paths, max_distance, denom = args
    local_mass = (i_ks[n1] + degree[n1]) / denom
    acc = 0.0
    for n2, dist in paths[n1].items():
        if n2 != n1 and dist <= max_distance:
            acc += (i_ks[n2] + degree[n2]) / dist
    return n1, (math.e ** local_mass) * acc  # using math.e


def _gli_new_worker(args: Tuple[int, Dict[int, int], Dict[int, int], Dict[int, set[int]], float]) -> Tuple[int, float]:
    n, degree, core_num, omega_vals, deg_max = args
    local = degree[n] + (omega_vals[n] / deg_max)
    return n, local + core_num[n]


def _compute_jaccard(a: set[int], b: set[int]) -> float:
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# ---------------------------------------------------------------------------#
# 1. Original GLI                                                             #
# ---------------------------------------------------------------------------#


def gli(
    G: nx.Graph,
    *,
    degree: Optional[dict[int, int]] = None,
    i_ks: Optional[dict[int, int]] = None,
    paths: Optional[dict[int, dict[int, int]]] = None,
    max_distance: int = 3,
    parallel: bool | None = None,
    processes: int | None = None,
) -> Dict[int, float]:
    """Global–local influence (exponential local term)."""

    degree = degree or dict(G.degree())
    i_ks = i_ks or i_kshell(G)
    paths = paths or dict(nx.all_pairs_shortest_path_length(G))
    use_mp = parallel if parallel is not None else len(G) >= 500

    denom = sum(i_ks[n] + degree[n] for n in G.nodes())

    payload: List[Tuple[int, Dict[int, int], Dict[int, int], dict[int, dict[int, int]], int, float]] = [
        (n1, degree, i_ks, paths, max_distance, denom) for n1 in G.nodes()
    ]
    return dict(_chunked_pool_map(_gli_worker, payload, use_mp, processes))


# ---------------------------------------------------------------------------#
# 2. Improved GLI (Jaccard-weighted)                                          #
# ---------------------------------------------------------------------------#


def gli_new(
    G: nx.Graph,
    *,
    degree: Optional[dict[int, int]] = None,
    core_num: Optional[dict[int, int]] = None,
    parallel: bool | None = None,
    processes: int | None = None,
) -> Dict[int, float]:
    """GLI-new combines neighbour degree, Jaccard similarity & k-core."""

    degree = degree or dict(G.degree())
    core_num = core_num or nx.core_number(G)
    use_mp = parallel if parallel is not None else len(G) >= 500

    # cache neighbour lists
    nbr_lists = {n: set(G.neighbors(n)) for n in G.nodes()}
    # precompute omega for each node
    omega_vals: Dict[int, float] = {}
    for n in G.nodes():
        omega_vals[n] = sum(
            degree[v] * _compute_jaccard(nbr_lists[n], nbr_lists[v]) + core_num[v]
            for v in nbr_lists[n]
        )

    deg_max = max(degree.values())

    payload: List[Tuple[int, Dict[int, int], Dict[int, int], Dict[int, set[int]], float]] = [
        (n, degree, core_num, omega_vals, deg_max) for n in G.nodes()
    ]
    return dict(_chunked_pool_map(_gli_new_worker, payload, use_mp, processes))
