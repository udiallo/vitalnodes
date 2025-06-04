"""density.py
================
Density-flavoured centrality measures.

* **Density Centrality** (Li et al., 2019) – mass = degree, r ≤ 3
* **Clustered Local Degree (CLD)** (Yan & Han, 2018) – local clustering
  coefficient modulates neighbour degree sum.
"""
from __future__ import annotations

import logging
import math
from multiprocessing import Pool, cpu_count
from typing import Dict, Optional, Tuple, List

from vitalnodes.metrics._utils import _chunked_pool_map

import networkx as nx

__all__ = ["density_centrality", "clustered_local_degree"]

_LOG = logging.getLogger(__name__)


def _all_pairs_paths(G: nx.Graph) -> dict[int, dict[int, int]]:
    _LOG.debug("Computing APSP for density metrics …")
    return dict(nx.all_pairs_shortest_path_length(G))


# ---------------------------------------------------------------------------#
# Worker functions                                                           #
# ---------------------------------------------------------------------------#


def _density_worker(args: Tuple[int, Dict[int, int], dict[int, dict[int, int]], int]) -> Tuple[int, float]:
    n1, degree, paths, max_distance = args
    val = 0.0
    for n2, dist in paths[n1].items():
        if n2 != n1 and dist <= max_distance:
            val += degree[n1] / (math.pi * (dist ** 2))
    return n1, val


def _cld_worker(args: Tuple[int, Dict[int, int], dict[int, dict[int, int]], Dict[int, float], int]) -> Tuple[int, float]:
    n1, degree, paths, clustering, max_distance = args
    acc = 0.0
    for n2, dist in paths[n1].items():
        if n2 != n1 and dist <= max_distance:
            acc += degree[n2]
    return n1, (1 + clustering[n1]) * acc


# ---------------------------------------------------------------------------#
# 1. Density Centrality                                                      #
# ---------------------------------------------------------------------------#


def density_centrality(
    G: nx.Graph,
    *,
    degree: Optional[dict[int, int]] = None,
    paths: Optional[dict[int, dict[int, int]]] = None,
    max_distance: int = 3,
    parallel: bool | None = None,
    processes: int | None = None,
) -> Dict[int, float]:
    """Inverse‐area “mass density” within radius 3."""

    degree = degree or dict(G.degree())
    paths = paths or _all_pairs_paths(G)
    use_mp = parallel if parallel is not None else len(G) >= 500

    payload: List[Tuple[int, Dict[int, int], dict[int, dict[int, int]], int]] = [
        (n1, degree, paths, max_distance) for n1 in G.nodes()
    ]
    return dict(_chunked_pool_map(_density_worker, payload, use_mp, processes))


# ---------------------------------------------------------------------------#
# 2. Clustered Local Degree (CLD)                                            #
# ---------------------------------------------------------------------------#


def clustered_local_degree(
    G: nx.Graph,
    *,
    degree: Optional[dict[int, int]] = None,
    paths: Optional[dict[int, dict[int, int]]] = None,
    clustering: Optional[dict[int, float]] = None,
    max_distance: int = 1,
    parallel: bool | None = None,
    processes: int | None = None,
) -> Dict[int, float]:
    """Local degree sum modulated by clustering coefficient."""

    degree = degree or dict(G.degree())
    paths = paths or _all_pairs_paths(G)
    clustering = clustering or nx.clustering(G)
    use_mp = parallel if parallel is not None else len(G) >= 500

    payload: List[Tuple[int, Dict[int, int], dict[int, dict[int, int]], Dict[int, float], int]] = [
        (n1, degree, paths, clustering, max_distance) for n1 in G.nodes()
    ]
    return dict(_chunked_pool_map(_cld_worker, payload, use_mp, processes))
