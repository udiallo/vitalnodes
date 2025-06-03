"""gravity.py
================
Gravity-family centrality measures.

This module bundles **all node-influence metrics derived from the gravity
analogy** – a node’s “mass” (degree, *k*-core, etc.) interacts with other
nodes’ masses inversely proportional to the square of the topological
distance between them.

Key design choices
------------------
* **Self-sufficient API** – every public function can be called with *just a
  NetworkX graph*. Expensive prerequisites (shortest-path lengths, k-core,
  degrees…) are lazily computed if the caller does not provide them, but you
  *may* pass pre-computed structures to reuse work across multiple metrics.
* **Multiprocessing-ready** – set ``parallel=True`` (default for |V| ≥ 500).
* **Naming** – neighbor-aggregated variants are suffixed ``_agg``.
* **Citations** – docstrings mention journal & year so users can look them up
  (the README will contain full references).
"""
from __future__ import annotations

import logging
import math
from multiprocessing import Pool, cpu_count
from statistics import median
from typing import Dict, Iterable, Optional, Tuple

import networkx as nx

__all__ = [
    "gravity_centrality",
    "gravity_centrality_agg",
    "improved_gravity_centrality",
    "improved_gravity_centrality_agg",
    "dk_gravity_centrality",
    "dk_gravity_centrality_agg",
    "local_gravity_centrality",
    "mcgm",
]

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _compute_paths(G: nx.Graph) -> dict[int, dict[int, int]]:
    _LOG.debug("Computing all-pairs shortest-path lengths …")
    return dict(nx.all_pairs_shortest_path_length(G))


def _neighbors_within_radius(
    node: int, paths: dict[int, dict[int, int]], radius: int
) -> Iterable[int]:
    """Yield neighbors of *node* whose shortest-path distance ≤ *radius*."""
    for nbr, dist in paths[node].items():
        if nbr != node and dist <= radius:
            yield nbr


def _chunked_pool_map(func, iterable, parallel: bool, processes: int | None):
    """Utility to run `func` over `iterable` either serially or via Pool()."""
    if not parallel:
        return map(func, iterable)
    procs = processes or max(cpu_count() - 1, 1)
    with Pool(procs) as pool:
        return pool.map(func, iterable)


# ---------------------------------------------------------------------------
# 1. Gravity Centrality (GC) – Wang et al., Physica A 452 (2016)
# ---------------------------------------------------------------------------

# https://arxiv.org/abs/1505.02476
# Identifying influential spreaders in complex networks based on gravity formula

def gravity_centrality(
    G: nx.Graph,
    *,
    paths: Optional[dict[int, dict[int, int]]] = None,
    core_num: Optional[dict[int, int]] = None,
    max_distance: int = 3,
    parallel: bool | None = None,
    processes: int | None = None,
) -> Dict[int, float]:
    """Classical gravity centrality (global variant)."""

    paths = paths or _compute_paths(G)
    core_num = core_num or nx.core_number(G)
    use_mp = parallel if parallel is not None else len(G) >= 500

    def _score(n1: int) -> Tuple[int, float]:
        s = 0.0
        for n2 in _neighbors_within_radius(n1, paths, max_distance):
            s += (core_num[n1] * core_num[n2]) / (paths[n1][n2] ** 2)
        return n1, s

    return dict(_chunked_pool_map(_score, G.nodes(), use_mp, processes))


def gravity_centrality_agg(
    G: nx.Graph,
    *,
    gc: Optional[Dict[int, float]] = None,
    **kwargs,
) -> Dict[int, float]:
    """Neighbor-aggregated GC (formerly *GC+*)."""

    gc = gc or gravity_centrality(G, **kwargs)
    return {n: sum(gc[nbr] for nbr in G.neighbors(n)) for n in G.nodes()}

# ---------------------------------------------------------------------------
# 2. Improved Gravity Centrality (IGC) – Yan et al., AMC 335 (2018)
# ---------------------------------------------------------------------------

# https://www.sciencedirect.com/science/article/abs/pii/S0096300318303461
# Improved centrality indicators to characterize the nodal spreading capability in complex networks

def improved_gravity_centrality(
    G: nx.Graph,
    *,
    paths: Optional[dict[int, dict[int, int]]] = None,
    core_num: Optional[dict[int, int]] = None,
    degree: Optional[dict[int, int]] = None,
    max_distance: int = 3,
    parallel: bool | None = None,
    processes: int | None = None,
) -> Dict[int, float]:
    """IGC: replace second node’s mass with its degree."""

    paths = paths or _compute_paths(G)
    core_num = core_num or nx.core_number(G)
    degree = degree or dict(G.degree())
    use_mp = parallel if parallel is not None else len(G) >= 500

    def _score(n1: int) -> Tuple[int, float]:
        s = 0.0
        for n2 in _neighbors_within_radius(n1, paths, max_distance):
            s += (core_num[n1] * degree[n2]) / (paths[n1][n2] ** 2)
        return n1, s

    return dict(_chunked_pool_map(_score, G.nodes(), use_mp, processes))


def improved_gravity_centrality_agg(
    G: nx.Graph,
    *,
    igc: Optional[Dict[int, float]] = None,
    **kwargs,
) -> Dict[int, float]:
    """Neighbor-aggregated IGC (ex *IGC+*)."""

    igc = igc or improved_gravity_centrality(G, **kwargs)
    return {n: sum(igc[nbr] for nbr in G.neighbors(n)) for n in G.nodes()}

# ---------------------------------------------------------------------------
# 3. Degree + k-shell Improved GC (DK-IGC) – Li & Huang, Sci. Rep. 11 (2021)
# ---------------------------------------------------------------------------

# https://www.nature.com/articles/s41598-021-01218-1
# "Identifying influential spreaders in complex networks by an improved gravity model" Zhe Li1* & Xinyu Huang2*

def dk_gravity_centrality(
    G: nx.Graph,
    *,
    paths: Optional[dict[int, dict[int, int]]] = None,
    degree: Optional[dict[int, int]] = None,
    i_kshell: Optional[dict[int, int]] = None,
    max_distance: int = 3,
    parallel: bool | None = None,
    processes: int | None = None,
) -> Dict[int, float]:
    """DK-IGC uses (degree + i-kshell) as node mass."""

    from vitalnodes.metrics._core import i_kshell as _i_kshell_helper

    paths = paths or _compute_paths(G)
    degree = degree or dict(G.degree())
    i_kshell = i_kshell or _i_kshell_helper(G)
    DK = {n: degree[n] + i_kshell[n] for n in G.nodes()}
    use_mp = parallel if parallel is not None else len(G) >= 500

    def _score(n1: int) -> Tuple[int, float]:
        s = 0.0
        for n2 in _neighbors_within_radius(n1, paths, max_distance):
            s += (DK[n1] * DK[n2]) / (paths[n1][n2] ** 2)
        return n1, s

    return dict(_chunked_pool_map(_score, G.nodes(), use_mp, processes))


def dk_gravity_centrality_agg(    
    # https://www.sciencedirect.com/science/article/abs/pii/S0950705121004603

    G: nx.Graph,
    *,
    dk_igc: Optional[Dict[int, float]] = None,
    **kwargs,
) -> Dict[int, float]:
    """Neighbor-aggregated DK-IGC (ex *DK-IGC+*)."""

    dk_igc = dk_igc or dk_gravity_centrality(G, **kwargs)
    return {n: sum(dk_igc[nbr] for nbr in G.neighbors(n)) for n in G.nodes()}


# Legacy aliases
improved_gravity_centrality2 = dk_gravity_centrality  # type: ignore
improved_gravity_centrality2_plus = dk_gravity_centrality_agg  # type: ignore
dk_gravity_centrality_local = dk_gravity_centrality_agg  # type: ignore

# ---------------------------------------------------------------------------
# 4. Local Gravity Centrality (LGC) – Wu et al., Sci. Rep. 9 (2019)
# ---------------------------------------------------------------------------

# Identifying influential spreaders by gravity model
# https://www.nature.com/articles/s41598-019-44930-9

def local_gravity_centrality(
    G: nx.Graph,
    *,
    paths: Optional[dict[int, dict[int, int]]] = None,
    degree: Optional[dict[int, int]] = None,
    avg_shortest_path: Optional[float] = None,
    parallel: bool | None = None,
    processes: int | None = None,
) -> Dict[int, float]:
    """LGC restricts interaction radius to ⌈L/2⌉ where *L* is the average path length."""

    paths = paths or _compute_paths(G)
    degree = degree or dict(G.degree())
    if avg_shortest_path is None:
        avg_shortest_path = nx.average_shortest_path_length(G)
    radius = round(avg_shortest_path / 2)
    use_mp = parallel if parallel is not None else len(G) >= 500

    def _score(n1: int) -> Tuple[int, float]:
        s = 0.0
        for n2 in _neighbors_within_radius(n1, paths, radius):
            s += (degree[n1] * degree[n2]) / (paths[n1][n2] ** 2)
        return n1, s

    return dict(_chunked_pool_map(_score, G.nodes(), use_mp, processes))


# ---------------------------------------------------------------------------
# 5. Multi-Characteristic Gravity Model (MCGM) – Liu et al., Sci. Rep. 12 (2022)
# ---------------------------------------------------------------------------

# Identifying influential spreaders by gravity model considering multi-characteristics of nodes
# https://www.nature.com/articles/s41598-022-14005-3

def mcgm(
    G: nx.Graph,
    *,
    paths: Optional[dict[int, dict[int, int]]] = None,
    degree: Optional[dict[int, int]] = None,
    core_num: Optional[dict[int, int]] = None,
    eigenvec: Optional[dict[int, float]] = None,
    avg_shortest_path: Optional[float] = None,
    parallel: bool | None = None,
    processes: int | None = None,
) -> Dict[int, float]:
    """MCGM blends normalised degree, k-core, and eigenvector centrality."""

    # Lazily compute prerequisites
    paths = paths or _compute_paths(G)
    degree = degree or dict(G.degree())
    core_num = core_num or nx.core_number(G)
    eigenvec = eigenvec or nx.eigenvector_centrality(G, max_iter=1000)

    if avg_shortest_path is None:
        avg_shortest_path = nx.average_shortest_path_length(G)
    radius = math.ceil(avg_shortest_path)

    # Normalisation factors
    dmax = max(degree.values())
    dmedian = median(degree.values())
    xmax = max(eigenvec.values())
    xmedian = median(eigenvec.values())
    kmax = max(core_num.values())
    kmedian = median(core_num.values())

    try:
        alpha = max(dmedian / dmax, xmedian / xmax) / (kmedian / kmax)
    except ZeroDivisionError:
        alpha = 0.0

    use_mp = parallel if parallel is not None else len(G) >= 500

    def _score(n1: int) -> Tuple[int, float]:
        s = 0.0
        for n2 in _neighbors_within_radius(n1, paths, radius):
            nd1 = degree[n1] / dmax if dmax else 0.0
            nd2 = degree[n2] / dmax if dmax else 0.0

            ne1 = eigenvec[n1] / xmax if xmax else 0.0
            ne2 = eigenvec[n2] / xmax if xmax else 0.0

            nk1 = alpha * core_num[n1] / kmax if kmax else 0.0
            nk2 = alpha * core_num[n2] / kmax if kmax else 0.0

            m1 = nd1 + nk1 + ne1
            m2 = nd2 + nk2 + ne2

            s += (m1 * m2) / (paths[n1][n2] ** 2)
        return n1, s

    return dict(_chunked_pool_map(_score, G.nodes(), use_mp, processes))
