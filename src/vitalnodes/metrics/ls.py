"""ls.py
================
Link-strength based influence (Li & Shu, Physica A 516 (2019)).

The LS measure uses neighbour similarity (1 – Jaccard distance) to modulate a
k-core-weighted propagation.

https://www.sciencedirect.com/science/article/pii/S0378437118310707
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, List, Set

import networkx as nx
from vitalnodes.metrics._utils import _chunked_pool_map

__all__ = ["ls_influence"]

_LOG = logging.getLogger(__name__)


def _jaccard(a: set[int], b: set[int]) -> float:
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# ---------------------------------------------------------------------------#
# Worker function (module-level for multiprocessing)                          #
# ---------------------------------------------------------------------------#

def _ls_worker(
    args: Tuple[
        int,                            # node
        Dict[int, int],                 # degree
        Dict[int, int],                 # core_num (k_S)
        Dict[int, Set[int]],            # nbr_sets (neighbor sets of each node)
        Dict[Tuple[int, int], float]    # ls_edge  (precomputed Ls for each undirected edge)
    ]
) -> Tuple[int, float]:
    """
    Worker for LS influence. Implements exactly:
      1) k'_S(v) = ( sum_{x in N_v} L_{v,x} ) / deg(v) * k_S(v)
      2) I_z   = sum_{v in N_z} [ L_{z,v} * k'_S(v) ].

    args = (node, degree, core_num, nbr_sets, ls_edge).
    """
    node, degree, core_num, nbr_sets, ls_edge = args
    nbrs = nbr_sets[node]
    if not nbrs:
        return node, 0.0

    influence = 0.0
    for v in nbrs:
        # 1) Compute k'_S(v):
        sum_Ls_to_v = 0.0
        for x in nbr_sets[v]:
            sum_Ls_to_v += ls_edge[(v, x)]
        k_prime_S_v = (sum_Ls_to_v / degree[v]) * core_num[v]

        # 2) Accumulate: I_node += L_{node,v} * k'_S(v)
        influence += ls_edge[(node, v)] * k_prime_S_v

    return node, influence


# ---------------------------------------------------------------------------#
# Main routine                                                               #
# ---------------------------------------------------------------------------#

def ls_influence(
    G: nx.Graph,
    *,
    degree: Optional[dict[int, int]] = None,
    core_num: Optional[dict[int, int]] = None,
    parallel: bool | None = None,
    processes: int | None = None,
) -> Dict[int, float]:
    """Compute the Link-Strength (LS) influence score exactly as in Li & Shu (2019)."""

    degree   = degree or dict(G.degree())
    core_num = core_num or nx.core_number(G)
    use_mp   = parallel if parallel is not None else len(G) >= 500

    # 1) Build neighbour‐set for each node:
    nbr_sets: Dict[int, Set[int]] = {n: set(G.neighbors(n)) for n in G.nodes()}

    # 2) Precompute Ls for each (undirected) edge:
    #    Ls_{u,v} = 1 - |N_u ∩ N_v| / |N_u ∪ N_v|.
    ls_edge: Dict[Tuple[int, int], float] = {}
    for u, v in G.edges():
        jacc = 1.0 - _jaccard(nbr_sets[u], nbr_sets[v])
        ls_edge[(u, v)] = ls_edge[(v, u)] = jacc

    # 3) Build payload and run worker in parallel/serial
    payload: List[
        Tuple[int, Dict[int, int], Dict[int, int], Dict[int, Set[int]], Dict[Tuple[int, int], float]]
    ] = [(n, degree, core_num, nbr_sets, ls_edge) for n in G.nodes()]

    return dict(_chunked_pool_map(_ls_worker, payload, use_mp, processes))
