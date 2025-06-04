"""hindex.py
================
* **H-index centrality** – node’s neighbours’ degree distribution
* **Local H-index**      – 1-hop aggregation of H-index scores
"""
from __future__ import annotations

import logging
from multiprocessing import Pool, cpu_count
from typing import Dict, Optional, Tuple, List

import networkx as nx

from vitalnodes.metrics._utils import _chunked_pool_map


__all__ = ["h_index", "local_h_index"]

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------#
# Worker functions                                                           #
# ---------------------------------------------------------------------------#


def _hindex_worker(args: Tuple[int, Dict[int, int], List[int]]) -> Tuple[int, int]:
    n, degrees, nbrs = args
    sorted_deg = sorted((degrees[v] for v in nbrs), reverse=True)
    h = 0
    for i, d in enumerate(sorted_deg, 1):
        if d >= i:
            h = i
        else:
            break
    return n, h


# ---------------------------------------------------------------------------#
# 1. H-index                                                                  #
# ---------------------------------------------------------------------------#


def h_index(
    G: nx.Graph,
    *,
    parallel: bool | None = None,
    processes: int | None = None,
) -> Dict[int, int]:
    """H-index of each node’s neighbour degree list."""

    degrees = dict(G.degree())
    use_mp = parallel if parallel is not None else len(G) >= 500

    # Precompute neighbour lists
    nbrs_dict: Dict[int, List[int]] = {n: list(G.neighbors(n)) for n in G.nodes()}

    payload: List[Tuple[int, Dict[int, int], List[int]]] = [
        (n, degrees, nbrs_dict[n]) for n in G.nodes()
    ]
    return dict(_chunked_pool_map(_hindex_worker, payload, use_mp, processes))


# ---------------------------------------------------------------------------#
# 2. Local H-index (aggregated)                                              #
# ---------------------------------------------------------------------------#


def local_h_index(
    G: nx.Graph,
    *,
    h_scores: Optional[Dict[int, int]] = None,
    **kwargs,
) -> Dict[int, int]:
    """Sum of a node’s own H and all neighbours’ H values."""
    h_scores = h_scores or h_index(G, **kwargs)
    return {n: h_scores[n] + sum(h_scores[v] for v in G.neighbors(n)) for n in G.nodes()}
