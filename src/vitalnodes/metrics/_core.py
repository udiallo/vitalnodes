"""_core.py
================
Foundational helpers shared across metric modules.

Only two public functions are exported for now:

* :func:`k_shell_alternative`  – returns both *k*-shell and iteration index
* :func:`i_kshell`             – “improved k-shell” (k + iteration)

They are intentionally kept *very thin* so other heavy utilities can live in
`vitalnodes.utils`.
"""
from __future__ import annotations

import networkx as nx
from typing import Dict, Tuple

__all__ = ["k_shell_alternative", "i_kshell"]


# ---------------------------------------------------------------------------#
# internal helpers exactly as in the research paper                          #
# ---------------------------------------------------------------------------#


def _check(G: nx.Graph, k: int) -> int:
    """Return 1 while there is at least one node with degree ≤ *k*."""
    for n in list(G.nodes()):
        if G.degree(n) <= k:
            return 1
    return 0


def _find_nodes(G: nx.Graph, k: int) -> list[int]:
    """Return nodes whose current degree ≤ *k*."""
    return [n for n in list(G.nodes()) if G.degree(n) <= k]


# ---------------------------------------------------------------------------#
# public hooks                                                               #
# ---------------------------------------------------------------------------#


def k_shell_alternative(G: nx.Graph) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Compute *k*-shell (core number) **and** the iteration at which each node
    was removed (`core-iteration index`), as required by DSR/ECRM.
    """
    h = G.copy()
    k = 1
    buckets: list[list[int]] = []
    tmp: list[int] = []
    iter_idx: dict[int, int] = {}
    count = 0

    while True:
        count += 1
        if _check(h, k) == 0:
            k += 1
            buckets.append(tmp)
            tmp = []
        else:
            nodes = _find_nodes(h, k)
            for v in nodes:
                h.remove_node(v)
                tmp.append(v)
                iter_idx[v] = count
        if h.number_of_nodes() == 0:
            buckets.append(tmp)
            break

    k_shell: dict[int, int] = {}
    for n in G.nodes():
        for idx, shell in enumerate(buckets, 1):
            if n in shell:
                k_shell[n] = idx
                break

    return k_shell, iter_idx


def i_kshell(
    G: nx.Graph,
    core_num: dict[int, int] | None = None,
    core_iter: dict[int, int] | None = None,
) -> Dict[int, int]:
    """Improved k-shell = k-core value + iteration removed."""
    if core_num is None or core_iter is None:
        core_num, core_iter = k_shell_alternative(G)
    return {n: core_num[n] + core_iter[n] for n in G.nodes()}
