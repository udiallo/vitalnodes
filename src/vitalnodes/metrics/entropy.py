"""entropy.py
================
Entropy-based influential-node measures.

This family relies on the **uncertainty of neighbours’ core/degree
distribution**.  All functions can be invoked with *just* a NetworkX graph;
they lazily compute prerequisites but accept pre-computed ones to
short-circuit duplicate work.

Public API
----------

* :func:`mcde`               – returns either MCDE **or** MCDWE
* :func:`erm`                – Entropy-Rank Metric
* :func:`dsr`                – first-order DSR
* :func:`dsr_agg` (alias **edsr**) – neighbour-aggregated DSR (second order)
* :func:`ecrm`               – Entropy Core-Rank Metric

Neighbour-aggregated variants follow the ``*_agg`` suffix convention.
"""
from __future__ import annotations

import logging
import math
from multiprocessing import Pool, cpu_count
from statistics import median
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx

# ---------------------------------------------------------------------------

__all__ = [
    "mcde",
    "mcde_weighted",
    "erm",
    "dsr",
    "dsr_agg",
    "edsr",
    "ecrm",
]

_LOG = logging.getLogger(__name__)

# common helpers ------------------------------------------------------------


def _chunked_pool_map(func, iterable, parallel: bool, processes: int | None):
    """Run *func* over *iterable* either serially or through a Pool()."""
    if not parallel:
        return map(func, iterable)
    procs = processes or max(cpu_count() - 1, 1)
    with Pool(procs) as pool:
        return pool.map(func, iterable)


def _shannon_entropy(vals: Iterable[float]) -> float:
    """Simple Shannon entropy (base-e)."""
    s = 0.0
    for p in vals:
        if p > 0:
            s -= p * math.log(p)
    return s


# ---------------------------------------------------------------------------
# 1. MCDE / MCDWE – Yang et al., Journal of IS 42 (2016)
# ---------------------------------------------------------------------------

# https://journals.sagepub.com/doi/10.1177/0165551516644171

def mcde(
    G: nx.Graph,
    *,
    degree: Optional[dict[int, int]] = None,
    core_num: Optional[dict[int, int]] = None,
    weighted: bool = False,
    parallel: bool | None = None,
    processes: int | None = None,
) -> Dict[int, float]:
    """
    **Multicore-degree Entropy (MCDE)**

    Parameters
    ----------
    weighted
        If *True*, returns the *weighted* variant (MCDWE).
    """
    degree = degree or dict(G.degree())
    core_num = core_num or nx.core_number(G)

    max_core = max(core_num.values())
    sorted_cores = sorted(set(core_num.values()))

    use_mp = parallel if parallel is not None else len(G) >= 500

    def _score(node: int) -> Tuple[int, float]:
        nbrs = list(G.neighbors(node))
        if not nbrs:  # isolate
            return node, 0.0

        # probability that a neighbour resides in core *c*
        probs: List[float] = []
        for c in sorted_cores:
            probs.append(sum(1 for v in nbrs if core_num[v] == c) / len(nbrs))

        entropy = _shannon_entropy(probs)

        if weighted:
            weights = [(1 / (max_core - c + 1)) for c in sorted_cores]
            w_entropy = _shannon_entropy(w * p for w, p in zip(weights, probs))
            entropy = w_entropy

        mcde_val = core_num[node] + degree[node] + entropy
        return node, mcde_val

    return dict(_chunked_pool_map(_score, G.nodes(), use_mp, processes))


# convenience alias
mcde_weighted = lambda *a, **kw: mcde(*a, weighted=True, **kw)  # noqa: E731

# ---------------------------------------------------------------------------
# 2. ERM – Zhang & Cui, Chaos 27 (2017)
# ---------------------------------------------------------------------------

# https://www.sciencedirect.com/science/article/abs/pii/S0960077917303788
# Influential nodes ranking in complex networks: An entropy-based approach

def erm(
    G: nx.Graph,
    *,
    degree: Optional[dict[int, int]] = None,
    parallel: bool | None = None,
    processes: int | None = None,
) -> Dict[int, float]:
    """Entropy-based Ranking Metric (two-layer neighbourhood)."""
    degree = degree or dict(G.degree())
    use_mp = parallel if parallel is not None else len(G) >= 500

    # pre-compute first- and second-layer degree sums
    d1 = {n: sum(degree[v] for v in G.neighbors(n)) for n in G.nodes()}
    d2 = {n: sum(d1[v] for v in G.neighbors(n)) for n in G.nodes()}

    def _entropy_contrib(node: int) -> Tuple[int, float]:
        nbrs = list(G.neighbors(node))
        if not nbrs:
            return node, 0.0

        E1 = -sum((degree[v] / d1[node]) * math.log(degree[v] / d1[node]) for v in nbrs)
        E2 = -sum((d1[v] / d2[node]) * math.log(d1[v] / d2[node]) for v in nbrs)

        lam = E2 / max(E2 for E2 in d2.values()) if max(d2.values()) else 0.0
        EC = sum(E1 + lam * E2 for v in nbrs)
        SI = sum(EC for v in nbrs)
        return node, SI

    return dict(_chunked_pool_map(_entropy_contrib, G.nodes(), use_mp, processes))


# ---------------------------------------------------------------------------
# 3. DSR / EDSR – Chen et al., FGCS 108 (2020)
# ---------------------------------------------------------------------------

# https://www.sciencedirect.com/science/article/abs/pii/S0167739X18319009

def dsr(
    G: nx.Graph,
    *,
    degree: Optional[dict[int, int]] = None,
    core_num: Optional[dict[int, int]] = None,
    core_iter: Optional[dict[int, int]] = None,
    parallel: bool | None = None,
    processes: int | None = None,
) -> Dict[int, float]:
    """Diversity-sensitive rank (1-hop)."""

    from vitalnodes.metrics._core import k_shell_alternative  # heavy but cached

    degree = degree or dict(G.degree())

    if core_num is None or core_iter is None:
        core_num, core_iter = k_shell_alternative(G)

    max_iter = max(core_iter.values())
    s_ni = {
        n: core_num[n] * (1 + core_iter[n] / max_iter) for n in G.nodes()
    }  # k-shell iteration factor

    # nsd: neighbour strength degree
    nsd = {
        n: sum(s_ni[v] * degree[v] for v in G.neighbors(n)) for n in G.nodes()
    }

    dsr_dict = {n: s_ni[n] * degree[n] + nsd[n] for n in G.nodes()}
    return dsr_dict


def dsr_agg(
    G: nx.Graph,
    *,
    dsr_scores: Optional[Dict[int, float]] = None,
    **kwargs,
) -> Dict[int, float]:
    """Aggregated DSR (== *E*DSR)."""
    dsr_scores = dsr_scores or dsr(G, **kwargs)
    return {n: sum(dsr_scores[v] for v in G.neighbors(n)) for n in G.nodes()}


# legacy alias
edsr = dsr_agg  # type: ignore

# ---------------------------------------------------------------------------
# 4. ECRM – Zhou et al., KBS 196 (2020)
# ---------------------------------------------------------------------------

# https://www.sciencedirect.com/science/article/abs/pii/S0950705120300630

def ecrm(
    G: nx.Graph,
    *,
    degree: Optional[dict[int, int]] = None,
    core_iter: Optional[dict[int, int]] = None,
    parallel: bool | None = None,
    processes: int | None = None,
) -> Dict[int, float]:
    """
    **Enhanced Core-Rank Metric (ECRM)**

    Uses correlation between core-iteration vectors of neighbours.
    """
    from vitalnodes.metrics._core import k_shell_alternative

    degree = degree or dict(G.degree())
    if core_iter is None:
        _, core_iter = k_shell_alternative(G)

    max_iter = max(core_iter.values())
    sorted_iters = sorted(set(core_iter.values()))

    # build “signature” vector for every node
    sv = {}
    for n in G.nodes():
        vec = []
        nbrs = list(G.neighbors(n))
        for c in sorted_iters:
            vec.append(sum(1 for v in nbrs if core_iter[v] == c))
        sv[n] = vec

    use_mp = parallel if parallel is not None else len(G) >= 500

    def _score(node: int) -> Tuple[int, float]:
        nbrs = list(G.neighbors(node))
        if not nbrs:
            return node, 0.0

        scc = 0.0
        for v in nbrs:
            # cosine-like similarity in core-iteration space
            num = sum(
                (sv[node][i] - degree[node] / max_iter)
                * (sv[v][i] - degree[v] / max_iter)
                for i in range(len(sorted_iters))
            )
            den = math.sqrt(
                sum((sv[node][i] - degree[node] / max_iter) ** 2 for i in range(len(sorted_iters)))
            ) * math.sqrt(
                sum((sv[v][i] - degree[v] / max_iter) ** 2 for i in range(len(sorted_iters)))
            )
            corr = num / den if den else 0.0
            scc += (2 - corr) + ((2 * degree[v] / max(degree.values())) + 1)

        crm = sum(scc for v in nbrs)  # neighbour aggregation
        ecrm_val = sum(crm for v in nbrs)  # two-hop aggregation
        return node, ecrm_val

    return dict(_chunked_pool_map(_score, G.nodes(), use_mp, processes))
