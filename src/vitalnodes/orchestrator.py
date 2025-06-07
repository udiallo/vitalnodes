# ──────────────────────────────────────────────────────────────────────────────
# src/vitalnodes/orchestrator.py
"""
Central façade for Vital-Node metrics.

>>> import networkx as nx
>>> from vitalnodes.orchestrator import compute_metric, compute_metrics
>>> G = nx.karate_club_graph()
>>> single = compute_metric(G, "gc", parallel=False)
>>> batch  = compute_metrics(G, ["gc", "erm", "ninl"], parallel=False)
"""
# -----------------------------------------------------------------------------
import inspect
import logging
from typing import Any, Dict, Iterable, List, Optional

import networkx as nx

# ── metric imports ───────────────────────────────────────────────────────────
from vitalnodes.metrics.gravity  import (
    gravity_centrality, gravity_centrality_agg,
    improved_gravity_centrality, improved_gravity_centrality_agg,
    dk_gravity_centrality, dk_gravity_centrality_agg,
    local_gravity_centrality, mcgm,
)
from vitalnodes.metrics.entropy  import mcde, mcde_weighted, erm, dsr, dsr_agg, ecrm
from vitalnodes.metrics.ninl     import ninl, ninl_layer0
from vitalnodes.metrics.density  import density_centrality, clustered_local_degree
from vitalnodes.metrics.gli      import gli, gli_new
from vitalnodes.metrics.hindex   import h_index, local_h_index
from vitalnodes.metrics.ls       import ls_influence

# helpers
from vitalnodes.metrics._core    import k_shell_alternative, i_kshell

_LOG = logging.getLogger(__name__)

# ── registry: key → function ─────────────────────────────────────────────────
_METRIC_REGISTRY: Dict[str, Any] = {
    # gravity
    "gc":   gravity_centrality,
    "gc+":  gravity_centrality_agg,
    "igc":  improved_gravity_centrality,
    "igc+": improved_gravity_centrality_agg,
    "dk":   dk_gravity_centrality,
    "dk+":  dk_gravity_centrality_agg,
    "lgc":  local_gravity_centrality,
    "mcgm": mcgm,
    # entropy
    "mcde":          mcde,
    "mcde_weighted": mcde_weighted,
    "erm":           erm,
    "dsr":           dsr,
    "dsr_agg":       dsr_agg,
    "ecrm":          ecrm,
    # ninl
    "ninl":        ninl,
    "ninl_layer0": ninl_layer0,
    # density
    "density": density_centrality,
    "cld":     clustered_local_degree,
    # GLI
    "gli":     gli,
    "gli_new": gli_new,
    # h-index
    "h_index":       h_index,
    "local_h_index": local_h_index,
    # link-strength
    "ls": ls_influence,
}

# ── public helpers ───────────────────────────────────────────────────────────
def get_metric_names() -> List[str]:
    """Return all supported metric keys."""
    return list(_METRIC_REGISTRY.keys())


def compute_metric(
    G: nx.Graph,
    name: str,
    *,
    parallel: Optional[bool] = None,
    processes: Optional[int] = None,
    **kwargs: Any,
) -> Dict[Any, float]:
    """Compute a single metric by key."""
    if name not in _METRIC_REGISTRY:
        raise ValueError(f"Unknown metric '{name}'. Available: {get_metric_names()}")
    fn = _METRIC_REGISTRY[name]
    return fn(G, parallel=parallel, processes=processes, **kwargs)


# ── batch runner ─────────────────────────────────────────────────────────────
def compute_metrics(
    G: nx.Graph,
    metrics: Iterable[str],
    *,
    parallel : Optional[bool] = None,
    processes: Optional[int] = None,
    **kwargs : Any,
) -> Dict[str, Dict[Any, float]]:
    """
    Compute several metrics; shared heavy helpers are done once.
    """
    # 1) collect functions + signatures
    funcs: List[tuple[str, Any]] = []
    sigs : List[Dict[str, inspect.Parameter]] = []
    for key in metrics:
        if key not in _METRIC_REGISTRY:
            raise ValueError(f"Unknown metric '{key}'. Available: {get_metric_names()}")
        fn = _METRIC_REGISTRY[key]
        funcs.append((key, fn))
        sigs.append(inspect.signature(fn).parameters)

    # 2) detect which pre-computes are needed at least once
    need_degree     = any("degree"              in p for sig in sigs for p in sig)
    need_core_iter  = any("core_iter"           in p for sig in sigs for p in sig)
    need_core_num   = any("core_num"            in p for sig in sigs for p in sig) or need_core_iter
    need_paths      = any("paths"               in p for sig in sigs for p in sig)
    need_avg_sp     = any("avg_shortest_path"   in p for sig in sigs for p in sig)
    need_i_ks       = any("i_ks" in p or "i_kshell" in p for sig in sigs for p in sig)
    need_clustering = any("clustering"          in p for sig in sigs for p in sig)

    # 3) run each helper once
    degree     = dict(G.degree())                          if need_degree     else None
    core_num   = core_iter = None
    if need_core_iter:
        core_num, core_iter = k_shell_alternative(G)
    elif need_core_num:
        core_num = nx.core_number(G)
    paths      = dict(nx.all_pairs_shortest_path_length(G)) if need_paths      else None
    avg_sp     = nx.average_shortest_path_length(G)         if need_avg_sp     else None
    i_ks       = i_kshell(G)                                if need_i_ks       else None
    clustering = nx.clustering(G)                           if need_clustering else None

    # 4) call each metric with *only* the args it accepts
    results: Dict[str, Dict[Any, float]] = {}
    for key, fn in funcs:
        common: Dict[str, Any] = {
            "parallel":           parallel,
            "processes":          processes,
            "degree":             degree,
            "core_num":           core_num,
            "core_iter":          core_iter,
            "paths":              paths,
            "avg_shortest_path":  avg_sp,
            "i_ks":               i_ks,
            "clustering":         clustering,
            **kwargs,
        }
        allowed = inspect.signature(fn).parameters
        filtered = {k: v for k, v in common.items() if k in allowed and v is not None}
        _LOG.debug("→ %s gets %s", key, list(filtered.keys()))
        results[key] = fn(G, **filtered)

    return results
