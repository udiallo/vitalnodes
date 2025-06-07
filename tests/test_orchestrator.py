# src/vitalnodes/orchestrator.py
"""
Central façade for computing all Vital‐Node metrics in a uniform way.

You can do:

    from vitalnodes.orchestrator import compute_metric, compute_metrics, get_metric_names

    G = nx.karate_club_graph()
    scores = compute_metric(G, "gc")
    batch  = compute_metrics(G, ["gc", "erm", "ninl"])
"""
from typing import Any, Dict, Iterable, Optional
import networkx as nx

# gravity
from vitalnodes.metrics.gravity import (
    gravity_centrality,
    gravity_centrality_agg,
    improved_gravity_centrality,
    improved_gravity_centrality_agg,
    dk_gravity_centrality,
    dk_gravity_centrality_agg,
    local_gravity_centrality,
    mcgm,
)

# entropy‐based
from vitalnodes.metrics.entropy import (
    mcde,
    mcde_weighted,
    erm,
    dsr,
    dsr_agg,
    ecrm,
)

# NINL
from vitalnodes.metrics.ninl import ninl, ninl_layer0

# density‐based
from vitalnodes.metrics.density import density_centrality, clustered_local_degree

# GLI
from vitalnodes.metrics.gli import gli, gli_new

# h-index
from vitalnodes.metrics.hindex import h_index, local_h_index

# LS
from vitalnodes.metrics.ls import ls_influence

# ---------------------------------------------------------------------------
# Registry: map string keys to functions
# ---------------------------------------------------------------------------
_METRIC_REGISTRY: Dict[str, Any] = {
    # gravity‐family
    "gc":   gravity_centrality,
    "gc+":  gravity_centrality_agg,
    "igc":  improved_gravity_centrality,
    "igc+": improved_gravity_centrality_agg,
    "dk":   dk_gravity_centrality,
    "dk+":  dk_gravity_centrality_agg,
    "lgc":  local_gravity_centrality,
    "mcgm": mcgm,

    # entropy‐family
    "mcde":            mcde,
    "mcde_weighted":   mcde_weighted,
    "erm":             erm,
    "dsr":             dsr,
    "edsr":            dsr_agg,
    "ecrm":            ecrm,

    # NINL
    "ninl":       ninl,
    "ninl_layer0": ninl_layer0,

    # density‐family
    "density": density_centrality,
    "cld":     clustered_local_degree,

    # GLI
    "gli":     gli,
    "gli_new": gli_new,

    # h-index
    "h_index":       h_index,
    "local_h_index": local_h_index,

    # LS
    "ls": ls_influence,
}


def get_metric_names() -> Iterable[str]:
    """
    Return all supported metric keys.
    """
    return list(_METRIC_REGISTRY.keys())


def compute_metric(
    G: nx.Graph,
    metric: str,
    *,
    parallel: Optional[bool] = None,
    processes: Optional[int] = None,
    **kwargs: Any
) -> Dict[Any, float]:
    """
    Compute a single centrality metric by name.

    Parameters
    ----------
    G
        A NetworkX graph.
    metric
        One of the keys returned by get_metric_names().
    parallel
        Override default multiprocessing logic (True/False).
    processes
        If parallel=True, number of worker processes to use.
    **kwargs
        Passed straight to the metric function (e.g. max_distance, weighted…).

    Returns
    -------
    A dict mapping node → score.
    """
    if metric not in _METRIC_REGISTRY:
        raise ValueError(
            f"Unknown metric '{metric}'. Available: {', '.join(get_metric_names())}"
        )
    func = _METRIC_REGISTRY[metric]
    # All our metrics accept signature (G, *, parallel, processes, **other)
    return func(G, parallel=parallel, processes=processes, **kwargs)


def compute_metrics(
    G: nx.Graph,
    metrics: Iterable[str],
    *,
    parallel: Optional[bool] = None,
    processes: Optional[int] = None,
    **kwargs: Any
) -> Dict[str, Dict[Any, float]]:
    """
    Compute multiple metrics in one go.

    Parameters
    ----------
    G
        A NetworkX graph.
    metrics
        An iterable of metric keys.
    parallel
        Override default multiprocessing logic for *all* metrics.
    processes
        If parallel=True, number of worker processes to use.
    **kwargs
        Passed to each metric call.

    Returns
    -------
    A dict mapping each metric key → (node → score) dict.
    """
    results: Dict[str, Dict[Any, float]] = {}
    for m in metrics:
        results[m] = compute_metric(
            G,
            m,
            parallel=parallel,
            processes=processes,
            **kwargs
        )
    return results
