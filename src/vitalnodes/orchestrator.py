# src/vitalnodes/orchestrator.py
from typing import Any, Dict, List
import networkx as nx

# import all your metric functions
from vitalnodes.metrics.gravity   import (
    gravity_centrality,
    gravity_centrality_agg,
    improved_gravity_centrality,
    improved_gravity_centrality_agg,
    dk_gravity_centrality,
    dk_gravity_centrality_agg,
    local_gravity_centrality,
    mcgm,
)
from vitalnodes.metrics.entropy   import mcde, mcde_weighted, erm, dsr, dsr_agg, edsr, ecrm
from vitalnodes.metrics.ninl      import ninl, ninl_layer0
from vitalnodes.metrics.density   import density_centrality, clustered_local_degree
from vitalnodes.metrics.gli       import gli, gli_new
from vitalnodes.metrics.hindex    import h_index, local_h_index
from vitalnodes.metrics.ls        import ls_influence

_METRIC_REGISTRY: Dict[str, Any] = {
    # gravity‐family
    "gravity":        gravity_centrality,
    "gravity_agg":    gravity_centrality_agg,
    "igc":            improved_gravity_centrality,
    "igc_agg":        improved_gravity_centrality_agg,
    "dk_igc":         dk_gravity_centrality,
    "dk_igc_agg":     dk_gravity_centrality_agg,
    "lgc":            local_gravity_centrality,
    "mcgm":           mcgm,
    # entropy‐family
    "mcde":           mcde,
    "mcde_w":         mcde_weighted,
    "erm":            erm,
    "dsr":            dsr,
    "dsr_agg":        dsr_agg,
    "edsr":           edsr, 
    "ecrm":           ecrm,
    # ninl
    "ninl":           ninl,
    "ninl0":          ninl_layer0,
    # density
    "density":        density_centrality,
    "cld":            clustered_local_degree,
    # gli
    "gli":            gli,
    "gli_new":        gli_new,
    # h‐index
    "h_index":        h_index,
    "h_index_agg":    local_h_index,
    # link‐strength
    "ls":             ls_influence,
}

def compute_metric(
    G: nx.Graph,
    name: str,
    **kwargs: Any
) -> Dict[int, float]:
    """
    Compute a single metric by name.
    Available: """ + ", ".join(sorted(_METRIC_REGISTRY.keys())) + """
    """
    try:
        fn = _METRIC_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown metric '{name}'. Choose from: {list(_METRIC_REGISTRY)}")
    return fn(G, **kwargs)

def compute_metrics(
    G: nx.Graph,
    names: List[str],
    **shared_kwargs: Any
) -> Dict[str, Dict[int, float]]:
    """Compute multiple metrics and return a dict name→scores."""
    return {n: compute_metric(G, n, **shared_kwargs) for n in names}
