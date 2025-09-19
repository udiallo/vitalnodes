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
from typing import Any, Dict, Iterable, List, Optional, Union, Mapping
from multiprocessing import Pool

import networkx as nx

# ── metric imports ───────────────────────────────────────────────────────────
from vitalnodes.metrics.gravity  import (
    gravity_centrality, gravity_centrality_agg,
    improved_gravity_centrality, improved_gravity_centrality_agg,
    dk_gravity_centrality, dk_gravity_centrality_agg,
    local_gravity_centrality, mcgm,
)
from vitalnodes.metrics.temporal_gravity import temporal_gravity_centrality
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
    "tgc":  temporal_gravity_centrality,
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

def _compute_metric_helper(fn, graph, time, kwargs):
    """Helper function for multiprocessing."""
    if not nx.is_connected(graph) and "avg_shortest_path" in inspect.signature(fn).parameters:
        _LOG.warning(f"Graph at index {time} is not connected; skipping avg_shortest_path and metric based on it (%s)", fn.__name__)
        return {n: None for n in graph.nodes()}
    return fn(graph, **kwargs)

def compute_metric(
    G: Union[nx.Graph, List[nx.Graph]],
    name: str,
    *,
    parallel: Optional[bool] = None,
    processes: Optional[int] = None,
    **kwargs: Any,
) -> Union[Dict[Any, float | None], List[Dict[Any, float | None]]]:
    """Compute a single metric by key."""
    if name not in _METRIC_REGISTRY:
        raise ValueError(f"Unknown metric '{name}'. Available: {get_metric_names()}")
    fn = _METRIC_REGISTRY[name]

    if isinstance(G, list):
        if parallel:
            if G[0].number_of_nodes() <= 500:
                _LOG.warning("You should maybe not enable parallel processing for graphs with < 500 nodes.")
            
            if name != 'tgc':
                # Use multiprocessing to compute metric in parallel for each graph
                with Pool(processes=processes) as pool:
                    results = pool.starmap(
                        _compute_metric_helper, [(fn, graph, time, kwargs) for time, graph in enumerate(G)]
                    )
            else:
                _LOG.info("Temporal Gravity Centrality (tgc) is much slower than other metrics")
                results = fn(G, **kwargs)
            return results
        else:
            if G[0].number_of_nodes() >= 500 or len(G) > 100:
                _LOG.info("You maybe want to enable parallel processing for graphs with ≥ 500 nodes or more than 100 graphs unless you set ``parallel=False``.")
            if name != 'tgc':
                return [fn(g, parallel=parallel, processes=processes, **kwargs) for g in G]
            else:
                _LOG.info("Temporal Gravity Centrality (tgc) is much slower than other metrics")
                return fn(G, **kwargs)
    else:
        if G.number_of_nodes() >= 500:
            _LOG.info("You maybe want to enable parallel processing for graphs with ≥ 500 nodes unless you set ``parallel=False``.")
        if not nx.is_connected(G) and "avg_shortest_path" in inspect.signature(fn).parameters:
            _LOG.warning(f"Graph is not connected; skipping avg_shortest_path and metric based on it (%s)", fn.__name__)
            return {n: None for n in G.nodes()}
        return fn(G, parallel=parallel, processes=processes, **kwargs)


# ── batch runner ─────────────────────────────────────────────────────────────
def compute_metrics(
        G: Union[nx.Graph, List[nx.Graph]],
        metrics: Iterable[str],
        *,
        parallel: Optional[bool] = None,
        processes: Optional[int] = None,
        **kwargs: Any,
        ) -> Union[Mapping[str, Mapping[Any, Union[float, None]]], Mapping[int, Mapping[str, Mapping[Any, Union[float, None]]]]]:
    """
    Compute several metrics; shared heavy helpers are done once.
    """
    # 1) collect functions + signatures
    funcs: List[tuple[str, Any]] = []
    param_names: List[List[str]] = []
    for key in metrics:
        if key not in _METRIC_REGISTRY:
            raise ValueError(f"Unknown metric '{key}'. Available: {get_metric_names()}")
        if key != 'tgc':
            fn = _METRIC_REGISTRY[key]
            funcs.append((key, fn))
            param_names.append(list(inspect.signature(fn).parameters.keys()))
        else:
            raise ValueError("Temporal Gravity Centrality (tgc) cannot be computed in batch mode; use compute_metric instead.")

    # 2) detect which pre-computes are needed at least once
    need_degree = any("degree" in p for params in param_names for p in params)
    need_core_iter = any("core_iter" in p for params in param_names for p in params)
    need_core_num = any("core_num" in p for params in param_names for p in params) or need_core_iter
    need_paths = any("paths" in p for params in param_names for p in params)
    need_avg_sp = any("avg_shortest_path" in p for params in param_names for p in params)
    need_i_ks = any("i_ks" in p or "i_kshell" in p for params in param_names for p in params)
    need_clustering = any("clustering" in p for params in param_names for p in params)

    if isinstance(G, list):
        if parallel:
            # Use multiprocessing to compute metrics in parallel for each graph
            with Pool(processes=processes) as pool:
                results = pool.starmap(
                    _compute_metrics_for_graph,
                    [(graph, funcs, param_names, need_degree, need_core_iter, need_core_num, need_paths, need_avg_sp, need_i_ks, need_clustering, metrics, kwargs, time)
                     for time, graph in enumerate(G)]
                )
            return {i: result for i, result in enumerate(results)}
        else:
            temporal_results: Dict[int, Mapping[str, Mapping[Any, float | None]]] = {}
            for time, graph in enumerate(G):
                temporal_results[time] = _compute_metrics_for_graph(
                    graph, funcs, param_names, need_degree, need_core_iter, need_core_num, need_paths, need_avg_sp, need_i_ks, need_clustering, metrics, kwargs, time
                )
            return temporal_results
    else:
        if G.number_of_nodes() >= 500:
            _LOG.info("You maybe want to enable parallel processing for graphs with ≥ 500 nodes unless you set ``parallel=False``.")
        return _compute_metrics_for_graph(
            G, funcs, param_names, need_degree, need_core_iter, need_core_num, need_paths, need_avg_sp, need_i_ks, need_clustering, metrics, kwargs, time=None
        )





def _compute_metrics_for_graph(
        graph: nx.Graph, 
        funcs: List[tuple[str, Any]],
        param_names: List[List[str]], 
        need_degree: bool, 
        need_core_iter: bool, 
        need_core_num: bool, 
        need_paths: bool, 
        need_avg_sp: bool, 
        need_i_ks: bool, 
        need_clustering: bool, 
        metrics: Iterable[str],
        kwargs: Any,
        time: Optional[int] = None
        ) -> Mapping[str, Mapping[Any, Optional[float]]]:
    """Compute metrics for a single graph."""
    degree = dict(graph.degree()) if need_degree else None
    core_num = core_iter = None
    if need_core_iter:
        core_num, core_iter = k_shell_alternative(graph)
    elif need_core_num:
        core_num = nx.core_number(graph)
    paths = dict(nx.all_pairs_shortest_path_length(graph)) if need_paths else None
    avg_sp = nx.average_shortest_path_length(graph) if need_avg_sp and nx.is_connected(graph) else None
    i_ks = i_kshell(graph) if need_i_ks else None
    clustering = nx.clustering(graph) if need_clustering else None

    # Handle disconnected graphs and skip metrics requiring avg_shortest_path
    remove_keys = []
    if avg_sp is None and need_avg_sp:
        remove_keys = [key for key, params in zip(metrics, param_names) if "avg_shortest_path" in params] + ["ninl_layer0"]
        _LOG.warning("Graph at index %d is not connected; skipping avg_shortest_path and metrics based on it (%s)", time, remove_keys)

    remove_keys += ["mcde"]
    static_results: Dict[str, Dict[Any, Optional[float]]] = {}
    for key, fn in funcs:
        if key not in remove_keys:
            common: Dict[str, Any] = {
                "degree": degree,
                "core_num": core_num,
                "core_iter": core_iter,
                "paths": paths,
                "avg_shortest_path": avg_sp,
                "i_ks": i_ks,
                "clustering": clustering,
                "time_step": time,
                **kwargs,
            }
            allowed = inspect.signature(fn).parameters
            filtered = {k: v for k, v in common.items() if k in allowed and v is not None}
            _LOG.debug("→ %s gets %s", key, list(filtered.keys()))
            static_results[key] = fn(graph, **filtered)
        else:
            static_results[key] = {n: None for n in graph.nodes()}
            _LOG.debug("→ %s skipped due to disconnected graph", key)

    return static_results