"""temporal_gravity.py
================
* **Temporal Gravity Centrality** – measures the influence of nodes over time
(Jialin Bi, Ji Jin, Cunquan Qu, Xiuxiu Zhan, Guanghui Wang, Guiying Yan,
Temporal gravity model for important node identification in temporal networks)
"""

from __future__ import annotations

import logging

from typing import List, Dict, Optional
from collections import deque
import networkx as nx
import numpy as np

__all__ = ["temporal_gravity_centrality"]
_LOG = logging.getLogger(__name__)


def matrix_to_numpy(matrix, nodes):
    """Convert matrix to numpy array (cost or path length)."""
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    result = np.full((n, n), np.inf)  # Initialize with inf

    for u in nodes:
        for v in nodes:
            value = matrix[u][v][0]  # cost or arrival time
            result[idx[u]][idx[v]] = value

    return result


def get_degree_vector(snapshots, temporal_or_static='static'):
    nodes = get_nodes(snapshots)  # Consistent ordering
    N = len(nodes)
    
    if temporal_or_static == 'average_over_time':
        D = []
        for t, G in enumerate(snapshots):
            degree_vector = [G.degree(node) for node in nodes]  # Use consistent ordering
            D.append(degree_vector)
        return np.mean(D, axis=0)/(N-1)
    elif temporal_or_static == 'static':
        static_graph = get_aggregated_static_network(snapshots)
        degree_vector = [static_graph.degree(node) for node in nodes]  # Use consistent ordering
        return np.array(degree_vector) / (N-1)


def get_closeness_vector(snapshots, temporal_or_static='static'):
    nodes = get_nodes(snapshots)  # Consistent ordering
    if temporal_or_static == 'average_over_time':
        C = []
        for t, G in enumerate(snapshots):
            closeness_vector = [nx.closeness_centrality(G, node) for node in nodes]
            C.append(closeness_vector)
        return np.mean(C, axis=0)
    elif temporal_or_static == 'static':
        static_graph = get_aggregated_static_network(snapshots)
        closeness_vector = [nx.closeness_centrality(static_graph, node) for node in nodes]
        return np.array(closeness_vector)

def get_betweenness_vector(snapshots, temporal_or_static='static'):
    nodes = get_nodes(snapshots)  # Get consistent node ordering
    
    if temporal_or_static == 'average_over_time':
        B = []
        for t, G in enumerate(snapshots):
            betweenness_dict = nx.betweenness_centrality(G)
            # Use consistent node ordering instead of dict.values()
            betweenness_vector = [betweenness_dict.get(node, 0) for node in nodes]
            B.append(betweenness_vector)
        return np.mean(B, axis=0)
    elif temporal_or_static == 'static':
        static_graph = get_aggregated_static_network(snapshots)
        betweenness_dict = nx.betweenness_centrality(static_graph)
        # Use consistent node ordering instead of dict.values()
        betweenness_vector = [betweenness_dict.get(node, 0) for node in nodes]
        return np.array(betweenness_vector)

def get_page_rank_vector(snapshots, temporal_or_static='static'):
    nodes = get_nodes(snapshots)  # Get consistent node ordering
    
    if temporal_or_static == 'average_over_time':
        P = []
        for t, G in enumerate(snapshots):
            page_rank_dict = nx.pagerank(G)
            # Use consistent node ordering instead of dict.values()
            page_rank_vector = [page_rank_dict.get(node, 0) for node in nodes]
            P.append(page_rank_vector)
        return np.mean(P, axis=0)
    elif temporal_or_static == 'static':
        static_graph = get_aggregated_static_network(snapshots)
        page_rank_dict = nx.pagerank(static_graph)
        # Use consistent node ordering instead of dict.values()
        page_rank_vector = [page_rank_dict.get(node, 0) for node in nodes]
        return np.array(page_rank_vector)
    
def get_time_degree_vector(snapshots):
    nodes = get_nodes(snapshots)  # Get consistent node ordering
    TD = []
    for t, G in enumerate(snapshots):
        degree_centrality_dict = nx.degree_centrality(G)
        # Use consistent node ordering instead of dict.values()
        degree_centrality_vector = [degree_centrality_dict.get(node, 0) for node in nodes]
        TD.append(np.exp(degree_centrality_vector))
    return np.sum(TD, axis=0)


def build_temporal_edges(graph_snapshots):
    """Creates a list of bidirectional time-stamped edges from undirected snapshots."""
    temporal_edges = []
    for t, G in enumerate(graph_snapshots):
        for u, v in G.edges():
            temporal_edges.append((u, v, t))
            temporal_edges.append((v, u, t))  # Add reverse edge
    return sorted(temporal_edges, key=lambda x: x[2])  # Sort by time

def get_nodes(graph_snapshots):
    """Returns the union of all nodes across all snapshots."""
    nodes = set()
    for G in graph_snapshots:
        nodes.update(G.nodes())
    return sorted(nodes)

def get_aggregated_static_network(snapshots):
    """Create an aggregated static network from temporal snapshots."""
    aggregated_graph = nx.Graph()
    
    # Add all nodes from all snapshots
    for G in snapshots:
        aggregated_graph.add_nodes_from(G.nodes())
    
    # Add all edges from all snapshots (duplicates will be ignored)
    for G in snapshots:
        aggregated_graph.add_edges_from(G.edges())
    
    return aggregated_graph

def _temporal_neighbors_within_radius(
    node, 
    distance_matrix, 
    radius
    ):
    """Yield nodes reachable within radius (temporal distance)."""
    for nbr, (dist, _) in distance_matrix[node].items():
        if nbr != node and dist <= radius:
            yield nbr


def _temporal_gravity_worker(args):
    node, distance_matrix, M, radius = args
    s = 0.0
    for nbr in _temporal_neighbors_within_radius(node, distance_matrix, radius):
        dist, _ = distance_matrix[node][nbr]
        if dist > 0:  # avoid div by zero
            s += (M[node] * M[nbr]) / (dist ** 2)
    return node, float(s)

def temporal_shortest_path(graph_snapshots, max_length=None):
    edges = build_temporal_edges(graph_snapshots)
    nodes = get_nodes(graph_snapshots)
    shortest_matrix = {u: {v: (float('inf'), []) for v in nodes} for u in nodes}

    for source in nodes:
        visited = {}
        queue = deque([(source, 0, [], 0)])  # (current_node, current_time, path, hops)

        while queue:
            u, t, path, hops = queue.popleft()

            #Do not explore further if hop limit is exceeded
            if max_length is not None and hops > max_length:
                continue

            if (u in visited and visited[u] <= hops):
                continue
            visited[u] = hops

            #Only update if within allowed hop limit
            if hops <= max_length or max_length is None:
                if hops < shortest_matrix[source][u][0]:
                    shortest_matrix[source][u] = (hops, path + [u])

            # Expand neighbors only if hop limit not reached yet
            if max_length is None or hops < max_length:
                for v, w, edge_time in edges:
                    if v == u and edge_time >= t:
                        queue.append((w, edge_time + 1, path + [u], hops + 1))

    return shortest_matrix



def temporal_gravity_centrality(
    snapshots: List[nx.Graph],
    mass_type: str = 'degree',  # 'degree', 'closeness', 'betweenness', 'pagerank', 'time_degree'
    aggregation: str = 'static',  # 'static' or 'average_over_time'
    R: int = 3,
    parallel: bool = False,
    processes: Optional[int] = None
) -> Dict:
    """
    Temporal Gravity Centrality based on temporal shortest or fastest arrival distances.

    Parameters
    ----------
    snapshots : List of NetworkX graphs (temporal snapshots)
    mass_type : str, optional
        Node masses (e.g., degree or other scalar per node). If None, use degree in aggregated static graph.
    aggregation : str, optional
        Type of aggregation to use ('static' or 'average_over_time').
    R : int, optional
        Max radius for gravity effect (default 3).
    parallel : bool, optional
        Whether to use multiprocessing.
    processes : int or None, optional
        Number of processes to spawn if parallel.

    Returns
    -------
    Dict[node, float]
        Gravity centrality scores for each node.
    """
    if isinstance(snapshots, nx.Graph):
        raise ValueError('Expected a list of snapshots to compute "tgc" metric, got a single graph.')

    # Get consistent node ordering (union of all snapshots)
    nodes = get_nodes(snapshots)

    # Prepare node mass M
    if mass_type == 'degree':
        _LOG.debug("Using degree vector for mass M")
        M = get_degree_vector(snapshots, temporal_or_static=aggregation)
    elif mass_type == 'closeness':
        _LOG.debug("Using closeness vector for mass M")
        M = get_closeness_vector(snapshots, temporal_or_static=aggregation)
    elif mass_type == 'betweenness':
        _LOG.debug("Using betweenness vector for mass M")
        M = get_betweenness_vector(snapshots, temporal_or_static=aggregation)
    elif mass_type == 'pagerank':
        _LOG.debug("Using pagerank vector for mass M")
        M = get_page_rank_vector(snapshots, temporal_or_static=aggregation)
    elif mass_type == 'time_degree':
        _LOG.debug("Using time degree vector for mass M")
        M = get_time_degree_vector(snapshots)
    else:
        raise ValueError(f"Unknown mass_type: {mass_type}")

    distance_matrix = temporal_shortest_path(snapshots, max_length=R)

    # Build payload for workers
    payload = [(node, distance_matrix, M, R) for node in nodes]

    if parallel:
        import multiprocessing as mp
        with mp.Pool(processes=processes) as pool:
            results = pool.map(_temporal_gravity_worker, payload)
    else:
        results = list(map(_temporal_gravity_worker, payload))

    # Convert results to dict
    return dict(results)



