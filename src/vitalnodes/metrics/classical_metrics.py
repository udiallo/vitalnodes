import networkx as nx
from typing import Dict, Any, Optional


def get_degree(graph: nx.Graph, parallel: Optional[bool] = None, processes: Optional[int] = None, normalized: bool = False) -> Dict[Any, float]:
    """Compute the degree centrality for each node in the graph.

    Args:
        graph (nx.Graph): The input graph.
        parallel (bool, optional): Whether to use parallel processing. Defaults to None.
        processes (int, optional): Number of processes to use for parallel processing. Defaults to None.
        normalized (bool, optional): Whether to normalize the degree values. Defaults to False.

    Returns:
        Dict[Any, float]: A dictionary mapping each node to its degree centrality.
    """
    N = graph.number_of_nodes()
    if N == 0:
        return {}
    
    if normalized:
        return {node: degree / (N - 1) for node, degree in graph.degree()}
    else:
        return dict(graph.degree())
    
def get_closeness(graph: nx.Graph, parallel: Optional[bool] = None, processes: Optional[int] = None) -> Dict[Any, float]:
    """Compute the closeness centrality for each node in the graph.

    Args:
        graph (nx.Graph): The input graph.
        parallel (bool, optional): Whether to use parallel processing. Defaults to None.
        processes (int, optional): Number of processes to use for parallel processing. Defaults to None.

    Returns:
        Dict[Any, float]: A dictionary mapping each node to its closeness centrality.
    """
    return dict(nx.closeness_centrality(graph))

def get_betweenness(graph: nx.Graph, parallel: Optional[bool] = None, processes: Optional[int] = None, normalized: bool = False) -> Dict[Any, float]:
    """Compute the betweenness centrality for each node in the graph.

    Args:
        graph (nx.Graph): The input graph.
        parallel (bool, optional): Whether to use parallel processing. Defaults to None.
        processes (int, optional): Number of processes to use for parallel processing. Defaults to None.
        normalized (bool, optional): Whether to normalize the betweenness values. Defaults to False.

    Returns:
        Dict[Any, float]: A dictionary mapping each node to its betweenness centrality.
    """
    return dict(nx.betweenness_centrality(graph, normalized=normalized))

def get_eigenvector(graph: nx.Graph, parallel: Optional[bool] = None, processes: Optional[int] = None, max_iter: int = 1000, tol: float = 1e-06) -> Dict[Any, float]:
    """Compute the eigenvector centrality for each node in the graph.

    Args:
        graph (nx.Graph): The input graph.
        parallel (bool, optional): Whether to use parallel processing. Defaults to None.
        processes (int, optional): Number of processes to use for parallel processing. Defaults to None.
        max_iter (int, optional): Maximum number of iterations for power method. Defaults to 1000.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-06.

    Returns:
        Dict[Any, float]: A dictionary mapping each node to its eigenvector centrality.
    """
    return dict(nx.eigenvector_centrality(graph, max_iter=max_iter, tol=tol))