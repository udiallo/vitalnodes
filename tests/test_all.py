import time
import networkx as nx

from vitalnodes.metrics import (
    gravity,
    entropy,
    ninl,
    density,
    gli,
    hindex,
    ls,
)

# === SETTINGS ===
GRAPH_TYPE = "erdos_renyi"   # options: "karate", "erdos_renyi", "barabasi"
N_NODES = 2000               # Only used for random graphs
P_ER = 0.05                  # ER graph edge probability
N_EDGES_BA = 2               # BA edges per new node

# === MULTIPROCESSING FLAGS ===
# Set PARALLEL to True to force multiprocessing,
# False to force serial execution, or None to let each metric
# decide based on graph size (>=500 nodes → parallel).
PARALLEL = False

# Number of worker processes to use when PARALLEL=True.
PROCESSES = None

def make_graph():
    if GRAPH_TYPE == "karate":
        return nx.karate_club_graph()
    elif GRAPH_TYPE == "erdos_renyi":
        return nx.erdos_renyi_graph(N_NODES, P_ER, seed=42)
    elif GRAPH_TYPE == "barabasi":
        return nx.barabasi_albert_graph(N_NODES, N_EDGES_BA, seed=42)
    else:
        raise ValueError(f"Unknown graph type: {GRAPH_TYPE}")

def _smoke_dict(d, G):
    assert isinstance(d, dict)
    assert set(d.keys()) == set(G.nodes())
    assert all(isinstance(v, (float, int)) for v in d.values())

def timed_measure(name, func, G, *args, **kwargs):
    print(f"\nRunning {name} (parallel={PARALLEL}, procs={PROCESSES}) …")
    t0 = time.time()
    result = func(G, *args, parallel=PARALLEL, processes=PROCESSES, **kwargs)
    elapsed = time.time() - t0
    _smoke_dict(result, G)
    items = list(result.items())[:5]
    print(f"{name:30} (first 5): {items}")
    print(f"{name:30} [OK] | Time: {elapsed:.4f} s")
    return elapsed

def main():
    G = make_graph()
    print(f"Network: {GRAPH_TYPE} (n={G.number_of_nodes()}, m={G.number_of_edges()})")
    print(f"Global PARALLEL = {PARALLEL}, PROCESSES = {PROCESSES}")

    times = {}

    times["gravity_centrality"] = timed_measure(
        "gravity_centrality", gravity.gravity_centrality, G)
    times["gravity_centrality_agg"] = timed_measure(
        "gravity_centrality_agg", gravity.gravity_centrality_agg, G)
    times["improved_gravity_centrality"] = timed_measure(
        "improved_gravity_centrality", gravity.improved_gravity_centrality, G)
    times["improved_gravity_centrality_agg"] = timed_measure(
        "improved_gravity_centrality_agg", gravity.improved_gravity_centrality_agg, G)
    times["mcde"] = timed_measure(
        "mcde", entropy.mcde, G)
    times["mcde_weighted"] = timed_measure(
        "mcde_weighted", entropy.mcde, G, weighted=True)
    times["erm"] = timed_measure(
        "erm", entropy.erm, G)
    times["dsr"] = timed_measure(
        "dsr", entropy.dsr, G)
    times["dsr_agg"] = timed_measure(
        "dsr_agg", entropy.dsr_agg, G)
    times["ecrm"] = timed_measure(
        "ecrm", entropy.ecrm, G)
    times["ninl"] = timed_measure(
        "ninl", ninl.ninl, G)
    times["ninl_layer0"] = timed_measure(
        "ninl_layer0", ninl.ninl_layer0, G)
    times["density_centrality"] = timed_measure(
        "density_centrality", density.density_centrality, G)
    times["clustered_local_degree"] = timed_measure(
        "clustered_local_degree", density.clustered_local_degree, G)
    times["gli"] = timed_measure(
        "gli", gli.gli, G)
    times["gli_new"] = timed_measure(
        "gli_new", gli.gli_new, G)
    times["h_index"] = timed_measure(
        "h_index", hindex.h_index, G)
    times["local_h_index"] = timed_measure(
        "local_h_index", hindex.local_h_index, G)
    times["ls_influence"] = timed_measure(
        "ls_influence", ls.ls_influence, G)

    print("\nSummary of timings:")
    print(f"{'Measure':30} | {'Time (s)':>9}")
    print("-" * 43)
    for k, v in times.items():
        print(f"{k:30} | {v:9.5f}")

if __name__ == "__main__":
    main()
