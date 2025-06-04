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

def _smoke_dict(d, G):
    assert isinstance(d, dict)
    assert set(d.keys()) == set(G.nodes())
    assert all(isinstance(v, (float, int)) for v in d.values())

def test_gravity_centrality():
    G = nx.karate_club_graph()
    result = gravity.gravity_centrality(G)
    print("gravity_centrality:", list(result.items())[:5])
    _smoke_dict(result, G)

def test_gravity_centrality_agg():
    G = nx.karate_club_graph()
    result = gravity.gravity_centrality_agg(G)
    print("gravity_centrality_agg:", list(result.items())[:5])
    _smoke_dict(result, G)

def test_improved_gravity_centrality():
    G = nx.karate_club_graph()
    result = gravity.improved_gravity_centrality(G)
    print("improved_gravity_centrality:", list(result.items())[:5])
    _smoke_dict(result, G)

def test_improved_gravity_centrality_agg():
    G = nx.karate_club_graph()
    result = gravity.improved_gravity_centrality_agg(G)
    print("improved_gravity_centrality_agg:", list(result.items())[:5])
    _smoke_dict(result, G)

def test_mcde():
    G = nx.karate_club_graph()
    result = entropy.mcde(G)
    print("mcde:", list(result.items())[:5])
    _smoke_dict(result, G)

def test_mcde_weighted():
    G = nx.karate_club_graph()
    result = entropy.mcde(G, weighted=True)
    print("mcde_weighted:", list(result.items())[:5])
    _smoke_dict(result, G)

def test_erm():
    G = nx.karate_club_graph()
    result = entropy.erm(G)
    print("erm:", list(result.items())[:5])
    _smoke_dict(result, G)

def test_dsr():
    G = nx.karate_club_graph()
    result = entropy.dsr(G)
    print("dsr:", list(result.items())[:5])
    _smoke_dict(result, G)

def test_dsr_agg():
    G = nx.karate_club_graph()
    result = entropy.dsr_agg(G)
    print("dsr_agg:", list(result.items())[:5])
    _smoke_dict(result, G)

def test_ecrm():
    G = nx.karate_club_graph()
    result = entropy.ecrm(G)
    print("ecrm:", list(result.items())[:5])
    _smoke_dict(result, G)

def test_ninl():
    G = nx.karate_club_graph()
    result = ninl.ninl(G)
    print("ninl:", list(result.items())[:5])
    _smoke_dict(result, G)

def test_ninl_layer0():
    G = nx.karate_club_graph()
    result = ninl.ninl_layer0(G)
    print("ninl_layer0:", list(result.items())[:5])
    _smoke_dict(result, G)

def test_density_centrality():
    G = nx.karate_club_graph()
    result = density.density_centrality(G)
    print("density_centrality:", list(result.items())[:5])
    _smoke_dict(result, G)

def test_clustered_local_degree():
    G = nx.karate_club_graph()
    result = density.clustered_local_degree(G)
    print("clustered_local_degree:", list(result.items())[:5])
    _smoke_dict(result, G)

def test_gli():
    G = nx.karate_club_graph()
    result = gli.gli(G)
    print("gli:", list(result.items())[:5])
    _smoke_dict(result, G)

def test_gli_new():
    G = nx.karate_club_graph()
    result = gli.gli_new(G)
    print("gli_new:", list(result.items())[:5])
    _smoke_dict(result, G)

def test_h_index():
    G = nx.karate_club_graph()
    result = hindex.h_index(G)
    print("h_index:", list(result.items())[:5])
    _smoke_dict(result, G)

def test_local_h_index():
    G = nx.karate_club_graph()
    result = hindex.local_h_index(G)
    print("local_h_index:", list(result.items())[:5])
    _smoke_dict(result, G)

def test_ls_influence():
    G = nx.karate_club_graph()
    result = ls.ls_influence(G)
    print("ls_influence:", list(result.items())[:5])
    _smoke_dict(result, G)

if __name__ == "__main__":
    # Run all manually for quick smoke-test
    import sys
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            print(f"Running {name} ...")
            func()
    print("All centrality measure smoke-tests passed!")
