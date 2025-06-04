import networkx as nx
from vitalnodes.metrics import gravity

def test_gravity_basic():
    G = nx.karate_club_graph()
    scores = gravity.gravity_centrality(G)
    print("Gravity scores:", list(scores.items())[:5])
    assert isinstance(scores, dict)
    assert set(scores.keys()) == set(G.nodes())
    assert all(isinstance(v, float) for v in scores.values())

if __name__ == "__main__":
    test_gravity_basic()
    print("Gravity metric basic test: PASSED")