import networkx as nx

from botsim.rng import create_rngs
from botsim.graphs import no_bot_network


def test_number_of_nodes():
    rngs = create_rngs(42, 0)

    graph = no_bot_network(850, 10, rngs["graph"])

    assert graph.number_of_nodes() == 850


def test_graph_is_undirected():
    rngs = create_rngs(42, 0)

    graph = no_bot_network(850, 10, rngs["graph"])

    assert not graph.is_directed()


def test_number_of_edges():
    rngs = create_rngs(42, 0)

    graph = no_bot_network(850, 10, rngs["graph"])

    # NetworkX BA starts with a star of m+1 nodes.
    # Total edges = m * (n_humans - m).
    assert graph.number_of_edges() == 10 * (850 - 10)


def test_reproducibility():
    rngs1 = create_rngs(42, 0)
    rngs2 = create_rngs(42, 0)

    graph1 = no_bot_network(850, 10, rngs1["graph"])
    graph2 = no_bot_network(850, 10, rngs2["graph"])

    assert nx.utils.graphs_equal(graph1, graph2)


def test_no_self_loops():
    rngs = create_rngs(42, 0)

    graph = no_bot_network(850, 10, rngs["graph"])

    assert nx.number_of_selfloops(graph) == 0