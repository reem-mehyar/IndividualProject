import numpy as np
import networkx as nx


def no_bot_network(n_humans: int, 
                   m: int, 
                   rng: np.random.Generator) -> nx.Graph:

    """Generate an undirected BA network.
    n_humans: number of human agents in the network
    m: number of edges each introduced node to the graph creates
    rng: random generator for reproducibility.
    """

    if not 1 <= m < n_humans:
        raise ValueError("Require 1 <= m < n_humans")

    graph = nx.barabasi_albert_graph(n=n_humans, m=m, seed=rng)

    return graph