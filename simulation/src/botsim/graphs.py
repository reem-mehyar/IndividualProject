"""Node labeling convention based on index.
humans are 0... n-humans - 1
bots are n_humans + n_bots -1. 
Graphs can either be undirected static Meng replications or of the directed 
adaptive kind, where an edge A -> B means 'A follows B'. 
Both nodes are converted to a 'followee matrix' where row i lists the accounts followed by node i.
Undirected edges are treated as reciprocal follows, so the 
in-degree in the followee matrix equals the undirected degree."""

from __future__ import annotations

import networkx as nx
import numpy as np 


CONDITIONS = ("no_bot", "random", "preferential", "no_bot_null")
NULL_CONDITION = "no_bot_null"

def no_bot_network(n_humans: int, m: int, rng: np.random.Generator) -> nx.Graph:
    """Undirected Barabasi-Albert network of humans only.
 
    n_humans: number of human agents
    m: number of edges each new node creates
    rng: random generator (use "graph" stream)
    """
    if not 1 <= m < n_humans:
        raise ValueError("Require 1 <= m < n_humans")
    graph = nx.barabasi_albert_graph(n=n_humans, m=m, seed=rng)
    nx.set_node_attributes(graph, False, "is_bot")
    return graph
 
 
def _human_degrees(human_graph: nx.Graph) -> np.ndarray:
    """Degree used for preferential targeting (in-degree if directed)."""
    n = human_graph.number_of_nodes()
    view = human_graph.in_degree if human_graph.is_directed() else human_graph.degree
    return np.array([view[i] for i in range(n)], dtype=float)
 
 
def _attach_bots(human_graph: nx.Graph, n_bots: int, bot_connections: int,
                 rng: np.random.Generator, probs: np.ndarray | None) -> nx.Graph:
    """Return a copy of human_graph with bots attached (input is not modified).
 
    Each bot connects to `bot_connections` distinct humans, drawn with
    probabilities `probs` (uniform if None). Bots never connect to each other.
    Undirected graph: edge (bot, human). Directed graph: edge (human, bot), i.e.
    the human follows the bot, and bots follow nobody.
    """
    n_humans = human_graph.number_of_nodes()
    if set(human_graph.nodes) != set(range(n_humans)):
        raise ValueError("Human nodes must be labelled 0..n_humans-1")
    if any(data.get("is_bot", False) for _, data in human_graph.nodes(data=True)):
        raise ValueError("Input graph must contain humans only")
    if n_bots < 0:
        raise ValueError("n_bots must be >= 0")
    if not 1 <= bot_connections <= n_humans:
        raise ValueError("Require 1 <= bot_connections <= n_humans")
 
    directed = human_graph.is_directed()
    graph = human_graph.copy()
    for bot in range(n_humans, n_humans + n_bots):
        graph.add_node(bot, is_bot=True)
        targets = rng.choice(n_humans, size=bot_connections, replace=False, p=probs)
        if directed:
            graph.add_edges_from((int(t), bot) for t in targets)
        else:
            graph.add_edges_from((bot, int(t)) for t in targets)
    return graph
 
 
def add_random_bots(human_graph: nx.Graph, n_bots: int, bot_connections: int,
                    rng: np.random.Generator) -> nx.Graph:
    """Bots connect to uniformly random humans (use the "bot_attachment" stream)."""
    return _attach_bots(human_graph, n_bots, bot_connections, rng, probs=None)
 
 
def add_preferential_bots(human_graph: nx.Graph, n_bots: int, bot_connections: int,
                          rng: np.random.Generator) -> nx.Graph:
    """Bots connect to humans with probability proportional to human degree.
 
    Degrees are taken from the human graph before any bots are added and are not
    updated as bots attach, so every bot draws from the same distribution.
    """
    degrees = _human_degrees(human_graph)
    return _attach_bots(human_graph, n_bots, bot_connections, rng,
                        probs=degrees / degrees.sum())
 
 
def build_network(condition: str, human_graph: nx.Graph, n_bots: int,
                  bot_connections: int, rng: np.random.Generator) -> nx.Graph:
    """Build one experimental arm from the shared human graph."""
    if condition == "no_bot":
        return human_graph.copy()
    if condition == "random":
        return add_random_bots(human_graph, n_bots, bot_connections, rng)
    if condition == "preferential":
        return add_preferential_bots(human_graph, n_bots, bot_connections, rng)
    if condition == "no_bot_null":
        return human_graph.copy()
    raise ValueError(f"Unknown condition: {condition}")
 
 
def to_followee_matrix(graph: nx.Graph) -> np.ndarray:
    """Convert a graph to a padded followee matrix of shape (n_nodes, width).
 
    Row i holds the accounts node i follows, padded with -1. For an undirected
    graph every edge is a mutual follow. For a directed graph, edge u -> v means
    u follows v. Rows of bots are inert in the simulation (bots never read), but
    they are kept so that in-degree equals the undirected degree.
    """
    n = graph.number_of_nodes()
    if set(graph.nodes) != set(range(n)):
        raise ValueError("Nodes must be labelled 0..n-1")
    neighbours = graph.successors if graph.is_directed() else graph.neighbors
    rows = [sorted(neighbours(i)) for i in range(n)]
    width = max(1, max((len(r) for r in rows), default=1))
    matrix = np.full((n, width), -1, dtype=np.int64)
    for i, row in enumerate(rows):
        matrix[i, :len(row)] = row
    return matrix
 
