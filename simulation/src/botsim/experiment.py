"""Runs the replicates and simulations, builds the network and 
records metrics. One replicate consists of building the human graph once
and running every condition on it so that the no-bot and bot pairs
share the same humans, initial opinions, and confidence thresholds."""

from __future__ import annotations
from pathlib import Path 
from typing import Callable

import numpy as np 
import networkx as nx

from . import metrics as m 
from .config import get
from.graphs import NULL_CONDITION, build_network, to_followee_matrix, no_bot_network
from .rng import create_rngs
from .model import Simulation

NULL_STREAM_OFFSET = 1_000_000

def static_human_graph(cfg: dict, rngs: dict) -> nx.Graph:
    """Human graph for the static Meng replication (undirected BA)."""
    net = cfg["network"]
    return no_bot_network(net["n_humans"], net["ba_m"], rngs["graph"])
 
 
def _bot_connections(cfg: dict, human_graph: nx.Graph) -> int:
    """Edges per bot: the configured number, else the mean human (in-)degree."""
    k = cfg["network"].get("bot_connections")
    if k is not None:
        return int(k)
    view = human_graph.in_degree if human_graph.is_directed() else human_graph.degree
    return int(round(np.mean([d for _, d in view])))
 
 
def run_replicate(cfg: dict, condition: str, replicate: int, human_graph: nx.Graph,
                  q: float | None = None, record_history: bool = True,
                  snapshot_steps: tuple[int, ...] = ()) -> dict:
    """Run one condition of one replicate and return all recorded quantities.
 
    Per-step arrays have length steps + 1 (index = time step, 0 = initial state).
    Exposure and unfollow arrays are NaN at step 0 because they describe a step.
    """
    sim_cfg, net, hum, bot = cfg["simulation"], cfg["network"], cfg["humans"], cfg["bots"]
    met = cfg["metrics"]
    steps, burn_in = sim_cfg["steps"], sim_cfg["burn_in"]
    q = get(cfg, "rewiring", "q", default=0.0) if q is None else q
 
    rngs = create_rngs(sim_cfg["master_seed"], replicate)
    if condition == NULL_CONDITION:
        # Same humans, graph and thresholds as no_bot, but different random attention
        # (and rewiring) draws: measures the noise floor of paired comparisons.
        other = create_rngs(sim_cfg["master_seed"], replicate + NULL_STREAM_OFFSET)
        for name in ("attention", "rewiring", "replacement"):
            rngs[name] = other[name]
    n_h = human_graph.number_of_nodes()
    n_b = 0 if condition in ("no_bot", NULL_CONDITION) else net["n_bots"]
    graph = build_network(condition, human_graph, n_b, _bot_connections(cfg, human_graph),
                          rngs["bot_attachment"])
    followees = to_followee_matrix(graph)
 
    # Exactly n_humans draws in every arm, so humans match across arms.
    x0 = rngs["opinions"].uniform(hum["opinion_min"], hum["opinion_max"], size=n_h)
    eps = np.clip(rngs["epsilon"].normal(hum["epsilon_mean"], hum["epsilon_sd"], size=n_h),
                  hum["epsilon_min"], None)
    x = np.concatenate([x0, np.full(n_b, bot["opinion"])])
    activity = np.concatenate([np.full(n_h, hum["activity"]), np.full(n_b, bot["activity"])])
 
    sim = Simulation(
        followees, n_h, x, eps, activity,
        rng_attention=rngs["attention"], rng_rewiring=rngs["rewiring"],
        rng_replacement=rngs["replacement"],
        sigma=cfg["attention"]["limit"], q=q,
        include_self=get(cfg, "opinion_update", "include_self", default=True),
        threshold_inclusive=get(cfg, "opinion_update", "threshold_inclusive", default=True),
        replacement_candidates=get(cfg, "rewiring", "replacement", "candidates",
                                   default="all_nodes"),
    )
 
    bins, base = met["entropy_bins"], met["baseline"]
    net_every = met.get("network_every", 10)
    net_steps = list(range(0, steps + 1, net_every))
    if net_steps[-1] != steps:
        net_steps.append(steps)
 
    nan = lambda: np.full(steps + 1, np.nan)
    rec = {k: nan() for k in ("entropy", "tendency", "polarization", "h2br",
                              "accepted_bot_exposure", "n_unfollow", "n_unfollow_bot",
                              "n_replaced_with_bot")}
    net_rec = {k: [] for k in ("mean_bot_degree", "opinion_assortativity", "in_degree_gini")}
    history = np.empty((steps + 1, n_h), dtype=np.float32) if record_history else None
    snapshots = {}
 
    def record(t: int) -> None:
        xh = sim.x[:n_h]
        rec["entropy"][t] = m.opinion_entropy(xh, bins)
        rec["tendency"][t] = m.opinion_tendency(xh, base)
        rec["polarization"][t] = m.opinion_polarization(xh, base)
        if history is not None:
            history[t] = xh
        if t in net_steps:
            net_rec["mean_bot_degree"].append(m.mean_bot_degree(sim.degree, n_h))
            net_rec["opinion_assortativity"].append(
                m.opinion_assortativity(sim.followees, sim.x, n_h))
            net_rec["in_degree_gini"].append(m.in_degree_gini(sim.degree, n_h))
        if t in snapshot_steps:
            snapshots[t] = sim.followees.copy()
 
    record(0)
    for t in range(1, steps + 1):
        info = sim.step()
        rec["h2br"][t] = info.h2br
        rec["accepted_bot_exposure"][t] = info.accepted_bot_exposure
        rec["n_unfollow"][t] = info.n_unfollow
        rec["n_unfollow_bot"][t] = info.n_unfollow_bot
        rec["n_replaced_with_bot"][t] = info.n_replaced_with_bot
        record(t)
 
    result = {
        "condition": condition, "replicate": replicate, "q": float(q),
        "n_humans": n_h, "n_bots": n_b, "steps": steps, "burn_in": burn_in,
        "x0": x0, "eps": eps, "x_final": sim.x[:n_h].copy(),
        "network_steps": np.array(net_steps),
        **rec, **{k: np.array(v) for k, v in net_rec.items()},
    }
    if history is not None:
        result["x_history"] = history
    for t, snap in snapshots.items():
        result[f"followees_t{t}"] = snap
    return result
 
 
def run_conditions(cfg: dict, replicate: int,
                   human_graph_fn: Callable[[dict, dict], nx.Graph] = static_human_graph,
                   q: float | None = None, conditions=None, **kwargs) -> dict[str, dict]:
    """Run every condition of one replicate on a shared human graph."""
    conditions = conditions or cfg["experiment"]["conditions"]
    rngs = create_rngs(cfg["simulation"]["master_seed"], replicate)
    human_graph = human_graph_fn(cfg, rngs)
    return {c: run_replicate(cfg, c, replicate, human_graph, q=q, **kwargs)
            for c in conditions}
 
 
def summary_row(res: dict) -> dict:
    """Scalar summary of one run: final-state metrics and post-burn-in exposure means."""
    post = slice(res["burn_in"] + 1, None)
    return {
        "condition": res["condition"], "replicate": res["replicate"], "q": res["q"],
        "final_entropy": res["entropy"][-1],
        "final_tendency": res["tendency"][-1],
        "final_polarization": res["polarization"][-1],
        "mean_h2br": np.nanmean(res["h2br"][post]),
        "mean_accepted_bot_exposure": np.nanmean(res["accepted_bot_exposure"][post]),
        "final_assortativity": res["opinion_assortativity"][-1],
        "final_mean_bot_degree": res["mean_bot_degree"][-1],
        "final_in_degree_gini": res["in_degree_gini"][-1],
        "total_unfollows": np.nansum(res["n_unfollow"]),
        "total_unfollows_of_bots": np.nansum(res["n_unfollow_bot"]),
        "total_replaced_with_bot": np.nansum(res["n_replaced_with_bot"]),
    }
 
 
def save_result(path: str | Path, res: dict) -> None:
    """Save a run as compressed .npz (arrays and scalars)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **res)
 
