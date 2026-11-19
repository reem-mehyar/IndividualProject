# Validate and load experimental YAML configs

from __future__ import annotations
import yaml
from pathlib import Path
from .graphs import CONDITIONS




def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get(cfg: dict, *keys, default=None):

    #Nested lookup: get(cfg, "rewiring", "replacement", "candidates", default="all_nodes")
    node = cfg
    for key in keys: 
        if not isinstance(node, dict) or key not in node or node[key] is None:
            return default
        node = node[key]
    return node


def find_nulls(node, path=""):

    if isinstance(node, dict):
        for k, v in node.items():
            yield from find_nulls(v, f"{path}.{k}" if path else k)
    elif node is None:
        yield path


def validate_config(cfg: dict, adaptive: bool = False) -> None:
    """Raise ValueError if the config is inconsistent. Call before running anything."""
    net, sim = cfg["network"], cfg["simulation"]
    n_h, n_b = net["n_humans"], net["n_bots"]
    if n_h < 1 or n_b < 0:
        raise ValueError("Need n_humans >= 1 and n_bots >= 0")
    if not set(cfg["experiment"]["conditions"]) <= set(CONDITIONS):
        raise ValueError(f"conditions must be a subset of {CONDITIONS}")
    if not 1 <= net["bot_connections"] <= n_h:
        raise ValueError("bot_connections must be between 1 and n_humans")
    if cfg["attention"]["limit"] < 1:
        raise ValueError("attention limit must be >= 1")
    if not 0 <= sim["burn_in"] < sim["steps"]:
        raise ValueError("Need 0 <= burn_in < steps")
    if sim["replicates"] < 1:
        raise ValueError("Need at least one replicate")
    h = cfg["humans"]
    if h["epsilon_mean"] <= 0 or h["epsilon_sd"] < 0 or h["epsilon_min"] <= 0:
        raise ValueError("Epsilon parameters must be positive")
    if cfg["bots"]["activity"] <= 0 or h["activity"] <= 0:
        raise ValueError("Activity must be positive")
 
    if adaptive:
        q_values = cfg["rewiring"]["q_values"]
        if not all(0.0 <= q <= 1.0 for q in q_values):
            raise ValueError("q_values must lie in [0, 1]")
        if 0.0 not in q_values:
            raise ValueError("q_values must include 0.0 (the static control)")
        nulls = list(find_nulls(net.get("generator", {}), "network.generator"))
        if nulls:
            raise ValueError(f"Fill in the directed-graph targets first: {nulls}")
    else:
        if get(cfg, "rewiring", "q", default=0.0) != 0.0:
            raise ValueError("The static undirected model has no unfollowing: rewiring.q must be 0")
        if net.get("directed", False):
            raise ValueError("Static config must be undirected")
        if net["ba_m"] >= n_h:
            raise ValueError("ba_m must be smaller than n_humans")
 