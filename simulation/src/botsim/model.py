


from __future__ import annotations
from dataclasses import dataclass
import numpy as np


MAX_REPLACEMENT_TRIES = 10_000



@dataclass
class StepInfo: #Metrics for one step

    h2br: float                    # share of humans with >= 1 bot in their attention set
    accepted_bot_exposure: float   # share with >= 1 bot that is also within their threshold
    n_unfollow: int                # unfollow events this step
    n_unfollow_bot: int            # ... of which the unfollowed account was a bot
    n_replaced_with_bot: int       # replacements that landed on a bot
    attended: np.ndarray           # (n_humans, k) attended node ids, -1 for empty slots
    in_range: np.ndarray           # (n_humans, k) bool, attended and within threshold


class Simulation: 

    def __init__(self, followees: np.ndarray, n_humans: int,
                x: np.ndarray,
                eps: np.ndarray, activity: np.ndarray,
                rng_attention: np.random.Generator,
                rng_rewiring: np.random.Generator,
                rng_replacement: np.random.Generator,
                sigma: int = 6, q: float = 0.0,
                include_self: bool = True,
                threshold_inclusive: bool = True,
                replacement_candidates: str = "all_nodes"):

        self.followees = np.array(followees, dtype=np.int64, copy=True)
        self.n_nodes, self.width = self.followees.shape
        self.n_humans = int(n_humans)
        self.n_bots = self.n_nodes - self.n_humans
        if not 0 < self.n_humans <= self.n_nodes:
            raise ValueError("n_humans must be between 1 and the number of nodes")
 
        self.x = np.array(x, dtype=float, copy=True)
        self.eps = np.array(eps, dtype=float, copy=True)
        self.activity = np.array(activity, dtype=float, copy=True)
        if self.x.shape != (self.n_nodes,) or self.activity.shape != (self.n_nodes,):
            raise ValueError("x and activity need one entry per node")
        if self.eps.shape != (self.n_humans,):
            raise ValueError("eps needs one entry per human")
        if sigma < 1:
            raise ValueError("sigma must be >= 1")
        if not 0.0 <= q <= 1.0:
            raise ValueError("q must be in [0, 1]")
        if replacement_candidates not in ("all_nodes", "humans_only"):
            raise ValueError("replacement_candidates must be 'all_nodes' or 'humans_only'")
 
        self.sigma, self.q = int(sigma), float(q)
        self.include_self = include_self
        self.threshold_inclusive = threshold_inclusive
        self.replacement_candidates = replacement_candidates
        self.rng_attention = rng_attention
        self.rng_rewiring = rng_rewiring
        self.rng_replacement = rng_replacement
 
        self.degree = self._in_degree()
        self._refresh_weights()
 

    def _in_degree(self) -> np.ndarray:
        """Number of accounts following each node (counts every row, bots included)."""
        return np.bincount(self.followees[self.followees >= 0], minlength=self.n_nodes)
 
    def _refresh_weights(self) -> None:
        """log(activity x degree) for every followee slot of every human."""
        rows = self.followees[:self.n_humans]
        valid = rows >= 0
        safe = np.where(valid, rows, 0)
        weight = self.activity[safe] * self.degree[safe]
        with np.errstate(divide="ignore"):
            log_weight = np.log(weight)
        self._log_weight = np.where(valid & (weight > 0), log_weight, -np.inf)
 
    
    def step(self) -> StepInfo:
        nh = self.n_humans
        rows = self.followees[:nh]
 
        # 1. limited attention: weighted sample without replacement (Gumbel top-k)
        keys = self._log_weight + self.rng_attention.gumbel(size=self._log_weight.shape)
        k = min(self.sigma, self.width)
        cols = np.argpartition(-keys, k - 1, axis=1)[:, :k]
        valid = np.isfinite(np.take_along_axis(keys, cols, axis=1))
        attended = np.where(valid, np.take_along_axis(rows, cols, axis=1), -1)
        safe = np.where(valid, attended, 0)
 
        # 2. which attended accounts are within the confidence threshold
        x_att = self.x[safe]
        x_own = self.x[:nh]
        dist = np.abs(x_att - x_own[:, None])
        close = dist <= self.eps[:, None] if self.threshold_inclusive else dist < self.eps[:, None]
        in_range = valid & close
 
        # 3. synchronous update from time-t opinions
        n_in = in_range.sum(axis=1)
        sum_in = np.where(in_range, x_att, 0.0).sum(axis=1)
        if self.include_self:
            x_new = (x_own + sum_in) / (1 + n_in)
        else:
            x_new = np.where(n_in > 0, sum_in / np.maximum(n_in, 1), x_own)
 
        # exposure metrics for this step
        is_bot = valid & (attended >= nh)
        h2br = float(is_bot.any(axis=1).mean())
        accepted = float((is_bot & in_range).any(axis=1).mean())
 
        # 4. disagreement-driven rewiring (decided on time-t opinions)
        n_unf = n_unf_bot = n_repl_bot = 0
        if self.q > 0.0:
            out_of_range = valid & ~in_range
            hit = out_of_range & (self.rng_rewiring.random(out_of_range.shape) < self.q)
            if hit.any():
                r, c = np.nonzero(hit)               # grouped by follower (row-major)
                slots = cols[r, c]
                old = rows[r, slots].copy()
                n_unf = len(r)
                n_unf_bot = int((old >= nh).sum())
                n_repl_bot = self._rewire(r, slots, old)
                self.degree = self._in_degree()
                self._refresh_weights()
 
        self.x[:nh] = x_new
        return StepInfo(h2br, accepted, n_unf, n_unf_bot, n_repl_bot, attended, in_range)
 
    def _rewire(self, followers: np.ndarray, slots: np.ndarray, old: np.ndarray) -> int:
        """Replace each unfollowed account by a uniformly random new one.
 
        A replacement can never be the follower itself, someone it already
        follows, or an account it unfollowed in this same step. Overwriting the
        slot keeps every follower's out-degree constant. Returns how many
        replacements were bots.
        """
        hi = self.n_nodes if self.replacement_candidates == "all_nodes" else self.n_humans
        n_bot = 0
        current, unfollowed = -1, set()
        for i, slot, gone in zip(followers.tolist(), slots.tolist(), old.tolist()):
            if i != current:
                current, unfollowed = i, set()
            unfollowed.add(gone)
            row = self.followees[i]
            for _ in range(MAX_REPLACEMENT_TRIES):
                cand = int(self.rng_replacement.integers(hi))
                if cand != i and cand not in unfollowed and not (row == cand).any():
                    break
            else:
                raise RuntimeError(f"No valid replacement found for node {i}")
            row[slot] = cand
            n_bot += cand >= self.n_humans
        return n_bot