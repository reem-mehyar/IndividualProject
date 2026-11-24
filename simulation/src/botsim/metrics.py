
# Population metrics computed on only human opinions. H2B and accepted bot exposure calculated in model.py. 


from __future__ import annotations

import numpy as np


#  Population level opinion metrics, calculated as Meng has
def opinion_entropy(x: np.ndarray, bins: int = 10, lo: float = 0.0, hi: float = 1.0) -> float:
    """compute Shannon entropy (natural log) of the opinion histogram; max is ln(bins)."""
    counts, _ = np.histogram(x, bins=bins, range=(lo, hi))
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def opinion_tendency(x: np.ndarray, baseline: float = 0.5) -> float:
    """Shift of the mean opinion from the neutral baseline (positive = toward 1)."""
    return float(np.mean(x) - baseline)


def opinion_polarization(x: np.ndarray, baseline: float = 0.5) -> float:
    """Root-mean-square distance of opinions from the baseline."""
    return float(np.sqrt(np.mean((x - baseline) ** 2)))


#  Network-level metrics 
def mean_bot_degree(degree: np.ndarray, n_humans: int) -> float:
    """Mean degree (in-degree, i.e. followers) of the bots; NaN if there are none."""
    bots = degree[n_humans:]
    return float(bots.mean()) if len(bots) else float("nan")


def opinion_assortativity(followees: np.ndarray, x: np.ndarray, n_humans: int) -> float:
    """Pearson correlation between a human's opinion and the opinion of each human it follows.

    Positive means people follow accounts with similar opinions. Only
    human-to-human follows are used. Returns NaN if undefined.
    """
    rows = followees[:n_humans]
    src, slot = np.nonzero((rows >= 0) & (rows < n_humans))
    if len(src) < 2:
        return float("nan")
    a, b = x[src], x[rows[src, slot]]
    if a.std() == 0.0 or b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def gini(values: np.ndarray) -> float:
    """Gini coefficient of non-negative values (0 = equal, near 1 = concentrated)."""
    v = np.sort(np.asarray(values, dtype=float))
    n, total = len(v), v.sum()
    if n == 0 or total == 0.0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * total))


def in_degree_gini(degree: np.ndarray, n_humans: int) -> float:
    """Gini of human in-degrees tracks centralization of followers/changes in hub structure."""
    return gini(degree[:n_humans])


# individual-level metrics
def paired_metrics(res_bot: dict, res_nobot: dict, threshold: float,
                   bot_sign: float = 1.0) -> dict:
    """Bot-attributable effects, comparing the same humans with and without bots.

    Both runs must share the same human graph, initial opinions and thresholds
    (guaranteed by using the same replicate seeds). All arrays are per human.
    bot_sign is +1 if bots sit above the neutral baseline (bot opinion 1.0) and
    -1 if below; it defines which direction counts as "toward the bot".
    """
    if not np.array_equal(res_bot["x0"], res_nobot["x0"]):
        raise ValueError("Runs are not paired: initial opinions differ")
    attributable = res_bot["x_final"] - res_nobot["x_final"]
    return {
        "individual_revision": res_bot["x_final"] - res_bot["x0"],
        "bot_attributable_revision": attributable,
        "mean_individual_revision": float(np.mean(res_bot["x_final"] - res_bot["x0"])),
        "mean_bot_attributable": float(attributable.mean()),
        "mean_abs_bot_attributable": float(np.abs(attributable).mean()),
        "proportion_influenced": float(np.mean(np.abs(attributable) > threshold)),
        "proportion_influenced_toward_bot": float(np.mean(bot_sign * attributable > threshold)),
    }