
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from .graphs import NULL_CONDITION

try:
    from scipy import stats
except ImportError:  
    stats = None


def ci95(values) -> tuple[float, float]:
    """Mean and 95% confidence half-width (t-interval if scipy is installed)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (float(v.mean()) if len(v) else float("nan")), float("nan")
    crit = stats.t.ppf(0.975, len(v) - 1) if stats else 1.96
    return float(v.mean()), float(crit * v.std(ddof=1) / np.sqrt(len(v)))


def write_csv(path, rows: list[dict]) -> None:
    if rows:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


SUMMARY_COLS = ["final_entropy", "final_tendency", "final_polarization",
                "mean_h2br", "mean_accepted_bot_exposure"]


def print_tables(conditions, summary: list[dict], paired: list[dict]) -> None:
    """Print final-state means +/- 95% CI per condition, then the paired comparison."""
    print("=== Final-state metrics, mean +/- 95% CI over replicates ===")
    for cond in conditions:
        rows = [s for s in summary if s["condition"] == cond]
        if not rows:
            continue
        parts = []
        for c in SUMMARY_COLS:
            m, h = ci95([s[c] for s in rows])
            parts.append(f"{c.replace('final_', '').replace('mean_', '')}={m:.3f}+/-{h:.3f}")
        print(f"{cond:13s} " + "  ".join(parts))
    print("--- paired against no_bot (same humans) ---")
    for cond in conditions:
        rows = [p for p in paired if p["condition"] == cond]
        if not rows:
            continue
        a = ci95([p["mean_bot_attributable"] for p in rows])
        b = ci95([p["proportion_influenced"] for p in rows])
        tag = "  <- noise floor (no bots)" if cond == NULL_CONDITION else ""
        print(f"{cond:13s} bot-attributable shift={a[0]:+.4f}+/-{a[1]:.4f}   "
              f"proportion influenced={b[0]:.3f}+/-{b[1]:.3f}{tag}")