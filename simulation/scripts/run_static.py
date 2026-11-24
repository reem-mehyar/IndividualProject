"""Run the static Meng replication: all conditions, all replicates.

    python scripts/run_static.py                      # uses config/static.yaml
    python scripts/run_static.py --replicates 5       # quick pilot
    python scripts/run_static.py --no-history         # skip saving per-step opinions

Saves one .npz per (condition, replicate) plus summary.csv and paired.csv in the
output directory, then prints means with 95% confidence intervals.
"""
import argparse
import time
from pathlib import Path
import numpy as np 
from botsim.config import load_config, validate_config  
from botsim.experiment import run_conditions, save_result, summary_row  
from botsim.metrics import paired_metrics
from botsim.reporting import print_tables, write_csv  


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/static.yaml")
    ap.add_argument("--replicates", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-history", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    validate_config(cfg)
    reps = args.replicates or cfg["simulation"]["replicates"]
    out = Path(args.out or cfg["output"]["dir"])
    keep_history = cfg["output"].get("save_opinion_history", True) and not args.no_history
    threshold = cfg["metrics"]["influenced_threshold"]
    bot_sign = float(np.sign(cfg["bots"]["opinion"] - cfg["metrics"]["baseline"]))

    summary, paired = [], []
    start = time.time()
    for r in range(reps):
        results = run_conditions(cfg, r, record_history=keep_history)
        for cond, res in results.items():
            save_result(out / f"{cond}_rep{r:03d}.npz", res)
            summary.append(summary_row(res))
        if "no_bot" in results:
            for cond, res in results.items():
                if cond == "no_bot":
                    continue
                pm = paired_metrics(res, results["no_bot"], threshold, bot_sign)
                paired.append({"condition": cond, "replicate": r,
                               **{k: v for k, v in pm.items() if np.isscalar(v)}})
        print(f"replicate {r + 1}/{reps} done ({time.time() - start:.0f}s)", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "paired.csv", paired)

    print()
    print_tables(cfg["experiment"]["conditions"], summary, paired)
    print("\nThe comparison between 'proportion influenced' and the no_bot_null row indicates how much "
          "opinion difference random starting configs alone produces.")

if __name__ == "__main__":
    main()