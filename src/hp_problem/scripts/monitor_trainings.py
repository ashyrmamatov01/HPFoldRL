#!/usr/bin/env python
"""Monitor recent training runs and plot episode rewards (raw + moving-average).

For each *job_name* the newest `training_log.csv` is loaded.  Two curves per job:
    • Thin, translucent line – per-episode reward.
    • Thick line             – MA-smoothed reward.

Distinct tab10 colors ensure clarity across six jobs.

Example
-------
python -m hp_problem.scripts.monitor_trainings \
       --runs-dir runs \
       --output runs/monitor_latest.png
"""
from __future__ import annotations
import argparse, pathlib
import pandas as pd
import matplotlib.pyplot as plt

DEF_NAMES = [
    "tabular_long",
    "tabular_short",
    "dqn_mlp_long",
    "dqn_mlp_short",
    "dqn_cnn_long",
    "dqn_cnn_short",
    "dqn_attn_long",
    "dqn_attn_short",
]

# -----------------------------------------------------------------------------

def moving_average(x: pd.Series, w: int) -> pd.Series:  # centered MA
    return x.rolling(w, min_periods=1, center=True).mean()


def latest_log(job_dir: pathlib.Path) -> pathlib.Path | None:
    logs = list(job_dir.rglob("training_log.csv"))
    return max(logs, key=lambda p: p.stat().st_mtime) if logs else None


def main(args):
    runs_dir = pathlib.Path(args.runs_dir).expanduser()
    job_names = args.job_names or DEF_NAMES

    logs: dict[str, pathlib.Path] = {}
    for name in job_names:
        newest = latest_log(runs_dir / name)
        if newest:
            logs[name] = newest
        else:
            print(f"[warn] missing log for '{name}' – skipped")

    if not logs:
        print("No logs found – aborting.")
        return

    # ------------------------------------------------------------- plotting
    colors = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (name, csv_path) in enumerate(sorted(logs.items())):
        df = pd.read_csv(csv_path)
        c = colors[i % len(colors)]
        # raw episode rewards
        ax.plot(df["episode"], df["Reward"], color=c, alpha=0.25, linewidth=0.8)
        # moving-average curve
        ax.plot(df["episode"], moving_average(df["Reward"], args.ma_window),
                color=c, linewidth=2.2, label=name.replace("_", "-"))

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode reward", fontsize=12)
    ax.set_title(f"Training progress (MA window = {args.ma_window})", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = pathlib.Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--job-names", nargs="*", default=None,
                   help="Job root dirs to plot (default: six preset names)")
    p.add_argument("--ma-window", type=int, default=500,
                   help="Moving-average window size")
    p.add_argument("--output", type=str, default="monitor_latest.png")
    args = p.parse_args()
    main(args)