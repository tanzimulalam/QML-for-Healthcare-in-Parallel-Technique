"""
Plot results/openmp/schedules.csv (written by openmp/run_schedules.sh):
speedup per schedule, and how unevenly the threads were loaded.
"""
import csv

import numpy as np

import plots
from config import RESULTS

OUT = RESULTS / "openmp"
STYLE = {"static": (plots.NAVY, "o"), "cyclic": (plots.ORANGE, "s"), "folded": (plots.BLUE, "D"),
         "dynamic": (plots.RED, "^"), "guided": (plots.YELLOW, "v")}


def main():
    with open(OUT / "schedules.csv") as f:
        rows = list(csv.DictReader(f))
    t1 = np.median([float(r["seconds"]) for r in rows if r["threads"] == "1"])
    n = rows[0]["N"]

    plt = plots.setup()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    p = np.array(sorted({int(r["threads"]) for r in rows}))
    a1.axvspan(0.5, 8.5, color="#F2F2F2", zorder=0)
    plots.ideal(a1, p)
    a1.plot(p, p ** 2 / (2 * p - 1), color=plots.GREY, lw=1.2, ls=(0, (5, 3)), label="static limit p²/(2p−1)")
    for s, (c, m) in STYLE.items():
        # idle share: how much of the run the least-loaded thread spent waiting at the end
        rs = sorted((int(r["threads"]), float(r["seconds"]), 1 - float(r["busy_min"]) / float(r["busy_max"]))
                    for r in rows if r["schedule"] == s)
        if not rs:
            continue
        a1.plot([x[0] for x in rs], [t1 / x[1] for x in rs], marker=m, color=c, label=s)
        a2.plot([x[0] for x in rs], [100 * x[2] for x in rs], marker=m, color=c, label=s)
    a1.set(xlabel="threads", ylabel="speedup (T1 / Tp)", xticks=p, ylim=(0, p[-1] + 1),
           title=f"OpenMP kernel, N = {n}")
    a1.legend(loc="upper left", fontsize=11)
    a2.set(xlabel="threads", ylabel="idle time of least-loaded thread (%)", xticks=p, ylim=(0, 100),
           title="Load imbalance (same colours)")
    fig.tight_layout()
    fig.savefig(OUT / "openmp_schedules.png")
    plt.close(fig)
    print("wrote", OUT / "openmp_schedules.png")


if __name__ == "__main__":
    main()
