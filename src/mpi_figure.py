"""
Figure for results/mpi/runs.csv (scripts/run_mpi.sh): speedup per mode, and
how the time of the slowest rank splits into computing and communicating.

    python src/mpi_figure.py
"""
import csv

import numpy as np

import plots
from config import RESULTS

OUT = RESULTS / "mpi"
STYLE = {"block": (plots.NAVY, "o"), "cyclic": (plots.ORANGE, "s"), "dynamic": (plots.RED, "^")}


def main():
    with open(OUT / "runs.csv") as f:
        rows = list(csv.DictReader(f))
    n = rows[-1]["n"]
    rows = [r for r in rows if r["n"] == n]           # only the latest problem size
    serial = [float(r["seconds"]) for r in rows if r["mode"] == "serial"]
    if not serial:
        raise SystemExit(f"no serial run for N={n}; run kernel_mpi.py --mode serial first")
    t1 = serial[-1]

    plt = plots.setup()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    p = np.array(sorted({int(r["ranks"]) for r in rows if r["mode"] != "serial"}))
    plots.ideal(a1, p)
    a1.plot(p, np.maximum(p - 1, 0), color=plots.GREY, lw=1.2, ls=(0, (5, 3)), label="P−1 workers (dynamic)")
    width = 0.27
    for k, (mode, (c, m)) in enumerate(STYLE.items()):
        rs = sorted((int(r["ranks"]), float(r["seconds"]), float(r["busy_max"]), float(r["comm_max"]))
                    for r in rows if r["mode"] == mode)
        if not rs:
            continue
        a1.plot([x[0] for x in rs], [t1 / x[1] for x in rs], marker=m, color=c, label=mode)
        # share of the run spent on messages (or waiting in them) by the rank that spent the most
        x = np.arange(len(rs)) + (k - 1) * width
        a2.bar(x, [100 * r[3] / r[1] for r in rs], width, color=c, label=mode)
    a1.set(xlabel="MPI ranks", ylabel="speedup (T1 / Tp)", xticks=p, ylim=(0, p[-1] + 1),
           title=f"Kernel matrix with MPI, N = {n}")
    a1.legend(loc="upper left", fontsize=11)
    a2.set_xticks(range(len(p)), [str(v) for v in p])
    a2.set(xlabel="MPI ranks", ylabel="% of run time in communication", ylim=(0, 100),
           title="Communication and waiting")
    a2.legend(loc="upper left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "mpi.png")
    plt.close(fig)
    print("wrote", OUT / "mpi.png")


if __name__ == "__main__":
    main()
