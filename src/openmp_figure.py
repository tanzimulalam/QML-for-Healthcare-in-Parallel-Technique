"""
Figures for the OpenMP sweeps in results/openmp/ (whichever CSVs exist):
  schedules.csv  (openmp/run_schedules.sh)  speedup per schedule, thread idle time
  gram.csv       (openmp/run_gram.sh)       naive vs tiled vs tasks, tile size
  bigstate.csv   (openmp/run_bigstate.sh)   memory-bound vs compute-bound layer

    python src/openmp_figure.py
"""
import csv

import numpy as np

import plots
from config import RESULTS

OUT = RESULTS / "openmp"
STYLE = {"static": (plots.NAVY, "o"), "cyclic": (plots.ORANGE, "s"), "folded": (plots.BLUE, "D"),
         "dynamic": (plots.RED, "^"), "guided": (plots.YELLOW, "v"), "taskloop": (plots.GREY, "P")}


def read(name):
    path = OUT / name
    if not path.exists():
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def schedules(plt, rows):
    t1 = np.median([float(r["seconds"]) for r in rows if r["threads"] == "1"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    p = np.array(sorted({int(r["threads"]) for r in rows}))
    a1.axvspan(0.5, 8.5, color="#F2F2F2", zorder=0)
    plots.ideal(a1, p)
    a1.plot(p, p ** 2 / (2 * p - 1), color=plots.GREY, lw=1.2, ls=(0, (5, 3)), label="static limit p²/(2p−1)")
    for s, (c, m) in STYLE.items():
        # idle share: how much of the run the least-loaded thread spent waiting
        rs = sorted((int(r["threads"]), float(r["seconds"]), 1 - float(r["busy_min"]) / float(r["busy_max"]))
                    for r in rows if r["schedule"] == s)
        if not rs:
            continue
        a1.plot([x[0] for x in rs], [t1 / x[1] for x in rs], marker=m, color=c, label=s)
        a2.plot([x[0] for x in rs], [100 * x[2] for x in rs], marker=m, color=c, label=s)
    a1.set(xlabel="threads", ylabel="speedup (T1 / Tp)", xticks=p, ylim=(0, p[-1] + 1),
           title=f"OpenMP kernel (version A), N = {rows[0]['N']}")
    a1.legend(loc="upper left", fontsize=11)
    a2.set(xlabel="threads", ylabel="idle time of least-loaded thread (%)", xticks=p, ylim=(0, 100),
           title="Load imbalance (same colours)")
    save(plt, fig, "openmp_schedules.png")


def gram(plt, rows):
    tri = [r for r in rows if r["method"] != "cross"]
    p = np.array(sorted({int(r["threads"]) for r in tri}))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    t1 = min(float(r["gram_s"]) for r in tri if r["threads"] == "1")      # best method on 1 thread
    for method, c, m in (("naive", plots.NAVY, "o"), ("tiled", plots.RED, "s"), ("tasks", plots.ORANGE, "D")):
        first = {}                          # the tile-size runs repeat tile 64; keep the thread sweep's value
        for r in tri:
            if r["method"] == method and r["tile"] == "64":
                first.setdefault(int(r["threads"]), float(r["gram_s"]))
        rs = sorted(first.items())
        a1.plot([x[0] for x in rs], [t1 / x[1] for x in rs], marker=m, color=c, label=method)
    cross = sorted((int(r["threads"]), float(r["gram_s"])) for r in rows if r["method"] == "cross")
    if cross:
        a1.plot([x[0] for x in cross], [cross[0][1] / x[1] for x in cross], marker="^", color=plots.BLUE,
                ls=(0, (4, 2)), label="test x train (own T1)")
    plots.ideal(a1, p)
    a1.set(xlabel="threads", ylabel="speedup vs best 1-thread time", xticks=p, ylim=(0, p[-1] + 1),
           title=f"Version B in C, N = {tri[0]['N']}")
    a1.legend(loc="upper left", fontsize=11)

    pmax = str(p[-1])
    tiles = sorted((int(r["tile"]), float(r["gram_s"])) for r in tri if r["method"] == "tiled" and r["threads"] == pmax)
    tiles = list(dict(tiles).items())       # one value per tile size
    naive = [float(r["gram_s"]) for r in tri if r["method"] == "naive" and r["threads"] == pmax]
    a2.bar(range(len(tiles)), [t for _, t in tiles], color=plots.BLUE)
    if naive:
        a2.axhline(naive[0], color=plots.NAVY, ls=(0, (5, 3)), lw=1.5, label="naive (no tiles)")
        a2.set_ylim(0, 1.3 * max([naive[0]] + [t for _, t in tiles]))
        a2.legend(loc="upper right", fontsize=11)
    a2.set_xticks(range(len(tiles)), [str(b) for b, _ in tiles])
    a2.set(xlabel="tile size B (beats)", ylabel="seconds", title=f"Tile size, {pmax} threads")
    save(plt, fig, "openmp_gram.png")


def bigstate(plt, rows):
    par = [r for r in rows if r["init"] == "parallel"]
    ser = {r["threads"]: r for r in rows if r["init"] == "serial"}
    p = np.array([int(r["threads"]) for r in par])
    h = np.array([float(r["h_s"]) for r in par])
    z = np.array([float(r["zz_s"]) for r in par])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    plots.ideal(a1, p)
    a1.plot(p, z[0] / z, "s-", color=plots.RED, label="ZZ layer (arithmetic)")
    a1.plot(p, h[0] / h, "o-", color=plots.NAVY, label="H layer (memory)")
    a1.set(xlabel="threads", ylabel="speedup (T1 / Tp)", xticks=p, ylim=(0, p[-1] + 1),
           title=f"One {par[0]['qubits']}-qubit state")
    a1.legend(loc="upper left", fontsize=11)

    a2.plot(p, [float(r["copy_GBps"]) for r in par], color=plots.GREY, ls=(0, (5, 3)), label="plain copy")
    a2.plot(p, [float(r["h_GBps"]) for r in par], "o-", color=plots.NAVY, label="H layer, parallel first touch")
    if ser:
        a2.plot(p, [float(ser[str(t)]["h_GBps"]) for t in p], "^-", color=plots.ORANGE,
                label="H layer, serial first touch")
    a2.set(xlabel="threads", ylabel="memory traffic (GB/s)", xticks=p, ylim=(0, None),
           title="Memory bandwidth")
    a2.legend(loc="lower right", fontsize=11)
    save(plt, fig, "openmp_bigstate.png")


def save(plt, fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


def main():
    plt = plots.setup()
    done = 0
    for name, fn in (("schedules.csv", schedules), ("gram.csv", gram), ("bigstate.csv", bigstate)):
        rows = read(name)
        if rows:
            fn(plt, rows)
            done += 1
    if not done:
        print("nothing to plot yet, run the scripts in openmp/ first")


if __name__ == "__main__":
    main()
