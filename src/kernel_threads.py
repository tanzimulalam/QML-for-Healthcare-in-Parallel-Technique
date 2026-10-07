"""
Version A of the kernel (one circuit per pair) on 1..24 threads, four schedules.

Row i of the upper triangle has N-1-i entries, so the same number of rows per
thread is not the same amount of work.
  block   - equal contiguous blocks of rows (Numba's default)
  cyclic  - rows dealt out like cards
  folded  - row t together with row N-1-t
  dynamic - a free thread takes the next 4 rows from a queue (qsim.WorkQueue)
The work per pair does not depend on the input values, so random inputs are used
unless --real is given (then the first N training beats).

    python src/kernel_threads.py                    # N chosen so that T1 is about 20 s
    python src/kernel_threads.py --n 800 --threads 1 4 8 --reps 1
"""
import argparse
import csv
import time

import numba
import numpy as np

import qsim
from config import DATA, RESULTS

OUT = RESULTS / "kernel"
MAXP = numba.config.NUMBA_NUM_THREADS


def timed(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def inputs(n, real=False, lam=0.1):
    if real:
        return qsim.encode(np.load(DATA / "features8.npz")["U_train"][:n], lam)
    return qsim.phase_table(np.random.default_rng(0).uniform(0, np.pi, (n, 8)))


def compile_all():
    D, K = inputs(16), np.zeros((16, 16))
    qsim.kernel_serial(D, K)
    qsim.kernel_block(D, K)
    qsim.kernel_cyclic(D, K, 2)
    qsim.kernel_folded(D, K)
    qsim._rows(D, K, 0, 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="beats (0 = pick N from --target)")
    ap.add_argument("--target", type=float, default=20.0, help="serial time to aim for, seconds")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 6, 8, 9, 12, 16, 20, 24])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--real", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    compile_all()
    print("numba threading layer:", numba.threading_layer(), "| threads available:", MAXP)

    n = args.n
    if n == 0:
        # time a small problem first; the cost grows with N^2
        D0 = inputs(400)
        t = timed(lambda: qsim.kernel_serial(D0, np.zeros((400, 400))), 1)
        n = int(round(np.sqrt(2 * args.target * 400 * 399 / 2 / t), -2))
        print(f"{t / (400 * 399 / 2) * 1e6:.2f} us per pair -> N = {n}")
    D = inputs(n, args.real)

    K1 = np.zeros((n, n))
    t1 = timed(lambda: qsim.kernel_serial(D, K1), args.reps)
    print(f"T1 = {t1:.2f} s for {n * (n - 1) // 2:,} pairs")

    rows = []
    for p in args.threads:
        p = min(p, MAXP)
        numba.set_num_threads(p)
        queue = qsim.WorkQueue(p, chunk=4)
        queue.run(D[:16], np.zeros((16, 16)))   # start the pool's threads before timing
        runs = {"block": lambda K: qsim.kernel_block(D, K),
                "cyclic": lambda K: qsim.kernel_cyclic(D, K, p),
                "folded": lambda K: qsim.kernel_folded(D, K),
                "dynamic": lambda K: queue.run(D, K)}
        for name, fn in runs.items():
            K = np.zeros((n, n))
            tp = timed(lambda: fn(K), args.reps)
            assert np.array_equal(K, K1), f"{name} on {p} threads differs from the serial result"
            rows.append(dict(threads=p, schedule=name, seconds=tp, speedup=t1 / tp, efficiency=t1 / tp / p))
            print(f"p={p:2d} {name:7s} {tp:7.3f} s  speedup {t1 / tp:5.2f}  efficiency {t1 / tp / p:.2f}", flush=True)
        queue.close()

    with open(OUT / "schedules.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows, n, t1)


def plot(rows, n, t1):
    import plots
    plt = plots.setup()
    fig, ax = plt.subplots(figsize=(8, 5))
    p = np.array(sorted({r["threads"] for r in rows}))
    ax.axvspan(0.5, 8.5, color="#F2F2F2", zorder=0)
    ax.text(4.5, p[-1] - 0.5, "8 P-cores", ha="center", color=plots.GREY, fontsize=12)
    plots.ideal(ax, p)
    ax.plot(p, p ** 2 / (2 * p - 1), color=plots.GREY, lw=1.2, ls=(0, (5, 3)), label="block limit p²/(2p−1)")
    for name, c, m in (("block", plots.NAVY, "o"), ("cyclic", plots.ORANGE, "s"),
                       ("folded", plots.BLUE, "D"), ("dynamic", plots.RED, "^")):
        rs = [r for r in rows if r["schedule"] == name]
        ax.plot([r["threads"] for r in rs], [r["speedup"] for r in rs], marker=m, color=c, label=name)
    ax.set(xlabel="threads", ylabel="speedup (T1 / Tp)", xticks=p, ylim=(0, p[-1] + 1),
           title=f"Kernel matrix, N = {n:,}, T1 = {t1:.1f} s")
    ax.legend(loc="upper left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "schedules.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
