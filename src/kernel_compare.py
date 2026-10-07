"""
Same kernel matrix, different ways of computing it:
  A  one circuit per pair, 1 core and all cores (dynamic queue)
  B  every state simulated once, then one complex GEMM, 1 core and all cores
  G  B on the GPU, complex128 and complex64
at a small N and a big N. At the big N version A would take hours, so its time
there is extrapolated from the measured time per pair (marked in the CSV).

Also times the three steps of B separately (states, GEMM, |.|^2) with the old
serial squaring and the parallel one, because the serial step was what held B
back on 24 cores in the pilot.

    python src/kernel_compare.py
    python src/kernel_compare.py --n 1000 --big 4000 --reps 1     # quick
"""
import argparse
import csv
import time

import numba
import numpy as np
from threadpoolctl import threadpool_limits

import qsim
from config import RESULTS
from kernel_threads import compile_all, inputs, timed

OUT = RESULTS / "kernel"
MAXP = numba.config.NUMBA_NUM_THREADS


def gpu_timed(fn, reps):
    import cupy as cp
    fn()                                  # first call also compiles the CUDA code
    cp.cuda.Device().synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        cp.cuda.Device().synchronize()    # kernel launches return immediately
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def gemm_timed(D, p, reps):
    numba.set_num_threads(p)
    with threadpool_limits(limits=p, user_api="blas"):
        return timed(lambda: qsim.kernel_gemm(D), reps)


def b_steps(D, p, reps):
    numba.set_num_threads(p)
    out = dict(N=len(D), cores=p)
    with threadpool_limits(limits=p, user_api="blas"):
        psi = qsim.states(D)
        G = psi.conj() @ psi.T
        K = np.empty(G.shape)
        out["states_s"] = timed(lambda: qsim.states(D), reps)
        out["gemm_s"] = timed(lambda: psi.conj() @ psi.T, reps)
        out["abs2_serial_s"] = timed(lambda: np.square(np.abs(G)), reps)
        out["abs2_parallel_s"] = timed(lambda: qsim.abs2(G, K), reps)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2900)
    ap.add_argument("--big", type=int, default=16000)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    compile_all()
    qsim.kernel_gemm(inputs(16))

    n = args.n
    D = inputs(n)
    K1 = np.zeros((n, n))
    numba.set_num_threads(MAXP)
    t1 = timed(lambda: qsim.kernel_serial(D, K1), args.reps)
    queue = qsim.WorkQueue(MAXP, chunk=4)
    Kp = np.zeros((n, n))
    tp = timed(lambda: queue.run(D, Kp), args.reps)
    queue.close()
    assert np.array_equal(Kp, K1)
    up = np.triu_indices(n, 1)
    assert np.abs(qsim.kernel_gemm(D)[up] - K1[up]).max() < 1e-12

    rows = [dict(N=n, method="A, 1 core", seconds=t1, measured=True),
            dict(N=n, method=f"A, {MAXP} threads", seconds=tp, measured=True)]
    per_pair, speedup = t1 / (n * (n - 1) / 2), t1 / tp
    big = args.big
    Db = inputs(big)
    pairs = big * (big - 1) / 2
    rows += [dict(N=big, method="A, 1 core", seconds=per_pair * pairs, measured=False),
             dict(N=big, method=f"A, {MAXP} threads", seconds=per_pair * pairs / speedup, measured=False)]
    for size, Dm in ((n, D), (big, Db)):
        for p in (1, MAXP):
            rows.append(dict(N=size, method=f"B, {p} core{'s' if p > 1 else ''}",
                             seconds=gemm_timed(Dm, p, args.reps), measured=True))

    try:
        import cupy as cp
        e = np.abs(cp.asnumpy(qsim.kernel_gpu(cp.asarray(D)))[up] - K1[up]).max()
        assert e < 1e-12, f"GPU kernel is off by {e}"
        for size, Dm in ((n, D), (big, Db)):
            for dt in (np.complex128, np.complex64):
                Dg = cp.asarray(Dm.astype(dt))
                rows.append(dict(N=size, method=f"GPU, {np.dtype(dt).name}",
                                 seconds=gpu_timed(lambda: qsim.kernel_gpu(Dg), args.reps), measured=True))
                del Dg
                cp.get_default_memory_pool().free_all_blocks()
    except Exception as e:  # no CuPy, no GPU, or not enough GPU memory
        print("GPU part skipped:", type(e).__name__, e)

    for r in rows:
        print(f"N={r['N']:6d}  {r['method']:20s} {r['seconds']:10.4f} s  {'' if r['measured'] else '(extrapolated)'}")
    write_csv(OUT / "compare.csv", rows)

    steps = [b_steps(Db, p, args.reps) for p in (1, MAXP)]
    for s in steps:
        print(f"B on {s['cores']:2d} cores, N={s['N']}: states {s['states_s']:.3f} s, GEMM {s['gemm_s']:.3f} s, "
              f"|.|^2 {s['abs2_serial_s']:.3f} s serial / {s['abs2_parallel_s']:.3f} s parallel")
    write_csv(OUT / "b_steps.csv", steps)
    plot(rows, n, big)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def nice(s):
    if s >= 3600:
        return f"{s / 3600:.1f} h"
    if s >= 60:
        return f"{s / 60:.1f} min"
    if s >= 1:
        return f"{s:.1f} s"
    return f"{s * 1000:.0f} ms" if s >= 1e-3 else f"{s * 1e6:.0f} µs"


def plot(rows, n, big):
    import plots
    plt = plots.setup()
    methods = list(dict.fromkeys(r["method"] for r in rows))
    y = np.arange(len(methods))[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for size, c, m in ((n, plots.NAVY, "o"), (big, plots.RED, "s")):
        for r in rows:
            if r["N"] != size:
                continue
            yi = y[methods.index(r["method"])]
            ax.plot(r["seconds"], yi, marker=m, ms=9, ls="none", color=c,
                    markerfacecolor=c if r["measured"] else "white")
            ax.annotate(nice(r["seconds"]), (r["seconds"], yi), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=10)
        ax.plot([], [], marker=m, ls="none", color=c, label=f"N = {size:,}")
    ax.plot([], [], marker="s", ls="none", color=plots.RED, markerfacecolor="white", label="extrapolated")
    ax.set_xscale("log")
    ax.set_yticks(y, methods)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="seconds for the whole kernel matrix (log scale)", title="Kernel matrix: methods compared")
    ax.legend(loc="lower right", fontsize=11)
    ax.margins(y=0.12)
    fig.tight_layout()
    fig.savefig(OUT / "compare.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
