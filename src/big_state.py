"""
One big state vector instead of many small ones.

Part 1: a 26-qubit state (1 GiB). One layer = a Hadamard on every qubit, which
reads and writes the whole array 26 times. Run it on 1..24 threads and compare
with a plain parallel copy, which is about as fast as the memory can go.
Part 2: 12 to 30 qubits on the CPU (1 and 24 threads) and the GPU, to see where
the GPU starts to win and where each one runs out of memory.

    python src/big_state.py
    python src/big_state.py --qubits 22 --nmax 22 --reps 1      # quick
"""
import argparse
import csv
import time

import numba
import numpy as np

import qsim
from config import RESULTS

OUT = RESULTS / "bigstate"
MAXP = numba.config.NUMBA_NUM_THREADS


def layer(psi, nq):
    for k in range(nq):
        qsim.h_gate(psi, k)


def timed(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def thread_sweep(nq, threads, reps):
    psi = np.zeros(1 << nq, np.complex128)
    psi[0] = 1
    dst = np.empty_like(psi)
    layer(psi, nq)
    # H on every qubit of |0...0> gives the same amplitude everywhere
    assert np.allclose(psi[:4096], 2.0 ** (-nq / 2))
    gb = nq * 2 * psi.nbytes / 1e9          # every gate reads and writes the whole state
    print(f"{nq} qubits: {psi.nbytes / 2**30:.2f} GiB, {gb:.1f} GB of memory traffic per layer")
    rows = []
    for p in threads:
        numba.set_num_threads(min(p, MAXP))
        t = timed(lambda: layer(psi, nq), reps)
        tc = timed(lambda: qsim.copy_par(dst, psi), reps)
        rows.append(dict(threads=p, s_per_layer=t, GBps=gb / t, copy_GBps=2 * psi.nbytes / 1e9 / tc))
        print(f"p={p:2d}  {t:6.3f} s per layer  speedup {rows[0]['s_per_layer'] / t:5.2f}  "
              f"{gb / t:5.1f} GB/s  (copy {rows[-1]['copy_GBps']:.1f} GB/s)", flush=True)
    norm = np.vdot(psi, psi).real
    assert abs(norm - 1) < 1e-9, f"norm drifted to {norm}"
    return rows


def qubit_sweep(nmin, nmax, reps):
    try:
        import cupy as cp
        h = np.zeros(8, np.complex128)
        qsim.h_gate_gpu(cp.asarray(h), 0)
        free = cp.cuda.Device().mem_info[0]
    except Exception as e:
        print("GPU skipped:", type(e).__name__)
        cp = None
    rows = []
    for nq in range(nmin, nmax + 1, 2):
        r = dict(qubits=nq, GiB=(16 << nq) / 2**30)
        k = reps if nq < 26 else 1
        for p in (1, MAXP):
            if p == 1 and nq > 28:      # would take minutes and tells us nothing new
                r["cpu_1"] = ""
                continue
            numba.set_num_threads(p)
            v = np.zeros(1 << nq, np.complex128)
            v[0] = 1
            layer(v, nq)
            r[f"cpu_{p}"] = timed(lambda: layer(v, nq), k)
            del v
        if cp is not None:
            for dt, key, size in ((cp.complex128, "gpu_c128", 16), (cp.complex64, "gpu_c64", 8)):
                if size << nq > 0.95 * free:
                    r[key] = "OOM"
                    continue
                g = cp.zeros(1 << nq, dt)
                g[0] = 1

                def run():
                    for q in range(nq):
                        qsim.h_gate_gpu(g, q)
                    cp.cuda.Device().synchronize()
                run()
                r[key] = timed(run, k)
                del g
                cp.get_default_memory_pool().free_all_blocks()
        rows.append(r)
        print(", ".join(f"{a}={b:.4g}" if isinstance(b, float) else f"{a}={b}" for a, b in r.items()), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=26)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 6, 8, 12, 16, 20, 24])
    ap.add_argument("--nmin", type=int, default=12)
    ap.add_argument("--nmax", type=int, default=30)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    w = np.zeros(8, np.complex128)
    qsim.h_gate(w, 0)
    qsim.copy_par(np.empty_like(w), w)

    threads = thread_sweep(args.qubits, args.threads, args.reps)
    write_csv(OUT / "threads.csv", threads)
    sweep = qubit_sweep(args.nmin, args.nmax, args.reps)
    write_csv(OUT / "qubits.csv", sweep, ["qubits", "GiB", "cpu_1", f"cpu_{MAXP}", "gpu_c128", "gpu_c64"])
    plot(threads, sweep, args.qubits)


def write_csv(path, rows, fields=None):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def plot(threads, sweep, nq):
    import plots
    plt = plots.setup()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    p = [r["threads"] for r in threads]
    a1.plot(p, [r["copy_GBps"] for r in threads], color=plots.GREY, ls=(0, (5, 3)), lw=1.5,
            label="plain copy (memory limit)")
    a1.plot(p, [r["GBps"] for r in threads], "o-", color=plots.NAVY, label="Hadamard layer")
    s = threads[0]["s_per_layer"] / threads[-1]["s_per_layer"]
    a1.set(xlabel="threads", ylabel="memory throughput (GB/s)", xticks=p, ylim=(0, None),
           title=f"{nq} qubits: {s:.1f}x faster on {p[-1]} threads")
    a1.legend(loc="lower right", fontsize=12)

    for key, c, m, label in (("cpu_1", plots.NAVY, "o", "CPU, 1 thread"),
                             (f"cpu_{MAXP}", plots.BLUE, "s", f"CPU, {MAXP} threads"),
                             ("gpu_c128", plots.RED, "D", "GPU, complex128"),
                             ("gpu_c64", plots.ORANGE, "^", "GPU, complex64")):
        pts = [(r["qubits"], r[key]) for r in sweep if isinstance(r.get(key), float)]
        if pts:
            a2.plot(*zip(*pts), marker=m, color=c, label=label)
    a2.set_yscale("log")
    a2.set(xlabel="qubits (memory = 16 bytes x 2^n)", ylabel="seconds per layer (log scale)",
           title="Each extra qubit doubles the cost")
    a2.legend(loc="upper left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "bigstate.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
