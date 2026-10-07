"""
Cut every annotated heartbeat out of the 44 MIT-BIH records and time it
with 1..24 worker processes (one record per task).

Writes data/beats_ds1.npz, data/beats_ds2.npz and results/prep/prep_speedup.{csv,png}

    python src/prep_beats.py
    python src/prep_beats.py --procs 2 4 8 --reps 1      # quicker
"""
import argparse
import csv
import os
import time
from multiprocessing import Pool

import numpy as np

from config import AAMI, DATA, DS1, DS2, FS, HALF_WIN, MITDB, RESULTS

OUT = RESULTS / "prep"


def extract(rec):
    # imports inside so each spawned worker only pays for them once
    import wfdb
    from scipy.ndimage import median_filter
    from scipy.signal import butter, filtfilt

    sig = wfdb.rdrecord(str(MITDB / str(rec)))
    x = sig.p_signal[:, sig.sig_name.index("MLII")]
    x = x - median_filter(median_filter(x, 71), 215)   # baseline wander (200 ms, then 600 ms)
    b, a = butter(4, 40 / (FS / 2))
    x = filtfilt(b, a, x)                               # 40 Hz low-pass, zero phase

    ann = wfdb.rdann(str(MITDB / str(rec)), "atr")
    all_peaks = np.array([s for s, c in zip(ann.sample, ann.symbol) if c in AAMI or c in "/fQ"])
    keep = [(s, AAMI[c]) for s, c in zip(ann.sample, ann.symbol)
            if c in AAMI and HALF_WIN <= s < len(x) - HALF_WIN]
    pos = np.array([s for s, _ in keep])
    beats = np.array([x[s - HALF_WIN:s + HALF_WIN] for s in pos], np.float32)
    labels = np.array([y for _, y in keep], np.int8)

    # RR intervals to the previous and next beat (any type), in seconds
    k = np.searchsorted(all_peaks, pos)
    rr_pre = np.where(k > 0, pos - all_peaks[np.maximum(k - 1, 0)], 0) / FS
    rr_post = np.where(k < len(all_peaks) - 1, all_peaks[np.minimum(k + 1, len(all_peaks) - 1)] - pos, 0) / FS
    return rec, beats, labels, pos, np.stack([rr_pre, rr_post], 1).astype(np.float32)


def _pid(_):
    time.sleep(0.05)
    return os.getpid()


def save(results, records, path):
    part = [r for r in results if r[0] in records]
    labels = np.concatenate([p[2] for p in part])
    np.savez_compressed(path,
                        beats=np.concatenate([p[1] for p in part]),
                        labels=labels,
                        record=np.concatenate([[p[0]] * len(p[2]) for p in part]).astype(np.int16),
                        sample=np.concatenate([p[3] for p in part]),
                        rr=np.concatenate([p[4] for p in part]))
    return np.bincount(labels, minlength=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, nargs="+", default=[2, 4, 6, 8, 11, 16, 22, 24])
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    records = sorted(DS1 + DS2)

    extract(records[0])  # warm the file cache before timing
    times = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        ref = [extract(r) for r in records]
        times.append(time.perf_counter() - t0)
    t1 = float(np.median(times))

    n1 = save(ref, DS1, DATA / "beats_ds1.npz")
    n2 = save(ref, DS2, DATA / "beats_ds2.npz")
    print(f"serial: {t1:.2f} s for {len(records)} records")
    print("DS1 N/S/V/F:", n1.tolist(), "total", int(n1.sum()))
    print("DS2 N/S/V/F:", n2.tolist(), "total", int(n2.sum()))

    rows = [dict(procs=1, startup_s=0.0, compute_s=t1, speedup=1.0)]
    for p in args.procs:
        with Pool(p) as pool:
            # On Windows every worker is a fresh interpreter, so wait until all of
            # them exist and have imported scipy/wfdb, and time that separately.
            t0 = time.perf_counter()
            while len(set(pool.map(_pid, range(4 * p), chunksize=1))) < p:
                pass
            pool.map(extract, records[:p], chunksize=1)
            startup = time.perf_counter() - t0

            times = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                out = pool.map(extract, records, chunksize=1)
                times.append(time.perf_counter() - t0)
        assert all(np.array_equal(u[1], v[1]) for u, v in zip(ref, out)), "parallel result differs"
        tp = float(np.median(times))
        rows.append(dict(procs=p, startup_s=startup, compute_s=tp, speedup=t1 / tp))
        print(f"p={p:2d}  start-up {startup:5.2f} s  compute {tp:5.2f} s  speedup {t1 / tp:5.2f}", flush=True)

    with open(OUT / "prep_speedup.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows)


def plot(rows):
    import plots
    plt = plots.setup()
    p = np.array([r["procs"] for r in rows])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    plots.ideal(a1, p)
    q = np.arange(1, p[-1] + 1)
    a1.plot(q, 44 / np.ceil(44 / q), color=plots.GREY, lw=1.2, ls=(0, (5, 3)), drawstyle="steps-post",
            label="limit if all 44 tasks took equal time")
    a1.plot(p, [r["speedup"] for r in rows], "o-", color=plots.NAVY, label="measured")
    a1.set(xlabel="worker processes", ylabel="speedup (T1 / Tp)", title="Beat extraction: speedup",
           xticks=p, ylim=(0, p[-1] + 1))
    a1.legend(loc="upper left")
    x = np.arange(len(rows))
    a2.bar(x, [r["compute_s"] for r in rows], color=plots.BLUE, label="computing")
    a2.bar(x, [r["startup_s"] for r in rows], bottom=[r["compute_s"] for r in rows],
           color=plots.LIGHT_BLUE, label="starting the worker processes")
    a2.set_xticks(x, [str(v) for v in p])
    a2.set(xlabel="worker processes", ylabel="seconds", title="Where the time goes")
    a2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "prep_speedup.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
