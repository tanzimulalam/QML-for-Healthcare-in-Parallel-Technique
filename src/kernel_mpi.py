"""
Version A of the kernel matrix with MPI (mpi4py), so it can run across the
cluster's nodes and not just on one machine's cores.

Every rank builds the same inputs by itself, so the only messages are row
numbers going out and finished kernel rows coming back.

  --mode dynamic   manager-worker. Rank 0 only hands out chunks of rows and
                   collects the results; a worker asks for the next chunk as
                   soon as it has sent the last one back. With P ranks there
                   are P-1 workers.
  --mode cyclic    no manager: rank r takes rows r, r+P, r+2P, ... and rank 0
                   collects everything with one Gatherv at the end
  --mode block     the same, but every rank takes one contiguous block of rows
  --mode serial    no MPI at all, one process; gives T1 for the speedups

Each run appends one line to results/mpi/runs.csv.

Windows (Intel MPI from pip), from the repository root in Git Bash:
    export I_MPI_ROOT="$PWD/../.venv/Library"     # wherever the venv is
    mpiexec -localonly -n 4 python src/kernel_mpi.py --n 800 --mode dynamic --check
Linux / cluster:
    mpirun -np 16 python src/kernel_mpi.py --n 3000 --mode dynamic
"""
import argparse
import csv
import time

import numpy as np

import qsim
from config import RESULTS
from kernel_threads import inputs

OUT = RESULTS / "mpi"
TAG_WORK, TAG_DONE, TAG_ROWS = 1, 2, 3
STOP = -1


def manager(comm, n, chunk):
    """Rank 0 in dynamic mode: hand out row chunks, collect the rows."""
    from mpi4py import MPI
    K = np.zeros((n, n))
    starts = list(range(0, n, chunk))        # longest rows first, which helps at the end
    nxt, active = 0, comm.Get_size() - 1
    status = MPI.Status()
    waiting = 0.0
    while active:
        t0 = MPI.Wtime()
        start = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_DONE, status=status)
        src = status.Get_source()
        if start is not None:                # None = "I'm ready" from a new worker
            comm.Recv(K[start:min(start + chunk, n)], source=src, tag=TAG_ROWS)
        waiting += MPI.Wtime() - t0
        if nxt < len(starts):
            comm.send(starts[nxt], dest=src, tag=TAG_WORK)
            nxt += 1
        else:
            comm.send(STOP, dest=src, tag=TAG_WORK)
            active -= 1
    return K, dict(busy=0.0, comm=waiting, tasks=0)


def worker(comm, D, chunk):
    """Every other rank in dynamic mode."""
    from mpi4py import MPI
    n = len(D)
    busy = sending = 0.0
    tasks = 0
    comm.send(None, dest=0, tag=TAG_DONE)
    while True:
        start = comm.recv(source=0, tag=TAG_WORK)
        if start == STOP:
            break
        rows = np.arange(start, min(start + chunk, n))
        t0 = MPI.Wtime()
        out = np.zeros((len(rows), n))
        qsim.kernel_rows(D, rows, out)
        t1 = MPI.Wtime()
        comm.send(start, dest=0, tag=TAG_DONE)
        comm.Send(out, dest=0, tag=TAG_ROWS)
        sending += MPI.Wtime() - t1
        busy += t1 - t0
        tasks += 1
    return dict(busy=busy, comm=sending, tasks=tasks)


def my_rows(n, size, rank, mode):
    if mode == "cyclic":
        return np.arange(rank, n, size)
    edges = np.linspace(0, n, size + 1).astype(int)
    return np.arange(edges[rank], edges[rank + 1])


def static_run(comm, D, mode):
    """block / cyclic: every rank computes its own rows, then one Gatherv."""
    from mpi4py import MPI
    n, size, rank = len(D), comm.Get_size(), comm.Get_rank()
    rows = my_rows(n, size, rank, mode)
    t0 = MPI.Wtime()
    out = np.zeros((len(rows), n))
    qsim.kernel_rows(D, rows, out)
    t1 = MPI.Wtime()
    counts = comm.gather(len(rows) * n, root=0)
    recv = None
    if rank == 0:
        buf = np.empty(n * n)
        displs = [0] + [int(c) for c in np.cumsum(counts)[:-1]]
        recv = [buf, counts, displs, MPI.DOUBLE]
    comm.Gatherv(out.reshape(-1), recv, root=0)
    stats = dict(busy=t1 - t0, comm=MPI.Wtime() - t1, tasks=1)
    K = None
    if rank == 0:
        # the gathered rows are in rank order; put them back in row order
        order = np.concatenate([my_rows(n, size, r, mode) for r in range(size)])
        K = np.empty((n, n))
        K[order] = buf.reshape(n, n)
    return K, stats


def append_csv(row):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "runs.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--mode", choices=["dynamic", "cyclic", "block", "serial"], default="dynamic")
    ap.add_argument("--chunk", type=int, default=4, help="rows per message in dynamic mode")
    ap.add_argument("--real", action="store_true", help="first N training beats instead of random inputs")
    ap.add_argument("--check", action="store_true", help="compare with the serial result (small N only)")
    args = ap.parse_args()

    D = inputs(args.n, args.real)
    small = inputs(8)
    qsim.kernel_rows(small, np.arange(8), np.zeros((8, 8)))     # compile before timing

    if args.mode == "serial":
        K = np.zeros((args.n, args.n))
        t0 = time.perf_counter()
        qsim.kernel_serial(D, K)
        t = time.perf_counter() - t0
        print(f"serial, N={args.n}: {t:.3f} s")
        append_csv(dict(n=args.n, ranks=1, mode="serial", chunk=0, seconds=t, busy_min=t, busy_max=t,
                        comm_max=0.0, ok=""))
        return

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if args.mode == "dynamic" and size < 2:
        raise SystemExit("dynamic mode needs at least 2 ranks (1 manager + workers)")

    comm.Barrier()
    t0 = MPI.Wtime()
    if args.mode == "dynamic":
        if rank == 0:
            K, stats = manager(comm, args.n, args.chunk)
        else:
            K, stats = None, worker(comm, D, args.chunk)
    else:
        K, stats = static_run(comm, D, args.mode)
    elapsed = MPI.Wtime() - t0
    everyone = comm.gather(stats, root=0)

    if rank == 0:
        workers = everyone[1:] if args.mode == "dynamic" else everyone
        busy = [s["busy"] for s in workers]
        ok = ""
        if args.check:
            ref = np.zeros((args.n, args.n))
            qsim.kernel_serial(D, ref)
            ok = bool(np.array_equal(np.triu(K, 1), ref))
        # workers only: the manager's "comm" is mostly waiting for results, not messages.
        # For block/cyclic it includes waiting in Gatherv for the slowest rank.
        comm_max = max(s["comm"] for s in workers)
        print(f"{args.mode}, {size} ranks, N={args.n}: {elapsed:.3f} s, workers busy {min(busy):.3f} to "
              f"{max(busy):.3f} s, communication up to {comm_max:.3f} s"
              + (f", same as serial: {ok}" if args.check else ""))
        if args.mode == "dynamic":
            print("chunks per worker:", [s["tasks"] for s in workers])
        append_csv(dict(n=args.n, ranks=size, mode=args.mode, chunk=args.chunk if args.mode == "dynamic" else 0,
                        seconds=elapsed, busy_min=min(busy), busy_max=max(busy), comm_max=comm_max, ok=ok))
        if args.check and not ok:
            comm.Abort(1)


if __name__ == "__main__":
    main()
