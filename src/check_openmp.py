"""
Check the C/OpenMP programs against qsim.py on real training beats:
  qkernel_omp  every schedule                      vs qsim.kernel_serial (version A)
  gram_omp     naive, tiled (odd tile size), tasks vs the same matrix
  gram_omp     cross (test x train)                vs qsim.kernel_cross
  bigstate_omp checks itself (amplitudes after H, norm), we only look at the exit code

    cd openmp && make && cd ..
    python src/check_openmp.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import qsim
from config import DATA, ROOT

N, M = 200, 70          # train beats, extra "test" beats for the cross kernel
LAM = 0.1


def exe(name):
    return ROOT / "openmp" / (name + ".exe" if sys.platform == "win32" else name)


def run(prog, args, out, shape):
    subprocess.run([str(exe(prog))] + [str(a) for a in args] + ["-o", str(out)], check=True, capture_output=True)
    return np.fromfile(out).reshape(shape)


def main():
    missing = [p for p in ("qkernel_omp", "gram_omp", "bigstate_omp") if not exe(p).exists()]
    if missing:
        sys.exit(f"not built: {', '.join(missing)} (run make in openmp/ first)")
    U = np.load(DATA / "features8.npz")["U_train"][:N + M].astype("<f8")
    D = qsim.encode(U, LAM)
    ref = np.zeros((N, N))
    qsim.kernel_serial(D[:N], ref)
    ref_cross = qsim.kernel_cross(D[N:], D[:N])

    ok = True

    def report(name, err, tol=1e-12):
        nonlocal ok
        print(f"{name:22s} max difference from Python: {err:.1e}")
        ok &= err < tol

    with tempfile.TemporaryDirectory() as tmp:
        feat, out = Path(tmp) / "u.bin", Path(tmp) / "K.bin"
        U.tofile(feat)
        common = ["-f", feat, "-l", LAM, "-r", 1, "-t", 4]
        for sched in ("static", "cyclic", "folded", "dynamic", "guided", "taskloop"):
            K = run("qkernel_omp", ["-n", N, "-s", sched, "-c", 4] + common, out, (N, N))
            report(f"qkernel_omp {sched}", np.abs(K - ref).max())
        # tile size 24 does not divide 200, so the edge tiles get tested too
        for method in ("naive", "tiled", "tasks"):
            K = run("gram_omp", ["-n", N, "-m", method, "-b", 24] + common, out, (N, N))
            report(f"gram_omp {method}", np.abs(K - ref).max())
        K = run("gram_omp", ["-n", N, "-x", M, "-m", "cross"] + common, out, (M, N))
        report("gram_omp cross", np.abs(K - ref_cross).max())

    r = subprocess.run([str(exe("bigstate_omp")), "-q", "16", "-t", "4", "-r", "1"], capture_output=True, text=True)
    print(f"{'bigstate_omp':22s} {r.stderr.strip()}")
    ok &= r.returncode == 0

    print("OpenMP matches Python" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
