"""
Check that the C/OpenMP program gives the same kernel matrix as qsim.py,
for every schedule, on real training beats.

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

N = 200
LAM = 0.1
EXE = ROOT / "openmp" / ("qkernel_omp.exe" if sys.platform == "win32" else "qkernel_omp")


def main():
    if not EXE.exists():
        sys.exit(f"{EXE} not found, run make in openmp/ first")
    U = np.load(DATA / "features8.npz")["U_train"][:N].astype("<f8")
    ref = np.zeros((N, N))
    qsim.kernel_serial(qsim.encode(U, LAM), ref)

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        feat, out = Path(tmp) / "u.bin", Path(tmp) / "K.bin"
        U.tofile(feat)
        for sched in ("static", "cyclic", "folded", "dynamic", "guided"):
            subprocess.run([str(EXE), "-n", str(N), "-t", "4", "-s", sched, "-c", "4", "-f", str(feat),
                            "-l", str(LAM), "-r", "1", "-o", str(out)], check=True, capture_output=True)
            K = np.fromfile(out).reshape(N, N)
            err = np.abs(K - ref).max()
            print(f"{sched:8s} max difference from Python: {err:.1e}")
            ok &= err < 1e-12
    print("OpenMP matches Python" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
