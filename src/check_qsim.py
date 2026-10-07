"""
Correctness checks for qsim.py. Run this before trusting any timing.

  1. statevectors vs brute-force dense matrices (2..8 qubits)
  2. kernel vs Qiskit's ZZFeatureMap + Statevector (needs qiskit)
  3. version A == version B (serial and parallel squaring), and the GPU if CuPy works
"""
import sys

import numpy as np

import qsim


def dense_state(x):
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    hn = np.array([[1.0]])
    for _ in range(len(x)):
        hn = np.kron(hn, h)
    u = np.diag(qsim.zz_phases(x)) @ hn
    psi = np.zeros(2 ** len(x), complex)
    psi[0] = 1
    for _ in range(qsim.REPS):
        psi = u @ psi
    return psi


def main():
    rng = np.random.default_rng(7)
    ok = True

    err = 0.0
    for nq in range(2, 9):
        X = rng.uniform(0, np.pi, (5, nq))
        err = max(err, np.abs(qsim.states(qsim.phase_table(X)) - np.array([dense_state(x) for x in X])).max())
    print(f"dense matrices, 2-8 qubits: max error {err:.1e}")
    ok &= err < 1e-12

    try:
        from qiskit.quantum_info import Statevector
        try:
            from qiskit.circuit.library import zz_feature_map
            fm = zz_feature_map(8, reps=qsim.REPS, entanglement="full")
        except ImportError:
            from qiskit.circuit.library import ZZFeatureMap
            fm = ZZFeatureMap(8, reps=qsim.REPS, entanglement="full")
        X = rng.uniform(0, np.pi, (20, 8))
        q = np.array([Statevector(fm.assign_parameters(x)).data for x in X])
        s = qsim.states(qsim.phase_table(X))
        err = np.abs(np.abs(q.conj() @ q.T) ** 2 - np.abs(s.conj() @ s.T) ** 2).max()
        print(f"Qiskit ZZFeatureMap, 8 qubits: max kernel error {err:.1e}")
        ok &= err < 1e-10
    except ImportError:
        print("qiskit not installed, skipping that check")

    D = qsim.phase_table(rng.uniform(0, np.pi, (60, 8)))
    KA = np.zeros((60, 60))
    qsim.kernel_serial(D, KA)
    up = np.triu_indices(60, 1)
    for name, KB in (("B", qsim.kernel_gemm(D)), ("B, serial square", qsim.kernel_gemm(D, serial_square=True))):
        err = np.abs(KA[up] - KB[up]).max()
        print(f"version A vs {name}: max error {err:.1e}")
        ok &= err < 1e-12

    try:
        import cupy as cp
        e128 = np.abs(cp.asnumpy(qsim.kernel_gpu(cp.asarray(D)))[up] - KA[up]).max()
        e64 = np.abs(cp.asnumpy(qsim.kernel_gpu(cp.asarray(D.astype(np.complex64))))[up] - KA[up]).max()
        print(f"GPU: complex128 error {e128:.1e}, complex64 error {e64:.1e}")
        ok &= e128 < 1e-12 and e64 < 1e-5
    except Exception as e:  # no CuPy or no GPU
        print("GPU check skipped:", type(e).__name__)

    print("all checks passed" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
