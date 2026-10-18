"""
Small state-vector simulator for the ZZ-feature-map quantum kernel.

n qubits are stored as an array of 2**n complex amplitudes. A Hadamard on qubit k
mixes the pairs (i, i | 2**k); no amplitude is in two pairs, so a gate can be split
over as many threads (or GPU threads) as we like. The ZZ part of the feature map is
diagonal, so one repetition of it is a single precomputed phase vector.

Two ways to build the kernel matrix K[i, j] = |<psi(x_i)|psi(x_j)>|^2:
  version A - one circuit per pair, U(x_j)^+ U(x_i)|0>, like real hardware would
  version B - simulate every sample once, then K = |Psi Psi^+|^2 (one GEMM)
Checked against Qiskit's ZZFeatureMap in check_qsim.py.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit, prange

REPS = 2  # feature map repetitions


def zz_phases(x):
    # qubit k = bit k of the basis index (Qiskit's ordering)
    nq = len(x)
    z = (np.arange(2 ** nq)[:, None] >> np.arange(nq)) & 1
    ph = 2.0 * (z @ x)
    for i in range(nq):
        for j in range(i + 1, nq):
            ph = ph + 2.0 * (np.pi - x[i]) * (np.pi - x[j]) * (z[:, i] ^ z[:, j])
    return np.exp(1j * ph)


@njit(parallel=True, cache=True)
def _phase_rows(X, out):
    # same formula as zz_phases, one sample per iteration, samples in parallel
    nq = X.shape[1]
    for r in prange(X.shape[0]):
        for s in range(out.shape[1]):
            ph = 0.0
            for i in range(nq):
                zi = (s >> i) & 1
                ph += 2.0 * X[r, i] * zi
                for j in range(i + 1, nq):
                    if zi != ((s >> j) & 1):
                        ph += 2.0 * (np.pi - X[r, i]) * (np.pi - X[r, j])
            out[r, s] = np.cos(ph) + 1j * np.sin(ph)


def phase_table(X):
    # A Python loop over zz_phases was fine for a few thousand beats, but for the
    # 49,691 test beats it took ~5 s, more than everything else in the QSVM.
    X = np.ascontiguousarray(X, dtype=np.float64)
    out = np.empty((X.shape[0], 2 ** X.shape[1]), np.complex128)
    _phase_rows(X, out)
    return out


def encode(U, lam=0.1):
    """Features in [0, 1] -> phase vectors, with bandwidth lam."""
    return phase_table(lam * np.pi * np.asarray(U, dtype=np.float64))


# ---- version A on the CPU ---------------------------------------------------

@njit(cache=True)
def _h_layer(psi):
    s = 1.0 / np.sqrt(2.0)
    n = psi.shape[0]
    stride = 1
    while stride < n:
        for base in range(0, n, 2 * stride):
            for off in range(stride):
                a = psi[base + off]
                b = psi[base + off + stride]
                psi[base + off] = (a + b) * s
                psi[base + off + stride] = (a - b) * s
        stride *= 2


@njit(cache=True)
def pair_kernel(di, dj, psi):
    for s in range(psi.shape[0]):
        psi[s] = 0.0
    psi[0] = 1.0
    for _ in range(REPS):          # U(x_i)
        _h_layer(psi)
        for s in range(psi.shape[0]):
            psi[s] *= di[s]
    for _ in range(REPS):          # U(x_j)^+
        for s in range(psi.shape[0]):
            psi[s] *= dj[s].conjugate()
        _h_layer(psi)
    return psi[0].real ** 2 + psi[0].imag ** 2


@njit(cache=True)
def _row(i, D, K, psi):
    for j in range(i + 1, D.shape[0]):
        K[i, j] = pair_kernel(D[i], D[j], psi)


@njit(cache=True)
def kernel_serial(D, K):
    psi = np.empty(D.shape[1], np.complex128)
    for i in range(D.shape[0]):
        _row(i, D, K, psi)


@njit(parallel=True, cache=True)
def kernel_block(D, K):
    # prange with Numba's default schedule: equal contiguous blocks of rows
    for i in prange(D.shape[0]):
        psi = np.empty(D.shape[1], np.complex128)
        _row(i, D, K, psi)


@njit(parallel=True, cache=True)
def kernel_cyclic(D, K, p):
    # thread t gets rows t, t+p, t+2p, ...
    n = D.shape[0]
    for t in prange(p):
        psi = np.empty(D.shape[1], np.complex128)
        for i in range(t, n, p):
            _row(i, D, K, psi)


@njit(parallel=True, cache=True)
def kernel_folded(D, K):
    # row t together with row n-1-t, so every task has n-1 entries
    n = D.shape[0]
    for t in prange((n + 1) // 2):
        psi = np.empty(D.shape[1], np.complex128)
        _row(t, D, K, psi)
        if n - 1 - t != t:
            _row(n - 1 - t, D, K, psi)


@njit(nogil=True, cache=True)
def _rows(D, K, start, stop):
    psi = np.empty(D.shape[1], np.complex128)
    for i in range(start, stop):
        _row(i, D, K, psi)


@njit(nogil=True, cache=True)
def kernel_rows(D, rows, out):
    # out[k, j] = K[rows[k], j] for j > rows[k]; the rest of out is left alone.
    # Used by the MPI version, where a worker only holds the rows it computed.
    psi = np.empty(D.shape[1], np.complex128)
    for k in range(rows.shape[0]):
        i = rows[k]
        for j in range(i + 1, D.shape[0]):
            out[k, j] = pair_kernel(D[i], D[j], psi)


class WorkQueue:
    """
    Dynamic schedule: chunks of rows go into a queue and a free thread takes the
    next one. Numba's chunksize setting did nothing for us on Windows, so this is
    done by hand with a thread pool (_rows releases the GIL).
    """

    def __init__(self, threads, chunk=4):
        self.chunk = chunk
        self.pool = ThreadPoolExecutor(max_workers=threads)

    def run(self, D, K):
        n, c = D.shape[0], self.chunk
        list(self.pool.map(lambda s: _rows(D, K, s, min(s + c, n)), range(0, n, c)))

    def close(self):
        self.pool.shutdown()


# ---- version B on the CPU ---------------------------------------------------

@njit(parallel=True, cache=True)
def h_gate(psi, k):
    # Hadamard on qubit k of a flat array. Works for one big state, or for a batch
    # of small states stored back to back as long as k < qubits per state.
    stride = 1 << k
    s = 1.0 / np.sqrt(2.0)
    for t in prange(psi.shape[0] // 2):
        i0 = ((t >> k) << (k + 1)) | (t & (stride - 1))
        i1 = i0 | stride
        a = psi[i0]
        b = psi[i1]
        psi[i0] = (a + b) * s
        psi[i1] = (a - b) * s


@njit(parallel=True, cache=True)
def _mul(psi, d):
    for i in prange(psi.shape[0]):
        psi[i] *= d[i]


@njit(parallel=True, cache=True)
def abs2(G, K):
    n, m = G.shape
    for i in prange(n):
        for j in range(m):
            K[i, j] = G[i, j].real ** 2 + G[i, j].imag ** 2


@njit(parallel=True, cache=True)
def copy_par(dst, src):
    # plain copy, used as the memory-bandwidth reference in big_state.py
    for i in prange(src.shape[0]):
        dst[i] = src[i]


def states(D):
    n, dim = D.shape
    nq = dim.bit_length() - 1
    flat = np.zeros(n * dim, np.complex128)
    flat[::dim] = 1.0
    d = D.reshape(-1)
    for _ in range(REPS):
        for k in range(nq):
            h_gate(flat, k)
        _mul(flat, d)
    return flat.reshape(n, dim)


def gram(psi_a, psi_b):
    # K[a, b] = |<psi_a|psi_b>|^2 for every pair of rows, one complex GEMM
    G = psi_a.conj() @ psi_b.T
    K = np.empty(G.shape)
    abs2(G, K)
    return K


def kernel_gemm(D, serial_square=False):
    """Version B. serial_square=True keeps the old single-threaded last step
    (the Amdahl bottleneck from the midterm report) for comparison."""
    psi = states(D)
    if serial_square:
        K = np.abs(psi.conj() @ psi.T)
        K *= K
        return K
    return gram(psi, psi)


def kernel_cross(DA, DB):
    """Kernel between two different sets of beats, e.g. test x train for the
    SVM's predictions. Rectangular, so there is no triangle to balance."""
    return gram(states(DA), states(DB))


# ---- version B on the GPU ---------------------------------------------------

CUDA_SRC = r"""
#include <cupy/complex.cuh>

// one thread per amplitude pair, same indexing as h_gate() on the CPU
template <typename T>
__device__ void h_pair(complex<T>* psi, long long half, int k) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= half) return;
    long long stride = 1LL << k;
    long long i0 = ((t >> k) << (k + 1)) | (t & (stride - 1));
    long long i1 = i0 | stride;
    complex<T> a = psi[i0], b = psi[i1];
    const T s = (T)0.70710678118654752440;
    psi[i0] = (a + b) * s;
    psi[i1] = (a - b) * s;
}

extern "C" __global__ void h_gate_c128(complex<double>* psi, long long half, int k) { h_pair<double>(psi, half, k); }
extern "C" __global__ void h_gate_c64(complex<float>* psi, long long half, int k) { h_pair<float>(psi, half, k); }

// Version A on the GPU, 8 qubits only: one thread block per pair (i, j).
// The 256 amplitudes sit in shared memory and each of the 128 threads does
// one butterfly per gate, with a barrier after every gate.
__device__ void h_layer_shared(complex<double>* psi, int t) {
    const double s = 0.70710678118654752440;
    for (int k = 0; k < 8; k++) {
        int stride = 1 << k;
        int i0 = ((t >> k) << (k + 1)) | (t & (stride - 1));
        int i1 = i0 | stride;
        complex<double> a = psi[i0], b = psi[i1];
        psi[i0] = (a + b) * s;
        psi[i1] = (a - b) * s;
        __syncthreads();
    }
}

extern "C" __global__ void pair_circuits(const complex<double>* D, double* K, int n, int row0) {
    int i = row0 + blockIdx.y, j = blockIdx.x;
    // half the grid is below the diagonal; those blocks leave straight away,
    // all threads together, so nobody is left waiting at a barrier
    if (j <= i || i >= n) return;
    __shared__ complex<double> psi[256];
    const complex<double>* di = D + (long long)i * 256;
    const complex<double>* dj = D + (long long)j * 256;
    int t = threadIdx.x;
    psi[t] = complex<double>(t == 0 ? 1.0 : 0.0, 0.0);
    psi[t + 128] = 0.0;
    __syncthreads();
    for (int r = 0; r < 2; r++) {           // U(x_i)
        h_layer_shared(psi, t);
        psi[t] *= di[t];
        psi[t + 128] *= di[t + 128];
        __syncthreads();
    }
    for (int r = 0; r < 2; r++) {           // U(x_j)^+
        psi[t] *= conj(dj[t]);
        psi[t + 128] *= conj(dj[t + 128]);
        __syncthreads();
        h_layer_shared(psi, t);
    }
    if (t == 0)
        K[(long long)i * n + j] = psi[0].real() * psi[0].real() + psi[0].imag() * psi[0].imag();
}
"""

_gpu = {}


def _gpu_fn(name):
    import cupy as cp
    if not _gpu:
        mod = cp.RawModule(code=CUDA_SRC, options=("--std=c++17",))
        for f in ("h_gate_c128", "h_gate_c64", "pair_circuits"):
            _gpu[f] = mod.get_function(f)
    return _gpu[name]


def h_gate_gpu(psi, k, block=256):
    import cupy as cp
    kern = _gpu_fn("h_gate_c64" if psi.dtype == cp.complex64 else "h_gate_c128")
    half = psi.size // 2
    kern(((half + block - 1) // block,), (block,), (psi, np.int64(half), np.int32(k)))


def states_gpu(D):
    import cupy as cp
    n, dim = D.shape
    nq = dim.bit_length() - 1
    flat = cp.zeros(n * dim, D.dtype)
    flat[::dim] = 1.0
    d = D.reshape(-1)
    for _ in range(REPS):
        for k in range(nq):
            h_gate_gpu(flat, k)
        flat *= d
    return flat.reshape(n, dim)


def gram_gpu(psi_a, psi_b):
    import cupy as cp
    K = cp.abs(psi_a.conj() @ psi_b.T)
    K *= K
    return K


def kernel_gpu(D):
    """Version B on the GPU. D is a CuPy array (complex128 or complex64)."""
    psi = states_gpu(D)
    return gram_gpu(psi, psi)


def kernel_pairs_gpu(D):
    """Version A on the GPU (one circuit per pair). D is a complex128 CuPy array
    of 8-qubit phase vectors; only the upper triangle of the result is filled."""
    import cupy as cp
    n, dim = D.shape
    assert dim == 256 and D.dtype == cp.complex128, "pair_circuits is written for 8 qubits, complex128"
    kern = _gpu_fn("pair_circuits")
    K = cp.zeros((n, n))
    for row0 in range(0, n, 65535):          # grid y is limited to 65535 blocks
        kern((n, min(65535, n - row0)), (128,), (D, K, np.int32(n), np.int32(row0)))
    return K
