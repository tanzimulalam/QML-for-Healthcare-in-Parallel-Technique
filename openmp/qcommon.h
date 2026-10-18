/*
 * qcommon.h - the parts of the simulator that all the OpenMP programs share.
 *
 * Same circuit as src/qsim.py: 8 qubits, ZZ feature map with 2 repetitions,
 * qubit k = bit k of the basis index (Qiskit's ordering).
 */
#ifndef QCOMMON_H
#define QCOMMON_H

#include <complex.h>
#include <stddef.h>

#define NQ   8
#define DIM  (1 << NQ)
#define REPS 2

typedef double complex cplx;

extern const double PI;

/* wall-clock seconds; QueryPerformanceCounter on Windows (omp_get_wtime only
   ticked once per millisecond there), clock_gettime everywhere else */
double now(void);

void *xmalloc(size_t bytes);

/* n x 8 angles: lambda * pi * u from a features file (src/features.py writes
   data/u_train.bin), or uniform random in [0, pi) if path is NULL */
double *load_inputs(const char *path, int n, double lam);

/* phase vector of one repetition of the ZZ part (it is diagonal) */
void zz_phases(const double *x, cplx *d);

/* zz_phases for n samples, in parallel; returns n x DIM */
cplx *phase_table(const double *X, int n);

/* Hadamard on every qubit of a DIM-amplitude state */
void h_layer(cplx *psi);

/* U(x)|0> for the phase vector d */
void encode_state(const cplx *d, cplx *psi);

/* |<0| U(x_j)^+ U(x_i) |0>|^2, one circuit (version A) */
double pair_kernel(const cplx *di, const cplx *dj);

/* K[i][j] for all j > i */
void kernel_row(int i, int n, const cplx *D, double *K);

/* median of v[0..n-1]; v is left untouched */
double median(const double *v, int n);

int write_doubles(const char *path, const double *v, size_t count);

#endif
