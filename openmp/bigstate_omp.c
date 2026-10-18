/*
 * bigstate_omp.c - one large state vector with OpenMP.
 *
 * The kernel programs simulate millions of tiny 8-qubit circuits. This one does
 * the opposite: a single state of q qubits (2^q amplitudes, 16 bytes each), split
 * between threads. Two kinds of layer are timed, because they behave differently:
 *
 *   H layer   a Hadamard on every qubit. Each gate reads and writes the whole
 *             array and does almost no arithmetic per byte, so it is limited by
 *             memory bandwidth and stops scaling once the memory is saturated.
 *   ZZ layer  the diagonal part of the feature map. Every amplitude gets a phase
 *             that takes q(q-1)/2 pair checks and a cexp to compute, so this one
 *             is limited by arithmetic and keeps scaling with the cores.
 *   copy      a plain parallel copy of the array, roughly the fastest the memory
 *             can go, as a reference for the H layer.
 *
 * Memory is first-touched by the threads that use it later (static schedule
 * everywhere). With -s the array is zeroed by one thread instead; on a laptop
 * that changes little, on a two-socket cluster node all pages end up on one
 * socket's memory and the H layer gets slower.
 *
 * Usage:   bigstate_omp -q qubits [-t threads] [-r reps] [-s]
 * Prints:  qubits,threads,init,h_s,h_GBps,zz_s,copy_GBps,norm_err
 */
#include <getopt.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qcommon.h"

static void h_gate(cplx *psi, long half, int k)
{
    const double s = 0.70710678118654752440;
    long stride = 1L << k;
    #pragma omp parallel for schedule(static)
    for (long t = 0; t < half; t++) {
        long i0 = ((t >> k) << (k + 1)) | (t & (stride - 1));
        long i1 = i0 | stride;
        cplx a = psi[i0], b = psi[i1];
        psi[i0] = (a + b) * s;
        psi[i1] = (a - b) * s;
    }
}

static void h_layer_big(cplx *psi, int q)
{
    for (int k = 0; k < q; k++)
        h_gate(psi, 1L << (q - 1), k);
}

static void zz_layer_big(cplx *psi, int q, const double *x)
{
    long dim = 1L << q;
    #pragma omp parallel for schedule(static)
    for (long s = 0; s < dim; s++) {
        double ph = 0.0;
        for (int i = 0; i < q; i++) {
            int zi = (s >> i) & 1;
            ph += 2.0 * x[i] * zi;
            for (int j = i + 1; j < q; j++)
                if (zi != ((s >> j) & 1))
                    ph += 2.0 * (PI - x[i]) * (PI - x[j]);
        }
        psi[s] *= cexp(I * ph);
    }
}

static void copy_big(cplx *dst, const cplx *src, long dim)
{
    #pragma omp parallel for schedule(static)
    for (long s = 0; s < dim; s++)
        dst[s] = src[s];
}

int main(int argc, char **argv)
{
    int q = 24, threads = omp_get_max_threads(), reps = 3, serial_init = 0;
    int c;
    while ((c = getopt(argc, argv, "q:t:r:s")) != -1) {
        switch (c) {
        case 'q': q = atoi(optarg); break;
        case 't': threads = atoi(optarg); break;
        case 'r': reps = atoi(optarg); break;
        case 's': serial_init = 1; break;
        default:
            fprintf(stderr, "usage: bigstate_omp -q qubits [-t threads] [-r reps] [-s]\n");
            return 1;
        }
    }
    if (q < 2 || q > 34 || threads < 1 || reps < 1)
        return 1;
    omp_set_num_threads(threads);

    long dim = 1L << q;
    cplx *psi = xmalloc(sizeof(cplx) * dim);
    cplx *tmp = xmalloc(sizeof(cplx) * dim);
    if (serial_init) {
        memset(psi, 0, sizeof(cplx) * dim);
        memset(tmp, 0, sizeof(cplx) * dim);
    } else {
        #pragma omp parallel for schedule(static)
        for (long s = 0; s < dim; s++)
            psi[s] = tmp[s] = 0.0;
    }
    psi[0] = 1.0;

    double x[64];
    srand(2);
    for (int i = 0; i < q; i++)
        x[i] = PI * rand() / ((double)RAND_MAX + 1);

    /* H on every qubit of |0...0> gives 2^(-q/2) everywhere: check a few */
    h_layer_big(psi, q);
    double expect = pow(2.0, -q / 2.0);
    for (long s = 0; s < dim; s += dim / 64)
        if (fabs(creal(psi[s]) - expect) > 1e-12 || fabs(cimag(psi[s])) > 1e-12) {
            fprintf(stderr, "wrong amplitude at %ld\n", s);
            return 1;
        }

    double *th = xmalloc(sizeof(double) * reps), *tz = xmalloc(sizeof(double) * reps),
           *tc = xmalloc(sizeof(double) * reps);
    for (int r = 0; r < reps; r++) {
        double t0 = now();
        h_layer_big(psi, q);
        double t1 = now();
        zz_layer_big(psi, q, x);
        double t2 = now();
        copy_big(tmp, psi, dim);
        th[r] = t1 - t0;
        tz[r] = t2 - t1;
        tc[r] = now() - t2;
    }

    /* every layer is unitary, so the norm must still be 1 */
    double norm = 0.0;
    #pragma omp parallel for reduction(+:norm) schedule(static)
    for (long s = 0; s < dim; s++)
        norm += creal(psi[s]) * creal(psi[s]) + cimag(psi[s]) * cimag(psi[s]);

    int team = 1;
    #pragma omp parallel
    #pragma omp single
    team = omp_get_num_threads();

    double bytes = (double)sizeof(cplx) * dim;
    double h = median(th, reps), z = median(tz, reps), cp = median(tc, reps);
    /* each gate reads and writes the array once: 2 * bytes per gate, q gates */
    printf("%d,%d,%s,%.6f,%.3f,%.6f,%.3f,%.2e\n", q, team, serial_init ? "serial" : "parallel",
           h, 2.0 * bytes * q / h / 1e9, z, 2.0 * bytes / cp / 1e9, fabs(norm - 1.0));
    fprintf(stderr, "%d qubits (%.2f GiB), %d threads: H layer %.3f s, ZZ layer %.3f s, norm error %.1e\n",
            q, bytes / (1 << 30), team, h, z, fabs(norm - 1.0));

    free(psi);
    free(tmp);
    free(th);
    free(tz);
    free(tc);
    return fabs(norm - 1.0) > 1e-9;
}
