/*
 * qkernel_omp.c - the quantum kernel matrix (version A) in C with OpenMP.
 *
 * Same circuit as pair_kernel() in src/qsim.py: 8 qubits, ZZ feature map with
 * 2 repetitions, and for every pair i < j
 *
 *     K[i][j] = |<0| U(x_j)^+ U(x_i) |0>|^2
 *
 * computed by simulating the circuit on a 256-entry state vector. Each thread
 * keeps its own state vector on its stack, and the pairs are independent, so the
 * only real question is how the rows get divided among the threads. Row i has
 * N-1-i entries, which is why the schedule matters:
 *
 *     static   equal contiguous blocks of rows (OpenMP default)
 *     cyclic   static with chunk 1, rows dealt out like cards
 *     dynamic  a free thread grabs the next chunk of rows
 *     guided   like dynamic but the chunks get smaller towards the end
 *     folded   row t and row N-1-t together, so every task has N-1 entries
 *
 * Usage:
 *     qkernel_omp -n N -t threads -s schedule [-c chunk] [-f features.bin]
 *                 [-l lambda] [-r reps] [-o K.bin]
 *
 * features.bin is what src/features.py writes (N x 8 doubles in [0, 1]); the
 * angles are lambda * pi * u. Without -f the inputs are random, which is fine
 * for timing because the work per pair does not depend on the values.
 *
 * Prints one CSV line to stdout:
 *     N,threads,schedule,chunk,seconds,busy_min,busy_max,checksum
 * seconds is the median over the repetitions. busy_min/busy_max are the shortest
 * and longest time any thread spent on its share in that same repetition, so
 * busy_max / busy_min shows the load imbalance directly.
 */
#include <complex.h>
#include <getopt.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NQ   8
#define DIM  (1 << NQ)
#define REPS 2

static const double PI = 3.14159265358979323846;

typedef double complex cplx;

/* one repetition of the ZZ part is diagonal: a phase per basis state */
static void zz_phases(const double *x, cplx *d)
{
    for (int s = 0; s < DIM; s++) {
        double ph = 0.0;
        for (int i = 0; i < NQ; i++) {
            int zi = (s >> i) & 1;
            ph += 2.0 * x[i] * zi;
            for (int j = i + 1; j < NQ; j++)
                if (zi != ((s >> j) & 1))
                    ph += 2.0 * (PI - x[i]) * (PI - x[j]);
        }
        d[s] = cexp(I * ph);
    }
}

/* Hadamard on every qubit */
static void h_layer(cplx *psi)
{
    const double s = 0.70710678118654752440;
    for (int stride = 1; stride < DIM; stride *= 2)
        for (int base = 0; base < DIM; base += 2 * stride)
            for (int k = base; k < base + stride; k++) {
                cplx a = psi[k], b = psi[k + stride];
                psi[k] = (a + b) * s;
                psi[k + stride] = (a - b) * s;
            }
}

static double pair_kernel(const cplx *di, const cplx *dj)
{
    cplx psi[DIM] = {0};
    psi[0] = 1.0;
    for (int r = 0; r < REPS; r++) {        /* U(x_i) */
        h_layer(psi);
        for (int s = 0; s < DIM; s++)
            psi[s] *= di[s];
    }
    for (int r = 0; r < REPS; r++) {        /* U(x_j)^+ */
        for (int s = 0; s < DIM; s++)
            psi[s] *= conj(dj[s]);
        h_layer(psi);
    }
    return creal(psi[0]) * creal(psi[0]) + cimag(psi[0]) * cimag(psi[0]);
}

static void row(int i, int n, const cplx *D, double *K)
{
    for (int j = i + 1; j < n; j++)
        K[(size_t)i * n + j] = pair_kernel(D + (size_t)i * DIM, D + (size_t)j * DIM);
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void usage(void)
{
    fprintf(stderr,
            "usage: qkernel_omp -n N -t threads -s static|cyclic|dynamic|guided|folded\n"
            "                   [-c chunk] [-f features.bin] [-l lambda] [-r reps] [-o K.bin]\n");
    exit(1);
}

int main(int argc, char **argv)
{
    int n = 1000, threads = omp_get_max_threads(), chunk = 0, reps = 3;
    const char *sched = "static", *feat = NULL, *out = NULL;
    double lam = 0.1;

    int c;
    while ((c = getopt(argc, argv, "n:t:s:c:f:l:r:o:")) != -1) {
        switch (c) {
        case 'n': n = atoi(optarg); break;
        case 't': threads = atoi(optarg); break;
        case 's': sched = optarg; break;
        case 'c': chunk = atoi(optarg); break;
        case 'f': feat = optarg; break;
        case 'l': lam = atof(optarg); break;
        case 'r': reps = atoi(optarg); break;
        case 'o': out = optarg; break;
        default: usage();
        }
    }
    if (n < 2 || threads < 1 || reps < 1)
        usage();

    int folded = 0;
    omp_sched_t kind = omp_sched_static;
    if (!strcmp(sched, "static"))
        kind = omp_sched_static;            /* chunk 0 = one block per thread */
    else if (!strcmp(sched, "cyclic"))
        chunk = 1;
    else if (!strcmp(sched, "dynamic"))
        kind = omp_sched_dynamic;
    else if (!strcmp(sched, "guided"))
        kind = omp_sched_guided;
    else if (!strcmp(sched, "folded")) {
        folded = 1;
        chunk = 0;                          /* not used, every task is the same size */
    }
    else
        usage();
    omp_set_schedule(kind, chunk);
    omp_set_num_threads(threads);

    double *X = malloc(sizeof(double) * n * NQ);
    if (feat) {
        FILE *f = fopen(feat, "rb");
        if (!f) {
            perror(feat);
            return 1;
        }
        size_t got = fread(X, sizeof(double) * NQ, n, f);
        fclose(f);
        if (got < (size_t)n) {
            fprintf(stderr, "%s has fewer than %d samples\n", feat, n);
            return 1;
        }
        for (long k = 0; k < (long)n * NQ; k++)
            X[k] *= lam * PI;
    } else {
        srand(1);
        for (long k = 0; k < (long)n * NQ; k++)
            X[k] = PI * rand() / ((double)RAND_MAX + 1);
    }

    cplx *D = malloc(sizeof(cplx) * DIM * n);
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        zz_phases(X + (size_t)i * NQ, D + (size_t)i * DIM);

    double *K = malloc(sizeof(double) * n * n);
    memset(K, 0, sizeof(double) * n * n);   /* touch the pages before timing */
    double *busy = calloc((size_t)reps * threads, sizeof(double));   /* per repetition, per thread */
    double *times = malloc(sizeof(double) * reps);
    int team = 1;

    for (int r = 0; r < reps; r++) {
        double t0 = omp_get_wtime();
        #pragma omp parallel
        {
            double start = omp_get_wtime();
            if (folded) {
                #pragma omp for schedule(static) nowait
                for (int t = 0; t < (n + 1) / 2; t++) {
                    row(t, n, D, K);
                    if (n - 1 - t != t)
                        row(n - 1 - t, n, D, K);
                }
            } else {
                #pragma omp for schedule(runtime) nowait
                for (int i = 0; i < n; i++)
                    row(i, n, D, K);
            }
            busy[(size_t)r * threads + omp_get_thread_num()] = omp_get_wtime() - start;
            #pragma omp single nowait
            team = omp_get_num_threads();
        }
        times[r] = omp_get_wtime() - t0;
    }

    /* busy times are taken from the repetition that gave the median time */
    double *sorted = malloc(sizeof(double) * reps);
    memcpy(sorted, times, sizeof(double) * reps);
    qsort(sorted, reps, sizeof(double), cmp_double);
    double median = sorted[reps / 2];
    int m = 0;
    while (times[m] != median)
        m++;
    const double *b = busy + (size_t)m * threads;
    double bmin = b[0], bmax = b[0], checksum = 0.0;
    for (int t = 1; t < team; t++) {
        bmin = fmin(bmin, b[t]);
        bmax = fmax(bmax, b[t]);
    }
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            checksum += K[(size_t)i * n + j];

    printf("%d,%d,%s,%d,%.6f,%.6f,%.6f,%.10f\n", n, team, sched, chunk,
           median, bmin, bmax, checksum);
    fprintf(stderr, "N=%d, %d threads, %s: %.3f s (threads busy %.3f to %.3f s)\n",
            n, team, sched, median, bmin, bmax);

    if (out) {
        FILE *f = fopen(out, "wb");
        if (!f) {
            perror(out);
            return 1;
        }
        fwrite(K, sizeof(double), (size_t)n * n, f);
        fclose(f);
    }
    free(X);
    free(D);
    free(K);
    free(busy);
    free(times);
    free(sorted);
    return 0;
}
