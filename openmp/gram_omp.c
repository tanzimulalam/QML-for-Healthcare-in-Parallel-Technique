/*
 * gram_omp.c - version B of the kernel in C with OpenMP.
 *
 * Step 1: simulate every beat's state once. The beats do not depend on each
 *         other, so this is one parallel loop.
 * Step 2: K[i][j] = |<psi_i|psi_j>|^2 for i < j. That is a matrix product, and
 *         the work is tiny per entry (256 multiply-adds), so the question is how
 *         much memory traffic each entry costs. Three ways to do it:
 *
 *   naive   one dot product per pair, rows handed out dynamically. psi_i stays
 *           in cache but every psi_j is read again for every row.
 *   tiled   the triangle is cut into B x B tiles (collapse(2) over the tile
 *           grid). The 2B states of a tile stay in cache while all B*B dot
 *           products are done, so each state is read from memory ~N/B times
 *           instead of ~N times.
 *   tasks   the same tiles, but each one is an OpenMP task created by one
 *           thread and picked up by whichever thread is free.
 *   cross   test x train kernel for the SVM's predictions: the first N beats
 *           are "train", the next M beats (-x M) are "test". Rectangular, so a
 *           collapse(2) static loop is already balanced.
 *
 * The states are stored as separate real and imaginary arrays so that the dot
 * product is plain real arithmetic and `omp simd` can vectorize it.
 *
 * Usage:
 *     gram_omp -n N -t threads -m naive|tiled|tasks|cross [-b tile] [-x M]
 *              [-f features.bin] [-l lambda] [-r reps] [-o K.bin]
 *
 * Prints one CSV line:  N,M,threads,method,tile,states_s,gram_s,checksum
 * (times are medians over the repetitions; M is 0 except for cross)
 */
#include <getopt.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qcommon.h"

/* |<a|b>|^2 for two states stored as (re, im) arrays */
static inline double overlap2(const double *restrict ar, const double *restrict ai,
                              const double *restrict br, const double *restrict bi)
{
    double re = 0.0, im = 0.0;
    #pragma omp simd reduction(+:re, im)
    for (int s = 0; s < DIM; s++) {
        re += ar[s] * br[s] + ai[s] * bi[s];
        im += ar[s] * bi[s] - ai[s] * br[s];
    }
    return re * re + im * im;
}

static void all_states(const cplx *D, int n, double *re, double *im)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        cplx psi[DIM];
        encode_state(D + (size_t)i * DIM, psi);
        for (int s = 0; s < DIM; s++) {
            re[(size_t)i * DIM + s] = creal(psi[s]);
            im[(size_t)i * DIM + s] = cimag(psi[s]);
        }
    }
}

static void tile(int bi, int bj, int b, int n, const double *re, const double *im, double *K)
{
    int i_end = (bi + 1) * b < n ? (bi + 1) * b : n;
    int j_end = (bj + 1) * b < n ? (bj + 1) * b : n;
    for (int i = bi * b; i < i_end; i++) {
        int j0 = bj * b > i + 1 ? bj * b : i + 1;
        const double *ar = re + (size_t)i * DIM, *ai = im + (size_t)i * DIM;
        for (int j = j0; j < j_end; j++)
            K[(size_t)i * n + j] = overlap2(ar, ai, re + (size_t)j * DIM, im + (size_t)j * DIM);
    }
}

static void gram_naive(int n, const double *re, const double *im, double *K)
{
    #pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < n; i++) {
        const double *ar = re + (size_t)i * DIM, *ai = im + (size_t)i * DIM;
        for (int j = i + 1; j < n; j++)
            K[(size_t)i * n + j] = overlap2(ar, ai, re + (size_t)j * DIM, im + (size_t)j * DIM);
    }
}

static void gram_tiled(int n, int b, const double *re, const double *im, double *K)
{
    int nt = (n + b - 1) / b;
    /* the full nt x nt grid so collapse(2) can be used; tiles below the
       diagonal are skipped right away, dynamic takes care of the imbalance */
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int bi = 0; bi < nt; bi++)
        for (int bj = 0; bj < nt; bj++)
            if (bj >= bi)
                tile(bi, bj, b, n, re, im, K);
}

static void gram_tasks(int n, int b, const double *re, const double *im, double *K)
{
    int nt = (n + b - 1) / b;
    #pragma omp parallel
    #pragma omp single
    for (int bi = 0; bi < nt; bi++)
        for (int bj = bi; bj < nt; bj++) {
            #pragma omp task firstprivate(bi, bj)
            tile(bi, bj, b, n, re, im, K);
        }
}

/* Kx[a][j] = |<test_a|train_j>|^2, test states start at row n of re/im */
static void gram_cross(int n, int m, const double *re, const double *im, double *Kx)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int a = 0; a < m; a++)
        for (int j = 0; j < n; j++)
            Kx[(size_t)a * n + j] = overlap2(re + (size_t)(n + a) * DIM, im + (size_t)(n + a) * DIM,
                                            re + (size_t)j * DIM, im + (size_t)j * DIM);
}

static void usage(void)
{
    fprintf(stderr,
            "usage: gram_omp -n N -t threads -m naive|tiled|tasks|cross [-b tile] [-x M]\n"
            "                [-f features.bin] [-l lambda] [-r reps] [-o K.bin]\n");
    exit(1);
}

int main(int argc, char **argv)
{
    int n = 2000, m = 0, threads = omp_get_max_threads(), b = 64, reps = 3;
    const char *method = "tiled", *feat = NULL, *out = NULL;
    double lam = 0.1;

    int c;
    while ((c = getopt(argc, argv, "n:x:t:m:b:f:l:r:o:")) != -1) {
        switch (c) {
        case 'n': n = atoi(optarg); break;
        case 'x': m = atoi(optarg); break;
        case 't': threads = atoi(optarg); break;
        case 'm': method = optarg; break;
        case 'b': b = atoi(optarg); break;
        case 'f': feat = optarg; break;
        case 'l': lam = atof(optarg); break;
        case 'r': reps = atoi(optarg); break;
        case 'o': out = optarg; break;
        default: usage();
        }
    }
    int cross = !strcmp(method, "cross");
    if (n < 2 || threads < 1 || reps < 1 || b < 1 || (cross && m < 1))
        usage();
    if (strcmp(method, "naive") && strcmp(method, "tiled") && strcmp(method, "tasks") && !cross)
        usage();
    if (!cross)
        m = 0;
    omp_set_num_threads(threads);

    int total = n + m;                       /* train beats, then test beats */
    double *X = load_inputs(feat, total, lam);
    cplx *D = phase_table(X, total);
    double *re = xmalloc(sizeof(double) * DIM * total);
    double *im = xmalloc(sizeof(double) * DIM * total);
    size_t kcount = cross ? (size_t)m * n : (size_t)n * n;
    double *K = xmalloc(sizeof(double) * kcount);
    memset(K, 0, sizeof(double) * kcount);

    double *ts = xmalloc(sizeof(double) * reps), *tg = xmalloc(sizeof(double) * reps);
    for (int r = 0; r < reps; r++) {
        double t0 = now();
        all_states(D, total, re, im);
        double t1 = now();
        if (cross)
            gram_cross(n, m, re, im, K);
        else if (!strcmp(method, "naive"))
            gram_naive(n, re, im, K);
        else if (!strcmp(method, "tiled"))
            gram_tiled(n, b, re, im, K);
        else
            gram_tasks(n, b, re, im, K);
        ts[r] = t1 - t0;
        tg[r] = now() - t1;
    }

    double checksum = 0.0;
    #pragma omp parallel for reduction(+:checksum)
    for (long k = 0; k < (long)kcount; k++)
        checksum += K[k];

    int team = 1;
    #pragma omp parallel
    #pragma omp single
    team = omp_get_num_threads();

    double s_med = median(ts, reps), g_med = median(tg, reps);
    printf("%d,%d,%d,%s,%d,%.6f,%.6f,%.6f\n", n, m, team, method, b, s_med, g_med, checksum);
    fprintf(stderr, "N=%d%s, %d threads, %s: states %.4f s, kernel %.4f s\n",
            n, cross ? " (cross)" : "", team, method, s_med, g_med);

    int rc = out ? write_doubles(out, K, kcount) : 0;
    free(X);
    free(D);
    free(re);
    free(im);
    free(K);
    free(ts);
    free(tg);
    return rc;
}
