/*
 * qkernel_omp.c - the quantum kernel matrix (version A) in C with OpenMP.
 *
 * For every pair i < j the circuit U(x_j)^+ U(x_i)|0> is simulated on a
 * 256-entry state vector and K[i][j] = |<0|...|0>|^2 (same as pair_kernel() in
 * src/qsim.py). Each thread keeps its own state vector on its stack and the pairs
 * are independent, so the only real question is how the rows get divided among
 * the threads. Row i has N-1-i entries, which is why the schedule matters:
 *
 *     static    equal contiguous blocks of rows (OpenMP default)
 *     cyclic    static with chunk 1, rows dealt out like cards
 *     dynamic   a free thread grabs the next chunk of rows
 *     guided    like dynamic but the chunks get smaller towards the end
 *     folded    row t and row N-1-t together, so every task has N-1 entries
 *     taskloop  the rows become OpenMP tasks (grainsize = chunk) that idle
 *               threads pick up
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
 * seconds is the median over the repetitions. busy_min/busy_max are the least
 * and the most time any thread spent computing rows in that same repetition,
 * so 1 - busy_min/busy_max is how long the least loaded thread sat idle.
 */
#include <getopt.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qcommon.h"

enum { LOOP, FOLDED, TASKLOOP };

static void usage(void)
{
    fprintf(stderr,
            "usage: qkernel_omp -n N -t threads -s static|cyclic|dynamic|guided|folded|taskloop\n"
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

    int how = LOOP;
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
        how = FOLDED;
        chunk = 0;                          /* not used, every task is the same size */
    } else if (!strcmp(sched, "taskloop")) {
        how = TASKLOOP;
        if (chunk < 1)
            chunk = 4;
    } else
        usage();
    omp_set_schedule(kind, chunk);
    omp_set_num_threads(threads);

    double *X = load_inputs(feat, n, lam);
    cplx *D = phase_table(X, n);
    double *K = xmalloc(sizeof(double) * n * n);
    memset(K, 0, sizeof(double) * n * n);   /* touch the pages before timing */

    double *work = xmalloc(sizeof(double) * reps * threads);   /* per repetition, per thread */
    double *times = xmalloc(sizeof(double) * reps);
    int team = 1;

    for (int r = 0; r < reps; r++) {
        double *w = work + (size_t)r * threads;
        for (int t = 0; t < threads; t++)
            w[t] = 0.0;
        double t0 = now();

        if (how == TASKLOOP) {
            #pragma omp parallel
            {
                #pragma omp single
                {
                    team = omp_get_num_threads();
                    /* whichever thread runs a task adds the time to its own slot */
                    #pragma omp taskloop grainsize(chunk)
                    for (int i = 0; i < n; i++) {
                        double s = now();
                        kernel_row(i, n, D, K);
                        w[omp_get_thread_num()] += now() - s;
                    }
                }
            }
        } else {
            #pragma omp parallel
            {
                double mine = 0.0;
                if (how == FOLDED) {
                    #pragma omp for schedule(static) nowait
                    for (int t = 0; t < (n + 1) / 2; t++) {
                        double s = now();
                        kernel_row(t, n, D, K);
                        if (n - 1 - t != t)
                            kernel_row(n - 1 - t, n, D, K);
                        mine += now() - s;
                    }
                } else {
                    #pragma omp for schedule(runtime) nowait
                    for (int i = 0; i < n; i++) {
                        double s = now();
                        kernel_row(i, n, D, K);
                        mine += now() - s;
                    }
                }
                w[omp_get_thread_num()] = mine;
                #pragma omp single nowait
                team = omp_get_num_threads();
            }
        }
        times[r] = now() - t0;
    }

    /* report the thread loads of the repetition that gave the median time */
    double med = median(times, reps);
    int m = 0;
    while (times[m] != med)
        m++;
    const double *w = work + (size_t)m * threads;
    double bmin = w[0], bmax = w[0];
    for (int t = 1; t < team; t++) {
        bmin = fmin(bmin, w[t]);
        bmax = fmax(bmax, w[t]);
    }

    /* the order of the additions depends on the thread count, so only the
       first ~10 digits of the checksum are comparable between runs */
    double checksum = 0.0;
    #pragma omp parallel for reduction(+:checksum) schedule(static)
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            checksum += K[(size_t)i * n + j];

    printf("%d,%d,%s,%d,%.6f,%.6f,%.6f,%.6f\n", n, team, sched, chunk, med, bmin, bmax, checksum);
    fprintf(stderr, "N=%d, %d threads, %s: %.3f s (threads busy %.3f to %.3f s)\n",
            n, team, sched, med, bmin, bmax);

    int rc = out ? write_doubles(out, K, (size_t)n * n) : 0;
    free(X);
    free(D);
    free(K);
    free(work);
    free(times);
    return rc;
}
