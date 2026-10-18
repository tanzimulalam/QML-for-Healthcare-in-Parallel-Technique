#include "qcommon.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const double PI = 3.14159265358979323846;

#ifdef _WIN32
#include <windows.h>
double now(void)
{
    LARGE_INTEGER t, f;
    QueryPerformanceCounter(&t);
    QueryPerformanceFrequency(&f);
    return (double)t.QuadPart / (double)f.QuadPart;
}
#else
#include <time.h>
double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
#endif

void *xmalloc(size_t bytes)
{
    void *p = malloc(bytes);
    if (!p) {
        fprintf(stderr, "could not allocate %.1f MB\n", bytes / 1e6);
        exit(1);
    }
    return p;
}

double *load_inputs(const char *path, int n, double lam)
{
    double *X = xmalloc(sizeof(double) * n * NQ);
    if (path) {
        FILE *f = fopen(path, "rb");
        if (!f) {
            perror(path);
            exit(1);
        }
        size_t got = fread(X, sizeof(double) * NQ, n, f);
        fclose(f);
        if (got < (size_t)n) {
            fprintf(stderr, "%s has fewer than %d samples\n", path, n);
            exit(1);
        }
        for (long k = 0; k < (long)n * NQ; k++)
            X[k] *= lam * PI;
    } else {
        srand(1);
        for (long k = 0; k < (long)n * NQ; k++)
            X[k] = PI * rand() / ((double)RAND_MAX + 1);
    }
    return X;
}

void zz_phases(const double *x, cplx *d)
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

cplx *phase_table(const double *X, int n)
{
    cplx *D = xmalloc(sizeof(cplx) * DIM * n);
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        zz_phases(X + (size_t)i * NQ, D + (size_t)i * DIM);
    return D;
}

void h_layer(cplx *psi)
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

void encode_state(const cplx *d, cplx *psi)
{
    memset(psi, 0, sizeof(cplx) * DIM);
    psi[0] = 1.0;
    for (int r = 0; r < REPS; r++) {
        h_layer(psi);
        for (int s = 0; s < DIM; s++)
            psi[s] *= d[s];
    }
}

double pair_kernel(const cplx *di, const cplx *dj)
{
    cplx psi[DIM];
    encode_state(di, psi);                  /* U(x_i)|0> */
    for (int r = 0; r < REPS; r++) {        /* then U(x_j)^+ */
        for (int s = 0; s < DIM; s++)
            psi[s] *= conj(dj[s]);
        h_layer(psi);
    }
    return creal(psi[0]) * creal(psi[0]) + cimag(psi[0]) * cimag(psi[0]);
}

void kernel_row(int i, int n, const cplx *D, double *K)
{
    for (int j = i + 1; j < n; j++)
        K[(size_t)i * n + j] = pair_kernel(D + (size_t)i * DIM, D + (size_t)j * DIM);
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

double median(const double *v, int n)
{
    double *c = xmalloc(sizeof(double) * n);
    memcpy(c, v, sizeof(double) * n);
    qsort(c, n, sizeof(double), cmp_double);
    double m = c[n / 2];
    free(c);
    return m;
}

int write_doubles(const char *path, const double *v, size_t count)
{
    FILE *f = fopen(path, "wb");
    if (!f) {
        perror(path);
        return 1;
    }
    fwrite(v, sizeof(double), count, f);
    fclose(f);
    return 0;
}
