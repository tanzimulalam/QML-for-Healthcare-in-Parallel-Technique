# gcc on Linux / the cluster, or w64devkit's gcc on Windows.
# Use "make CC=clang" etc. to change the compiler.
# -fcx-limited-range: plain complex multiply, without the inf/nan checks of C99 Annex G
CC = gcc
CFLAGS = -O3 -march=native -fopenmp -fcx-limited-range -std=gnu11 -Wall -Wextra
LDLIBS = -lm

qkernel_omp: qkernel_omp.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f qkernel_omp qkernel_omp.exe

.PHONY: clean
