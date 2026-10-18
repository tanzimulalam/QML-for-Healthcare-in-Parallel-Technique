# gcc on Linux / the cluster, or w64devkit's gcc on Windows.
# Use "make CC=clang" etc. to change the compiler.
# -fcx-limited-range: plain complex multiply, without the inf/nan checks of C99 Annex G
CC = gcc
CFLAGS = -O3 -march=native -fopenmp -fcx-limited-range -std=gnu11 -Wall -Wextra
LDLIBS = -lm

PROGS = qkernel_omp gram_omp bigstate_omp

all: $(PROGS)

qkernel_omp: qkernel_omp.c qcommon.c qcommon.h
	$(CC) $(CFLAGS) -o $@ qkernel_omp.c qcommon.c $(LDLIBS)

gram_omp: gram_omp.c qcommon.c qcommon.h
	$(CC) $(CFLAGS) -o $@ gram_omp.c qcommon.c $(LDLIBS)

bigstate_omp: bigstate_omp.c qcommon.c qcommon.h
	$(CC) $(CFLAGS) -o $@ bigstate_omp.c qcommon.c $(LDLIBS)

clean:
	rm -f $(PROGS) $(addsuffix .exe,$(PROGS))

.PHONY: all clean
