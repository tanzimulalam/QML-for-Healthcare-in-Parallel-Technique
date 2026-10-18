#!/bin/sh
# Version B in C: naive vs tiled vs tasks on 1..24 threads, plus a few tile sizes
# and the test x train kernel. Run from the openmp/ folder after "make":
#
#     sh run_gram.sh
#     N=8000 THREADS="1 4 8" sh run_gram.sh
#
# Writes ../results/openmp/gram.csv

N=${N:-8000}
THREADS=${THREADS:-"1 2 4 8 12 16 24"}
TILES=${TILES:-"16 32 64 128 256"}
FEAT=${FEAT:-../data/u_train.bin}
OUT=../results/openmp
mkdir -p "$OUT"

INPUT=""
[ -f "$FEAT" ] && INPUT="-f $FEAT"

CSV="$OUT/gram.csv"
echo "N,M,threads,method,tile,states_s,gram_s,checksum" > "$CSV"

for p in $THREADS; do
    for m in naive tiled tasks; do
        ./gram_omp -n "$N" -t "$p" -m "$m" -b 64 $INPUT -r 3 >> "$CSV" || exit 1
    done
done

# tile size, all threads
PMAX=$(echo $THREADS | awk '{print $NF}')
for b in $TILES; do
    ./gram_omp -n "$N" -t "$PMAX" -m tiled -b "$b" $INPUT -r 3 >> "$CSV" || exit 1
done

# test x train: N train beats against 8000 test beats
for p in $THREADS; do
    ./gram_omp -n "$N" -x 8000 -t "$p" -m cross $INPUT -r 3 >> "$CSV" || exit 1
done
echo "wrote $CSV" >&2
