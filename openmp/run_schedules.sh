#!/bin/sh
# Time the OpenMP kernel with every schedule on 1..24 threads.
# Run from the openmp/ folder after "make":
#
#     sh run_schedules.sh                   # N = 2000 real beats
#     N=4000 THREADS="1 2 4 8" sh run_schedules.sh
#
# Writes ../results/openmp/schedules.csv; plot it with python ../src/openmp_figure.py

N=${N:-2000}
THREADS=${THREADS:-"1 2 4 6 8 12 16 20 24"}
FEAT=${FEAT:-../data/u_train.bin}
OUT=../results/openmp
mkdir -p "$OUT"

if [ -f "$FEAT" ]; then
    INPUT="-f $FEAT"
else
    echo "no $FEAT, using random inputs" >&2
    INPUT=""
fi

echo "N,threads,schedule,chunk,seconds,busy_min,busy_max,checksum" > "$OUT/schedules.csv"
for p in $THREADS; do
    for s in static cyclic folded dynamic guided taskloop; do
        chunk=0
        [ "$s" = dynamic ] || [ "$s" = taskloop ] && chunk=4
        ./qkernel_omp -n "$N" -t "$p" -s "$s" -c "$chunk" $INPUT -r 3 >> "$OUT/schedules.csv" || exit 1
    done
done
echo "wrote $OUT/schedules.csv" >&2
