#!/bin/sh
# One big state on 1..24 threads: memory-bound H layer vs compute-bound ZZ
# layer, with parallel and serial first touch. Run from openmp/ after "make":
#
#     sh run_bigstate.sh                    # 26 qubits = 1 GiB per array, 2 arrays
#     Q=24 sh run_bigstate.sh               # smaller and quicker
#
# Writes ../results/openmp/bigstate.csv

Q=${Q:-26}
THREADS=${THREADS:-"1 2 4 6 8 12 16 20 24"}
OUT=../${QML_RESULTS:-results}/openmp
mkdir -p "$OUT"

CSV="$OUT/bigstate.csv"
echo "qubits,threads,init,h_s,h_GBps,zz_s,copy_GBps,norm_err" > "$CSV"
for p in $THREADS; do
    ./bigstate_omp -q "$Q" -t "$p" -r 3 >> "$CSV" || exit 1
    ./bigstate_omp -q "$Q" -t "$p" -r 3 -s >> "$CSV" || exit 1
done
echo "wrote $CSV" >&2
