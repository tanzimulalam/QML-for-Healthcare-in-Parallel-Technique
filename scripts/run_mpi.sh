#!/bin/sh
# MPI sweep: the serial time first, then block / cyclic / dynamic on more and
# more ranks. Run from the repository root.
#
#   Linux / cluster:
#       sh scripts/run_mpi.sh
#       N=3000 RANKS="2 4 8 16 32" MPIRUN="mpirun --oversubscribe" sh scripts/run_mpi.sh
#   Windows, Git Bash, Intel MPI from pip (venv one folder up, as on my laptop):
#       export I_MPI_ROOT="$(cygpath -w ../.venv/Library)"
#       MPIRUN="../.venv/Library/bin/mpiexec.exe -localonly" PYTHON=../.venv/Scripts/python.exe sh scripts/run_mpi.sh
#
# Writes results/mpi/runs.csv and results/mpi/mpi.png

N=${N:-2000}
RANKS=${RANKS:-"2 4 8 12 16 24"}
PYTHON=${PYTHON:-python}
MPIRUN=${MPIRUN:-mpirun}
EXTRA=${EXTRA:-}            # e.g. EXTRA=--real for the real ECG features

rm -f results/mpi/runs.csv
"$PYTHON" src/kernel_mpi.py --n "$N" --mode serial $EXTRA || exit 1
for p in $RANKS; do
    for m in block cyclic dynamic; do
        $MPIRUN -n "$p" "$PYTHON" src/kernel_mpi.py --n "$N" --mode "$m" $EXTRA || exit 1
    done
done
"$PYTHON" src/mpi_figure.py
