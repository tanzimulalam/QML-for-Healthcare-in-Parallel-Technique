#!/bin/sh
# Every step once, with small sizes, to check that the whole thing works
# (a few minutes). Output goes to results/quick_test/, so the real results
# in results/ are not touched. Run from the repository root:
#
#     sh scripts/quick_test.sh
#
# On my Windows laptop (Git Bash, venv one folder up, w64devkit for gcc/make,
# Intel MPI from pip):
#     export PATH="$HOME/.local/w64devkit/bin:$PATH"
#     export I_MPI_ROOT="$(cygpath -w ../.venv/Library)"
#     PYTHON=../.venv/Scripts/python.exe MPIRUN="../.venv/Library/bin/mpiexec.exe -localonly" sh scripts/quick_test.sh
#
# Steps that need something missing (gcc, MPI, a GPU) are skipped with a message.

set -e
PYTHON=${PYTHON:-python}
MPIRUN=${MPIRUN:-mpirun}
export QML_RESULTS=results/quick_test
rm -rf "$QML_RESULTS"
START=$(date +%s)

step() {
    echo
    echo "=== $* ==="
}

step "1. data"
"$PYTHON" scripts/get_data.py
"$PYTHON" src/prep_beats.py --procs 2 4 --reps 1
"$PYTHON" src/features.py

step "2. classical baselines"
"$PYTHON" src/baselines.py --sizes 1000 --seeds 2 --trees 50 --cores 1 4

step "3. simulator checks"
"$PYTHON" src/check_qsim.py

step "4. kernel with threads (Numba)"
"$PYTHON" src/kernel_threads.py --n 400 --threads 1 4 --reps 1
"$PYTHON" src/kernel_compare.py --n 400 --big 1500 --reps 1
"$PYTHON" src/big_state.py --qubits 20 --threads 1 4 --nmin 12 --nmax 20 --reps 1

step "5. quantum SVM"
"$PYTHON" src/qsvm.py --sizes 500 1000 --seeds 2 --tune-size 600

step "6. OpenMP (C)"
if command -v make >/dev/null 2>&1 && command -v gcc >/dev/null 2>&1; then
    (cd openmp && make)
    "$PYTHON" src/check_openmp.py
    (cd openmp && N=400 THREADS="1 4" sh run_schedules.sh)
    (cd openmp && N=1000 THREADS="1 4" TILES="32 64" sh run_gram.sh)
    (cd openmp && Q=20 THREADS="1 4" sh run_bigstate.sh)
    "$PYTHON" src/openmp_figure.py
else
    echo "no gcc/make on the PATH, skipping"
fi

step "7. MPI"
if "$PYTHON" -c "import mpi4py" >/dev/null 2>&1 && command -v ${MPIRUN%% *} >/dev/null 2>&1; then
    N=300 RANKS="2 4" PYTHON="$PYTHON" MPIRUN="$MPIRUN" sh scripts/run_mpi.sh
    $MPIRUN -n 4 "$PYTHON" src/kernel_mpi.py --n 200 --mode dynamic --check
else
    echo "mpi4py or $MPIRUN not found, skipping"
fi

echo
echo "all steps finished in $(( $(date +%s) - START )) s; output is in $QML_RESULTS/"
