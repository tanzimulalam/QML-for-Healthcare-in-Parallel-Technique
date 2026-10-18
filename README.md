# The Computational Cost of Healthcare Quantum Machine Learning

Md Tanzimul Alam Fahim. Course project for CSCI 6330, Middle Tennessee State University, Fall 2026.
Code: https://github.com/tanzimulalam/QML-for-Healthcare-in-Parallel-Technique

I train a quantum kernel SVM to pick out ventricular heartbeats in the MIT-BIH Arrhythmia Database and measure what it costs to run. No quantum computer is involved. The circuits are simulated with a small state-vector simulator I wrote for this project, and most of the work is in making that simulation run in parallel. The same features also go into an ordinary SVM and a random forest, so the quantum model's cost can be compared with something.

The project is not finished yet (see "Still to do" at the end), but every step below runs, and each parallel version is checked against the serial one before any timing is trusted.

## The parallel parts at a glance

The quantum kernel matrix K[i, j] = |⟨ψ(x_i)|ψ(x_j)⟩|² can be computed in two ways. **Version A** simulates one circuit per pair of beats, the way real hardware would. **Version B** simulates every beat's state once and gets the whole matrix from one complex matrix product. Both give the same numbers. Version A is a triangle of independent jobs with very uneven rows, version B is a matrix product, and a single big state is memory-bound, so each one is a different kind of parallel problem.

| What | Where | Parallel model |
|------|-------|----------------|
| Cutting beats out of 44 ECG records | `src/prep_beats.py` | process pool (multiprocessing) |
| Random forest training | `src/baselines.py` | scikit-learn threads, 1 to 24 cores |
| Version A, schedules | `src/kernel_threads.py` | Numba `prange` + my own work queue |
| Version A, schedules | `openmp/qkernel_omp.c` | OpenMP: static, cyclic, dynamic, guided, folded, taskloop |
| Version A, distributed | `src/kernel_mpi.py` | MPI (mpi4py): manager-worker, block, cyclic |
| Version A on the GPU | `src/qsim.py` (`pair_circuits`) | CUDA, one thread block per pair, shared memory |
| Version B | `src/qsim.py` (`kernel_gemm`) | Numba + multithreaded BLAS |
| Version B | `openmp/gram_omp.c` | OpenMP: cache tiling with `collapse(2)`, `simd` reductions, tasks |
| Version B on the GPU | `src/qsim.py` (`kernel_gpu`) | CUDA kernels + cuBLAS, double and single precision |
| One big state (up to 30 qubits) | `src/big_state.py`, `openmp/bigstate_omp.c` | Numba / OpenMP / CUDA, memory-bound vs compute-bound |
| Quantum SVM on the ECG features | `src/qsvm.py` | the kernels above, test kernel built in blocks |

## What's in here

```
scripts/get_data.py       download MIT-BIH from PhysioNet into data/
scripts/quick_test.sh     run every step with small sizes (~2 minutes)
scripts/run_mpi.sh        MPI timing sweep
src/config.py             paths, record lists, label mapping
src/plots.py              shared figure style
src/prep_beats.py         filter the ECG and cut out every beat
src/features.py           PCA to 8 features, fitted on the training patients only
src/baselines.py          RBF SVM and random forest, V vs non-V
src/qsim.py               the simulator: feature map, versions A and B, CUDA kernels
src/check_qsim.py         simulator vs dense matrices, Qiskit and the GPU
src/kernel_threads.py     version A on 1..24 threads, four schedules
src/kernel_compare.py     A vs B vs GPU, and where B spends its time
src/big_state.py          one large state vector, CPU vs GPU
src/kernel_mpi.py         version A with MPI
src/mpi_figure.py         figure for the MPI sweep
src/qsvm.py               the quantum kernel SVM on the real features
src/check_openmp.py       all OpenMP programs vs the Python simulator
src/openmp_figure.py      figures for the OpenMP sweeps
openmp/qcommon.c, .h      simulator code shared by the C programs
openmp/qkernel_omp.c      version A in C, six schedules
openmp/gram_omp.c         version B in C: naive, tiled, tasks, test x train
openmp/bigstate_omp.c     one big state in C
openmp/Makefile
openmp/run_*.sh           OpenMP timing sweeps
results/week1/            data, features and baselines (week 1)
results/pilot/            first simulator runs, with random inputs
```

## Setup

You need Python 3.10 or newer (I used 3.12), and for the C part a compiler with OpenMP.

```
python -m venv .venv
.venv\Scripts\activate             # Linux: source .venv/bin/activate
pip install -r requirements.txt
```

- **C/OpenMP:** gcc on Linux and on the cluster. On Windows I used [w64devkit](https://github.com/skeeto/w64devkit), which needs no admin rights: unzip it and put its `bin` folder on the PATH.
- **MPI:** `pip install mpi4py` on a machine that has MPI (on the cluster, load its MPI module first). On Windows, `pip install mpi4py impi-rt` installs Intel MPI into the venv; set `I_MPI_ROOT` to the venv's `Library` folder and use the `mpiexec.exe` in `Library\bin`. Windows Firewall may ask whether `hydra_pmi_proxy` can accept connections. Clicking Cancel is fine, because everything runs on one machine.
- **GPU (optional):** an NVIDIA GPU and driver. `cupy-cuda12x[ctk]` brings the CUDA libraries along, and the CUDA kernels are compiled at run time, so no CUDA toolkit install is needed. Without a GPU, the GPU parts are skipped.
- **Qiskit** is only used by `check_qsim.py`, to compare my simulator with it.
- **Long folder paths on Windows:** Numba keeps compiled code in `src/__pycache__`. If the folder is buried very deep, those file names can go past Windows' 260-character path limit and Numba fails with a `FileNotFoundError` on an `.nbc` file. Move the folder closer to `C:\`, or set `NUMBA_CACHE_DIR` to a short folder.

## Quick test

To see that everything works without waiting for the real experiments, run this from the repository root:

```
sh scripts/quick_test.sh
```

It runs every step below with small sizes, writes to `results/quick_test/` (the real results are not touched), and skips anything that needs a missing tool. It takes about two minutes on my laptop. On Windows, run it in Git Bash and tell it where things are:

```
export PATH="$HOME/.local/w64devkit/bin:$PATH"
export I_MPI_ROOT="$(cygpath -w .venv/Library)"
PYTHON=.venv/Scripts/python.exe MPIRUN=".venv/Library/bin/mpiexec.exe -localonly" sh scripts/quick_test.sh
```

## Running each step

All commands run from the repository root. Each script writes its CSV files and figures to `results/<step>/`. The defaults are the full experiments. The "quick" line under each step gives a small version that finishes in seconds. Scripts that take options list them with `--help`.

### Step 0. Get the data

```
python scripts/get_data.py
```

Downloads the MIT-BIH Arrhythmia Database (about 73 MB) from PhysioNet and unpacks it into `data/`. The data is not part of the repository.

### Step 1. Cut out the heartbeats

```
python src/prep_beats.py
quick: python src/prep_beats.py --procs 2 4 --reps 1
```

Every record is filtered (median filters for baseline wander, 40 Hz low-pass), and a 0.5 s window is cut around every annotated beat. Labels are mapped to the AAMI classes and the records are split into DS1 (training patients) and DS2 (test patients). The 44 records are independent, so a process pool handles them one record per task, timed with 1 to 24 worker processes. Pool start-up is timed separately, because on Windows every worker is a new Python interpreter. Before any time is recorded, the parallel output is compared with the serial output.

Output: `data/beats_ds1.npz`, `data/beats_ds2.npz`, `results/prep/prep_speedup.csv` and `.png`.

### Step 2. Features

```
python src/features.py
```

Standardizes every beat and reduces it to 8 numbers with PCA, fitted on DS1 only. It also writes a copy scaled to [0, 1] (for the quantum circuit), and `data/u_train.bin`, the same numbers as raw doubles for the C programs. Takes a few seconds.

Output: `data/features8.npz`, `data/u_train.bin`, `results/features/pca_features.png`.

### Step 3. Classical baselines

```
python src/baselines.py
quick: python src/baselines.py --sizes 1000 --seeds 2 --trees 50 --cores 1 4
```

The task is V (ventricular) vs everything else, training on DS1 and testing on all of DS2. The SVM's C and gamma are chosen by 5-fold cross-validation grouped by patient. Both models are trained on 1k to 20k beats with 5 seeds, and the forest is also trained on all of DS1, on 1 to 24 cores.

Output: `results/baselines/` (runs, forest scaling, chosen SVM parameters, two figures).

### Step 4. Check the simulator

```
python src/check_qsim.py
```

Compares the simulator with brute-force dense matrices (2 to 8 qubits) and with Qiskit's `ZZFeatureMap`. It also checks that version A, version B, the test x train kernel and the GPU versions all agree. Run this before trusting any timing. It prints the largest error for each check, and exits with an error if one fails.

### Step 5. Version A on threads

```
python src/kernel_threads.py
quick: python src/kernel_threads.py --n 400 --threads 1 4 --reps 1
```

Fills the kernel matrix one circuit per pair, on 1 to 24 threads, with four ways of splitting the rows: equal blocks, cyclic, folded (row t together with row N−1−t) and a dynamic queue. By default N is picked so that the serial run takes about 20 s, and `--real` uses real beats instead of random inputs. Every parallel result must be identical to the serial one.

Output: `results/kernel/schedules.csv` and `.png`.

### Step 6. Version A vs version B vs the GPU

```
python src/kernel_compare.py
quick: python src/kernel_compare.py --n 400 --big 1500 --reps 1
```

Times version A on 1 core, all cores and the GPU, and version B on 1 core, all cores and the GPU in double and single precision, at N = 2,900 and N = 16,000. Version A at the big N on the CPU would take hours, so that time is extrapolated (marked in the CSV). The script also splits version B into its three steps (states, matrix product, squaring) to show which one stops scaling.

Output: `results/kernel/compare.csv`, `b_steps.csv`, `compare.png`.

### Step 7. One big state vector

```
python src/big_state.py
quick: python src/big_state.py --qubits 20 --threads 1 4 --nmin 12 --nmax 20 --reps 1
```

A single 26-qubit state (1 GiB) on 1 to 24 threads, compared with a plain parallel copy (the memory limit). Then 12 to 30 qubits on the CPU and the GPU, to find where the GPU starts to win and where each one runs out of memory. The 30-qubit state needs 16 GB of free memory; use `--nmax 28` on a smaller machine.

Output: `results/bigstate/`.

### Step 8. OpenMP (C)

Build all three programs and check them against Python first:

```
cd openmp
make
cd ..
python src/check_openmp.py
```

`check_openmp.py` runs every schedule and every method on 200 real beats and compares the matrices with `qsim.py`. They agree to about 3e-14.

The three programs, each run from inside `openmp/`:

```
./qkernel_omp -n 2000 -t 8 -s dynamic -c 4 -f ../data/u_train.bin
./gram_omp -n 8000 -t 8 -m tiled -b 64 -f ../data/u_train.bin
./bigstate_omp -q 24 -t 8
```

- **`qkernel_omp`** is version A. `-s` picks the schedule: `static`, `cyclic`, `dynamic`, `guided`, `folded` or `taskloop`. The first four use `schedule(runtime)` with `omp_set_schedule()`, so one loop covers all of them. Besides the time, it prints how long the least and the most loaded thread were busy, which shows the load imbalance directly.
- **`gram_omp`** is version B. `-m naive` does one dot product per pair. `-m tiled` cuts the triangle into B x B tiles (`collapse(2)` over the tile grid), so a tile's states stay in cache. `-m tasks` uses the same tiles as OpenMP tasks. `-m cross -x M` computes a test x train kernel, which is what the SVM needs for its predictions. The dot product is plain real arithmetic with `#pragma omp simd reduction`, so it vectorizes.
- **`bigstate_omp`** is one state of q qubits. It times a Hadamard layer (memory-bound), a ZZ phase layer (compute-bound) and a plain copy. With `-s` the memory is first touched by a single thread, which matters on two-socket cluster nodes.

Without `-f` the inputs are random; the work per pair does not depend on the values. All three print one CSV line. The timing sweeps:

```
cd openmp
sh run_schedules.sh                    # all schedules, 1..24 threads, N = 2000
sh run_gram.sh                         # naive / tiled / tasks, tile sizes, test x train
sh run_bigstate.sh                     # 26 qubits, both first-touch variants
cd ..
python src/openmp_figure.py
quick: N=400 THREADS="1 4" sh run_schedules.sh   (and the same for the other two)
```

Output: `results/openmp/*.csv` and `openmp_schedules.png`, `openmp_gram.png`, `openmp_bigstate.png`.

### Step 9. MPI

```
mpirun -np 4 python src/kernel_mpi.py --n 800 --mode dynamic --check
```

Version A spread over MPI processes. Every rank builds the inputs itself, so the only messages are row numbers going out and finished rows coming back.

- `--mode dynamic` is manager-worker: rank 0 hands out chunks of rows on request and collects them, so with P ranks there are P−1 workers.
- `--mode block` and `--mode cyclic` split the rows up front and collect them with one `Gatherv`.
- `--mode serial` runs without MPI and gives T1.
- `--check` compares the result with the serial matrix.

The sweep and its figure:

```
sh scripts/run_mpi.sh
N=3000 RANKS="2 4 8 16 32" sh scripts/run_mpi.sh
```

On Windows, use `mpiexec -localonly -n 4` instead of `mpirun -np 4`; `run_mpi.sh` has an example at the top.

Output: `results/mpi/runs.csv`, `mpi.png`.

### Step 10. The quantum SVM

```
python src/qsvm.py                 # kernels on the CPU
python src/qsvm.py --gpu           # kernels on the GPU (--single for complex64)
quick: python src/qsvm.py --sizes 500 1000 --seeds 2 --tune-size 600
```

First the bandwidth λ and C are picked by 5-fold cross-validation grouped by patient, on a DS1 subset. For each λ the script prints how spread out the kernel values are: when λ is too large they all collapse towards 0 (kernel concentration) and the SVM has nothing to work with. Then the QSVM is trained on 1k, 2k and 5k DS1 beats with 5 seeds and scored on all 49,691 DS2 beats, next to an RBF SVM trained on exactly the same beats. The prediction only needs the kernel against the support vectors, so the test kernel is built for those columns only, in blocks of 8,192 test beats. The script checks that this shortcut gives the same scores as scikit-learn. Train kernel, fit, test kernel and prediction are timed separately.

Output: `results/qsvm/tuning.csv`, `runs.csv`, `settings.json`, `tuning.png`, `qsvm.png`.

## How the code developed

The full history is in the git log. In short:

1. **Pilot (late September).** A first Python version of the simulator, checked against Qiskit, plus timing scripts for version A schedules, version B and the GPU. Their results are in `results/pilot/`.
2. **Week 1.** Data pipeline, features and classical baselines (`results/week1/`).
3. **Midterm (early October).** The code was cleaned up into this repository, and the first OpenMP program was written (version A, five schedules).
4. **After the midterm.** Version A on the GPU, the test x train kernel, two more OpenMP programs (version B with tiling, one big state) plus a taskloop schedule, the MPI version, the quantum SVM on the real features, and the quick test.

A few things I only learned by measuring:

- **The squaring step.** In version B, squaring the matrix product ran on one core and capped the speedup on 24 cores (Amdahl's law). It is parallel now: 0.034 s down to 0.003 s on 24 cores at N = 2,000.
- **The phase table.** Turning features into phase vectors was a Python loop. That was fine for a few thousand beats, but it took about 5 s of the 5.4 s of a whole QSVM run with 49,691 test beats. With a Numba `prange` loop the whole run takes 0.35 s.
- **Process start-up.** On Windows, starting 24 worker processes took longer than the work they did, so the pool start-up is timed on its own.
- **The timer.** `omp_get_wtime()` only ticked once per millisecond with w64devkit on Windows. The C programs now use `QueryPerformanceCounter` there and `clock_gettime` on Linux.
- **The schedule matters as much as the core count.** The same 24 threads gave a speedup of 9.1 with equal blocks of rows and 14.1 with a dynamic queue.

## Results so far

From the midterm report, on my laptop: an Intel Core Ultra 9 285HX (8 performance and 16 efficiency cores), 64 GB of memory and an NVIDIA RTX PRO 4000 Blackwell laptop GPU, running Windows 11.

- **Beat extraction (44 records):** 3.46 s serial and 0.26 s with 24 processes (13.3x). Starting those processes took another 2.3 s.
- **Random forest (all 51,002 training beats):** 11.2 s on 1 core and 0.75 s on 24 (14.9x).
- **Classical models:** AUROC of about 0.97 on the 22 unseen test patients.
- **Version A, N = 2,900 (4.2 million circuits):** 19.7 s on one core. On 24 threads the speedup was 9.1x with block, 13.6x with cyclic and folded, and 14.1x with dynamic.
- **Version B, N = 16,000:** 9.4 s on one core, 2.0 s on 24 cores, 0.92 s on the GPU in double precision and 0.053 s in single precision.
- **Simulator vs Qiskit:** the largest kernel difference is 1.6e-15.

First numbers from the newer code. These come from short test runs; the full sweeps are still to be done.

- **OpenMP version A:** one pair takes about 2.7 µs in C, against 4.7 µs with Numba.
- **OpenMP version B (N = 4,000, 8 threads):** the tiled Gram matrix took 0.053 s and the naive one 0.098 s.
- **One 22-qubit state, 1 to 8 threads:** the compute-bound ZZ layer got 8.3x faster, the memory-bound Hadamard layer only 4x.
- **GPU version A (N = 2,000):** 0.29 s, against about 0.75 s estimated for 24 CPU threads. Version B on the GPU is still far faster.
- **MPI, 4 ranks, N = 600:** block took 0.44 s because its first rank had most of the work; cyclic took 0.27 s, and dynamic 0.35 s with only 3 workers.
- **Quantum SVM, 500 to 1,000 training beats:** AUROC 0.91 to 0.93, about the same as the RBF SVM on the same beats, with a higher AUPRC. A run takes about 0.35 s, against about 0.2 s for the RBF SVM.

`results/week1` was made with an earlier version of `baselines.py` that searched a smaller SVM grid (it picked C = 100, gamma = 0.01).

## Still to do

- Full OpenMP and MPI sweeps on a stable machine (plugged in, or the department cluster)
- QSVM up to 10,000 training beats with all seeds, on the CPU and the GPU
- Qubit scaling study from 2 to about 30 qubits
- Final comparison of the QSVM and the classical models, with confidence intervals

## Data

MIT-BIH Arrhythmia Database v1.0.0 from PhysioNet, under the Open Data Commons Attribution License v1.0. It is not included here; `scripts/get_data.py` downloads it.

- G. B. Moody and R. G. Mark, "The impact of the MIT-BIH Arrhythmia Database," *IEEE Eng. Med. Biol. Mag.*, vol. 20, no. 3, pp. 45–50, 2001.
- A. L. Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet," *Circulation*, vol. 101, no. 23, pp. e215–e220, 2000.
- Train/test split: P. de Chazal, M. O'Dwyer and R. B. Reilly, "Automatic classification of heartbeats using ECG morphology and heartbeat interval features," *IEEE Trans. Biomed. Eng.*, vol. 51, no. 7, pp. 1196–1206, 2004.
