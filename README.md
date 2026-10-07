# The Computational Cost of Healthcare Quantum Machine Learning

Md Tanzimul Alam Fahim. Course project for CSCI 6330, Middle Tennessee State University, Fall 2026.

I train a quantum kernel SVM to pick out ventricular heartbeats in the MIT-BIH Arrhythmia Database and measure what it costs to run. No quantum computer is involved. The circuits are simulated with a small state-vector simulator I wrote for this project, and most of the work is in making that simulation run in parallel: threads (Numba and OpenMP), a GPU through CuPy, and later MPI. The same features also go into an ordinary SVM and a random forest, so the quantum model's cost can be compared with something.

## What's in here

```
scripts/get_data.py      download MIT-BIH from PhysioNet into data/
src/config.py            paths, record lists, label mapping
src/prep_beats.py        filter the ECG and cut out every beat, one worker process per record
src/features.py          PCA to 8 features, fitted on the training patients only
src/baselines.py         RBF SVM and random forest (V vs non-V), forest on 1..24 cores
src/qsim.py              the simulator: ZZ feature map, kernel versions A and B, CUDA kernels
src/check_qsim.py        simulator vs dense matrices, Qiskit and the GPU
src/kernel_threads.py    version A on 1..24 threads with four schedules
src/kernel_compare.py    A vs B vs GPU, and where B spends its time
src/big_state.py         one large state vector (up to 30 qubits), CPU vs GPU
src/check_openmp.py      OpenMP program vs the Python simulator
src/openmp_figure.py     figure for the OpenMP timings
src/plots.py             shared figure style
openmp/qkernel_omp.c     version A in C with OpenMP, five schedules
openmp/Makefile
openmp/run_schedules.sh  OpenMP timing sweep
results/week1/           data, features and baselines (week 1)
results/pilot/           first simulator runs, with random inputs
```

Version A builds the kernel matrix the way real hardware would, one circuit per pair of beats. Version B simulates each beat's state once and gets the whole matrix from one complex matrix product, `K = |Ψ Ψ†|²`. Both give the same numbers (checked in `check_qsim.py`), but they parallelize very differently.

## Setup

```
python -m venv .venv
.venv\Scripts\activate             # Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/get_data.py
```

The dataset is about 100 MB unpacked and goes into `data/`, which is not part of the repository. Qiskit is only used by `check_qsim.py`, and CuPy only by the GPU parts; everything else runs without them.

## Running it

From the repository root, in this order:

```
python src/prep_beats.py           # data/beats_ds1.npz, data/beats_ds2.npz
python src/features.py             # data/features8.npz, data/u_train.bin
python src/baselines.py
python src/check_qsim.py           # run this before trusting any timing
python src/kernel_threads.py
python src/kernel_compare.py
python src/big_state.py
```

Each script writes its CSV files and figures to `results/<step>/`. Most of them can be made smaller and quicker from the command line; the docstring at the top of each file has an example. A few things to keep in mind:

- T1 is always compiled code on one core (Numba or C), never plain Python, so speedups are not inflated.
- Numba compiles a function the first time it is called, so every script calls each one once before timing.
- Every parallel result is compared with the serial one before its time is recorded.
- On Windows, `multiprocessing` starts a new Python interpreter for every worker. `prep_beats.py` times that start-up separately because it turned out to be bigger than the work itself.
- `big_state.py` needs 16 GB of free memory for the 30-qubit state. Use `--nmax 28` on a smaller machine.

## OpenMP version

`openmp/qkernel_omp.c` computes the same kernel matrix as version A in `qsim.py`, in C with OpenMP. Every thread keeps its own 256-amplitude state vector on its stack. Row i of the upper triangle has N−1−i entries, so the interesting part is how the rows are split between threads:

| schedule | how the rows are divided |
|----------|--------------------------|
| static   | equal contiguous blocks (the OpenMP default) |
| cyclic   | `static,1`: rows dealt out like cards |
| dynamic  | a free thread takes the next chunk of rows (chunk 4 in the sweep) |
| guided   | like dynamic, but the chunks shrink towards the end |
| folded   | row t together with row N−1−t, so every task is the same size |

The first four use `schedule(runtime)` with `omp_set_schedule()`, so one loop covers all of them.

Build with gcc on Linux or on the cluster:

```
cd openmp
make
```

On Windows I used the gcc from [w64devkit](https://github.com/skeeto/w64devkit), which needs no admin rights: put its `bin` folder on the PATH and run `make` the same way.

Then check it and time it:

```
python src/check_openmp.py         # same matrix as the Python simulator, every schedule
cd openmp
./qkernel_omp -n 2000 -t 8 -s dynamic -c 4 -f ../data/u_train.bin
sh run_schedules.sh                # all schedules on 1..24 threads
cd ..
python src/openmp_figure.py
```

The program prints one CSV line: the median time of three runs and, from that same run, the busy time of the least and the most loaded thread. Without `-f` it uses random inputs, which is fine for timing because the work per pair does not depend on the values. On Windows `omp_get_wtime()` only had millisecond resolution in my tests, so keep N at 2000 or more there.

## Results so far

These are the numbers from my midterm report. They were measured on my laptop: an Intel Core Ultra 9 285HX (8 performance and 16 efficiency cores), 64 GB of memory and an NVIDIA RTX PRO 4000 Blackwell laptop GPU, running Windows 11.

- **Beat extraction (44 records):** 3.46 s serial and 0.26 s with 24 processes (13.3x). Starting those 24 processes took another 2.3 s.
- **Random forest (all 51,002 training beats):** 11.2 s on 1 core and 0.75 s on 24 (14.9x).
- **Classical accuracy:** both classical models reach an AUROC of about 0.97 on the 22 unseen test patients.
- **Quantum kernel, version A (N = 2,900, 4.2 million circuits):** 19.7 s on one core. With 24 threads the speedup depended on the schedule: block 9.1x, cyclic 13.6x, folded 13.6x, dynamic 14.1x.
- **Version B (N = 16,000):** 9.4 s on one core, 2.0 s on 24 cores, 0.92 s on the GPU in double precision and 0.053 s in single precision.
- **Simulator vs Qiskit:** the largest kernel difference is 1.6e-15.

Two things changed after the report:

- The last step of version B, squaring the matrix product, now runs in parallel. Before, it was the serial part that held B back on 24 cores.
- The OpenMP version is new. It matches the Python simulator to 3e-14, and one pair takes about 2.7 µs in C against 4.7 µs with Numba. Its full timing sweep is still to be run.

`results/week1` was made with an earlier version of `baselines.py` that searched a smaller SVM grid (it picked C = 100, gamma = 0.01).

## Still to do

- MPI version (manager–worker over chunks of rows with mpi4py), so the long runs can move to the department cluster
- QSVM on the real ECG features with 1,000 to 10,000 training beats, after tuning the bandwidth λ on DS1
- GPU scaling from 2 up to about 30 qubits
- QSVM vs classical models with several seeds and confidence intervals

## Data

MIT-BIH Arrhythmia Database v1.0.0 from PhysioNet, under the Open Data Commons Attribution License v1.0. It is not included here; `scripts/get_data.py` downloads it.

- G. B. Moody and R. G. Mark, "The impact of the MIT-BIH Arrhythmia Database," *IEEE Eng. Med. Biol. Mag.*, vol. 20, no. 3, pp. 45–50, 2001.
- A. L. Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet," *Circulation*, vol. 101, no. 23, pp. e215–e220, 2000.
- Train/test split: P. de Chazal, M. O'Dwyer and R. B. Reilly, "Automatic classification of heartbeats using ECG morphology and heartbeat interval features," *IEEE Trans. Biomed. Eng.*, vol. 51, no. 7, pp. 1196–1206, 2004.
