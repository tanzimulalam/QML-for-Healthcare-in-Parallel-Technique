"""
The quantum kernel SVM on the real ECG features: ventricular (V) vs everything
else, trained on DS1 patients, tested on all 49,691 beats of the DS2 patients.

1. Tuning. The bandwidth lambda (angles = lambda * pi * u) and C are picked with
   5-fold cross-validation grouped by patient on a DS1 subset. For each lambda the
   kernel matrix is computed once and the folds just take slices of it. The script
   also prints how spread out the kernel values are: with a large lambda they all
   collapse towards 0 (kernel concentration) and the SVM has nothing to work with.
2. Training on 1k-5k DS1 beats with a few seeds, scored on all of DS2. The test
   kernel only needs the columns of the support vectors, and it is built in
   blocks of test beats so it never has to be in memory all at once.
3. An RBF SVM is trained on exactly the same beats, so the comparison is fair.

Train kernel, SVM fit, test kernel and prediction are timed separately.

    python src/qsvm.py                                     # CPU (version B)
    python src/qsvm.py --gpu                               # kernels on the GPU
    python src/qsvm.py --sizes 500 1000 --seeds 2 --tune-size 800      # quick
"""
import argparse
import json
import time

import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import qsim
from baselines import mean_ci, subsample, write_csv
from config import DATA, RESULTS, V

OUT = RESULTS / "qsvm"
LAMBDAS = [0.025, 0.05, 0.1, 0.2, 0.4, 0.8]
CS = [0.1, 1, 10, 100, 1000]
BLOCK = 8192          # test beats per block


class Kernel:
    """Kernel matrices with version B, on the CPU (Numba + BLAS threads) or the GPU."""

    def __init__(self, gpu=False, single=False):
        self.gpu = gpu
        if gpu:
            import cupy as cp
            self.cp = cp
            self.dtype = cp.complex64 if single else cp.complex128

    def states(self, U, lam):
        D = qsim.encode(U, lam)
        if self.gpu:
            return qsim.states_gpu(self.cp.asarray(D, dtype=self.dtype))
        return qsim.states(D)

    def gram(self, psi_a, psi_b):
        if self.gpu:
            return self.cp.asnumpy(qsim.gram_gpu(psi_a, psi_b)).astype(np.float64)
        return qsim.gram(psi_a, psi_b)

    def matrix(self, U, lam):
        psi = self.states(U, lam)
        return self.gram(psi, psi)


def metrics(y, score):
    pred = score > 0
    return dict(auroc=roc_auc_score(y, score), auprc=average_precision_score(y, score),
                sensitivity=recall_score(y, pred), ppv=precision_score(y, pred, zero_division=0))


def tune(kern, U, y, groups, size):
    idx = subsample(y, size, seed=321)
    Us, ys, gs = U[idx], y[idx], groups[idx]
    folds = [(tr, va) for tr, va in GroupKFold(5).split(Us, ys, gs) if ys[va].any()]
    rows = []
    for lam in LAMBDAS:
        t0 = time.perf_counter()
        K = kern.matrix(Us, lam)
        tk = time.perf_counter() - t0
        off = K[np.triu_indices(len(K), 1)]
        for C in CS:
            scores = []
            for tr, va in folds:
                svm = SVC(C=C, kernel="precomputed", class_weight="balanced").fit(K[np.ix_(tr, tr)], ys[tr])
                scores.append(average_precision_score(ys[va], svm.decision_function(K[np.ix_(va, tr)])))
            rows.append(dict(lam=lam, C=C, cv_auprc=float(np.mean(scores)), k_mean=float(off.mean()),
                             k_std=float(off.std()), kernel_s=tk))
        best = max((r for r in rows if r["lam"] == lam), key=lambda r: r["cv_auprc"])
        print(f"lambda={lam:<6} kernel values {off.mean():.3f} +- {off.std():.3f}   "
              f"best C={best['C']:<6} CV AUPRC {best['cv_auprc']:.3f}   ({tk:.2f} s for the kernel)", flush=True)
    return max(rows, key=lambda r: r["cv_auprc"]), rows


def qsvm_run(kern, U, y, Ute, yte, idx, lam, C):
    t = {}
    t0 = time.perf_counter()
    psi = kern.states(U[idx], lam)
    K = kern.gram(psi, psi)
    t["train_kernel_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    svm = SVC(C=C, kernel="precomputed", class_weight="balanced").fit(K, y[idx])
    t["fit_s"] = time.perf_counter() - t0

    # f(x) = sum_k dual_k K(x, sv_k) + b, so only the support vector columns are needed
    sv = svm.support_
    psi_sv = psi[sv]
    dual, b = svm.dual_coef_[0], svm.intercept_[0]
    score = np.empty(len(yte))
    t["test_kernel_s"] = t["predict_s"] = 0.0
    for start in range(0, len(yte), BLOCK):
        stop = min(start + BLOCK, len(yte))
        t0 = time.perf_counter()
        Kb = kern.gram(kern.states(Ute[start:stop], lam), psi_sv)
        t1 = time.perf_counter()
        score[start:stop] = Kb @ dual + b
        t["test_kernel_s"] += t1 - t0
        t["predict_s"] += time.perf_counter() - t1

    # once: the shortcut must agree with scikit-learn's own prediction
    full = kern.gram(kern.states(Ute[:200], lam), psi)
    assert np.allclose(svm.decision_function(full), score[:200], atol=1e-8), "support vector shortcut is off"
    return dict(**metrics(yte, score), **t, n_sv=len(sv))


def rbf_run(X, y, Xte, yte, idx, C, gamma):
    model = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma, class_weight="balanced"))
    t0 = time.perf_counter()
    model.fit(X[idx], y[idx])
    fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    score = model.decision_function(Xte)
    pred = time.perf_counter() - t0
    return dict(**metrics(yte, score), train_kernel_s=0.0, fit_s=fit, test_kernel_s=0.0, predict_s=pred,
                n_sv=int(model[-1].n_support_.sum()))


def rbf_params():
    for path in (RESULTS / "baselines" / "svm_params.json", RESULTS / "week1" / "svm_params.json"):
        if path.exists():
            p = json.loads(path.read_text())
            return p["C"], p["gamma"]
    return 100, 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 2000, 5000])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--tune-size", type=int, default=2000)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--single", action="store_true", help="complex64 on the GPU")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    d = np.load(DATA / "features8.npz")
    U, X, y, groups = d["U_train"], d["X_train"], (d["y_train"] == V).astype(int), d["rec_train"]
    Ute, Xte, yte = d["U_test"], d["X_test"], (d["y_test"] == V).astype(int)
    kern = Kernel(args.gpu, args.single)
    kern.matrix(U[:16], 0.1)                     # compile / warm up before timing
    print(f"backend: {'GPU' if args.gpu else 'CPU'}, train {len(y)} beats, test {len(yte)} beats ({yte.sum()} V)")

    best, tuning = tune(kern, U, y, groups, args.tune_size)
    lam, C = best["lam"], best["C"]
    print(f"picked lambda={lam}, C={C} (CV AUPRC {best['cv_auprc']:.3f})")
    write_csv(OUT / "tuning.csv", tuning)
    rC, rgamma = rbf_params()

    runs = []
    for n in args.sizes:
        for seed in range(args.seeds):
            idx = subsample(y, n, seed)
            for model, fn in (("QSVM", lambda: qsvm_run(kern, U, y, Ute, yte, idx, lam, C)),
                              ("RBF SVM", lambda: rbf_run(X, y, Xte, yte, idx, rC, rgamma))):
                r = dict(model=model, n_train=n, seed=seed, **fn())
                runs.append(r)
                total = r["train_kernel_s"] + r["fit_s"] + r["test_kernel_s"] + r["predict_s"]
                print(f"{model:7s} N={n:5d} seed={seed}  AUROC {r['auroc']:.3f}  AUPRC {r['auprc']:.3f}  "
                      f"sens {r['sensitivity']:.2f}  PPV {r['ppv']:.2f}  {r['n_sv']:5d} SVs  {total:6.2f} s",
                      flush=True)
    write_csv(OUT / "runs.csv", runs)
    with open(OUT / "settings.json", "w") as f:
        json.dump(dict(lam=lam, C=C, rbf_C=rC, rbf_gamma=rgamma, backend="gpu" if args.gpu else "cpu",
                       single=args.single), f)
    plot(tuning, runs, lam)


def log_ticks(ax, values):
    # log axis labelled with the values that were actually run
    ax.set_xscale("log")
    ax.set_xticks(values, [f"{v:g}" for v in values])
    ax.minorticks_off()


def plot(tuning, runs, lam_best):
    import plots
    plt = plots.setup()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    lams = sorted({r["lam"] for r in tuning})
    best = [max(r["cv_auprc"] for r in tuning if r["lam"] == lam) for lam in lams]
    a1.plot(lams, best, "o-", color=plots.NAVY)
    a1.axvline(lam_best, color=plots.GREY, ls=(0, (5, 3)), lw=1.2)
    log_ticks(a1, lams)
    a1.set(xlabel="bandwidth λ (log scale)", ylabel="CV AUPRC (best C)", title="Tuning on DS1, grouped by patient")
    mean = np.array([next(r["k_mean"] for r in tuning if r["lam"] == lam) for lam in lams])
    std = np.array([next(r["k_std"] for r in tuning if r["lam"] == lam) for lam in lams])
    a2.fill_between(lams, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), color=plots.LIGHT_BLUE, alpha=0.6,
                    label="mean ± 1 sd")
    a2.plot(lams, mean, "o-", color=plots.BLUE, label="mean")
    log_ticks(a2, lams)
    a2.set(xlabel="bandwidth λ (log scale)", ylabel="kernel value between two beats", ylim=(0, 1.02),
           title="Kernel concentration")
    a2.legend(loc="lower left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "tuning.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, key in zip(axes[:2], ("auroc", "auprc")):
        for model, c, m in (("QSVM", plots.RED, "o"), ("RBF SVM", plots.NAVY, "s")):
            s = mean_ci(runs, model, key)
            ax.errorbar([x[0] for x in s], [x[1] for x in s], yerr=[x[2] for x in s], color=c, marker=m,
                        capsize=4, label=model)
        log_ticks(ax, sorted({r["n_train"] for r in runs}))
        ax.set(xlabel="training beats (log scale)", title=f"{key.upper()} on the 22 test patients")
        ax.legend(loc="lower right")
    ax = axes[2]
    sizes = sorted({r["n_train"] for r in runs})
    x = np.arange(len(sizes))
    bottom = np.zeros(len(sizes))
    for key, c, label in (("train_kernel_s", plots.RED, "train kernel"), ("fit_s", plots.ORANGE, "SVM fit"),
                          ("test_kernel_s", plots.YELLOW, "test kernel"), ("predict_s", plots.GREY, "predict")):
        v = np.array([np.mean([r[key] for r in runs if r["model"] == "QSVM" and r["n_train"] == n]) for n in sizes])
        ax.bar(x, v, bottom=bottom, color=c, width=0.6, label=label)
        bottom += v
    rbf = [np.mean([r["fit_s"] + r["predict_s"] for r in runs if r["model"] == "RBF SVM" and r["n_train"] == n])
           for n in sizes]
    ax.plot(x, rbf, "s", color=plots.NAVY, ms=9, label="RBF SVM, total")
    ax.set_xticks(x, [f"{n:,}" for n in sizes])
    ax.set(xlabel="training beats", ylabel="seconds", title="Where the QSVM's time goes")
    ax.legend(loc="upper left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "qsvm.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
