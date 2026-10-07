"""
Classical baselines on the 8 features: RBF SVM and random forest.

Task: ventricular (V) beats vs everything else. Train on DS1, test on all of DS2.
SVM hyperparameters come from 5-fold CV grouped by patient inside DS1.
Training sets of 1k-20k beats (5 seeds each) because the quantum kernel SVM will
only be able to use subsets of this size; the forest is also trained on all of DS1.
Also times the forest on 1..24 cores and the SVM against training size.

    python src/baselines.py
    python src/baselines.py --sizes 1000 5000 --seeds 2 --cores 1 4 8   # quick run
"""
import argparse
import csv
import json
import time

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import DATA, RESULTS, V

OUT = RESULTS / "baselines"


def subsample(y, n, seed):
    # keep the V / non-V ratio of the full training set
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    k = round(n * len(pos) / len(y))
    return np.sort(np.r_[rng.choice(pos, k, replace=False), rng.choice(neg, n - k, replace=False)])


def tune_svm(X, y, groups):
    idx = subsample(y, 5000, seed=123)
    grid = GridSearchCV(
        make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced")),
        {"svc__C": [0.1, 1, 10, 100, 1000], "svc__gamma": [0.003, 0.01, 0.03, 0.1, 0.3]},
        scoring="average_precision", cv=GroupKFold(5), n_jobs=-1)
    grid.fit(X[idx], y[idx], groups=groups[idx])
    return grid.best_params_["svc__C"], grid.best_params_["svc__gamma"], grid.best_score_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 2000, 5000, 10000, 20000])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--cores", type=int, nargs="+", default=[1, 2, 4, 6, 8, 12, 16, 20, 24])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    d = np.load(DATA / "features8.npz")
    Xtr, ytr, groups = d["X_train"], (d["y_train"] == V).astype(int), d["rec_train"]
    Xte, yte = d["X_test"], (d["y_test"] == V).astype(int)
    print(f"train {len(ytr)} beats ({ytr.sum()} V), test {len(yte)} beats ({yte.sum()} V)")

    C, gamma, cv_score = tune_svm(Xtr, ytr, groups)
    print(f"SVM: C={C}, gamma={gamma} (patient-grouped CV AUPRC {cv_score:.3f})")
    with open(OUT / "svm_params.json", "w") as f:
        json.dump({"C": C, "gamma": gamma, "cv_auprc": cv_score}, f)

    runs = []
    for n in args.sizes + [len(ytr)]:
        full = n == len(ytr)
        for seed in range(1 if full else args.seeds):
            idx = np.arange(len(ytr)) if full else subsample(ytr, n, seed)
            models = {"RF": RandomForestClassifier(args.trees, class_weight="balanced_subsample",
                                                   n_jobs=-1, random_state=seed)}
            if not full:
                models["SVM"] = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma, class_weight="balanced"))
            for name, model in models.items():
                t0 = time.perf_counter()
                model.fit(Xtr[idx], ytr[idx])
                fit_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                if name == "SVM":
                    score = model.decision_function(Xte)
                    pred = score > 0
                else:
                    score = model.predict_proba(Xte)[:, 1]
                    pred = score > 0.5
                pred_s = time.perf_counter() - t0
                runs.append(dict(model=name, n_train=n, seed=seed, fit_s=fit_s, predict_s=pred_s,
                                 auroc=roc_auc_score(yte, score), auprc=average_precision_score(yte, score),
                                 sensitivity=recall_score(yte, pred), ppv=precision_score(yte, pred, zero_division=0)))
                r = runs[-1]
                print(f"{name:3s} N={n:6d} seed={seed}  AUROC {r['auroc']:.3f}  AUPRC {r['auprc']:.3f}  "
                      f"sens {r['sensitivity']:.2f}  PPV {r['ppv']:.2f}  fit {fit_s:.2f} s", flush=True)
    write_csv(OUT / "baseline_runs.csv", runs)

    # random forest on all of DS1 with an increasing number of cores
    scaling = []
    for p in args.cores:
        times = []
        for _ in range(2):
            rf = RandomForestClassifier(args.trees, class_weight="balanced_subsample", n_jobs=p, random_state=0)
            t0 = time.perf_counter()
            rf.fit(Xtr, ytr)
            times.append(time.perf_counter() - t0)
        scaling.append(dict(cores=p, fit_s=float(np.median(times))))
        print(f"RF on {p:2d} cores: {scaling[-1]['fit_s']:.2f} s "
              f"(speedup {scaling[0]['fit_s'] / scaling[-1]['fit_s']:.2f})", flush=True)
    write_csv(OUT / "rf_scaling.csv", scaling)
    plot(runs, scaling, len(ytr))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def mean_ci(runs, model, key):
    out = []
    for n in sorted({r["n_train"] for r in runs if r["model"] == model}):
        v = np.array([r[key] for r in runs if r["model"] == model and r["n_train"] == n])
        half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        out.append((n, v.mean(), half))
    return out


def plot(runs, scaling, n_full):
    import plots
    plt = plots.setup()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, key in zip(axes, ("auroc", "auprc")):
        for model, c, m, label in (("SVM", plots.NAVY, "o", "SVM"), ("RF", plots.RED, "s", "random forest")):
            s = mean_ci(runs, model, key)
            ax.errorbar([x[0] for x in s], [x[1] for x in s], yerr=[x[2] for x in s],
                        color=c, marker=m, capsize=4, label=label)
        ax.set_xscale("log")
        ax.set(xlabel="training beats (log scale)", title=f"{key.upper()} on the 22 test patients")
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "baseline_accuracy.png")
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    s = mean_ci(runs, "SVM", "fit_s")
    if s:
        n = np.array([x[0] for x in s], float)
        t = np.array([x[1] for x in s])
        a1.plot(n, t, "o-", color=plots.NAVY, label="SVM (1 core)")
        a1.plot(n, t[-1] * (n / n[-1]) ** 2, color=plots.GREY, ls=(0, (5, 3)), lw=1.2, label="N squared")
    r = mean_ci(runs, "RF", "fit_s")
    a1.plot([x[0] for x in r], [x[1] for x in r], "s-", color=plots.RED, label="random forest (all cores)")
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set(xlabel="training beats", ylabel="training time (s)", title="Training time vs data size")
    a1.legend(loc="upper left", fontsize=12)

    p = np.array([x["cores"] for x in scaling])
    sp = scaling[0]["fit_s"] / np.array([x["fit_s"] for x in scaling])
    a2.axvspan(0.5, 8.5, color="#F2F2F2", zorder=0)
    plots.ideal(a2, p)
    a2.plot(p, sp, "o-", color=plots.RED, label="random forest")
    a2.text(4.5, 0.6, "8 P-cores", ha="center", color=plots.GREY, fontsize=12)
    a2.set(xlabel="cores", ylabel="speedup (T1 / Tp)", xticks=p, ylim=(0, p[-1] + 1),
           title=f"Random forest on all {n_full:,} beats")
    a2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "baseline_timing.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
