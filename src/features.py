"""
Reduce every beat (180 samples) to 8 numbers with PCA.

PCA is fitted on the training patients (DS1) only and then applied to DS2 as is.
The same 8 features feed every model. A copy scaled to [0, 1] with DS1's
1st/99th percentiles is kept for the quantum circuit, and is also written as a
raw float64 file so the C/OpenMP code can read it.

Writes data/features8.npz, data/u_train.bin and results/features/pca_features.png
"""
import numpy as np
from sklearn.decomposition import PCA

from config import DATA, RESULTS

N_FEATURES = 8
OUT = RESULTS / "features"


def standardize(beats):
    b = beats.astype(np.float64)
    return (b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-9)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d1 = np.load(DATA / "beats_ds1.npz")
    d2 = np.load(DATA / "beats_ds2.npz")
    b1, b2 = standardize(d1["beats"]), standardize(d2["beats"])

    pca = PCA(N_FEATURES, random_state=0).fit(b1)
    x1, x2 = pca.transform(b1), pca.transform(b2)
    var = pca.explained_variance_ratio_
    print("variance per component:", np.round(var, 3).tolist())
    print(f"kept {var.sum():.1%} of the variance")

    lo, hi = np.percentile(x1, 1, axis=0), np.percentile(x1, 99, axis=0)
    u1 = np.clip((x1 - lo) / (hi - lo), 0, 1)
    u2 = np.clip((x2 - lo) / (hi - lo), 0, 1)

    np.savez_compressed(DATA / "features8.npz",
                        X_train=x1.astype(np.float32), y_train=d1["labels"], rec_train=d1["record"],
                        X_test=x2.astype(np.float32), y_test=d2["labels"], rec_test=d2["record"],
                        U_train=u1, U_test=u2,   # float64, so the .bin file below is the same numbers
                        rr_train=d1["rr"], rr_test=d2["rr"],
                        explained_variance_ratio=var)
    u1.astype("<f8").tofile(DATA / "u_train.bin")   # N x 8 doubles, row-major, for openmp/qkernel_omp
    plot(pca, x1, d1["labels"])


def plot(pca, x1, y1):
    import plots
    plt = plots.setup()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    t = (np.arange(pca.components_.shape[1]) - pca.components_.shape[1] // 2) / 360 * 1000
    for k, c in zip(range(3), (plots.NAVY, plots.RED, plots.ORANGE)):
        a1.plot(t, pca.components_[k], color=c, label=f"component {k + 1} ({pca.explained_variance_ratio_[k]:.0%})")
    a1.set(xlabel="time around the beat (ms)", ylabel="weight", title="First three PCA components")
    a1.legend(loc="lower left", fontsize=12)

    rng = np.random.default_rng(0)
    for lab, c, name in ((0, plots.BLUE, "normal (N)"), (2, plots.RED, "ventricular (V)")):
        idx = rng.choice(np.flatnonzero(y1 == lab), 1500, replace=False)
        a2.scatter(x1[idx, 0], x1[idx, 1], s=6, alpha=0.35, color=c, label=name, edgecolors="none")
    a2.set(xlabel="feature 1", ylabel="feature 2", title="DS1 beats, first two features")
    leg = a2.legend(loc="upper right", markerscale=3)
    for h in leg.legend_handles:
        h.set_alpha(1)
    fig.tight_layout()
    fig.savefig(OUT / "pca_features.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
