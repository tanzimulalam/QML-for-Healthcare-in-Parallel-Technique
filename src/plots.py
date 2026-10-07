"""Common look for all figures (Times New Roman so they match the report)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, BLUE, LIGHT_BLUE = "#1F3864", "#2E5C9A", "#8FAADC"
RED, ORANGE, YELLOW = "#C0392B", "#C55A11", "#E0A100"
GREY, GRID = "#7F7F7F", "#D9D9D9"


def setup():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#8C8C8C",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "lines.markersize": 7,
        "savefig.dpi": 200,
    })
    return plt


def ideal(ax, xs):
    ax.plot(xs, xs, color=GREY, lw=1.2, ls=(0, (1, 3)), label="ideal")
