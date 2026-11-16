"""Figures comparing the a + b mod 113 baseline with the elliptic-curve run.

  figures/check_baseline_vs_nanda.png  the baseline against the curves saved in Nanda et al.'s
                                       notebook (data/nanda2023_fig7_traces.json)
  figures/post_modadd_vs_ec.png        both runs: loss, accuracy, Gini, weight norm,
                                       shared y-axes
  figures/nanda_vs_ec.png              Nanda et al. Figure 7 next to the elliptic-curve run

Run from the repository root, after progress_measures.py:
    python compare.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from transformer import CKPT

HERE = Path(__file__).parent
FIGURES = HERE / "figures"
NANDA_TRACES = HERE / "data" / "nanda2023_fig7_traces.json"
BLUE, RED, GREEN = "#636EFA", "#EF553B", "#00CC96"

NANDA_PHASES = [1400, 9400, 14000]     # Nanda et al., Figure 7
XMAX = 40_000


def load(task):
    return torch.load(CKPT / f"{task}_seed0" / "measures.pt")


def style(ax, title, ylabel, phases, log=False, xmax=XMAX):
    ax.set_facecolor("#E5ECF6")
    ax.grid(color="white", linewidth=1)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, color="#2a3f5f", fontsize=13)
    ax.set_xlabel("Epoch", color="#2a3f5f")
    ax.set_ylabel(ylabel, color="#2a3f5f")
    ax.set_xlim(0, xmax)
    ticks = list(range(0, xmax + 1, 5000))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t // 1000}k" if t else "0" for t in ticks])
    if log:
        ax.set_yscale("log")
    for x in phases:
        ax.axvline(x, color="#555", linestyle="--", linewidth=1.2)



def check_baseline(m):
    ref = json.loads(NANDA_TRACES.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    pairs = [
        (axes[0], [("Train Loss", "train", BLUE), ("Test Loss", "test", RED), ("Excluded Loss", "excluded", GREEN)],
         ref["excluded_panel"], "Train, test and excluded loss", True),
        (axes[1], [("Restricted loss", "restricted", GREEN)], ref["restricted_panel"], "Restricted loss", True),
        (axes[2], [("value", "sumsq", BLUE)], ref["sumsq_panel"], "Total sum of squared weights", False),
    ]
    for ax, series, panel, title, log in pairs:
        for ref_key, key, color in series:
            ax.plot(panel[ref_key]["x"], panel[ref_key]["y"], color=color, linewidth=3, alpha=0.35)
            ax.plot(m["epoch"], [max(v, 1e-12) for v in m[key]], color=color, linewidth=1)
        style(ax, title, "", NANDA_PHASES, log=log, xmax=40_000)
    axes[0].plot([], [], color="gray", linewidth=3, alpha=0.35, label="Nanda et al. (saved in their notebook)")
    axes[0].plot([], [], color="gray", linewidth=1, label="this baseline")
    axes[0].legend(fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig(FIGURES / "check_baseline_vs_nanda.png", dpi=150)
    plt.close(fig)



def history(task):
    return torch.load(CKPT / f"{task}_seed0" / "history.pt")["history"]


def post_figure(base, ec):
    """Loss, accuracy, Gini and weight norm for both runs, with shared y-axes.

    Excluded and restricted loss are left out: their key frequencies are taken from the final
    model, and for a model that has not grokked they are not defined."""
    hists = {"modadd": history("modadd"), "ec": history("ec")}
    fig = plt.figure(figsize=(20, 9), layout="constrained")
    sub = fig.subfigures(1, 2, wspace=0.03)
    halves = [
        (sub[0], base, hists["modadd"], NANDA_PHASES,
         "a + b mod 113, input  a b =\nNanda et al. (2023): their architecture, data split and initialisation, rerun"),
        (sub[1], ec, hists["ec"], [],
         "P + Q on E(F$_{113}$), E = 61a1, input  x$_1$ y$_1$ x$_2$ y$_2$ =\nsame architecture, data split and initialisation"),
    ]
    all_axes = []
    for sf, m, h, phases, title in halves:
        axes = sf.subplots(2, 2).flatten()
        all_axes.append(axes)
        e = m["epoch"]
        ax = axes[0]
        ax.plot(e, m["train"], color=BLUE, label="Train loss")
        ax.plot(e, m["test"], color=RED, label="Test loss")
        style(ax, "Train and Test Loss", "Loss", phases, log=True)
        ax.legend(fontsize=9, loc="center right")
        ax = axes[1]
        ax.plot(h["epoch"], h["train_acc"], color=BLUE, label="Train accuracy")
        ax.plot(h["epoch"], h["test_acc"], color=RED, label="Test accuracy")
        style(ax, "Train and Test Accuracy", "Accuracy", phases)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, loc="center right")
        ax = axes[2]
        ax.plot(e, m["gini_E"], color=BLUE, label="Embed matrix")
        ax.plot(e, m["gini_L"], color=RED, label="Neuron-logit map")
        style(ax, "Gini Coefficients of Embed Matrix and Neuron-logit Map", "Gini coefficient", phases)
        ax.title.set_fontsize(11)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9, loc="upper left")
        ax = axes[3]
        ax.plot(e, m["sumsq"], color=BLUE)
        style(ax, "Total Sum of Squared Weights", "Sum of squared weights", phases)
        sf.suptitle(title, fontsize=15)
    for i in (0, 3):
        lo = min(min(a[i].get_ylim()) for a in all_axes)
        hi = max(max(a[i].get_ylim()) for a in all_axes)
        for a in all_axes:
            a[i].set_ylim(lo, hi)
    fig.savefig(FIGURES / "post_modadd_vs_ec.png", dpi=150)
    plt.close(fig)



if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    base, ec = load("modadd"), load("ec")
    check_baseline(base)
    post_figure(base, ec)
    print("saved figures in", FIGURES)
