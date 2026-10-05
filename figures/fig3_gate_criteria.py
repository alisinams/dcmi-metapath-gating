# -*- coding: utf-8 -*-
"""Fig. 3: gate criteria against the gate the ranking loss learns without a prior.

(a) item-side gate weight on the route the free gate selects; (b) NDCG@20 relative to
the DCMI anchor.  Bars are means over the three seeds of the gate study, error bars one
standard deviation (numpy default, as in the manuscript tables) and open circles the
individual seeds (n < 30, SVIS 9.2).
"""
import glob
import os

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import figcommon as C
import sciviz as sv

DATASETS = ["ml1m", "lastfm", "amazon"]
VARIANT = {"DCMI": "DCMI (full)", "raw MI": "raw MI", "density": "density prior", "uniform": "uniform",
           "free gate": "free gate"}
ORDER = ["DCMI", "raw MI", "density", "uniform", "free gate"]
W = 0.155


def load():
    rows = sum((C.jl(f) for f in sorted(glob.glob(os.path.join(C.RESULTS, "ablation_gate_*.jsonl")))), [])
    return [r for r in rows if r["dataset"] in DATASETS]


def build():
    C.setup()
    rows = load()
    fig, axes = sv.figure(C.FINAL_MM, aspect=0.44, ncols=2)
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    record = {"route": {}, "a": {}, "b": {}, "n_seeds": {}}
    x = np.arange(len(DATASETS))

    for k, d in enumerate(DATASETS):
        free = [r["gate_i"] for r in rows if r["dataset"] == d and r["variant"] == VARIANT["free gate"]]
        mean = {m: np.mean([g[m] for g in free]) for m in free[0]}
        route = max(mean, key=mean.get)
        record["route"][d] = route
        base = [r["ndcg@20"] for r in rows if r["dataset"] == d and r["variant"] == VARIANT["DCMI"]]
        for j, cond in enumerate(ORDER):
            sel = [r for r in rows if r["dataset"] == d and r["variant"] == VARIANT[cond]]
            record["n_seeds"].setdefault(d, {})[cond] = len(sel)
            xc = k + (j - 2) * W
            # (a) weight on the free gate's route
            g = np.array([r["gate_i"][route] for r in sel])
            ax_a.bar(xc, g.mean(), W * 0.92, **C.bar_kw(cond), zorder=2)
            ax_a.errorbar(xc, g.mean(), yerr=g.std(), color=sv.INK, capsize=3, lw=1.5, zorder=4)
            C.seed_points(ax_a, xc, g, W)
            record["a"].setdefault(d, {})[cond] = dict(mean=g.mean(), sd=g.std(), seeds=g)
            # (b) accuracy relative to the DCMI anchor
            v = np.array([r["ndcg@20"] for r in sel])
            rel = 100 * (v / np.mean(base) - 1)
            ax_b.bar(xc, rel.mean(), W * 0.92, **C.bar_kw(cond), zorder=2)
            ax_b.errorbar(xc, rel.mean(), yerr=rel.std(), color=sv.INK, capsize=3, lw=1.5, zorder=4)
            C.seed_points(ax_b, xc, rel, W)
            record["b"].setdefault(d, {})[cond] = dict(mean=rel.mean(), sd=rel.std(), seeds=rel)
        f = record["a"][d]["free gate"]["mean"]
        ax_a.plot([k - 2.6 * W, k + 2.6 * W], [f, f], color=C.color("free gate"), lw=1.6, ls=(0, (4, 2)), zorder=3)

    ax_a.set_xticks(x, ["%s\n%s" % (C.DS_NAME[d], record["route"][d]) for d in DATASETS])
    ax_a.set_ylim(0, 1.05)
    ax_a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_a.set_ylabel("Weight on free-gate route")
    C.exact_decimals(ax_a, "y")

    ax_b.axhline(0, color=sv.INK, lw=1.0, ls="--", zorder=1)
    ax_b.set_xticks(x, [C.DS_NAME[d] for d in DATASETS])
    lo = min(min(np.min(v["seeds"]) for v in record["b"][d].values()) for d in DATASETS)
    hi = max(max(np.max(v["seeds"]) for v in record["b"][d].values()) for d in DATASETS)
    ax_b.set_ylim(np.floor(lo / 2) * 2 - 1, np.ceil(hi / 2) * 2 + 1)
    ax_b.set_ylabel("NDCG@20 relative to DCMI (%)")
    C.exact_decimals(ax_b, "y")
    for ax in (ax_a, ax_b):
        ax.set_xlim(-0.55, len(DATASETS) - 0.45)
        sv.enforce_closed_frame(ax)

    handles = [Patch(**{kk: vv for kk, vv in C.bar_kw(c).items() if kk != "color"}, facecolor=C.color(c))
               for c in ORDER]
    labels = [C.LABEL[c] for c in ORDER]
    handles.append(Line2D([], [], color=sv.INK, lw=0, marker="o", markerfacecolor="#ffffff",
                          markeredgecolor=sv.INK, markersize=6))
    labels.append("Seed")
    sv.figure_legend(fig, handles, labels, where="top", ncol=6)
    sv.panel_labels(fig, [ax_a, ax_b], lowercase=True)
    return fig, record


def main():
    fig, record = build()
    C.write_json("fig3_values.json", record)
    notes = dict(title="Gate criteria versus the free gate", script="fig3_gate_criteria.py",
                 data="results/ablation_gate_<dataset>.jsonl",
                 encodes="bars = mean over n = 3 seeds; error bars = 1 SD; circles = seeds; dashed line = free-gate mean",
                 exceptions="none")
    return C.deliver(fig, 3, "Figure_3", notes)


if __name__ == "__main__":
    print(main())
