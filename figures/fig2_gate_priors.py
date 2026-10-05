# -*- coding: utf-8 -*-
"""Fig. 2: gate prior that each scoring criterion assigns to every meta-path.

Rows are benchmarks, columns node sides.  Horizontal grouped bars keep the meta-path
names horizontal (SVIS 7.3); all panels share the 0 to 1 weight axis.  The priors are
deterministic functions of the DCMI statistics (Table 5), so no error bars apply.
"""
import os

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.patches import Patch

import figcommon as C
import sciviz as sv

DATASETS = ["ml100k", "ml1m", "lastfm", "amazon"]
CRITERIA = [("dcmi", "DCMI"), ("mi", "raw MI"), ("density", "density")]     # data order = legend order
H_BAR = 0.25


def build():
    from dcmi import gate_prior
    C.setup()
    stats = {d: C.jload(os.path.join(C.CACHE, "%s_dcmi.json" % d)) for d in DATASETS}
    rows_n = [max(len(stats[d]["u"]), len(stats[d]["i"])) for d in DATASETS]
    fig, axes = sv.figure(C.FINAL_MM, aspect=0.80, nrows=4, ncols=2, sharex=True,
                          height_ratios=rows_n)
    record = {}
    for r, d in enumerate(DATASETS):
        for c, side in enumerate(("u", "i")):
            ax = axes[r, c]
            names = list(stats[d][side])
            off = (rows_n[r] - len(names)) / 2.0
            ys = -(np.arange(len(names)) + off)
            for j, (kind, cond) in enumerate(CRITERIA):
                pr = gate_prior(stats[d][side], kind)
                vals = [pr[n] for n in names]
                record.setdefault(d, {}).setdefault(side, {})[cond] = dict(zip(names, map(float, vals)))
                ax.barh(ys + (1 - j) * H_BAR, vals, height=H_BAR * 0.92, **C.bar_kw(cond))
            ax.set_yticks(ys, names)
            ax.set_ylim(-rows_n[r] + 0.45, 0.55)
            ax.set_xlim(0, 1.0)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.tick_params(axis="y", length=0)
            if c == 0:
                ax.set_ylabel(C.DS_NAME[d])
            if r == 0:
                ax.set_title("User side" if side == "u" else "Item side")
            if r == len(DATASETS) - 1:
                ax.set_xlabel("Gate prior weight")
            sv.enforce_closed_frame(ax)
            C.exact_decimals(ax, "x")
    handles = [Patch(**{k: v for k, v in C.bar_kw(cond).items() if k != "color"},
                     facecolor=C.color(cond)) for _, cond in CRITERIA]
    sv.figure_legend(fig, handles, [C.LABEL[cond] for _, cond in CRITERIA], where="top", ncol=3)
    fig.align_ylabels(axes[:, 0])
    sv.panel_labels(fig, axes, lowercase=True)
    return fig, record


def main():
    fig, record = build()
    C.write_json("fig2_values.json", record)
    notes = dict(title="Gate priors by criterion", script="fig2_gate_priors.py",
                 data="cache/<dataset>_dcmi.json through code/dcmi.py gate_prior()",
                 encodes="prior weight per meta-path; deterministic, no error bars",
                 exceptions="horizontal bars so that meta-path names stay horizontal (SVIS 7.3)")
    return C.deliver(fig, 2, "Figure_2", notes)


if __name__ == "__main__":
    print(main())
