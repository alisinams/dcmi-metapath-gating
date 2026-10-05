# -*- coding: utf-8 -*-
"""Fig. 4: NDCG@20 by user-activity quintile of the training window.

Quintiles are cut on the training interaction count of the evaluated users, exactly as
in code/tables.py; lines are means over the five seeds of the main study and bands one
standard deviation.  Panels keep independent y scales (the benchmarks differ by an order
of magnitude in NDCG) and zero is shown on every panel.
"""
import glob
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import numpy as np

import figcommon as C
import sciviz as sv

DATASETS = ["ml1m", "lastfm", "amazon"]
MODELS = ["MostPop", "LightGCN", "XSimGCL", "HDGCN", "MIDGaP"]


def activity(d):
    p = os.path.join(C.RESULTS, "train_activity.json")
    if os.path.exists(p):
        return np.array(C.jload(p)[d])
    R = pickle.load(open(os.path.join(C.CACHE, d + ".pkl"), "rb"))["R"].tocsr()
    return np.asarray(R.sum(1)).ravel()


def quintile_curves(d, model, deg):
    curves, counts = [], None
    for f in sorted(glob.glob(os.path.join(C.RESULTS, "per_user_npz", "%s_%s_s*.npz" % (d, model)))):
        z = np.load(f)
        u, r = z["users"], z["ndcg20"]
        q = np.quantile(deg[u], [0, .2, .4, .6, .8, 1.0])
        q[-1] += 1
        idx = np.clip(np.digitize(deg[u], q[1:-1]), 0, 4)
        curves.append([r[idx == g].mean() if (idx == g).any() else np.nan for g in range(5)])
        counts = [int((idx == g).sum()) for g in range(5)]
    return np.array(curves), counts


def build():
    C.setup()
    fig, axes = sv.figure(C.FINAL_MM, aspect=0.40, ncols=3)
    record = {}
    handles = []
    x = np.arange(5)
    for k, d in enumerate(DATASETS):
        ax = axes[0, k]
        deg = activity(d)
        for m in MODELS:
            c, counts = quintile_curves(d, m, deg)
            st = C.series(m)
            mu, sd = c.mean(0), c.std(0)
            ax.fill_between(x, mu - sd, mu + sd, color=sv.with_alpha(C.color(m), "band_sd"), linewidth=0, zorder=1)
            h, = ax.plot(x, mu, color=st["color"], marker=st["marker"], linestyle=st["linestyle"], lw=2.0,
                         markersize=7, markerfacecolor=st["color"], markeredgecolor="#ffffff",
                         markeredgewidth=1.0, label=m, zorder=3)
            if k == 0:
                handles.append(h)
            record.setdefault(d, {})[m] = dict(mean=mu, sd=sd, n_seeds=len(c))
            record[d]["n_users_per_quintile"] = counts
            record[d]["n_users"] = int(sum(counts))
        top = max(np.nanmax(record[d][m]["mean"] + record[d][m]["sd"]) for m in MODELS)
        ax.set_ylim(-0.035 * top, top * 1.08)      # zero stays a labelled tick; markers at 0 are not clipped
        ax.set_xticks(x, ["Q%d" % (g + 1) for g in range(5)])
        ax.set_xlim(-0.3, 4.3)
        ax.set_title(C.DS_NAME[d])
        ax.set_xlabel("Activity quintile")
        if k == 0:
            ax.set_ylabel("NDCG@20")
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))
        sv.enforce_closed_frame(ax)
        C.exact_decimals(ax, "y")
    sv.figure_legend(fig, handles, MODELS, where="top", ncol=5)
    sv.panel_labels(fig, axes, lowercase=True)
    return fig, record


def main():
    fig, record = build()
    C.write_json("fig4_values.json", record)
    notes = dict(title="Accuracy by user-activity quintile", script="fig4_activity_quintiles.py",
                 data="results/per_user_npz/<dataset>_<model>_s<seed>.npz, results/train_activity.json",
                 encodes="line = mean over n = 5 seeds; band = 1 SD; quintiles of training activity among evaluated users",
                 exceptions="independent y scales across panels, all from zero (SVIS 7.4 exception, stated in caption)")
    return C.deliver(fig, 4, "Figure_4", notes)


if __name__ == "__main__":
    print(main())
