# -*- coding: utf-8 -*-
"""Fig. 5: predictive variance as an uncertainty diagnostic on ML-1M.

(a) seed-averaged top-20 predictive variance of MIDGaP by user-activity decile, mean and
one standard deviation across the users of each decile; (b) per-user NDCG@20 difference
between MIDGaP (uncertainty runs) and LightGCN (five main-study seeds) by predictive-
variance decile, mean and t-based 95% confidence interval.  Decile cuts follow the
original analysis.
"""
import glob
import os

import matplotlib

matplotlib.use("Agg")
import numpy as np
from scipy import stats

import figcommon as C
import sciviz as sv

DATASET = "ml1m"


def deciles(v):
    q = np.quantile(v, np.linspace(0, 1, 11))
    q[-1] += 1
    return np.clip(np.digitize(v, q[1:-1]), 0, 9)


def build():
    C.setup()
    runs = C.jload(os.path.join(C.RESULTS, "uncertainty_%s.json" % DATASET))
    users = np.array(runs[0]["users"])
    deg = np.array(runs[0]["deg"])
    unc = np.mean([r["unc"] for r in runs], 0)
    nd_mid = np.mean([r["ndcg"] for r in runs], 0)
    base = []
    for f in sorted(glob.glob(os.path.join(C.RESULTS, "per_user_npz", "%s_LightGCN_s*.npz" % DATASET))):
        z = np.load(f)
        o = np.argsort(z["users"])
        uu = z["users"][o]
        base.append(z["ndcg20"][o][np.isin(uu, users)])
    gain = nd_mid - np.mean(base, 0)

    fig, axes = sv.figure(C.FINAL_MM, aspect=0.40, ncols=2)
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    st = C.series("MIDGaP")
    x = np.arange(1, 11)

    ia = deciles(deg)
    m_a = np.array([unc[ia == g].mean() for g in range(10)])
    s_a = np.array([unc[ia == g].std() for g in range(10)])
    n_a = [int((ia == g).sum()) for g in range(10)]
    ax_a.errorbar(x, m_a, yerr=s_a, color=st["color"], marker=st["marker"], linestyle="-", lw=2.0,
                  markersize=7, markeredgecolor="#ffffff", markeredgewidth=1.0, ecolor=sv.INK,
                  elinewidth=1.5, capsize=4, zorder=3)
    ax_a.set_xticks(x)
    ax_a.set_xlim(0.4, 10.6)
    ax_a.set_ylim(0, (m_a + s_a).max() * 1.08)
    ax_a.set_xlabel("User-activity decile")
    ax_a.set_ylabel("Top-20 predictive variance")

    ib = deciles(unc)
    m_b, ci_b, n_b = [], [], []
    for g in range(10):
        v = gain[ib == g]
        m_b.append(v.mean())
        ci_b.append(stats.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size))
        n_b.append(int(v.size))
    m_b, ci_b = np.array(m_b), np.array(ci_b)
    ax_b.bar(x, m_b, 0.68, color=st["color"], edgecolor=sv.INK, linewidth=1.2, zorder=2)
    ax_b.errorbar(x, m_b, yerr=ci_b, fmt="none", ecolor=sv.INK, elinewidth=1.5, capsize=4, zorder=3)
    ax_b.axhline(0, color=sv.INK, lw=1.0, ls="--", zorder=1)
    ax_b.set_xticks(x)
    ax_b.set_xlim(0.4, 10.6)
    ax_b.set_ylim((m_b - ci_b).min() * 1.12, 0.02)
    ax_b.set_xlabel("Predictive-variance decile")
    ax_b.set_ylabel(r"$\Delta$NDCG@20 vs LightGCN")
    for ax in (ax_a, ax_b):
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))
        sv.enforce_closed_frame(ax)
        C.exact_decimals(ax, "y")
    sv.panel_labels(fig, [ax_a, ax_b], lowercase=True)
    record = dict(dataset=DATASET, n_users=int(users.size), n_seeds_midgap=len(runs), n_seeds_lightgcn=len(base),
                  a=dict(mean=m_a, sd=s_a, n_per_decile=n_a), b=dict(mean=m_b, ci95=ci_b, n_per_decile=n_b))
    return fig, record


def main():
    fig, record = build()
    C.write_json("fig5_values.json", record)
    notes = dict(title="Predictive variance diagnostic", script="fig5_uncertainty.py",
                 data="results/uncertainty_ml1m.json, results/per_user_npz/ml1m_LightGCN_s<seed>.npz",
                 encodes="(a) mean +/- 1 SD across users per decile; (b) mean +/- t-based 95% CI per decile",
                 exceptions="single-series panels: no legend (SVIS 8, level 1)")
    return C.deliver(fig, 5, "Figure_5", notes)


if __name__ == "__main__":
    print(main())
