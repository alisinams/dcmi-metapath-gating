# -*- coding: utf-8 -*-
"""Fig. 1: the MIDGaP pipeline (schematic).

Two swim lanes: the offline estimator (meta-path subgraphs, density-corrected mutual
information, gate prior) and the recommender trained end to end (embeddings, gated
meta-path propagation, variational bottleneck), whose posterior gives both the ranking
score and the predictive variance.  Connectors are orthogonal; their line style is
declared inside the graphic.
"""
import matplotlib

matplotlib.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import figcommon as C
import sciviz as sv

W = C.FINAL_MM / sv.MAX_REDUCTION / 25.4      # design canvas width (in), 9.79
H = 4.32                                      # design canvas height (in)
FILL_OFFLINE, FILL_MODEL = 3, 4               # light swatches for diagram nodes (SVIS 11)


def build():
    pal = C.setup()
    fig, ax = sv.figure(C.FINAL_MM, aspect=H / W)
    fig.set_layout_engine("none")
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    pairs = []

    def box(x, y, w, h, title, body, fill):
        face = "#ffffff" if fill is None else sv.get_sequential_color(fill)
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                           linewidth=sv.FRAME_LW, edgecolor=sv.INK, facecolor=face, zorder=3)
        ax.add_patch(p)
        n_t, n_b = title.count("\n") + 1, body.count("\n") + 1
        gap = 0.07
        t_h = n_t * 0.21
        b_h = n_b * 0.19
        top = y + h / 2 + (t_h + gap + b_h) / 2
        t = ax.text(x + w / 2, top - t_h / 2, title, ha="center", va="center", fontsize=sv.FS["L2"],
                    fontweight="bold", color=sv.INK, linespacing=1.05, zorder=4)
        b = ax.text(x + w / 2, top - t_h - gap - b_h / 2, body, ha="center", va="center",
                    fontsize=sv.FS["L1"], color=sv.INK, linespacing=1.15, zorder=4)
        pairs.append((p, [t, b]))
        return dict(x0=x, x1=x + w, y0=y, y1=y + h, xm=x + w / 2, ym=y + h / 2)

    def connector(pts, ls="-"):
        """Orthogonal polyline; the shaft carries the line style, the head is always solid."""
        xs, ys = zip(*pts)
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        d = 0.16
        hx = x2 - d * (1 if x2 > x1 else -1 if x2 < x1 else 0)
        hy = y2 - d * (1 if y2 > y1 else -1 if y2 < y1 else 0)
        ax.add_line(Line2D(list(xs[:-1]) + [hx], list(ys[:-1]) + [hy], color=sv.INK, lw=2.0,
                           linestyle=ls, solid_capstyle="butt", dash_capstyle="butt", zorder=2))
        ax.add_patch(FancyArrowPatch((hx, hy), (x2, y2), arrowstyle="-|>", mutation_scale=18,
                                     lw=2.0, color=sv.INK, shrinkA=0, shrinkB=0, zorder=2))

    # swim lanes: tinted bands with vertical labels at the left edge, outside the flow column
    lanes = [(0.02, 1.86, W - 0.04, 2.42, FILL_MODEL, "Recommender"),
             (0.02, 0.18, 6.62, 1.48, FILL_OFFLINE, "Offline")]
    for x, y, w, h, idx, lab in lanes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=sv.with_alpha(sv.get_sequential_color(idx), 0.30),
                               edgecolor="none", zorder=0))
        ax.text(0.24, y + h / 2, lab, rotation=90, ha="center", va="center", fontsize=sv.FS["L2"],
                fontweight="bold", color=sv.INK, zorder=1)

    ym, yo = 2.95, 0.92                      # lane centre lines
    bh_m, bh_o = 1.06, 1.02
    A = dict(x=0.56, w=1.45)
    B = dict(x=2.41, w=1.75)
    Cc = dict(x=4.56, w=1.95)
    D = dict(x=6.91, w=1.25)
    E = dict(x=8.58, w=1.17)

    g = box(A["x"], ym - bh_m / 2, A["w"], bh_m, "Heterogeneous\ngraph", "users, items,\nattributes", None)
    emb = box(B["x"], ym - bh_m / 2, B["w"], bh_m, "Content and ID\nembeddings",
              r"$\mathit{e_v}=\mathbf{E}_\mathit{v}+\mathit{P\,f_v}$", FILL_MODEL)
    gate = box(Cc["x"], ym - bh_m / 2, Cc["w"], bh_m, "Gated meta-path\npropagation",
               r"$\mathit{h_v}=\mathit{h}_\mathit{v}^{0}+\Sigma_\mathit{m}\,\mathit{g_m}\,\hat{\mathit{A}}^{\mathit{P_m}}\mathit{e_v}$",
               FILL_MODEL)
    vib = box(D["x"], ym - bh_m / 2, D["w"], bh_m, "Variational\nbottleneck", r"$\mu_\mathit{v},\ \sigma_\mathit{v}^{2}$", FILL_MODEL)
    rank = box(E["x"], 3.30, E["w"], 0.88, "Ranking", r"$\mathit{z}_\mathit{u}^{\top}\mathit{z_i}$", None)
    unc = box(E["x"], 1.98, E["w"], 1.06, "Predictive\nvariance", r"$\mathit{S}=16$ draws", None)

    mp = box(A["x"], yo - bh_o / 2, A["w"], bh_o, "Meta-path\nsubgraphs",
             r"$\mathit{A}^{\mathit{P}_1},\dots,\mathit{A}^{\mathit{P_M}}$", FILL_OFFLINE)
    dc = box(B["x"], yo - bh_o / 2, B["w"], bh_o, "DCMI score",
             "MI $\\mathit{z}$-score against\n40 null graphs", FILL_OFFLINE)
    pr = box(Cc["x"], yo - bh_o / 2, Cc["w"], bh_o, r"Gate prior $\pi$",
             "tempered softmax\nof $\\mathit{z}$-scores", FILL_OFFLINE)

    connector([(g["x1"], ym), (emb["x0"], ym)])
    connector([(g["xm"], g["y0"]), (g["xm"], mp["y1"])])
    connector([(mp["x1"], yo), (dc["x0"], yo)])
    connector([(dc["x1"], yo), (pr["x0"], yo)])
    connector([(emb["x1"], ym), (gate["x0"], ym)])
    connector([(gate["x1"], ym), (vib["x0"], ym)])
    connector([(pr["xm"], pr["y1"]), (pr["xm"], gate["y0"])], ls=(0, (1, 1.6)))
    xj = (vib["x1"] + rank["x0"]) / 2
    connector([(vib["x1"], ym + 0.24), (xj, ym + 0.24), (xj, rank["ym"]), (rank["x0"], rank["ym"])])
    connector([(vib["x1"], ym - 0.24), (xj, ym - 0.24), (xj, unc["ym"] - 0.12), (unc["x0"], unc["ym"] - 0.12)],
              ls=(0, (4, 2)))
    ax.text(pr["xm"] + 0.10, (pr["y1"] + gate["y0"]) / 2, r"$\mathrm{KL}(\mathit{g}\,\|\,\pi)$", ha="left",
            va="center", fontsize=sv.FS["L1"], color=sv.INK, zorder=4)

    # line semantics, declared inside the graphic (right of the offline lane)
    handles = [Line2D([], [], color=sv.INK, lw=2.0, ls="-"),
               Line2D([], [], color=sv.INK, lw=2.0, ls=(0, (1, 1.6))),
               Line2D([], [], color=sv.INK, lw=2.0, ls=(0, (4, 2)))]
    labels = ["Forward pass", "Training-time anchor", "Inference-time sampling"]
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(6.86 / W, yo / H), ncol=1, frameon=False,
              fontsize=sv.FS["L2"], handlelength=2.6, borderaxespad=0.0, labelspacing=0.55)
    sv.finalize(fig)
    return fig, C.text_inside(fig, pairs)


def main():
    fig, inside = build()
    notes = dict(title="MIDGaP pipeline", script="fig1_pipeline.py", data="none (schematic)",
                 encodes="schematic; solid = forward pass, dotted = training-time KL anchor, dashed = inference-time sampling",
                 exceptions="diagram: no data axes, 2 pt outlines on boxes (SVIS 6, 11)")
    return C.deliver(fig, 1, "Figure_1", notes, extra_problems=inside)


if __name__ == "__main__":
    print(main())
