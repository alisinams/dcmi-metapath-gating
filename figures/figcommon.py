# -*- coding: utf-8 -*-
"""Shared configuration for the manuscript figures.

Paths, the condition-to-colour mapping that every figure reuses, bar and line
encodings, a few drawing helpers and the journal export (Springer: Fig<N>.eps /
.pdf / .tif / .png at the final width of 174 mm).  The house style itself lives
in sciviz.py and is not modified here.
"""
import io
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import Collection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import sciviz as sv

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.environ.get("FIG_RESULTS", os.path.join(REPO, "results"))
CACHE = os.environ.get("FIG_CACHE", os.path.join(REPO, "cache"))
CODE = os.environ.get("FIG_CODE", os.path.join(REPO, "code"))
OUT = os.environ.get("FIG_OUT", os.path.join(HERE, "output"))
JOURNAL_OUT = os.environ.get("FIG_JOURNAL_OUT", os.path.join(OUT, "journal"))
sys.path.insert(0, CODE)

FINAL_MM = 174.0          # Springer large-sized journal, full text width
JOURNAL_DPI = 600         # combination art (colour plots with lettering)
DS_NAME = {"ml100k": "ML-100K", "ml1m": "ML-1M", "lastfm": "Last.FM", "amazon": "Amazon-VG"}

# One palette index per condition, used in every figure (SVIS 4.2.7).  The proposed
# estimator and the model built on it share an index, as do the density prior and the
# density-weighted InfoMax model that uses it.
COND_INDEX = {
    "DCMI": 1, "MIDGaP": 1,
    "raw MI": 3,
    "density": 6, "HDGCN": 6,
    "uniform": 4,
    "free gate": 8,
    "MostPop": 0,
    "XSimGCL": 2,
    "LightGCN": 9,
}
LABEL = {"DCMI": "DCMI", "raw MI": "Raw MI", "density": "Density prior", "uniform": "Uniform",
         "free gate": "Free gate"}
# Fill patterns give bars a second, colour-independent code (greyscale print, colour-vision deficiency).
HATCH = {"DCMI": "", "raw MI": "//", "density": "\\\\", "uniform": "..", "free gate": "xx"}


def setup():
    """Palette resolution (RULES/Ocean Sunset.ase when the project tree is present) and global style."""
    rules = os.environ.get("RULES_DIR") or sv.find_rules_dir(start=HERE)
    pal = sv.setup(rules_dir=str(rules) if rules else None)
    if pal.source.startswith("ASE file "):           # keep machine paths out of file metadata and reports
        pal.source = "ASE file RULES/" + os.path.basename(pal.source[len("ASE file "):])
    return pal


def color(cond):
    return sv.get_sequential_color(COND_INDEX[cond])


def series(cond, alpha=1.0):
    """Colour, marker and line style of a condition (redundant encoding, SVIS 4.2.6)."""
    return sv.active_palette().series(COND_INDEX[cond], alpha)


def bar_kw(cond):
    """Solid palette fill, black outline, pattern in white on dark fills and in black on light ones."""
    c = color(cond)
    dark = sv.srgb_to_lab(c)[0] < 50
    return dict(color=c, edgecolor=sv.INK, linewidth=1.2, hatch=HATCH[cond],
                hatchcolor="#ffffff" if dark else sv.INK)


def seed_points(ax, x, values, width):
    """Every observation (n < 30): open circles spread evenly across the bar."""
    v = np.asarray(values, float)
    xs = x + (np.linspace(-0.22, 0.22, v.size) * width if v.size > 1 else np.zeros(1))
    ax.scatter(xs, v, s=22, marker="o", facecolors="#ffffff", edgecolors=sv.INK, linewidths=1.1, zorder=5)


def exact_decimals(ax, axis="y", max_dec=4):
    """One decimal convention per axis (SVIS 5.3) with labels that equal the tick values:
    the smallest number of decimals that represents every tick exactly, true minus sign.
    (sv.fixed_decimals derives decimals from the step and prints a 1.5 step as integers.)"""
    from matplotlib.ticker import FuncFormatter
    target = ax.yaxis if axis == "y" else ax.xaxis
    lo, hi = sorted(target.get_view_interval())
    ticks = [t for t in target.get_majorticklocs() if lo - 1e-12 <= t <= hi + 1e-12]
    dec = next(d for d in range(max_dec + 1) if all(abs(round(t, d) - t) < 1e-9 for t in ticks) or d == max_dec)
    target.set_major_formatter(FuncFormatter(lambda v, _p: ("%.*f" % (dec, v)).replace("-", "−")
                                             if abs(v) >= 0.5 * 10 ** -dec else "%.*f" % (dec, 0.0)))
    return dec


def jl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")] if os.path.exists(path) else []


def jload(path):
    return json.load(open(path, encoding="utf-8"))


def py(obj):
    """numpy scalars and arrays to plain Python, for JSON."""
    if isinstance(obj, dict):
        return {str(k): py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return py(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def write_json(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(py(obj), f, indent=1, ensure_ascii=False)


# ---------------------------------------------------------------------------- journal export
def _tight_width_in(fig, pad_in=0.05):
    fig.canvas.draw()
    return fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * pad_in


def _blend(rgba, bg=(1.0, 1.0, 1.0)):
    r, g, b, a = mcolors.to_rgba(rgba)
    return (r * a + bg[0] * (1 - a), g * a + bg[1] * (1 - a), b * a + bg[2] * (1 - a), 1.0)


def flatten_alpha(fig):
    """EPS has no transparency: replace every translucent colour by its blend on white
    (visually identical on a white page; only used for the EPS file, written last)."""
    for art in fig.findobj():
        if isinstance(art, Line2D):
            for get, put in ((art.get_color, art.set_color), (art.get_markerfacecolor, art.set_markerfacecolor),
                             (art.get_markeredgecolor, art.set_markeredgecolor)):
                c = get()
                if isinstance(c, str) and c in ("none", "None"):
                    continue
                a = art.get_alpha()
                rgba = mcolors.to_rgba(c, a if a is not None else None)
                if rgba[3] < 1:
                    put(_blend(rgba))
            art.set_alpha(None)
        elif isinstance(art, Collection):
            a = art.get_alpha()
            for get, put in ((art.get_facecolor, art.set_facecolor), (art.get_edgecolor, art.set_edgecolor)):
                cols = np.atleast_2d(get())
                if cols.size == 0:
                    continue
                cols = [mcolors.to_rgba(c[:3], c[3] if a is None else c[3] * a) for c in cols]
                put([_blend(c) for c in cols])
            art.set_alpha(None)
        elif isinstance(art, Patch):
            a = art.get_alpha()
            for get, put in ((art.get_facecolor, art.set_facecolor), (art.get_edgecolor, art.set_edgecolor)):
                c = get()
                rgba = mcolors.to_rgba(c[:3], c[3] if a is None else c[3] * a)
                if rgba[3] == 0:
                    continue
                put(_blend(rgba))
            art.set_alpha(None)


def _scale_eps(raw, s):
    """Scale an EPS produced by matplotlib by s about the origin: bounding boxes are
    rescaled and one 'scale' operator is inserted before the page content."""
    lines = raw.decode("latin-1").split("\n")
    out = []
    after_prolog = inserted = False
    for ln in lines:
        if ln.startswith("%%BoundingBox:"):
            x0, y0, x1, y1 = (float(v) for v in ln.split()[1:5])
            ln = "%%%%BoundingBox: %d %d %d %d" % (int(np.floor(x0 * s)), int(np.floor(y0 * s)),
                                                   int(np.ceil(x1 * s)), int(np.ceil(y1 * s)))
        elif ln.startswith("%%HiResBoundingBox:"):
            x0, y0, x1, y1 = (float(v) for v in ln.split()[1:5])
            ln = "%%%%HiResBoundingBox: %.6f %.6f %.6f %.6f" % (x0 * s, y0 * s, x1 * s, y1 * s)
        out.append(ln)
        if ln.strip() == "%%EndProlog":
            after_prolog = True
        elif after_prolog and not inserted and ln.strip() == "mpldict begin":
            out.append("%.6f %.6f scale" % (s, s))          # page content starts here
            inserted = True
    if not inserted:
        raise RuntimeError("unexpected EPS layout; cannot insert the scale operator")
    return "\n".join(out).encode("latin-1")


def journal_export(fig, n, outdir=None, pad_in=0.05):
    """Fig<n>.pdf (vector), Fig<n>.tif (600 dpi RGB, LZW), Fig<n>.png (600 dpi RGB) and
    Fig<n>.eps (vector, alpha pre-blended), all exactly FINAL_MM wide.  The design canvas is
    scaled by s = FINAL_MM / tight width (about 0.70, SVIS 5.2), so 12 pt prints at 8.4 pt."""
    import fitz
    from PIL import Image

    outdir = outdir or JOURNAL_OUT
    os.makedirs(outdir, exist_ok=True)
    sv.finalize(fig)
    w_in = _tight_width_in(fig, pad_in)
    s = (FINAL_MM / 25.4) / w_in
    stem = os.path.join(outdir, "Fig%d" % n)
    meta = {"Title": "Fig%d" % n, "Creator": "sciviz / matplotlib " + matplotlib.__version__}

    # vector PDF at design size, then the page is scaled to the final width (fonts stay embedded)
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight", pad_inches=pad_in, facecolor="white",
                metadata={**meta, "CreationDate": None})
    src = fitz.open("pdf", buf.getvalue())
    r = src[0].rect
    dst = fitz.open()
    page = dst.new_page(width=r.width * s, height=r.height * s)
    page.show_pdf_page(page.rect, src, 0)
    dst.set_metadata({"title": "Fig%d" % n, "creator": meta["Creator"], "producer": "PyMuPDF"})
    dst.save(stem + ".pdf", garbage=4, deflate=True)

    # raster at exactly JOURNAL_DPI over the final width
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=JOURNAL_DPI * s, bbox_inches="tight", pad_inches=pad_in,
                facecolor="white")
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    im.save(stem + ".tif", compression="tiff_lzw", dpi=(JOURNAL_DPI, JOURNAL_DPI))
    im.save(stem + ".png", dpi=(JOURNAL_DPI, JOURNAL_DPI), optimize=True)

    # EPS last, because flattening alpha changes the figure
    flatten_alpha(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="eps", bbox_inches="tight", pad_inches=pad_in, facecolor="white")
    with open(stem + ".eps", "wb") as f:
        f.write(_scale_eps(buf.getvalue(), s))
    return dict(scale=round(s, 4), width_mm=round(im.size[0] / JOURNAL_DPI * 25.4, 1),
                height_mm=round(im.size[1] / JOURNAL_DPI * 25.4, 1), pixels=list(im.size))


def text_inside(fig, pairs, tol_px=0.5):
    """Diagram check (SVIS 11): every text must sit fully inside its box."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    bad = []
    for box, texts in pairs:
        bb = box.get_window_extent(rend)
        for t in texts:
            tb = t.get_window_extent(rend)
            if (tb.x0 < bb.x0 - tol_px or tb.x1 > bb.x1 + tol_px or
                    tb.y0 < bb.y0 - tol_px or tb.y1 > bb.y1 + tol_px):
                bad.append("TEXT OUTSIDE BOX: %r" % t.get_text()[:40])
    return bad


def deliver(fig, n, stem, notes, extra_problems=()):
    """Strict QC, SVIS masters (3 formats x 2 backgrounds) and the journal files."""
    probs = list(extra_problems) + sv.qc_figure(fig, final_width_mm=FINAL_MM, strict=False)
    _, scale = sv.audit_min_font(fig, FINAL_MM)
    min_pt = min(t.get_fontsize() for _, _, t in sv._visible_texts(fig)) * scale
    if probs:
        for p in probs:
            print("  QC:", p)
        raise RuntimeError("Fig%d failed QC (%d problem(s))" % (n, len(probs)))
    masters = sv.export_figure(fig, stem, os.path.join(OUT, "masters"))
    info = journal_export(fig, n)
    plt.close(fig)
    return py(dict(figure=n, stem=stem, qc="passed", print_scale=round(float(scale), 3),
                   min_text_pt=round(float(min_pt), 1), masters=[os.path.basename(str(m)) for m in masters],
                   journal=info, palette=sv.active_palette().source, **notes))
