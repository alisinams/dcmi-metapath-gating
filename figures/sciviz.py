# -*- coding: utf-8 -*-
"""
sciviz.py - publication-grade scientific visualisation engine.

Single import for every figure script:
    import matplotlib; matplotlib.use("Agg")
    import sciviz as sv
    sv.setup()                      # palette resolution + global style
    fig, axes = sv.figure(final_width_mm=180, aspect=0.42, ncols=3)
    ...
    sv.qc_figure(fig, final_width_mm=180)          # overlap / font / palette audit
    sv.export_figure(fig, "Figure_2", "figures")   # 3 formats x 2 backgrounds

Sections
  1. Palette: ASE reader, resolution order, sequential colour rule, alpha control
  2. Colour maps derived from the palette (+ perceptual lightness check)
  3. Style: 4-tier typography, closed frame, ticks, 3D axes
  4. Layout: figure factory, outside legends, panel labels, number formatting
  5. Annotation: significance brackets, direct labels, label repulsion
  6. QC: text overlap, legend placement, minimum font, palette compliance
  7. Export: PDF / TIFF / PNG x white / transparent, metadata, verification
  8. Animation: time-based engine, GIF / MP4 / WebM-alpha / APNG, key frames
"""
from __future__ import annotations

import io
import logging
import shutil
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.text import Text

LOG = logging.getLogger("sciviz")
if not LOG.handlers:                      # own handler: never touch the root logger
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[sciviz] %(levelname)s %(message)s"))
    LOG.addHandler(_h)
    LOG.setLevel(logging.INFO)
    LOG.propagate = False

# =============================================================== 1. PALETTE
DEFAULT_PALETTE = ["#001219", "#005f73", "#0a9396", "#94d2bd", "#e9d8a6",
                   "#ee9b00", "#ca6702", "#bb3e03", "#ae2012", "#9b2226"]
ASE_FILENAME = "Ocean Sunset.ase"
LIGHTEN_FACTOR = 0.20
INK = "#000000"                     # frames, ticks, text: never taken from the palette

# Named opacity presets. Hue is fixed by the palette; ONLY alpha may vary.
ALPHA = {
    "solid": 1.00,       # lines, markers, bars
    "surface": 0.92,     # 3D surfaces
    "hist": 0.55,        # overlapping histograms / densities
    "scatter": 0.70,     # ordinary scatter
    "dense": 0.25,       # >2k points
    "context": 0.30,     # de-emphasised background series
    "band_ci": 0.22,     # 95% CI ribbons
    "band_sd": 0.14,     # SD ribbons
    "fill": 0.12,        # area under curve
    "grid": 0.12,        # optional grid lines (INK at low alpha)
}


def _lab_d50_to_srgb(L, a, b):
    """CIELAB (D50, as stored in ASE files) -> sRGB [0, 1] via Bradford D50->D65."""
    fy = (L + 16.0) / 116.0
    fx, fz = fy + a / 500.0, fy - b / 200.0
    eps, kappa = 216 / 24389, 24389 / 27

    def finv(t):
        return t ** 3 if t ** 3 > eps else (116 * t - 16) / kappa
    xyz = np.array([0.96422 * finv(fx),
                    1.0 * (fy ** 3 if L > kappa * eps else L / kappa),
                    0.82521 * finv(fz)])
    bradford = np.array([[0.9555766, -0.0230393, 0.0631636],
                         [-0.0282895, 1.0099416, 0.0210077],
                         [0.0122982, -0.0204830, 1.3299098]])
    m = np.array([[3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660, 1.8760108, 0.0415560],
                  [0.0556434, -0.2040259, 1.0572252]])
    lin = np.clip(m @ (bradford @ xyz), 0, 1)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return tuple(np.clip(srgb, 0, 1))


def read_ase(path):
    """Parse an Adobe Swatch Exchange file. Returns swatches in file order:
    [{'name', 'hex', 'model', 'group'}]. Supports RGB, CMYK, LAB and Gray."""
    data = Path(path).read_bytes()
    if data[:4] != b"ASEF":
        raise ValueError(f"{path} is not an ASE file")
    _, _, n_blocks = struct.unpack(">HHI", data[4:12])
    off, group, out = 12, None, []
    for _ in range(n_blocks):
        btype, blen = struct.unpack(">HI", data[off:off + 6])
        blk = data[off + 6:off + 6 + blen]
        off += 6 + blen
        if btype == 0xC002:                       # group end
            group = None
            continue
        nlen = struct.unpack(">H", blk[:2])[0]
        name = blk[2:2 + 2 * nlen].decode("utf-16-be").rstrip("\x00")
        p = 2 + 2 * nlen
        if btype == 0xC001:                       # group start
            group = name
            continue
        if btype != 0x0001:
            continue
        model = blk[p:p + 4].decode("ascii")
        p += 4
        k = {"RGB ": 3, "CMYK": 4, "LAB ": 3, "Gray": 1}[model]
        v = struct.unpack(">" + "f" * k, blk[p:p + 4 * k])
        if model == "RGB ":
            rgb = v
        elif model == "CMYK":
            c, m_, y, kk = v
            rgb = ((1 - c) * (1 - kk), (1 - m_) * (1 - kk), (1 - y) * (1 - kk))
        elif model == "LAB ":
            rgb = _lab_d50_to_srgb(v[0] * 100.0, v[1], v[2])
        else:
            rgb = (v[0],) * 3
        out.append({"name": name, "hex": mcolors.to_hex(np.clip(rgb, 0, 1)),
                    "model": model.strip(), "group": group})
    return out


def find_rules_dir(start=None, name="RULES"):
    """Walk up from `start` (default: cwd) and return the first folder named RULES."""
    here = Path(start or Path.cwd()).resolve()
    for d in [here, *here.parents]:
        cand = d / name
        if cand.is_dir():
            return cand
    return None


@dataclass
class Palette:
    colors: list
    source: str
    lighten_factor: float = LIGHTEN_FACTOR
    markers: list = field(default_factory=lambda: ["o", "s", "^", "D", "v", "P", "X", "h", "<", ">"])
    linestyles: list = field(default_factory=lambda: ["-", "--", "-.", ":", (0, (5, 1)),
                                                      (0, (3, 1, 1, 1)), (0, (1, 1)),
                                                      (0, (5, 2, 1, 2)), (0, (8, 2)), (0, (2, 2))])

    def __len__(self):
        return len(self.colors)

    def color(self, index):
        return get_sequential_color(index, self.colors, self.lighten_factor)

    def rgba(self, index, alpha=1.0):
        return with_alpha(self.color(index), alpha)

    def marker(self, index):
        return self.markers[index % len(self.markers)]

    def linestyle(self, index):
        return self.linestyles[index % len(self.linestyles)]

    def series(self, index, alpha=1.0):
        """Redundant encoding for series `index`: colour + marker + line style."""
        return dict(color=self.rgba(index, alpha), marker=self.marker(index),
                    linestyle=self.linestyle(index))


_ACTIVE = Palette(list(DEFAULT_PALETTE), "default (built-in Ocean Sunset hex list)")


def resolve_palette(user_palette=None, rules_dir=None, ase_name=ASE_FILENAME):
    """Resolution order: (1) palette given explicitly by the user for this task,
    (2) RULES/<ase_name> in the ROOT of the RULES folder, (3) DEFAULT_PALETTE."""
    if user_palette:
        cols = [mcolors.to_hex(c) for c in user_palette]
        return Palette(cols, "user-supplied list")
    rules = Path(rules_dir) if rules_dir else find_rules_dir()
    if rules is not None:
        hits = [p for p in rules.iterdir() if p.is_file() and p.name.lower() == ase_name.lower()]
        if hits:
            sw = read_ase(hits[0])
            cols = [s["hex"] for s in sw]
            if len(cols) >= 2:
                return Palette(cols, f"ASE file {hits[0]}")
            LOG.warning("ASE file %s has fewer than 2 colours; using default", hits[0])
    return Palette(list(DEFAULT_PALETTE), "default (built-in Ocean Sunset hex list)")


def use_palette(palette):
    global _ACTIVE
    _ACTIVE = palette
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=palette.colors)
    LOG.info("palette: %d colours from %s", len(palette), palette.source)
    return palette


def active_palette():
    return _ACTIVE


def get_sequential_color(index, base_palette=None, lighten_factor=LIGHTEN_FACTOR):
    """Exact colour for data series `index` (0-based, in input order).
    Cycle 0: base colours. Cycle c >= 1: RGB + (1 - RGB) * (1 - (1 - f)^c)."""
    base_palette = base_palette or _ACTIVE.colors
    if index < 0:
        raise ValueError("series index must be >= 0")
    cycle = index // len(base_palette)
    base_hex = base_palette[index % len(base_palette)]
    if cycle == 0:
        return mcolors.to_hex(base_hex)
    rgb = np.array(mcolors.to_rgb(base_hex))
    tint = 1.0 - (1.0 - lighten_factor) ** cycle
    return mcolors.to_hex(rgb + (1.0 - rgb) * tint)


def with_alpha(color, alpha=1.0):
    """The ONLY sanctioned colour modification: opacity. Hue stays untouched."""
    if isinstance(alpha, str):
        alpha = ALPHA[alpha]
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    r, g, b = mcolors.to_rgb(color)
    return (r, g, b, float(alpha))


# ======================================================= 2. COLOUR MAPS
def srgb_to_lab(hex_or_rgb):
    rgb = np.array(mcolors.to_rgb(hex_or_rgb))
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = (m @ lin) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return 116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])


def lightness_profile(cmap, n=64):
    return np.array([srgb_to_lab(cmap(x))[0] for x in np.linspace(0, 1, n)])


def is_monotonic(cmap, tol=1.0):
    L = lightness_profile(cmap)
    d = np.diff(L)
    return bool(np.all(d >= -tol) or np.all(d <= tol))


def build_colormaps(palette=None):
    """All continuous maps are built from palette hues only.
    full      : palette in order (user default for fields, FELs, heatmaps)
    div       : diverging, lightest palette colour fixed at the centre
    seq_low   : lightest -> first palette colour (monotonic, for magnitudes)
    seq_high  : lightest -> last palette colour (monotonic, for magnitudes)
    """
    cols = (palette or _ACTIVE).colors
    Ls = [srgb_to_lab(c)[0] for c in cols]
    k = int(np.argmax(Ls))                                      # lightest swatch
    maps = {"full": LinearSegmentedColormap.from_list("pub_full", cols, N=256)}
    if 0 < k < len(cols) - 1:
        left, right = cols[:k + 1], cols[k:]
        nodes = [(0.5 * i / k, c) for i, c in enumerate(left)]
        nodes += [(0.5 + 0.5 * i / (len(right) - 1), c) for i, c in enumerate(right)][1:]
        maps["div"] = LinearSegmentedColormap.from_list("pub_div", nodes, N=256)
        maps["seq_low"] = LinearSegmentedColormap.from_list("pub_seq_low", left[::-1], N=256)
        maps["seq_high"] = LinearSegmentedColormap.from_list("pub_seq_high", right, N=256)
    else:
        order = sorted(cols, key=lambda c: -srgb_to_lab(c)[0])
        maps["seq_low"] = maps["seq_high"] = LinearSegmentedColormap.from_list("pub_seq", order, N=256)
        maps["div"] = maps["full"]
    for name, cm in maps.items():
        if name.startswith("seq") and not is_monotonic(cm):
            LOG.warning("colormap %s is not monotonic in lightness", name)
        if cm.name in mpl.colormaps:
            mpl.colormaps.unregister(cm.name)
        mpl.colormaps.register(cm)
    return maps


CMAPS = build_colormaps()
CUSTOM_CMAP = CMAPS["full"]


# ============================================================ 3. STYLE
FS = {"L4": 16, "L3": 14, "L2": 12, "L1": 10}     # design-size typography (pt)
FRAME_LW, TICK_W, TICK_LEN = 2.0, 2.0, 6.0
MINOR_W, MINOR_LEN = 1.5, 3.5
FONT_STACK = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]


def set_isi_style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT_STACK,
        "font.size": FS["L2"], "font.weight": "normal",
        "mathtext.fontset": "custom", "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "mathtext.default": "regular", "axes.unicode_minus": True,
        "axes.titlesize": FS["L4"], "axes.titleweight": "bold", "axes.titlepad": 10,
        "axes.labelsize": FS["L3"], "axes.labelweight": "bold", "axes.labelpad": 6,
        "axes.linewidth": FRAME_LW, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.grid": False, "axes.axisbelow": True,
        "axes.formatter.use_mathtext": True, "axes.formatter.limits": (-3, 4),
        "axes.formatter.useoffset": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": TICK_W, "ytick.major.width": TICK_W,
        "xtick.major.size": TICK_LEN, "ytick.major.size": TICK_LEN,
        "xtick.minor.width": MINOR_W, "ytick.minor.width": MINOR_W,
        "xtick.minor.size": MINOR_LEN, "ytick.minor.size": MINOR_LEN,
        "xtick.major.pad": 4, "ytick.major.pad": 4,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.labelsize": FS["L2"], "ytick.labelsize": FS["L2"],
        "legend.fontsize": FS["L2"], "legend.title_fontsize": FS["L3"],
        "legend.frameon": False, "legend.handlelength": 2.2, "legend.borderaxespad": 0.0,
        "legend.labelspacing": 0.45, "legend.columnspacing": 1.4,
        "lines.linewidth": 2.0, "lines.markersize": 7, "lines.markeredgewidth": 1.6,
        "patch.linewidth": 1.2, "hatch.linewidth": 1.0,
        "errorbar.capsize": 4, "grid.color": INK, "grid.alpha": ALPHA["grid"],
        "grid.linewidth": 0.6,
        "figure.dpi": 100, "figure.facecolor": "white", "axes.facecolor": "white",
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.w_pad": 6 / 72, "figure.constrained_layout.h_pad": 6 / 72,
        "figure.constrained_layout.wspace": 0.04, "figure.constrained_layout.hspace": 0.06,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "image.interpolation": "nearest", "image.cmap": "pub_full",
        "animation.embed_limit": 200,
    })
    ff = shutil.which("ffmpeg")
    if ff:
        mpl.rcParams["animation.ffmpeg_path"] = ff


def setup(user_palette=None, rules_dir=None):
    """Call once at the top of every figure script."""
    global CMAPS, CUSTOM_CMAP
    pal = use_palette(resolve_palette(user_palette, rules_dir))
    CMAPS = build_colormaps(pal)
    CUSTOM_CMAP = CMAPS["full"]
    set_isi_style()
    return pal


def enforce_closed_frame(ax, minor=False, bold_ticklabels=True):
    for s in ("left", "right", "bottom", "top"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(FRAME_LW)
    ax.tick_params(which="major", direction="out", width=TICK_W, length=TICK_LEN,
                   color=INK, labelsize=FS["L2"], labelcolor=INK)
    if minor:
        ax.minorticks_on()
        ax.tick_params(which="minor", direction="out", width=MINOR_W, length=MINOR_LEN, color=INK)
    if bold_ticklabels:
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontweight("bold")
    return ax


def style_3d_axes(ax, transparent_panes=False, grid=False, ortho=True, zoom=0.88):
    """3D axes have no spines: the closed-frame rule maps to black axis lines,
    black pane edges and outward ticks."""
    pane = (1, 1, 1, 0) if transparent_panes else (1, 1, 1, 1)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(pane)
        axis.pane.set_edgecolor(INK)
        axis.pane.set_linewidth(1.0)
        axis.line.set_color(INK)
        axis.line.set_linewidth(FRAME_LW)
        try:                                  # private but stable since mpl 3.3
            axis._axinfo["tick"]["inward_factor"] = 0.0
            axis._axinfo["tick"]["outward_factor"] = 0.35
            axis._axinfo["tick"]["linewidth"] = {True: TICK_W, False: MINOR_W}
            axis._axinfo["grid"]["color"] = (0, 0, 0, ALPHA["grid"])
        except (KeyError, TypeError):
            pass
    ax.grid(grid)
    ax.tick_params(labelsize=FS["L2"], pad=2)
    if ortho:
        ax.set_proj_type("ortho")
    try:
        ax.set_box_aspect(None, zoom=zoom)    # leaves room for axis labels
    except TypeError:
        pass
    return ax


# ============================================================ 4. LAYOUT
MM = 1 / 25.4
COLUMN_MM = {"single": 85, "onehalf": 120, "double": 180}
MAX_REDUCTION = 0.70        # 10 pt design text must stay >= 7 pt after reduction


def design_width_in(final_width_mm):
    """Canvas width so that the smallest tier (10 pt) prints at >= 7 pt."""
    return final_width_mm * MM / MAX_REDUCTION


def figure(final_width_mm=180, aspect=0.45, nrows=1, ncols=1, projection=None,
           width_ratios=None, height_ratios=None, sharex=False, sharey=False, **kw):
    """Constrained-layout figure sized for a journal column.
    aspect = height / width of the whole canvas."""
    if isinstance(final_width_mm, str):
        final_width_mm = COLUMN_MM[final_width_mm]
    w = design_width_in(final_width_mm)
    fig = plt.figure(figsize=(w, w * aspect), layout="constrained")
    fig._sciviz_final_width_mm = final_width_mm
    subplot_kw = {"projection": projection} if projection else {}
    axes = fig.subplots(nrows, ncols, squeeze=False, sharex=sharex, sharey=sharey,
                        subplot_kw=subplot_kw,
                        gridspec_kw={k: v for k, v in (("width_ratios", width_ratios),
                                                       ("height_ratios", height_ratios)) if v}, **kw)
    for ax in axes.ravel():
        if projection == "3d":
            style_3d_axes(ax)
        else:
            enforce_closed_frame(ax)
    return fig, (axes[0, 0] if axes.size == 1 else axes)


def outside_legend(ax, where="right", ncol=None, title=None, handles=None, labels=None, **kw):
    """Legend strictly outside the data area of one axes."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    loc = {"right": ("upper left", (1.02, 1.0)), "top": ("lower left", (0.0, 1.02)),
           "bottom": ("upper center", (0.5, -0.22))}[where]
    if ncol is None:
        ncol = 1 if where == "right" else min(len(handles), 4)
    leg = ax.legend(handles, labels, loc=loc[0], bbox_to_anchor=loc[1], ncol=ncol,
                    frameon=False, title=title, fontsize=FS["L2"], borderaxespad=0.0, **kw)
    if title:
        leg.get_title().set_fontweight("bold")
    return leg


def figure_legend(fig, handles, labels, where="right", ncol=None, title=None, **kw):
    """One shared legend for a multi-panel figure, placed outside all panels.
    Requires constrained layout ('outside ...' locations)."""
    loc = {"right": "outside right upper", "top": "outside upper center",
           "bottom": "outside lower center", "left": "outside left upper"}[where]
    if ncol is None:
        ncol = 1 if where in ("right", "left") else min(len(handles), 5)
    leg = fig.legend(handles, labels, loc=loc, ncol=ncol, frameon=False,
                     fontsize=FS["L2"], title=title, **kw)
    if title:
        leg.get_title().set_fontweight("bold")
    return leg


def panel_labels(fig, axes, letters="ABCDEFGHIJKLMNOP", dx_pt=0, dy_pt=2, lowercase=False):
    """Bold 16 pt panel letters aligned per column (left edge of each column's
    tight bbox) and per row (top edge). Freezes the layout afterwards."""
    axes = list(np.ravel(axes))
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    boxes = [inv.transform(ax.get_tightbbox(r).get_points()) for ax in axes]
    pos = [ax.get_position() for ax in axes]
    cols = {}
    rows = {}
    for i, p in enumerate(pos):
        cols.setdefault(round(p.x0, 2), []).append(i)
        rows.setdefault(round(p.y1, 2), []).append(i)
    xleft = {i: min(boxes[j][0, 0] for j in members) for members in cols.values() for i in members}
    ytop = {i: max(boxes[j][1, 1] for j in members) for members in rows.values() for i in members}
    fig.set_layout_engine("none")
    out = []
    for i, ax in enumerate(axes):
        s = letters[i].lower() if lowercase else letters[i]
        t = fig.text(xleft[i] + dx_pt / 72 / fig.get_figwidth(),
                     ytop[i] + dy_pt / 72 / fig.get_figheight(), s,
                     fontsize=FS["L4"], fontweight="bold", ha="left", va="bottom", color=INK)
        out.append(t)
    return out


def finalize(fig, bold_ticks=True, opaque_legend_handles=True):
    """Last styling pass (idempotent; called by qc_figure and export_figure).
    Re-applies bold tick labels (set_xticks() creates fresh, unstyled labels)
    and makes legend keys fully opaque so faint data never yields faint keys."""
    fig.canvas.draw()
    for ax in fig.axes:
        axes_list = [ax.xaxis, ax.yaxis] + ([ax.zaxis] if hasattr(ax, "zaxis") else [])
        for axis in axes_list:
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo <= tick.get_loc() <= hi:         # skip undrawn (stale) ticks
                    continue
                for lab in (tick.label1, tick.label2):
                    lab.set_fontweight("bold" if bold_ticks else "normal")
                    lab.set_fontsize(FS["L2"])
        leg = ax.get_legend()
        for lg in ([leg] if leg else []):
            if opaque_legend_handles:
                for h in lg.legend_handles:
                    try:
                        h.set_alpha(1.0)
                    except Exception:
                        pass
    for lg in fig.legends:
        if opaque_legend_handles:
            for h in lg.legend_handles:
                try:
                    h.set_alpha(1.0)
                except Exception:
                    pass
    return fig


def fixed_decimals(ax, axis="y", decimals=None):
    """Same number of decimals on every tick label of an axis."""
    from matplotlib.ticker import FormatStrFormatter
    target = ax.yaxis if axis == "y" else ax.xaxis
    ticks = target.get_majorticklocs()
    if decimals is None:
        step = np.min(np.diff(ticks)) if len(ticks) > 1 else 1.0
        decimals = max(0, int(-np.floor(np.log10(step))) if step < 1 else 0)
    target.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))


# ========================================================= 5. ANNOTATION
def sig_bracket(ax, x1, x2, y, text, h=None, lw=1.5, fs=None):
    """Significance bracket in data coordinates; returns the top y used so
    stacked brackets can be placed above one another without overlap."""
    ylo, yhi = ax.get_ylim()
    h = h if h is not None else 0.025 * (yhi - ylo)
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=INK, lw=lw, clip_on=False)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom",
            fontsize=fs or FS["L1"], color=INK, clip_on=False)
    return y + 3.2 * h


def p_to_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def direct_label(ax, x, y, text, color, dx_pt=6, fs=None):
    """Label a series at its end point, outside the data region if x is at the edge."""
    return ax.annotate(text, (x, y), xytext=(dx_pt, 0), textcoords="offset points",
                       ha="left", va="center", fontsize=fs or FS["L2"], color=color,
                       fontweight="bold", annotation_clip=False)


def repel_texts(fig, annotations, max_iter=200, pad_px=2.0):
    """Separate overlapping call-outs. `annotations` must be ax.annotate(...)
    objects using textcoords='offset points'; they are moved vertically in
    point units until no two bounding boxes intersect."""
    for _ in range(max_iter):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        boxes = [a.get_window_extent(r).padded(pad_px) for a in annotations]
        moved = False
        for i in range(len(annotations)):
            for j in range(i + 1, len(annotations)):
                bi, bj = boxes[i], boxes[j]
                if bi.overlaps(bj):
                    shift_pt = ((min(bi.y1, bj.y1) - max(bi.y0, bj.y0)) / 2 + 1) * 72 / fig.dpi
                    up, down = (i, j) if bi.y0 >= bj.y0 else (j, i)
                    for k, sgn in ((up, 1), (down, -1)):
                        x0, y0 = annotations[k].xyann
                        annotations[k].xyann = (x0, y0 + sgn * shift_pt)
                    moved = True
        if not moved:
            break
    return annotations


# ================================================================= 6. QC
def _visible_texts(fig):
    """Every text actually drawn on the canvas (tick labels filtered to drawn ticks)."""
    out = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        axes_list = [ax.xaxis, ax.yaxis] + ([ax.zaxis] if hasattr(ax, "zaxis") else [])
        for axis in axes_list:
            if hasattr(ax, "zaxis"):
                ticks = axis.get_major_ticks()          # 3D: keep positions from the last draw
            else:
                try:
                    ticks = axis._update_ticks()        # 2D: only ticks that are drawn
                except Exception:
                    ticks = axis.get_major_ticks()
            lo, hi = sorted(axis.get_view_interval())
            eps = 1e-9 * max(1.0, abs(hi - lo))
            for t in ticks:
                if not lo - eps <= t.get_loc() <= hi + eps:      # tick not drawn
                    continue
                for lab in (t.label1, t.label2):
                    if lab.get_visible() and lab.get_text().strip():
                        out.append(("tick", ax, lab))
            if axis.label.get_visible() and axis.label.get_text().strip():
                out.append(("axislabel", ax, axis.label))
            off = axis.get_offset_text()
            if off.get_visible() and off.get_text().strip():
                out.append(("offset", ax, off))
        for t in (ax.title, getattr(ax, "_left_title", None), getattr(ax, "_right_title", None)):
            if t is not None and t.get_visible() and t.get_text().strip():
                out.append(("title", ax, t))
        for t in ax.texts:
            if t.get_visible() and t.get_text().strip():
                out.append(("text", ax, t))
        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            for t in leg.get_texts() + [leg.get_title()]:
                if t.get_text().strip():
                    out.append(("legend", ax, t))
    for t in fig.texts:
        if t.get_visible() and t.get_text().strip():
            out.append(("figtext", None, t))
    for leg in fig.legends:
        for t in leg.get_texts() + [leg.get_title()]:
            if t.get_text().strip():
                out.append(("legend", None, t))
    return out


def _text_polygon(t, renderer):
    """Exact (rotated) rectangle of a text in display pixels: 4 x 2 array.
    The axis-aligned extent of a rotated text is far larger than the glyphs,
    so overlap tests use the true rotated rectangle."""
    bb = t.get_window_extent(renderer)
    rot = t.get_rotation() % 180
    cx, cy = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2
    if rot in (0, 90):
        return np.array([[bb.x0, bb.y0], [bb.x1, bb.y0], [bb.x1, bb.y1], [bb.x0, bb.y1]])
    saved = t.get_rotation()
    t.set_rotation(0)
    flat = t.get_window_extent(renderer)
    t.set_rotation(saved)
    w, h = flat.width / 2, flat.height / 2
    th = np.deg2rad(saved)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    corners = np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) @ R.T
    return corners + [cx, cy]


def _polygons_overlap(P, Q, tol):
    """Separating-axis test for two convex quadrilaterals (tolerance in px)."""
    for poly in (P, Q):
        for i in range(4):
            edge = poly[(i + 1) % 4] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            n = np.linalg.norm(axis)
            if n == 0:
                continue
            axis /= n
            p, q = P @ axis, Q @ axis
            if p.max() - tol <= q.min() or q.max() - tol <= p.min():
                return False
    return True


def audit_overlaps(fig, tol_px=0.5):
    """Every pair of drawn texts must be disjoint (rotation-aware)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    canvas = fig.bbox
    items = []
    for k, ax, t in _visible_texts(fig):
        bb = t.get_window_extent(r)
        if not bb.overlaps(canvas):       # stale, never-drawn labels (e.g. 3D axes)
            continue
        items.append((k, t, bb, _text_polygon(t, r)))
    problems = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if not items[i][2].overlaps(items[j][2]):
                continue
            if _polygons_overlap(items[i][3], items[j][3], tol_px):
                problems.append(f"TEXT OVERLAP: '{items[i][1].get_text()[:30]}' ({items[i][0]}) x "
                                f"'{items[j][1].get_text()[:30]}' ({items[j][0]})")
    return problems


def audit_legends_outside(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    legs = [(ax, ax.get_legend()) for ax in fig.axes if ax.get_legend()] + [(None, l) for l in fig.legends]
    probs = []
    for owner, leg in legs:
        lb = leg.get_window_extent(r)
        for ax in fig.axes:
            if getattr(ax, "_colorbar", None) is not None:
                continue
            ab = ax.get_window_extent(r)
            if lb.overlaps(ab) and ax.get_visible() and ax.axison:
                probs.append(f"LEGEND INSIDE AXES: legend of {owner.get_title() if owner else 'figure'!r} "
                             f"overlaps axes {ax.get_title()!r}")
    return probs


def audit_min_font(fig, final_width_mm=None, min_pt=7.0):
    final_width_mm = final_width_mm or getattr(fig, "_sciviz_final_width_mm", 180)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tight_w_in = fig.get_tightbbox(r).width
    scale = (final_width_mm * MM) / tight_w_in
    small = {round(t.get_fontsize() * scale, 1) for _, _, t in _visible_texts(fig)
             if t.get_fontsize() * scale < min_pt}
    return ([f"FONT TOO SMALL at {final_width_mm} mm: {sorted(small)} pt (scale {scale:.2f})"]
            if small else []), scale


def _allowed_rgbs(palette, cycles=4):
    cols = [palette.color(i) for i in range(len(palette) * cycles)]
    cols += [INK, "#ffffff"]
    return np.array([mcolors.to_rgb(c) for c in cols])


def _cmap_on_palette(cmap, n=33, tol=0.02):
    """True if every sampled colour of `cmap` lies on one palette-derived map
    (covers reversed, truncated and level-resampled versions)."""
    got = np.array([cmap(x)[:3] for x in np.linspace(0, 1, n)])
    discrete = _allowed_rgbs(_ACTIVE)
    if np.all(np.min(np.max(np.abs(got[:, None, :] - discrete[None]), axis=2), axis=1) < tol):
        return True                               # e.g. contour lines drawn in INK
    for ref in CMAPS.values():
        curve = ref(np.linspace(0, 1, 1024))[:, :3]
        d = np.min(np.max(np.abs(got[:, None, :] - curve[None, :, :]), axis=2), axis=1)
        if np.all(d < tol):
            return True
    return False


def audit_palette(fig, palette=None, tol=2.5 / 255):
    """Every data colour must be a palette colour (any cycle tint) at any alpha."""
    palette = palette or _ACTIVE
    allowed = _allowed_rgbs(palette)
    bad = set()

    def check(rgba, who):
        rgba = np.atleast_2d(rgba)
        for c in rgba:
            if len(c) == 4 and c[3] == 0:
                continue
            if np.min(np.max(np.abs(allowed - c[:3]), axis=1)) > tol:
                bad.add(f"{who}: {mcolors.to_hex(c[:3])}")
    from matplotlib.collections import Collection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    for ax in fig.axes:
        for ln in ax.get_lines():
            check(mcolors.to_rgba(ln.get_color()), "line")
            mfc = ln.get_markerfacecolor()
            if mfc not in ("none", None):
                check(mcolors.to_rgba(mfc), "marker")
        for coll in ax.collections:
            if coll.get_array() is not None:
                continue                                      # colour-mapped: checked via cmap
            for getter in (coll.get_facecolor, coll.get_edgecolor):
                try:
                    check(getter(), type(coll).__name__)
                except Exception:
                    pass
        for p in ax.patches:
            check(p.get_facecolor(), "patch")
        for im in ax.images + [c for c in ax.collections if c.get_array() is not None]:
            if not _cmap_on_palette(im.get_cmap()):
                bad.add(f"colormap not derived from palette: {getattr(im.get_cmap(), 'name', '?')}")
    return [f"OFF-PALETTE COLOUR {b}" for b in sorted(bad)]


def qc_figure(fig, final_width_mm=None, strict=True, check_palette=True):
    """Run all audits. strict=True raises on any problem (use in CI / batch)."""
    finalize(fig)
    probs = audit_overlaps(fig) + audit_legends_outside(fig)
    font_probs, scale = audit_min_font(fig, final_width_mm)
    probs += font_probs
    if check_palette:
        probs += audit_palette(fig)
    for p in probs:
        LOG.warning(p)
    if not probs:
        LOG.info("QC passed (print scale %.2f)", scale)
    if probs and strict:
        raise RuntimeError(f"{len(probs)} QC problem(s); see log")
    return probs


# ============================================================== 7. EXPORT
RASTER_DPI = {"png": 300, "tiff": 600}


def _set_background(fig, mode):
    """Switch figure, axes and 3D panes between white and transparent."""
    saved = []
    face = (1, 1, 1, 0) if mode == "transparent" else (1, 1, 1, 1)
    saved.append((fig.patch.set_facecolor, fig.patch.get_facecolor()))
    fig.patch.set_facecolor(face)
    for ax in fig.axes:
        saved.append((ax.patch.set_facecolor, ax.patch.get_facecolor()))
        ax.patch.set_facecolor(face)
        if hasattr(ax, "zaxis"):
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                saved.append((axis.set_pane_color, axis.pane.get_facecolor()))
                axis.set_pane_color(face)
    return saved


def _restore(saved):
    for setter, value in reversed(saved):
        setter(value)


def _render(fig, dpi, transparent):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=transparent,
                facecolor="none" if transparent else "white")
    buf.seek(0)
    from PIL import Image
    im = Image.open(buf)
    im.load()
    return im


def export_figure(fig, stem, outdir, formats=("pdf", "tiff", "png"),
                  backgrounds=("white", "transparent"), dpi=None, metadata=None):
    """Write <stem>_<background>.<ext> for every format x background.
    PDF  : vector, TrueType fonts embedded, raster children at 600 dpi.
    TIFF : LZW, RGB (white) or RGBA (transparent), default 600 dpi.
    PNG  : RGB (white) or RGBA (transparent), default 300 dpi."""
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    finalize(fig)
    dpi = {**RASTER_DPI, **(dpi or {})}
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {"Title": stem, "Creator": "sciviz / matplotlib " + mpl.__version__,
            "Subject": f"palette: {_ACTIVE.source}"}
    meta.update(metadata or {})
    written = []
    for bg in backgrounds:
        transparent = bg == "transparent"
        saved = _set_background(fig, bg)
        try:
            for fmt in formats:
                path = outdir / f"{stem}_{bg}.{fmt}"
                if fmt == "pdf":
                    fig.savefig(path, format="pdf", dpi=600, transparent=transparent,
                                facecolor="none" if transparent else "white",
                                metadata={**meta, "CreationDate": None})
                elif fmt in ("png", "tiff"):
                    im = _render(fig, dpi[fmt], transparent)
                    im = im.convert("RGBA" if transparent else "RGB")
                    if not transparent:
                        flat = Image.new("RGB", im.size, (255, 255, 255))
                        flat.paste(im)
                        im = flat
                    if fmt == "png":
                        info = PngInfo()
                        for k, v in meta.items():
                            info.add_text(k, str(v))
                        im.save(path, dpi=(dpi[fmt], dpi[fmt]), pnginfo=info, optimize=True)
                    else:
                        im.save(path, dpi=(dpi[fmt], dpi[fmt]), compression="tiff_lzw")
                elif fmt in ("svg", "eps"):
                    fig.savefig(path, format=fmt, transparent=transparent)
                else:
                    raise ValueError(fmt)
                written.append(path)
        finally:
            _restore(saved)
    verify_exports(written)
    return written


def verify_exports(paths, max_mb=10):
    from PIL import Image
    for p in map(Path, paths):
        size_mb = p.stat().st_size / 1e6
        if size_mb > max_mb:
            LOG.warning("%s is %.1f MB (> %d MB journal limit)", p.name, size_mb, max_mb)
        if p.suffix.lower() in (".png", ".tiff"):
            with Image.open(p) as im:
                want = "RGBA" if "_transparent" in p.stem else "RGB"
                if im.mode != want:
                    raise RuntimeError(f"{p.name}: mode {im.mode}, expected {want}")
                if want == "RGBA" and im.getpixel((0, 0))[3] != 0:
                    raise RuntimeError(f"{p.name}: corner pixel is not transparent")
                if want == "RGB" and im.getpixel((0, 0)) != (255, 255, 255):
                    raise RuntimeError(f"{p.name}: corner pixel is not white")
        elif p.suffix.lower() == ".pdf":
            head = p.read_bytes()[:2048]
            if not head.startswith(b"%PDF"):
                raise RuntimeError(f"{p.name}: not a PDF")
            if b"/Type3" in p.read_bytes():
                raise RuntimeError(f"{p.name}: contains Type 3 fonts (set pdf.fonttype=42)")
    LOG.info("verified %d file(s)", len(paths))


# =========================================================== 8. ANIMATION
def _gif_palette(img, n=255):
    """Shared GIF colour table: exact white, exact ink and every palette colour
    are pinned; the remaining slots come from a median-cut of the first frame
    (anti-aliasing shades). Pinning prevents a grey 'white' background."""
    from PIL import Image
    pinned = [(255, 255, 255), (0, 0, 0)] + [tuple(int(round(255 * v)) for v in mcolors.to_rgb(c))
                                             for c in _ACTIVE.colors]
    q = img.quantize(colors=n - len(pinned), method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    flat = q.getpalette()[: 3 * (n - len(pinned))]
    entries = pinned + [tuple(flat[i:i + 3]) for i in range(0, len(flat), 3)]
    entries = (entries + [(255, 255, 255)] * 256)[:256]
    pal_img = Image.new("P", (1, 1))
    pal_img.putpalette([v for rgb in entries for v in rgb])
    return pal_img


class Animator:
    """Time-based animation engine.

    init()      -> list of artists, draws the static frame (axes, limits, legend)
    update(t)   -> list of artists changed at time t (seconds), updated IN PLACE
    One update function drives every output so all formats show identical frames.
    """

    def __init__(self, fig, init, update, duration_s, fps=25, dpi=150):
        self.fig, self.init, self.update = fig, init, update
        self.duration, self.fps, self.dpi = float(duration_s), int(fps), int(dpi)
        self._frozen = False

    @property
    def times(self):
        n = int(round(self.duration * self.fps))
        return np.arange(n) / self.fps

    def freeze(self):
        """Run the layout engine once, then disable it: no jitter between frames."""
        if not self._frozen:
            self.fig.set_dpi(self.dpi)
            self.init()
            self.fig.canvas.draw()
            self.fig.set_layout_engine("none")
            self._frozen = True

    def funcanimation(self, blit=True):
        """Matplotlib FuncAnimation for on-screen preview or anim.save()."""
        from matplotlib.animation import FuncAnimation
        self.freeze()
        return FuncAnimation(self.fig, lambda i: self.update(i / self.fps),
                             frames=len(self.times), init_func=self.init, blit=blit,
                             interval=1000 / self.fps, cache_frame_data=False)

    def frames(self, transparent=False):
        """Yield RGBA uint8 arrays, one per frame (generator: constant memory)."""
        self.freeze()
        saved = _set_background(self.fig, "transparent" if transparent else "white")
        try:
            self.init()
            for t in self.times:
                self.update(float(t))
                self.fig.canvas.draw()
                yield np.asarray(self.fig.canvas.buffer_rgba()).copy()
        finally:
            _restore(saved)

    # ---- writers ------------------------------------------------------
    def _ffmpeg(self, out, args_in, args_out, transparent):
        ff = shutil.which("ffmpeg") or mpl.rcParams.get("animation.ffmpeg_path")
        if not ff:
            raise RuntimeError("ffmpeg not found on PATH")
        self.freeze()                                   # fixes dpi, hence the frame size
        w, h = self.fig.canvas.get_width_height()
        cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
               "-s", f"{w}x{h}", "-r", str(self.fps), "-i", "-", *args_out, str(out)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for fr in self.frames(transparent):
            proc.stdin.write(fr.tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {out}")

    def save_mp4(self, out, crf=18):
        self._ffmpeg(out, [], ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white,format=yuv420p",
                               "-c:v", "libx264", "-crf", str(crf), "-preset", "slow",
                               "-movflags", "+faststart"], transparent=False)

    def save_webm_alpha(self, out, crf=30):
        self._ffmpeg(out, [], ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0",
                               "-crf", str(crf), "-auto-alt-ref", "0"], transparent=True)

    def save_mov_alpha(self, out):
        self._ffmpeg(out, [], ["-c:v", "prores_ks", "-profile:v", "4444",
                               "-pix_fmt", "yuva444p10le"], transparent=True)

    def save_gif(self, out, transparent=False, alpha_threshold=128):
        """GIF = 256-colour table + 1-bit transparency. All frames share ONE table
        (no flicker); white, ink and palette colours are pinned exactly; pixels
        are mapped with an exact nearest-colour search; semi-transparent edge
        pixels are matted on white."""
        from PIL import Image
        from scipy.spatial import cKDTree
        frames, table, tree = [], None, None
        for fr in self.frames(transparent):
            rgba = fr.astype(np.float32) / 255.0
            a = rgba[..., 3:4]
            rgb = np.round((rgba[..., :3] * a + (1 - a)) * 255).astype(np.uint8)   # matte on white
            if table is None:
                table = np.array(_gif_palette(Image.fromarray(rgb, "RGB")).getpalette()[:768],
                                 np.uint8).reshape(256, 3)
                tree = cKDTree(table[:255].astype(np.float32))                    # 255 = transparent
            flat = rgb.reshape(-1, 3)
            uniq, inv = np.unique(flat, axis=0, return_inverse=True)
            idx = tree.query(uniq.astype(np.float32))[1][inv.ravel()].astype(np.uint8)
            idx = idx.reshape(rgb.shape[:2])
            if transparent:
                idx[fr[..., 3] < alpha_threshold] = 255
            q = Image.fromarray(idx, "P")
            q.putpalette(table.ravel().tolist())
            frames.append(q)
        ms = int(round(1000 / self.fps / 10.0)) * 10            # GIF timing is in 10 ms units
        kw = dict(save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=False)
        if transparent:
            kw.update(transparency=255, disposal=2)
        frames[0].save(out, **kw)

    def save_apng(self, out, transparent=True):
        from PIL import Image
        frames = [Image.fromarray(fr, "RGBA") if transparent else
                  Image.fromarray(fr[..., :3], "RGB") for fr in self.frames(transparent)]
        frames[0].save(out, save_all=True, append_images=frames[1:],
                       duration=int(round(1000 / self.fps)), loop=0, format="PNG")

    def save_all(self, stem, outdir, formats=("gif", "mp4", "webm", "apng"),
                 backgrounds=("white", "transparent"), keyframes=None):
        """Default bundle: white GIF + MP4, transparent GIF + WebM(alpha) + APNG,
        plus static key frames (PDF/TIFF/PNG, both backgrounds) for the paper."""
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        written = []
        for bg in backgrounds:
            tr = bg == "transparent"
            if "gif" in formats:
                p = outdir / f"{stem}_{bg}.gif"
                self.save_gif(p, transparent=tr)
                written.append(p)
            if "mp4" in formats and not tr:
                p = outdir / f"{stem}_{bg}.mp4"
                self.save_mp4(p)
                written.append(p)
            if "webm" in formats and tr:
                p = outdir / f"{stem}_{bg}.webm"
                self.save_webm_alpha(p)
                written.append(p)
            if "apng" in formats and tr:
                p = outdir / f"{stem}_{bg}.png"
                self.save_apng(p, transparent=True)
                written.append(p)
        for t in (keyframes or []):
            self.init()
            self.update(float(t))
            written += export_figure(self.fig, f"{stem}_t{t:06.2f}s", outdir / "keyframes")
        LOG.info("animation: %d file(s), %d frames at %d fps", len(written), len(self.times), self.fps)
        return written


def lttb(x, y, n_out):
    """Largest-Triangle-Three-Buckets downsampling for DISPLAY of long series
    (analysis must always use the full data). Returns indices."""
    n = len(x)
    if n_out >= n or n_out < 3:
        return np.arange(n)
    idx = [0]
    bucket = (n - 2) / (n_out - 2)
    a = 0
    for i in range(n_out - 2):
        s, e = int(np.floor((i + 1) * bucket)) + 1, int(np.floor((i + 2) * bucket)) + 1
        e = min(e, n)
        nxt_s, nxt_e = e, min(int(np.floor((i + 3) * bucket)) + 1, n)
        avg_x, avg_y = (x[nxt_s:nxt_e].mean(), y[nxt_s:nxt_e].mean()) if nxt_e > nxt_s else (x[-1], y[-1])
        xs, ys = x[s:e], y[s:e]
        area = np.abs((x[a] - avg_x) * (ys - y[a]) - (x[a] - xs) * (avg_y - y[a]))
        a = s + int(np.argmax(area))
        idx.append(a)
    idx.append(n - 1)
    return np.array(idx)
