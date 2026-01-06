#!/usr/bin/env python3
# gcode_visualize.py
#
# Publication-ready G-code toolpath visualization (PNG + SVG),
# styled like the S3/S4-Slicer paper examples: thin polylines colored by layer index.
#
# 3D oblique view + paper-like discrete per-layer color stripes.

import os
import re
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from mpl_toolkits.mplot3d.art3d import Line3DCollection


# -----------------------------
# Parsing helpers
# -----------------------------

@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    e: float = 0.0
    is_abs_xyz: bool = True   # G90/G91
    is_abs_e: bool = False    # M82/M83


_move_re = re.compile(r'^(G0|G1)\s')
_param_re = re.compile(r'([XYZEFR])\s*([-+]?\d*\.?\d+)')


def _strip_comment(line: str) -> str:
    if ";" in line:
        return line.split(";", 1)[0].strip()
    return line.strip()


def _parse_params(cmd: str) -> Dict[str, float]:
    out = {}
    for m in _param_re.finditer(cmd):
        out[m.group(1)] = float(m.group(2))
    return out


# -----------------------------
# Toolpath extraction
# -----------------------------

@dataclass
class Segment:
    x0: float
    y0: float
    x1: float
    y1: float
    z: float
    extruding: bool


def extract_segments(
    gcode_path: str,
    only_extruding: bool = True,
    z_tol: float = 1e-4,
) -> Tuple[List[Segment], List[float]]:
    st = State()
    segments: List[Segment] = []

    decimals = max(0, int(round(-math.log10(z_tol)))) if z_tol > 0 else 4

    def snap_z(z: float) -> float:
        return round(z, decimals)

    with open(gcode_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = _strip_comment(raw)
            if not line:
                continue

            # Mode switches
            if line.startswith("G90"):
                st.is_abs_xyz = True
                continue
            if line.startswith("G91"):
                st.is_abs_xyz = False
                continue
            if line.startswith("M82"):
                st.is_abs_e = True
                continue
            if line.startswith("M83"):
                st.is_abs_e = False
                continue

            if not _move_re.match(line):
                continue

            params = _parse_params(line)

            x_new, y_new, z_new, e_new = st.x, st.y, st.z, st.e

            # XYZ update
            if "X" in params:
                x_new = params["X"] if st.is_abs_xyz else (st.x + params["X"])
            if "Y" in params:
                y_new = params["Y"] if st.is_abs_xyz else (st.y + params["Y"])
            if "Z" in params:
                z_new = params["Z"] if st.is_abs_xyz else (st.z + params["Z"])

            # E update
            extruding = False
            if "E" in params:
                if st.is_abs_e:
                    e_target = params["E"]
                    extruding = (e_target - st.e) > 1e-9
                    e_new = e_target
                else:
                    e_inc = params["E"]
                    extruding = e_inc > 1e-9
                    e_new = st.e + e_inc

            # Create a segment if XY changes
            if (x_new != st.x) or (y_new != st.y):
                z_seg = snap_z(z_new)
                seg = Segment(
                    x0=st.x, y0=st.y,
                    x1=x_new, y1=y_new,
                    z=z_seg,
                    extruding=extruding
                )
                if (not only_extruding) or seg.extruding:
                    segments.append(seg)

            st.x, st.y, st.z, st.e = x_new, y_new, z_new, e_new

    layers = sorted({s.z for s in segments})
    return segments, layers


# -----------------------------
# Paper-like discrete palette
# -----------------------------

def _paper_pastel_palette(n: int = 80) -> np.ndarray:
    """
    Create a paper-like pastel rainbow palette (discrete stripes),
    returning RGBA colors shape (n,4).
    """
    # Evenly spaced hues, moderate saturation, high value for pastel look
    hues = np.linspace(0.0, 1.0, n, endpoint=False)
    s = 0.55
    v = 0.95

    # HSV -> RGB (vectorized)
    h = hues
    i = np.floor(h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6

    r = np.zeros(n)
    g = np.zeros(n)
    b = np.zeros(n)

    mask = (i == 0); r[mask], g[mask], b[mask] = v, t[mask], p
    mask = (i == 1); r[mask], g[mask], b[mask] = q[mask], v, p
    mask = (i == 2); r[mask], g[mask], b[mask] = p, v, t[mask]
    mask = (i == 3); r[mask], g[mask], b[mask] = p, q[mask], v
    mask = (i == 4); r[mask], g[mask], b[mask] = t[mask], p, v
    mask = (i == 5); r[mask], g[mask], b[mask] = v, p, q[mask]

    rgba = np.stack([r, g, b, np.ones(n)], axis=1)
    return rgba


# -----------------------------
# Plotting — 3D + discrete per-layer colors
# -----------------------------

def plot_toolpath_layer_colored(
    segments: List[Segment],
    layers: List[float],
    out_png: str,
    out_svg: str,
    dpi: int = 600,
    linewidth: float = 0.55,   # tuned for 0.4mm nozzle look
    margin_mm: float = 2.0,
    bg: str = "white",
    equal_aspect: bool = True,
    show_axes: bool = False,
    elev: float = 20.0,
    azim: float = -55.0,
    palette_size: int = 80,
):
    if not segments:
        raise RuntimeError("No segments to plot. Try --only-extruding off or check the G-code path.")

    # Map z -> layer index
    z_to_idx = {z: i for i, z in enumerate(layers)}
    idx_vals = np.array([z_to_idx[s.z] for s in segments], dtype=int)

    # Build 3D line geometry: constant Z per segment
    lines3d = np.array([[[s.x0, s.y0, s.z], [s.x1, s.y1, s.z]] for s in segments], dtype=float)

    xs = lines3d[:, :, 0].ravel()
    ys = lines3d[:, :, 1].ravel()
    zs = lines3d[:, :, 2].ravel()

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    zmin, zmax = zs.min(), zs.max()

    xmin -= margin_mm; xmax += margin_mm
    ymin -= margin_mm; ymax += margin_mm
    zmin -= margin_mm * 0.25; zmax += margin_mm * 0.25

    w = max(1e-6, xmax - xmin)
    h = max(1e-6, ymax - ymin)
    d = max(1e-6, zmax - zmin)

    # Figure sizing based on XY footprint
    base_in = 6.5
    aspect = h / w
    fig_w = base_in
    fig_h = max(3.5, min(8.0, base_in * aspect))

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=bg)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(bg)

    try:
        ax.set_proj_type("persp")
    except Exception:
        pass

    # DISCRETE "paper-like" colors: stripe per layer (repeat palette)
    palette = _paper_pastel_palette(palette_size)
    colors = palette[idx_vals % palette_size]

    lc = Line3DCollection(
        lines3d,
        linewidths=linewidth,
        antialiased=True,
    )
    lc.set_color(colors)
    ax.add_collection3d(lc)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    # Camera (paper-like oblique)
    ax.view_init(elev=elev, azim=azim)

    if equal_aspect:
        try:
            ax.set_box_aspect((w, h, d))
        except Exception:
            pass

    if not show_axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        try:
            ax.xaxis.pane.set_visible(False)
            ax.yaxis.pane.set_visible(False)
            ax.zaxis.pane.set_visible(False)
        except Exception:
            pass
        ax.grid(False)
        try:
            ax.set_axis_off()
        except Exception:
            pass
    else:
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_zlabel("Z [mm]")

    plt.tight_layout(pad=0.05)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_svg) or ".", exist_ok=True)

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.02, facecolor=bg)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.02, facecolor=bg)
    plt.close(fig)

    print("[gcode_visualize] Saved PNG:", out_png)
    print("[gcode_visualize] Saved SVG:", out_svg)
    print(f"[gcode_visualize] Layers: {len(layers)} | Segments: {len(segments)}")


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Publication-ready G-code toolpath visualization (paper-like discrete layer colors, 3D view)."
    )
    ap.add_argument("--gcode", required=True, help="Path to input .gcode (combined or per-part).")
    ap.add_argument("--outdir", default="vis", help="Output folder for PNG + SVG.")
    ap.add_argument("--name", default=None, help="Base name for outputs (default: gcode file stem).")

    ap.add_argument("--only-extruding", choices=["on", "off"], default="on",
                    help="Plot only extruding moves (clean contours) or all moves.")
    ap.add_argument("--z-tol", type=float, default=1e-4,
                    help="Z snapping tolerance for layer grouping (mm).")
    ap.add_argument("--dpi", type=int, default=600, help="PNG DPI (publication-ready).")
    ap.add_argument("--linewidth", type=float, default=0.55,
                    help="Polyline width in points (default tuned for 0.4mm nozzle look).")
    ap.add_argument("--margin", type=float, default=2.0, help="Plot margin around bounds (mm).")

    # Camera controls (optional)
    ap.add_argument("--elev", type=float, default=20.0, help="Camera elevation angle.")
    ap.add_argument("--azim", type=float, default=-55.0, help="Camera azimuth angle.")

    # Palette controls (optional)
    ap.add_argument("--palette-size", type=int, default=80,
                    help="Number of discrete pastel colors before repeating.")

    ap.add_argument("--show-axes", action="store_true", help="Show axes labels/ticks.")
    ap.add_argument("--no-equal", action="store_true", help="Disable equal aspect ratio.")

    args = ap.parse_args()

    gcode_path = os.path.abspath(args.gcode)
    if not os.path.isfile(gcode_path):
        raise FileNotFoundError(f"Input G-code not found: {gcode_path}")

    stem = args.name or os.path.splitext(os.path.basename(gcode_path))[0]

    outdir = os.path.abspath(args.outdir)
    out_png = os.path.join(outdir, f"{stem}_toolpath.png")
    out_svg = os.path.join(outdir, f"{stem}_toolpath.svg")

    segments, layers = extract_segments(
        gcode_path,
        only_extruding=(args.only_extruding == "on"),
        z_tol=args.z_tol,
    )

    plot_toolpath_layer_colored(
        segments=segments,
        layers=layers,
        out_png=out_png,
        out_svg=out_svg,
        dpi=args.dpi,
        linewidth=args.linewidth,
        margin_mm=args.margin,
        equal_aspect=(not args.no_equal),
        show_axes=args.show_axes,
        elev=args.elev,
        azim=args.azim,
        palette_size=args.palette_size,
    )


if __name__ == "__main__":
    main()
