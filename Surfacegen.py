# Surfacegen_v6_footprint.py
"""
Transform surface aligned to model footprint (no centering, no external deps beyond numpy-stl + scipy):
- Plane spans exactly the STL's XY bounds
- Supported region = rasterized union of triangles near the minimum Z (zmin + tol)
- Signed distance via EDT on the raster -> Z = max(0, dist) * sin(angle)
- Exports an open surface STL (heightfield), no cap

Inputs : test/stl_parts/test_4.stl
Outputs: Surfacegen/test_4_transform_v6.stl
"""

import os
import numpy as np
from math import radians, sin
from stl import mesh
from scipy.ndimage import distance_transform_edt as edt


# ----------------- helpers -----------------

def triangulate_heightfield(X, Y, Z):
    ny, nx = X.shape
    tris = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            p1 = [X[j, i],     Y[j, i],     Z[j, i]]
            p2 = [X[j, i+1],   Y[j, i+1],   Z[j, i+1]]
            p3 = [X[j+1, i+1], Y[j+1, i+1], Z[j+1, i+1]]
            p4 = [X[j+1, i],   Y[j+1, i],   Z[j+1, i]]
            tris.append([p1, p2, p3])
            tris.append([p1, p3, p4])
    return np.asarray(tris, dtype=np.float64)


def grid_over_bbox(xmin, xmax, ymin, ymax, nx, ny):
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    return np.meshgrid(xs, ys)


def rasterize_supported_triangles(triangles_xyz, X, Y, zmin, tol=0.2):
    """
    Rasterize into a boolean mask over grid (X,Y) the union of projected
    triangles whose MEAN z <= zmin + tol.
    Efficient enough for typical meshes: only loops supported tris and their AABBs.
    """
    inside = np.zeros(X.shape, dtype=bool)
    dx = X[0, 1] - X[0, 0] if X.shape[1] > 1 else 1.0
    dy = Y[1, 0] - Y[0, 0] if Y.shape[0] > 1 else 1.0

    ny, nx = X.shape

    # Precompute to locate grid indices from XY
    x0, x1 = X[0, 0], X[0, -1]
    y0, y1 = Y[0, 0], Y[-1, 0]

    # Select supported triangles
    tri = triangles_xyz  # (T,3,3)
    mean_z = tri[:, :, 2].mean(axis=1)
    sup_mask = (mean_z <= (zmin + tol))
    sup_tris = tri[sup_mask]
    if sup_tris.shape[0] == 0:
        return inside  # empty; caller should adjust tol

    def clamp_idx(ix, max_ix):
        return 0 if ix < 0 else (max_ix if ix > max_ix else ix)

    for t in sup_tris:
        # triangle vertices projected to XY
        x = t[:, 0]; y = t[:, 1]

        # AABB in XY -> grid index window
        xmin = float(x.min()); xmax = float(x.max())
        ymin = float(y.min()); ymax = float(y.max())

        # quickly skip if AABB outside grid
        if (xmax < x0) or (xmin > x1) or (ymax < y0) or (ymin > y1):
            continue

        i0 = int(np.floor((xmin - x0) / dx))
        i1 = int(np.ceil ((xmax - x0) / dx))
        j0 = int(np.floor((ymin - y0) / dy))
        j1 = int(np.ceil ((ymax - y0) / dy))

        i0 = clamp_idx(i0, nx - 1); i1 = clamp_idx(i1, nx - 1)
        j0 = clamp_idx(j0, ny - 1); j1 = clamp_idx(j1, ny - 1)
        if i1 < i0 or j1 < j0:
            continue

        # Barycentric test on the subgrid window
        # tri in 2D: p0,p1,p2
        p0 = np.array([x[0], y[0]])
        p1 = np.array([x[1], y[1]])
        p2 = np.array([x[2], y[2]])

        v0 = p2 - p0
        v1 = p1 - p0
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot11 = np.dot(v1, v1)
        denom = (dot00 * dot11 - dot01 * dot01)
        if abs(denom) < 1e-14:
            continue
        inv_d = 1.0 / denom

        # Build local arrays of grid points
        subX = X[j0:j1+1, i0:i1+1]
        subY = Y[j0:j1+1, i0:i1+1]
        q = np.stack([subX, subY], axis=-1)  # (..., 2)
        v2 = q - p0  # (..., 2)
        # barycentric u,v; w=1-u-v
        dot02 = v2[..., 0] * v0[0] + v2[..., 1] * v0[1]
        dot12 = v2[..., 0] * v1[0] + v2[..., 1] * v1[1]
        u = (dot11 * dot02 - dot01 * dot12) * inv_d
        v = (dot00 * dot12 - dot01 * dot02) * inv_d
        w = 1.0 - u - v
        mask = (u >= -1e-6) & (v >= -1e-6) & (w >= -1e-6)

        # OR into final mask
        inside[j0:j1+1, i0:i1+1] |= mask

    return inside


# ----------------- main -----------------

def generate_transform_surface_v6(
    in_stl: str,
    out_stl: str,
    z_eps_mm: float = 0.20,     # tolerance band for "supported" (zmin + tol)
    grid_nx: int = 300,
    grid_ny: int = 300,
    margin_mm: float = 0.0,     # optional plane margin around footprint
    angle_deg: float = 20.0,    # Z = dist * sin(angle)
    smooth_sigma_cells: float = 0.0  # optional gaussian smoothing in grid cells
):
    print(f"[v6] Loading {in_stl}")
    m_in = mesh.Mesh.from_file(in_stl)
    V = m_in.vectors.reshape(-1, 3)

    # Footprint bounds from ORIGINAL coordinates (no centering)
    xmin, xmax = float(V[:, 0].min()), float(V[:, 0].max())
    ymin, ymax = float(V[:, 1].min()), float(V[:, 1].max())
    zmin = float(V[:, 2].min())

    if margin_mm > 0:
        xmin -= margin_mm; ymin -= margin_mm
        xmax += margin_mm; ymax += margin_mm

    print(f"[v6] XY footprint: x[{xmin:.3f},{xmax:.3f}]  y[{ymin:.3f},{ymax:.3f}]  zmin={zmin:.3f}")

    # Build regular grid EXACTLY over the model footprint
    X, Y = grid_over_bbox(xmin, xmax, ymin, ymax, nx=grid_nx, ny=grid_ny)
    dx = (xmax - xmin) / max(grid_nx - 1, 1)
    dy = (ymax - ymin) / max(grid_ny - 1, 1)

    # Rasterize supported (red) region from triangles whose mean z<=zmin+z_eps_mm
    inside = rasterize_supported_triangles(m_in.vectors, X, Y, zmin, tol=z_eps_mm)
    if not inside.any():
        raise RuntimeError(
            "Supported area mask empty. Increase z_eps_mm (e.g., 0.3–0.6) or check STL."
        )

    # Signed distance via two EDTs on raster (mm units via sampling=(dy,dx))
    dist_out = edt(~inside, sampling=(dy, dx))
    dist_in  = edt( inside, sampling=(dy, dx))
    signed   = dist_out - dist_in  # >0 outside, <0 inside

    # Heightfield: zero inside, positive ramp outside
    gain = sin(radians(angle_deg))
    Z = np.maximum(0.0, signed) * gain

    if smooth_sigma_cells and smooth_sigma_cells > 0:
        from scipy.ndimage import gaussian_filter
        Z = gaussian_filter(Z, sigma=float(smooth_sigma_cells))

    # Triangulate and export open surface STL
    tris = triangulate_heightfield(X, Y, Z)
    out_m = mesh.Mesh(np.zeros(tris.shape[0], dtype=mesh.Mesh.dtype))
    out_m.vectors[:] = tris

    os.makedirs(os.path.dirname(out_stl), exist_ok=True)
    out_m.save(out_stl)

    print(f"[v6] Grid {grid_nx}x{grid_ny} | sin({angle_deg}°)={gain:.6f}")
    print(f"[v6] Output -> {out_stl}")


# --------------- run ---------------

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    in_path  = os.path.join(base, "test", "stl_parts", "test_2.stl")
    out_dir  = os.path.join(base, "Surfacegen")
    out_path = os.path.join(out_dir, "test_2_transform_v6.stl")

    generate_transform_surface_v6(
        in_stl=in_path,
        out_stl=out_path,
        z_eps_mm=0.25,      # try 0.2–0.6 depending on your base band thickness
        grid_nx=300,
        grid_ny=300,
        margin_mm=0.0,
        angle_deg=20.0,
        smooth_sigma_cells=0.0  # try 0.8–1.5 if you want a gentler ramp
    )
