# deform_column_freeze.py
# Direct mesh deformation with COLUMN FREEZE over supported XY,
# smooth blend at boundary, no deformation inside supported footprint.
# Inputs  : test/stl_parts/test_4.stl
# Outputs : Surfacegen/test_4_deformed_columnfreeze.stl

import os
import numpy as np
from math import radians, sin, cos, pi
from stl import mesh
from scipy.ndimage import distance_transform_edt as edt

# ---------------- utils ----------------

def grid_over_bbox(xmin, xmax, ymin, ymax, nx, ny):
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    return np.meshgrid(xs, ys)

def rasterize_supported_mask(triangles_xyz, X, Y, zmin, z_tol):
    """
    Boolean mask of supported (True) cells over (X,Y).
    Supported triangles are those with mean(z) <= zmin + z_tol.
    """
    inside = np.zeros(X.shape, dtype=bool)

    dx = (X[0, 1] - X[0, 0]) if X.shape[1] > 1 else 1.0
    dy = (Y[1, 0] - Y[0, 0]) if X.shape[0] > 1 else 1.0
    x0, y0 = X[0, 0], Y[0, 0]
    nx, ny = X.shape[1], X.shape[0]

    tri = triangles_xyz  # (T,3,3)
    mean_z = tri[:, :, 2].mean(axis=1)
    sup_tris = tri[mean_z <= (zmin + z_tol)]
    if sup_tris.shape[0] == 0:
        return inside  # empty

    for t in sup_tris:
        x = t[:, 0]; y = t[:, 1]
        xmin = float(x.min()); xmax = float(x.max())
        ymin = float(y.min()); ymax = float(y.max())

        # compute grid window
        i0 = int(np.clip(np.floor((xmin - x0) / dx), 0, nx - 1))
        i1 = int(np.clip(np.ceil ((xmax - x0) / dx), 0, nx - 1))
        j0 = int(np.clip(np.floor((ymin - y0) / dy), 0, ny - 1))
        j1 = int(np.clip(np.ceil ((ymax - y0) / dy), 0, ny - 1))
        if i1 < i0 or j1 < j0:
            continue

        # barycentric inclusion on subgrid
        p0 = np.array([x[0], y[0]])
        p1 = np.array([x[1], y[1]])
        p2 = np.array([x[2], y[2]])
        v0 = p2 - p0
        v1 = p1 - p0
        denom = (v0[0]*v1[1] - v0[1]*v1[0])
        if abs(denom) < 1e-12:
            continue
        inv_d = 1.0 / denom

        subX = X[j0:j1+1, i0:i1+1]
        subY = Y[j0:j1+1, i0:i1+1]
        qx = subX - p0[0]
        qy = subY - p0[1]
        u = (qx * v1[1] - qy * v1[0]) * inv_d
        v = (qy * v0[0] - qx * v0[1]) * inv_d
        w = 1.0 - u - v
        mask = (u >= 0) & (v >= 0) & (w >= 0)
        inside[j0:j1+1, i0:i1+1] |= mask

    return inside

def bilinear_sample(Z, x, y, xmin, ymin, dx, dy):
    """Bilinear sample Z on a regular grid for world (x,y)."""
    u = (x - xmin) / dx
    v = (y - ymin) / dy
    i0 = int(np.floor(u)); j0 = int(np.floor(v))
    i1 = i0 + 1; j1 = j0 + 1

    nx = Z.shape[1]; ny = Z.shape[0]
    i0 = max(0, min(nx-1, i0))
    i1 = max(0, min(nx-1, i1))
    j0 = max(0, min(ny-1, j0))
    j1 = max(0, min(ny-1, j1))

    fu = u - np.floor(u)
    fv = v - np.floor(v)

    z00 = Z[j0, i0]; z10 = Z[j0, i1]
    z01 = Z[j1, i0]; z11 = Z[j1, i1]
    z0 = z00*(1-fu) + z10*fu
    z1 = z01*(1-fu) + z11*fu
    return z0*(1-fv) + z1*fv

def smoothstep_cos(t):
    """0..1 smooth step using a raised cosine."""
    t = np.clip(t, 0.0, 1.0)
    return 0.5 - 0.5*cos(pi*t)

# -------------- main --------------

def deform_mesh_column_freeze(
    in_stl: str,
    out_stl: str,
    grid_nx: int = 400,
    grid_ny: int = 400,
    z_tol: float = 0.05,       # tight base band
    angle_deg: float = 20.0,   # slope
    blend_mm: float = 0.4,     # smooth transition width from 0 to full deformation
    margin_mm: float = 0.0
):
    print(f"[column-freeze] Loading {in_stl}")
    m_in = mesh.Mesh.from_file(in_stl)
    Vtri = m_in.vectors.copy()             # (T,3,3)
    V = Vtri.reshape(-1, 3)                # (N,3)

    xmin = float(V[:,0].min()); xmax = float(V[:,0].max())
    ymin = float(V[:,1].min()); ymax = float(V[:,1].max())
    zmin = float(V[:,2].min())

    if margin_mm > 0:
        xmin -= margin_mm; ymin -= margin_mm
        xmax += margin_mm; ymax += margin_mm

    # Build XY grid over footprint
    X, Y = grid_over_bbox(xmin, xmax, ymin, ymax, grid_nx, grid_ny)
    dx = (xmax - xmin) / max(grid_nx - 1, 1)
    dy = (ymax - ymin) / max(grid_ny - 1, 1)

    # 1) Supported mask at the cut plane
    supported = rasterize_supported_mask(Vtri, X, Y, zmin, z_tol)
    if not supported.any():
        raise RuntimeError("No supported footprint found. Increase z_tol slightly.")

    # 2) Outside distance field (in mm) — only meaningful outside supported area
    dist_out = edt(~supported, sampling=(dy, dx))  # shape (ny,nx)

    # 3) Deform vertices, but FREEZE any vertex whose (x,y) falls in supported==True
    s = sin(radians(angle_deg))
    blend = max(1e-6, float(blend_mm))  # avoid divide by zero

    # convenience for nearest mask lookup
    def mask_at(x, y):
        u = int(round((x - xmin) / dx))
        v = int(round((y - ymin) / dy))
        u = max(0, min(grid_nx-1, u))
        v = max(0, min(grid_ny-1, v))
        return supported[v, u]

    V_new = V.copy()
    for idx in range(V.shape[0]):
        x, y, z = V[idx]

        # Column freeze: if (x,y) is supported, DO NOT deform this vertex at all.
        if mask_at(x, y):
            continue

        # Otherwise, sample the outside distance field with bilinear interpolation
        d = bilinear_sample(dist_out, x, y, xmin, ymin, dx, dy)

        # Smoothly ramp deformation from 0 at boundary to 1 past blend_mm
        w = smoothstep_cos(d / blend)

        dz = (d * s) * w
        V_new[idx, 2] = z + dz

    # 4) Write out deformed STL with original faces
    out_mesh = mesh.Mesh(np.zeros(Vtri.shape[0], dtype=mesh.Mesh.dtype))
    out_mesh.vectors[:] = V_new.reshape((-1, 3, 3))
    os.makedirs(os.path.dirname(out_stl), exist_ok=True)
    out_mesh.save(out_stl)

    print(f"[column-freeze] Done → {out_stl}")
    print(f"  grid: {grid_nx}x{grid_ny}  z_tol={z_tol}  blend_mm={blend_mm}  angle={angle_deg}°")

# -------------- run example --------------

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    in_path  = os.path.join(base, "test", "stl_parts", "test_2.stl")
    out_dir  = os.path.join(base, "Surfacegen")
    out_stl  = os.path.join(out_dir, "test_2_deformed_columnfreeze.stl")

    deform_mesh_column_freeze(
        in_stl=in_path,
        out_stl=out_stl,
        grid_nx=420,
        grid_ny=420,
        z_tol=0.05,        # very tight base band
        angle_deg=20.0,
        blend_mm=0.35,     # try 0.25–0.50 if needed
        margin_mm=0.0
    )
