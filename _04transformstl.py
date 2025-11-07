# _04transformstl.py
import os
import time
import numpy as np
from math import radians, sin, cos, pi
from stl import mesh
from scipy.ndimage import distance_transform_edt as edt


# ---------------- utils ----------------

def _grid_over_bbox(xmin, xmax, ymin, ymax, nx, ny):
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    return np.meshgrid(xs, ys)

def _rasterize_supported_mask(triangles_xyz, X, Y, zmin, z_tol):
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
        return inside  # empty mask

    for t in sup_tris:
        x = t[:, 0]; y = t[:, 1]
        xmin = float(x.min()); xmax = float(x.max())
        ymin = float(y.min()); ymax = float(y.max())

        # grid window
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

def _bilinear_sample(Z, x, y, xmin, ymin, dx, dy):
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

def _smoothstep_cos(t):
    """0..1 smooth step using a raised cosine."""
    t = np.clip(t, 0.0, 1.0)
    return 0.5 - 0.5*cos(pi*t)


# ---------------- main API ----------------

def transformSTL(in_body, in_transform, out_dir,
                 grid_nx=420, grid_ny=420,
                 z_tol=0.05, angle_deg=20.0,
                 blend_mm=0.35, margin_mm=0.0):
    """
    Deform (lift/warp) an STL mesh 'in_body' using the embedded COLUMN-FREEZE
    heightmap math (no external surface). The legacy 'in_transform' argument is
    accepted for compatibility but IGNORED.

    Output is written to 'out_dir' with the same base filename.
    """
    start = time.time()
    print("[transformSTL] START (embedded column-freeze)")
    print(f"[transformSTL]   in_body      = {in_body}")
    if in_transform is not None:
        print("[transformSTL]   NOTE: 'in_transform' provided but ignored (column-freeze mode)")
    print(f"[transformSTL]   out_dir      = {out_dir}")
    print(f"[transformSTL]   params       = grid={grid_nx}x{grid_ny}  z_tol={z_tol}  "
          f"blend_mm={blend_mm}  angle={angle_deg}°  margin={margin_mm}")

    # --- Load STL to be deformed -------------------------------------------
    m_in = mesh.Mesh.from_file(in_body)
    Vtri = m_in.vectors.copy()             # (T,3,3)
    V = Vtri.reshape(-1, 3)                # (N,3)

    # Bounding box & base
    xmin = float(V[:,0].min()); xmax = float(V[:,0].max())
    ymin = float(V[:,1].min()); ymax = float(V[:,1].max())
    zmin = float(V[:,2].min())

    if margin_mm > 0:
        xmin -= margin_mm; ymin -= margin_mm
        xmax += margin_mm; ymax += margin_mm

    # --- Build XY grid over footprint --------------------------------------
    X, Y = _grid_over_bbox(xmin, xmax, ymin, ymax, int(grid_nx), int(grid_ny))
    dx = (xmax - xmin) / max(int(grid_nx) - 1, 1)
    dy = (ymax - ymin) / max(int(grid_ny) - 1, 1)

    # --- Supported mask at the cut plane -----------------------------------
    supported = _rasterize_supported_mask(Vtri, X, Y, zmin, float(z_tol))
    if not supported.any():
        raise RuntimeError("No supported footprint found. Try increasing z_tol slightly.")

    # --- Outside distance field (mm) ----------------------------------------
    dist_out = edt(~supported, sampling=(dy, dx))  # (ny,nx)

    # --- Column-freeze deformation -----------------------------------------
    s = sin(radians(angle_deg))
    blend = max(1e-6, float(blend_mm))  # guard

    # nearest-cell lookup for mask
    ny, nx = supported.shape
    def mask_at(x, y):
        u = int(round((x - xmin) / dx))
        v = int(round((y - ymin) / dy))
        u = max(0, min(nx-1, u))
        v = max(0, min(ny-1, v))
        return supported[v, u]

    V_new = V.copy()
    for idx in range(V.shape[0]):
        x, y, z = V[idx]

        # Freeze any column whose (x,y) lies inside supported XY footprint
        if mask_at(x, y):
            continue

        # Distance from supported boundary (bilinear sampled)
        d = _bilinear_sample(dist_out, x, y, xmin, ymin, dx, dy)

        # Smooth ramp from 0 at boundary to 1 past blend_mm
        w = _smoothstep_cos(d / blend)

        dz = (d * s) * w
        V_new[idx, 2] = z + dz

    # --- Save deformed STL --------------------------------------------------
    new_vecs = V_new.reshape((-1, 3, 3))
    out_mesh = mesh.Mesh(np.zeros(new_vecs.shape[0], dtype=mesh.Mesh.dtype))
    out_mesh.vectors[:] = new_vecs

    os.makedirs(out_dir, exist_ok=True)
    file_name = os.path.basename(in_body)
    output_path = os.path.join(out_dir, file_name)
    out_mesh.save(output_path)

    end = time.time()
    print(f"[transformSTL]   WROTE: {output_path}")
    print(f"[transformSTL]   DONE in {end - start:.2f}s")
    print("[transformSTL] END\n")
    return output_path


# ---------------------------------------------------------------------------
# Standalone test hook
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    example_in_body      = os.path.join("stl_parts", "test_2.stl")
    example_in_transform = os.path.join("tf_surfaces", "ignored_surface.stl")  # ignored
    example_out_dir      = "stl_tf"

    transformSTL(
        in_body=example_in_body,
        in_transform=example_in_transform,  # compat, ignored
        out_dir=example_out_dir,
        grid_nx=420, grid_ny=420,
        z_tol=0.05, angle_deg=20.0,
        blend_mm=0.35, margin_mm=0.0
    )
