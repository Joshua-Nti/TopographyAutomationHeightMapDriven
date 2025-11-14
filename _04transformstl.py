# _04transformstl.py (column-freeze + heightmap visualization for multiple models)
import os
import time
import numpy as np
from math import radians, sin, cos, pi
from stl import mesh
from scipy.ndimage import distance_transform_edt as edt
import matplotlib.pyplot as plt

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

def _triangulate_heightfield(X, Y, Z):
    """Convert (X,Y,Z) grid into triangles for STL visualization."""
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


# ---------------- main API ----------------

def transformSTL(in_body, in_transform, out_dir,
                 grid_nx=420, grid_ny=420,
                 z_tol=0.05, angle_deg=20.0,
                 blend_mm=0.35, margin_mm=0.0):
    """
    Deform (lift/warp) an STL mesh 'in_body' using the embedded COLUMN-FREEZE
    heightmap math (no external surface). Exports per-model heightmap:
      - heightmap_<model>.npy
      - heightmap_<model>_preview.stl
      - heightmap_<model>.png (2D color plot)
    plus the deformed STL itself.
    """
    start = time.time()
    base_name = os.path.splitext(os.path.basename(in_body))[0]

    print("[transformSTL] START (column-freeze + heightmap export)")
    print(f"  model  = {base_name}")
    print(f"  in_body = {in_body}")
    print(f"  out_dir = {out_dir}")
    print(f"  grid={grid_nx}x{grid_ny}  z_tol={z_tol}  blend={blend_mm}  angle={angle_deg}°")

    # --- Load STL ----------------------------------------------------------
    m_in = mesh.Mesh.from_file(in_body)
    Vtri = m_in.vectors.copy()
    V = Vtri.reshape(-1, 3)

    xmin = float(V[:,0].min()); xmax = float(V[:,0].max())
    ymin = float(V[:,1].min()); ymax = float(V[:,1].max())
    zmin = float(V[:,2].min())

    if margin_mm > 0:
        xmin -= margin_mm; ymin -= margin_mm
        xmax += margin_mm; ymax += margin_mm

    # --- Build grid --------------------------------------------------------
    X, Y = _grid_over_bbox(xmin, xmax, ymin, ymax, int(grid_nx), int(grid_ny))
    dx = (xmax - xmin) / max(int(grid_nx) - 1, 1)
    dy = (ymax - ymin) / max(int(grid_ny) - 1, 1)

    # --- Supported mask ----------------------------------------------------
    supported = _rasterize_supported_mask(Vtri, X, Y, zmin, float(z_tol))
    if not supported.any():
        raise RuntimeError("No supported footprint found. Try increasing z_tol slightly.")

    # --- Distance field = raw heightmap (mm) -------------------------------
    dist_out = edt(~supported, sampling=(dy, dx))  # 2D field of distances
    s = sin(radians(angle_deg))
    heightmap = dist_out * s
    heightmap[supported] = 0.0

    # --- Export heightmap data & STL preview -------------------------------
    os.makedirs(out_dir, exist_ok=True)

    npy_path = os.path.join(out_dir, f"heightmap_{base_name}.npy")
    np.save(npy_path, heightmap)

    tris = _triangulate_heightfield(X, Y, heightmap)
    hm_mesh = mesh.Mesh(np.zeros(tris.shape[0], dtype=mesh.Mesh.dtype))
    hm_mesh.vectors[:] = tris
    hm_stl_path = os.path.join(out_dir, f"heightmap_{base_name}_preview.stl")
    hm_mesh.save(hm_stl_path)

    print(f"[transformSTL]   Heightmap NPY     → {npy_path}")
    print(f"[transformSTL]   Heightmap preview → {hm_stl_path}")

    # --- Plot heightmap and save PNG ---------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    # extent maps array indices back to world X/Y for nicer axes
    im = ax.imshow(heightmap,
                   origin='lower',
                   extent=[xmin, xmax, ymin, ymax],
                   aspect='equal')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Height [mm]")
    ax.set_title(f"Heightmap for {base_name}")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    fig.tight_layout()
    png_path = os.path.join(out_dir, f"heightmap_{base_name}.png")
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    print(f"[transformSTL]   Heightmap plot   → {png_path}")

    # --- Column-freeze deformation -----------------------------------------
    blend = max(1e-6, float(blend_mm))
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
        if mask_at(x, y):  # freeze columns above supported XY
            continue
        d = _bilinear_sample(dist_out, x, y, xmin, ymin, dx, dy)
        w = _smoothstep_cos(d / blend)
        dz = (d * s) * w
        V_new[idx, 2] = z + dz

    # --- Save deformed STL -------------------------------------------------
    new_vecs = V_new.reshape((-1, 3, 3))
    out_mesh = mesh.Mesh(np.zeros(new_vecs.shape[0], dtype=mesh.Mesh.dtype))
    out_mesh.vectors[:] = new_vecs

    file_name = os.path.basename(in_body)
    output_path = os.path.join(out_dir, file_name)
    out_mesh.save(output_path)

    end = time.time()
    print(f"[transformSTL]   Deformed STL     → {output_path}")
    print(f"[transformSTL]   DONE in {end - start:.2f}s\n")
    return output_path


# ---------------------------------------------------------------------------
# Batch run for test_2, test_4, test_5
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    example_out_dir = os.path.join("stl_tf")

    for name in ["test_2.stl", "test_4.stl", "test_5.stl"]:
        in_body = os.path.join("stl_parts", name)
        transformSTL(
            in_body=in_body,
            in_transform=None,          # kept for legacy API, ignored internally
            out_dir=example_out_dir,
            grid_nx=420, grid_ny=420,
            z_tol=0.05,
            angle_deg=20.0,
            blend_mm=0.35,
            margin_mm=0.0,
        )
