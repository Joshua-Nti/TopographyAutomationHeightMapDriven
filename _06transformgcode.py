# _06transformgcode.py
# Reverse-transform G-code sliced from a HEIGHTMAP-deformed STL back to printer space,
# using the same column-freeze heightmap math as heightmap_gen.py.
#
# Behavior:
#  - For most models: original stable behavior
#       * XY segmentation in backtransform_data (maximal_length)
#       * slowdown on downward-facing perimeters
#  - For ONE specific file: test/gcode_parts/test_2.gcode
#       * Pre-segmentation of XY moves BEFORE reverse mapping
#       * Internal segmentation in backtransform_data effectively disabled
#
# Inputs:
#   - in_file: G-code produced by slicing the DEFORMED mesh
#   - stl_for_heightmap: ORIGINAL (pre-deformation) STL used to rebuild Δz(x,y)
#   - surface_for_slowdown: FINAL geometry STL (for downward perimeter slowdown)
#
# Outputs:
#   - Writes backtransformed G-code to out_dir with the same basename
#
# Dependencies: numpy, numpy-stl, scipy.ndimage (for EDT)

import re
import os
import time
import math
import numpy as np
from stl import mesh
from scipy.ndimage import distance_transform_edt as edt

# ---------------------------------------------------------------------------
#  CONFIG: which file gets pre-segmentation
# ---------------------------------------------------------------------------

# Max segment length (XY) for pre-segmentation of test_4.gcode.
# This only affects test/gcode_parts/test_4.gcode.
PRESEG_MAX_LEN = 0.25
SEGMENT_TOGGLE = True

# Global ramp length for feedrate transitions (number of motion lines
# used to smooth fast->slow and slow->fast feedrate jumps).
FEEDRATE_RAMP_LEN = 7  # you can set this to 3, 5, 7, ...

def _should_presegment(in_file: str) -> bool:
    """
    Return True ONLY for these exact files:
        test/gcode_tf/test_2.gcode
        test/gcode_tf/test_4.gcode
        test/gcode_tf/test_5.gcode

    Any other G-code file → pre-segmentation DISABLED.
    """
    norm_path = os.path.normpath(os.path.abspath(in_file))

    # --- Exact tails we want to match ---
    TARGET_TAILS = [
        os.path.normpath(os.path.join("test", "gcode_tf", "test_2.gcode")),
        os.path.normpath(os.path.join("test", "gcode_tf", "test_4.gcode")),
        os.path.normpath(os.path.join("test", "gcode_tf", "test_5.gcode")),
    ]

    # Check if this file matches any exact tail
    for tail in TARGET_TAILS:
        if norm_path.endswith(tail):
            print(f"[transformGCode] Pre-segmentation ENABLED for {tail}")
            return SEGMENT_TOGGLE

    print("[transformGCode] Pre-segmentation DISABLED for this file.")
    return False




# ---------------------------------------------------------------------------
# Heightmap (column-freeze) utilities — SAME MATH AS heightmap_gen.py
# ---------------------------------------------------------------------------

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
    nx = X.shape[1]
    ny = X.shape[0]

    tri = triangles_xyz  # (T,3,3)
    mean_z = tri[:, :, 2].mean(axis=1)
    sup_tris = tri[mean_z <= (zmin + z_tol)]
    if sup_tris.shape[0] == 0:
        return inside  # empty

    for t in sup_tris:
        x = t[:, 0]
        y = t[:, 1]
        xmin_t = float(x.min())
        xmax_t = float(x.max())
        ymin_t = float(y.min())
        ymax_t = float(y.max())

        # compute grid window
        i0 = int(np.clip(np.floor((xmin_t - x0) / dx), 0, nx - 1))
        i1 = int(np.clip(np.ceil((xmax_t - x0) / dx), 0, nx - 1))
        j0 = int(np.clip(np.floor((ymin_t - y0) / dy), 0, ny - 1))
        j1 = int(np.clip(np.ceil((ymax_t - y0) / dy), 0, ny - 1))
        if i1 < i0 or j1 < j0:
            continue

        # barycentric inclusion on subgrid
        p0 = np.array([x[0], y[0]])
        p1 = np.array([x[1], y[1]])
        p2 = np.array([x[2], y[2]])
        v0 = p2 - p0
        v1 = p1 - p0
        denom = (v0[0] * v1[1] - v0[1] * v1[0])
        if abs(denom) < 1e-12:
            continue
        inv_d = 1.0 / denom

        subX = X[j0:j1 + 1, i0:i1 + 1]
        subY = Y[j0:j1 + 1, i0:i1 + 1]
        qx = subX - p0[0]
        qy = subY - p0[1]
        u = (qx * v1[1] - qy * v1[0]) * inv_d
        v = (qy * v0[0] - qx * v0[1]) * inv_d
        w = 1.0 - u - v
        mask = (u >= 0) & (v >= 0) & (w >= 0)
        inside[j0:j1 + 1, i0:i1 + 1] |= mask

    return inside


def bilinear_sample(Z, x, y, xmin, ymin, dx, dy):
    """Bilinear sample Z on a regular grid for world (x,y)."""
    u = (x - xmin) / dx
    v = (y - ymin) / dy
    i0 = int(np.floor(u))
    j0 = int(np.floor(v))
    i1 = i0 + 1
    j1 = j0 + 1

    nx = Z.shape[1]
    ny = Z.shape[0]
    i0 = max(0, min(nx - 1, i0))
    i1 = max(0, min(nx - 1, i1))
    j0 = max(0, min(ny - 1, j0))
    j1 = max(0, min(ny - 1, j1))

    fu = u - np.floor(u)
    fv = v - np.floor(v)

    z00 = Z[j0, i0]
    z10 = Z[j0, i1]
    z01 = Z[j1, i0]
    z11 = Z[j1, i1]
    z0 = z00 * (1 - fu) + z10 * fu
    z1 = z01 * (1 - fu) + z11 * fu
    return z0 * (1 - fv) + z1 * fv


def smoothstep_cos(t):
    t = np.clip(t, 0.0, 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * t)


def build_heightmap_field(in_stl, grid_nx, grid_ny, z_tol, angle_deg, blend_mm, margin_mm):
    """
    Rebuild Δz(x,y) field: supported mask -> outside distance field -> Δz components.
    Returns: (xmin, ymin, dx, dy, dist_out, supported, sin(angle), blend)
    """
    m_in = mesh.Mesh.from_file(in_stl)
    Vtri = m_in.vectors.copy()  # (T,3,3)
    V = Vtri.reshape(-1, 3)      # (N,3)

    xmin = float(V[:, 0].min())
    xmax = float(V[:, 0].max())
    ymin = float(V[:, 1].min())
    ymax = float(V[:, 1].max())
    zmin = float(V[:, 2].min())

    if margin_mm > 0:
        xmin -= margin_mm
        ymin -= margin_mm
        xmax += margin_mm
        ymax += margin_mm

    X, Y = grid_over_bbox(xmin, xmax, ymin, ymax, int(grid_nx), int(grid_ny))
    dx = (xmax - xmin) / max(int(grid_nx) - 1, 1)
    dy = (ymax - ymin) / max(int(grid_ny) - 1, 1)

    supported = rasterize_supported_mask(Vtri, X, Y, zmin, float(z_tol))
    if not supported.any():
        raise RuntimeError("No supported footprint found; try increasing z_tol.")

    dist_out = edt(~supported, sampling=(dy, dx))
    s = math.sin(math.radians(angle_deg))
    blend = max(1e-6, float(blend_mm))
    return xmin, ymin, dx, dy, dist_out, supported, s, blend


def dz_at(x, y, xmin, ymin, dx, dy, dist_out, supported, s, blend):
    """Compute Δz(x,y) with column-freeze (0 inside supported footprint)."""
    # freeze: nearest-cell lookup in supported mask
    u = int(round((x - xmin) / dx))
    v = int(round((y - ymin) / dy))
    ny, nx = supported.shape
    u = max(0, min(nx - 1, u))
    v = max(0, min(ny - 1, v))
    if supported[v, u]:
        return 0.0

    d = bilinear_sample(dist_out, x, y, xmin, ymin, dx, dy)
    w = smoothstep_cos(d / blend)
    return (d * s) * w


# ---------------------------------------------------------------------------
# Geometry helpers for slowdown (DOWNWARD-facing perimeters in FINAL geometry)
# ---------------------------------------------------------------------------

def triangle_data_from_mesh(stl_path, max_angle_deg=10.0):
    """
    Load an STL (FINAL geometry) and keep triangles whose normal is within
    max_angle_deg of -Z (downward-facing).
    """
    body = mesh.Mesh.from_file(stl_path).vectors  # shape (N,3,3)
    triangles = []
    cos_max = np.cos(np.deg2rad(max_angle_deg))

    total_tris = 0
    downward_tris = 0
    for tri in body:
        total_tris += 1
        p0, p1, p2 = tri
        n = np.cross(p1 - p0, p2 - p0)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        if n[2] <= -cos_max:
            downward_tris += 1
            aabb_min = np.min(tri, axis=0)
            aabb_max = np.max(tri, axis=0)
            triangles.append({
                "p0": p0.copy(),
                "p1": p1.copy(),
                "p2": p2.copy(),
                "normal": n.copy(),
                "aabb_min": aabb_min,
                "aabb_max": aabb_max
            })

    print("=== DEBUG triangle_data_from_mesh ===")
    print("Source STL:", stl_path)
    print("Total triangles:", total_tris)
    print(f"Downward-facing triangles (<= {max_angle_deg:.1f}° from -Z): {downward_tris}")
    return triangles


def build_triangle_spatial_index(triangles, cell_size=2.0):
    grid = {}

    def cid(x, y, z):
        return (int(np.floor(x / cell_size)),
                int(np.floor(y / cell_size)),
                int(np.floor(z / cell_size)))

    for tri in triangles:
        mn = tri["aabb_min"]
        mx = tri["aabb_max"]
        PAD = 1.0
        x0, y0, z0 = mn - PAD
        x1, y1, z1 = mx + PAD
        ix0, iy0, iz0 = cid(x0, y0, z0)
        ix1, iy1, iz1 = cid(x1, y1, z1)
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                for iz in range(iz0, iz1 + 1):
                    key = (ix, iy, iz)
                    if key not in grid:
                        grid[key] = []
                    grid[key].append(tri)
    return grid, cell_size


def query_triangles_near_point(p, grid, cell_size):
    x, y, z = p
    ix = int(np.floor(x / cell_size))
    iy = int(np.floor(y / cell_size))
    iz = int(np.floor(z / cell_size))
    # 1-cell neighborhood is enough for speed (can expand if needed)
    candidates = []
    key = (ix, iy, iz)
    if key in grid:
        candidates.extend(grid[key])
    return candidates


def point_near_downward_surface(midpoint, tri_grid, cell_size, dist_tol=0.4):
    nearby = query_triangles_near_point(midpoint, tri_grid, cell_size)
    if not nearby:
        return False
    mx, my, mz = midpoint
    for tri in nearby:
        p0 = tri["p0"]
        p1 = tri["p1"]
        p2 = tri["p2"]
        n = tri["normal"]
        aabb_min = tri["aabb_min"]
        aabb_max = tri["aabb_max"]
        PAD = 0.5
        if (mx < aabb_min[0] - PAD or mx > aabb_max[0] + PAD or
                my < aabb_min[1] - PAD or my > aabb_max[1] + PAD or
                mz < aabb_min[2] - PAD or mz > aabb_max[2] + PAD):
            continue
        v_mid = midpoint - p0
        d_signed = np.dot(v_mid, n)
        if abs(d_signed) > dist_tol:
            continue
        proj = midpoint - d_signed * n
        v0 = p2 - p0
        v1 = p1 - p0
        v2 = proj - p0
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)
        denom = (dot00 * dot11 - dot01 * dot01)
        if abs(denom) < 1e-16:
            continue
        inv = 1.0 / denom
        u = (dot11 * dot02 - dot01 * dot12) * inv
        v = (dot00 * dot12 - dot01 * dot02) * inv
        w = 1.0 - u - v
        if (u >= -1e-4 and v >= -1e-4 and w >= -1e-4):
            return True
    return False


def segment_near_downward_surface(p_start, p_mid, p_end, tri_grid, cell_size, dist_tol=0.4):
    return (point_near_downward_surface(p_start, tri_grid, cell_size, dist_tol) or
            point_near_downward_surface(p_mid,   tri_grid, cell_size, dist_tol) or
            point_near_downward_surface(p_end,   tri_grid, cell_size, dist_tol))


# ---------------------------------------------------------------------------
# G-code helpers (unchanged from stable pipeline)
# ---------------------------------------------------------------------------

def insert_Z(row, z_value):
    pattern_Z = r'Z[-0-9.]+[.]?[0-9]*'
    m = re.search(pattern_Z, row)
    if m is not None:
        return re.sub(pattern_Z, ' Z' + str(round(z_value, 3)), row)
    # else insert after Y or X if present
    mY = re.search(r'Y[-0-9.]+[.]?[0-9]*', row)
    mX = re.search(r'X[-0-9.]+[.]?[0-9]*', row)
    if mY is not None:
        return row[:mY.end(0)] + ' Z' + str(round(z_value, 3)) + row[mY.end(0):]
    if mX is not None:
        return row[:mX.end(0)] + ' Z' + str(round(z_value, 3)) + row[mX.end(0):]
    return 'Z' + str(round(z_value, 3)) + ' ' + row


def replace_E(row, corr_value):
    pattern_E = r'E[-0-9.]+[.]?[0-9]*'
    m = re.search(pattern_E, row)
    if m is None:
        return row
    e_old = float(m.group(0).replace('E', ''))
    if corr_value == 0:
        e_new = 0.0
    else:
        e_new = round(e_old / corr_value, 6)
        if abs(e_new) < 1e-3:
            e_new = 0.0
    return row[:m.start(0)] + ('E' + str(e_new)) + row[m.end(0):]


def clean_and_set_feedrate(row, feed_mm_min):
    row_noF = re.sub(r'F[-0-9.]+[.]?[0-9]*', '', row).rstrip()
    return f"{row_noF} F{feed_mm_min:.1f}\n"


def extract_feedrate(row):
    m = re.search(r'F([0-9.]+)', row)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None



def smooth_feedrate_transitions(lines, slow_feedrate, ramp_len=None):
    """
    Post-process a list of G-code lines and smooth abrupt feedrate jumps

    1) FAST -> SLOW (deceleration):
       - Detect transitions from a higher feedrate down to `slow_feedrate`.
       - Apply a backward ramp over up to `ramp_len` motion lines
         (previous lines + the first slow line).
       - This keeps the slow region itself at `slow_feedrate`.

    2) SLOW -> FAST (acceleration):
       - Detect transitions from `slow_feedrate` to a higher feedrate.
       - If there is another explicit slow feedrate soon after (within about
         2 * ramp_len motion lines), we SKIP the acceleration ramp and let the
         later deceleration ramp handle the transition.
       - Otherwise, apply a forward ramp over up to `ramp_len` motion lines
         (starting at the first faster line).

    If ramp_len is None, FEEDRATE_RAMP_LEN is used.
    """
    if ramp_len is None:
        ramp_len = FEEDRATE_RAMP_LEN

    EPS = 1e-6

    # Work on a modifiable copy
    out = list(lines)

    # ----------------------------------------------------------------------
    # PASS 1: FAST -> SLOW (backward ramp)
    # ----------------------------------------------------------------------
    last_F = None

    for i in range(len(out)):
        line = out[i]
        F_here = extract_feedrate(line)

        if F_here is not None:
            # Detect jump from fast -> exactly slow_feedrate
            if (
                last_F is not None
                and last_F > slow_feedrate + EPS
                and abs(F_here - slow_feedrate) < EPS
            ):
                # We want to ramp backwards over up to 'ramp_len' motion lines:
                # some previous G0/G1 lines (fast) + this first slow line.
                ramp_indices = []

                j = i
                while j >= 0 and len(ramp_indices) < ramp_len:
                    l_j = out[j]
                    stripped_j = l_j.lstrip()
                    if stripped_j.startswith("G0") or stripped_j.startswith("G1"):
                        ramp_indices.append(j)
                    j -= 1

                # ramp_indices currently in descending order -> sort ascending
                ramp_indices.sort()
                total = len(ramp_indices)

                if total > 0:
                    F_start = last_F
                    F_target = slow_feedrate

                    for k, idx in enumerate(ramp_indices):
                        t = float(k + 1) / float(total)  # 1/total .. 1
                        F_k = F_start + (F_target - F_start) * t
                        out[idx] = clean_and_set_feedrate(out[idx], F_k)

                    # After the ramp, effective feedrate is slow
                    last_F = slow_feedrate
                else:
                    # No valid motion lines found; just keep the explicit slow feed
                    last_F = F_here
            else:
                # Normal update of last_F
                last_F = F_here

    # ----------------------------------------------------------------------
    # PASS 2: SLOW -> FAST (forward ramp with lookahead)
    # ----------------------------------------------------------------------
    result = out
    last_F = None
    i = 0
    n = len(result)

    while i < n:
        line = result[i]
        F_here = extract_feedrate(line)

        if F_here is not None:
            # Detect jump from slow -> faster
            if (
                last_F is not None
                and abs(last_F - slow_feedrate) < EPS
                and F_here > slow_feedrate + EPS
            ):
                # Look ahead: is there another explicit slow feedrate
                # within about 2 * ramp_len motion lines?
                motion_seen = 0
                next_slow_index = None
                j = i + 1

                while j < n and motion_seen < 2 * ramp_len:
                    l_j = result[j]
                    stripped_j = l_j.lstrip()
                    if stripped_j.startswith("G0") or stripped_j.startswith("G1"):
                        motion_seen += 1
                        F_j = extract_feedrate(l_j)
                        if F_j is not None and abs(F_j - slow_feedrate) < EPS:
                            next_slow_index = j
                            break
                    j += 1

                if next_slow_index is not None and motion_seen < 2 * ramp_len:
                    # Not enough room for a full accel ramp AND a clean decel
                    # ramp before the next slow region.
                    # In this case, we KEEP the slow speed and do NOT jump up:
                    # overwrite this faster F with slow_feedrate.
                    result[i] = clean_and_set_feedrate(result[i], slow_feedrate)
                    last_F = slow_feedrate
                    i += 1
                    continue


                # Otherwise: perform normal forward ramp from slow -> F_here
                F_start = slow_feedrate
                F_target = F_here

                ramp_indices = []
                j = i
                while j < n and len(ramp_indices) < ramp_len:
                    l_j = result[j]
                    stripped_j = l_j.lstrip()
                    if stripped_j.startswith("G0") or stripped_j.startswith("G1"):
                        ramp_indices.append(j)
                    j += 1

                total = len(ramp_indices)
                if total > 0:
                    for k, idx in enumerate(ramp_indices):
                        t = float(k + 1) / float(total)  # 1/total .. 1
                        F_k = F_start + (F_target - F_start) * t
                        result[idx] = clean_and_set_feedrate(result[idx], F_k)

                    last_F = F_target
                    i = ramp_indices[-1] + 1
                    continue
                else:
                    # No motion lines to ramp; just set the faster feed here
                    result[i] = clean_and_set_feedrate(result[i], F_target)
                    last_F = F_target
                    i += 1
                    continue
            else:
                # Normal explicit feedrate update (no ramp)
                last_F = F_here

        # No feedrate on this line, keep last_F as-is
        i += 1

    return result



# ---------------------------------------------------------------------------
# Core: backtransform_data using HEIGHTMAP Δz(x,y)
# ---------------------------------------------------------------------------

def backtransform_data(
    data,
    zmin_clamp,
    maximal_length,
    # heightmap field components:
    xmin, ymin, dx, dy, dist_out, supported, s, blend,
    # slowdown geometry:
    tri_grid,
    cell_size,
    slow_feedrate=180.0
):
    """
    Convert deformed-space G-code 'data' into final printer-space toolpaths.
    - Rewrites Z as Z := Z - Δz(x,y) using the provided heightmap field.
    - Segments long XY moves (unless maximal_length is huge, e.g. for test_2.gcode).
    - Perimeter slowdown on downward-facing geometry.
    """
    new_data = []

    pattern_X = r'X[-0-9.]+[.]?[0-9]*'
    pattern_Y = r'Y[-0-9.]+[.]?[0-9]*'
    pattern_Z = r'Z[-0-9.]+[.]?[0-9]*'
    pattern_G = r'\AG[01]\s'   # G0/G1

    x_old, y_old = 0.0, 0.0
    z_layer = 0.0
    in_perimeter = False

    last_normal_F = None
    was_slow = False

    perimeter_true_count = 0
    total_subsegments = 0
    subsegments_near_downward = 0
    subsegments_slow_candidates = 0

    for row in data:
        stripped = row.strip()

        # Region classification via comments
        if stripped.startswith(";"):
            lower = stripped.lower()
            prev = in_perimeter
            if "perimeter" in lower:
                in_perimeter = True
            if "infill" in lower or ("fill" in lower and "perimeter" not in lower):
                in_perimeter = False
            if (in_perimeter is True) and (prev is False):
                perimeter_true_count += 1
            new_data.append(row)
            continue

        # Non-move lines
        if re.search(pattern_G, row) is None:
            fval = extract_feedrate(row)
            if fval is not None and abs(fval - slow_feedrate) > 1e-6:
                last_normal_F = fval
                was_slow = False
            new_data.append(row)
            continue

        # Move line (G0/G1 ... )
        x_match = re.search(pattern_X, row)
        y_match = re.search(pattern_Y, row)
        z_match = re.search(pattern_Z, row)

        if (x_match is None and y_match is None and z_match is None):
            fval = extract_feedrate(row)
            if fval is not None and abs(fval - slow_feedrate) > 1e-6:
                last_normal_F = fval
                was_slow = False
            new_data.append(row)
            continue

        if z_match is not None:
            z_layer = float(z_match.group(0).replace('Z', ''))

        x_new = x_old
        y_new = y_old
        if x_match is not None:
            x_new = float(x_match.group(0).replace('X', ''))
        if y_match is not None:
            y_new = float(y_match.group(0).replace('Y', ''))

        # Segment long XY moves
        dist_xy = np.linalg.norm([x_new - x_old, y_new - y_old])
        if maximal_length <= 0:
            num_segm = 1
        else:
            num_segm = max(int(dist_xy // maximal_length + 1), 1)

        x_vals = np.linspace(x_old, x_new, num_segm + 1)
        y_vals = np.linspace(y_old, y_new, num_segm + 1)

        # Compute final Z for each segment endpoint: Z := max(z_layer - Δz(x,y), zmin)
        z_vals = np.empty(num_segm + 1, dtype=float)
        for i, (xx, yy) in enumerate(zip(x_vals, y_vals)):
            delta_z = dz_at(xx, yy, xmin, ymin, dx, dy, dist_out, supported, s, blend)
            z_vals[i] = max(z_layer - delta_z, zmin_clamp)

        # Base row: inject starting Z and rescale E across subsegments
        base_row = insert_Z(row, z_vals[0])
        base_row = replace_E(base_row, num_segm)

        replacement_rows = ""

        for j in range(num_segm):
            sub_x = x_vals[j + 1]
            sub_y = y_vals[j + 1]
            sub_z = z_vals[j + 1]

            # Build one sub-move
            single_row = re.sub(pattern_X, 'X' + str(round(sub_x, 3)), base_row)
            single_row = re.sub(pattern_Y, 'Y' + str(round(sub_y, 3)), single_row)
            single_row = re.sub(pattern_Z, 'Z' + str(round(sub_z, 3)), single_row)

            # Probe geometry (in FINAL printer space after reverse Z)
            p_start = np.array([x_vals[j],     y_vals[j],     z_vals[j]])
            p_mid   = np.array([
                0.5 * (x_vals[j] + x_vals[j + 1]),
                0.5 * (y_vals[j] + y_vals[j + 1]),
                0.5 * (z_vals[j] + z_vals[j + 1])
            ])
            p_end   = np.array([x_vals[j + 1], y_vals[j + 1], z_vals[j + 1]])

            if in_perimeter:
                near_downward = segment_near_downward_surface(
                    p_start, p_mid, p_end, tri_grid, cell_size, dist_tol=0.4
                )
                if near_downward:
                    subsegments_near_downward += 1
            else:
                near_downward = False

            slow_this = (in_perimeter and near_downward)
            if slow_this:
                subsegments_slow_candidates += 1

            # Feedrate state machine
            if slow_this:
                if not was_slow:
                    single_row = clean_and_set_feedrate(single_row, slow_feedrate)
                    was_slow = True
                else:
                    if not single_row.endswith("\n"):
                        single_row = single_row.rstrip() + "\n"
            else:
                fval_here = extract_feedrate(single_row)
                if was_slow:
                    if fval_here is not None and abs(fval_here - slow_feedrate) > 1e-6:
                        last_normal_F = fval_here
                        was_slow = False
                        if not single_row.endswith("\n"):
                            single_row = single_row.rstrip() + "\n"
                    else:
                        if last_normal_F is not None:
                            single_row = clean_and_set_feedrate(single_row, last_normal_F)
                        else:
                            if not single_row.endswith("\n"):
                                single_row = single_row.rstrip() + "\n"
                        was_slow = False
                else:
                    if fval_here is not None and abs(fval_here - slow_feedrate) > 1e-6:
                        last_normal_F = fval_here
                    if not single_row.endswith("\n"):
                        single_row = single_row.rstrip() + "\n"

            replacement_rows += single_row
            total_subsegments += 1

        # update XY
        x_old = x_new
        y_old = y_new
        new_data.append(replacement_rows)

    # Debug
    print("=== DEBUG backtransform_data ===")
    print("perimeter_true_count:", perimeter_true_count)
    print("total_subsegments:", total_subsegments)
    print("subsegments_near_downward:", subsegments_near_downward)
    print("subsegments_slow_candidates:", subsegments_slow_candidates)

    return new_data


# ---------------------------------------------------------------------------
# Pre-segmentation of G-code moves (ONLY for test_2.gcode)
# ---------------------------------------------------------------------------

def pre_subdivide_gcode_moves(data, max_seg_len=0.15):
    """
    Pre-subdivide XY moves in raw slicer G-code BEFORE applying reverse deformation.

    This is used ONLY for test/gcode_parts/test_2.gcode.

    Assumes RELATIVE extrusion (M83): each motion line's E value is an increment.
    That increment is split evenly across subsegments.
    """

    out = []

    pat_cmd = re.compile(r'^(G0|G1)\s')
    patX = re.compile(r'X([-0-9.]+)')
    patY = re.compile(r'Y([-0-9.]+)')
    patE = re.compile(r'E([-0-9.]+)')

    x_old = None
    y_old = None

    for row in data:
        stripped = row.strip()

        # Pass non G0/G1 lines unchanged
        if not pat_cmd.match(stripped):
            out.append(row)
            continue

        mX = patX.search(row)
        mY = patY.search(row)
        mE = patE.search(row)

        # If no XY on this line, keep as-is
        if mX is None and mY is None:
            out.append(row)
            continue

        x_new = x_old if mX is None and x_old is not None else (float(mX.group(1)) if mX is not None else None)
        y_new = y_old if mY is None and y_old is not None else (float(mY.group(1)) if mY is not None else None)

        # First XY occurrence: just record and pass through
        if x_old is None or y_old is None or x_new is None or y_new is None:
            out.append(row)
            x_old = x_new
            y_old = y_new
            continue

        dist = math.hypot(x_new - x_old, y_new - y_old)

        # No subdivision needed
        if dist <= max_seg_len or max_seg_len <= 0:
            out.append(row)
            x_old = x_new
            y_old = y_new
            continue

        # Need subdivision
        N = int(math.ceil(dist / max_seg_len))
        xs = np.linspace(x_old, x_new, N + 1)
        ys = np.linspace(y_old, y_new, N + 1)

        e_total = float(mE.group(1)) if mE is not None else None
        e_per = (e_total / N) if e_total is not None else None

        for i in range(1, N + 1):
            xi = xs[i]
            yi = ys[i]
            new_line = row

            # Replace X/Y
            if mX is not None:
                new_line = patX.sub(f"X{xi:.3f}", new_line)
            else:
                new_line = new_line.rstrip() + f" X{xi:.3f}"

            if mY is not None:
                new_line = patY.sub(f"Y{yi:.3f}", new_line)
            else:
                new_line = new_line.rstrip() + f" Y{yi:.3f}"

            # Split extrusion
            if mE is not None and e_per is not None:
                new_line = patE.sub(f"E{e_per:.5f}", new_line)

            if not new_line.endswith("\n"):
                new_line = new_line + "\n"

            out.append(new_line)

        x_old = x_new
        y_old = y_new

    print(f"[pre_subdivide_gcode_moves] Finished. Output lines: {len(out)}")
    return out


# ---------------------------------------------------------------------------
# Public entry: transformGCode (HEIGHTMAP version)
# ---------------------------------------------------------------------------

def transformGCode(
    in_file: str,
    stl_for_heightmap: str,
    out_dir: str,
    surface_for_slowdown: str,
    maximal_length: float = 1.0,
    x_shift: float = 0.0,   # kept for interface parity; not applied here
    y_shift: float = 0.0,   # kept for interface parity; not applied here
    z_desired: float = 0.1,
    downward_angle_deg: float = 10.0,
    slow_feedrate: float = 180.0,
    # HEIGHTMAP params (must match the ones used during deformation):
    grid_nx: int = 420,
    grid_ny: int = 420,
    z_tol: float = 0.05,
    angle_deg: float = 20.0,
    blend_mm: float = 0.35,
    margin_mm: float = 0.0
):
    """
    Reverse-transform G-code sliced from a HEIGHTMAP-deformed mesh back to printer space.

    in_file            : path to G-code sliced from the DEFORMED STL
    stl_for_heightmap  : ORIGINAL (pre-deformation) STL used to rebuild Δz(x,y)
    out_dir            : output directory for final G-code
    surface_for_slowdown : FINAL geometry STL for downward-facing slowdown detection

    For most files: original stable behavior.
    For test/gcode_parts/test_2.gcode:
        - pre-segmentation before reverse mapping
        - internal segmentation disabled by setting maximal_length huge.
    """
    start = time.time()

    # 1) Read deformed-space G-code
    with open(in_file, 'r', encoding='utf-8', errors='ignore') as f_gcode:
        data = f_gcode.readlines()

    # 1b) Optional pre-segmentation ONLY for test_2.gcode
    use_preseg = _should_presegment(in_file)
    if use_preseg:
        data = pre_subdivide_gcode_moves(data, max_seg_len=PRESEG_MAX_LEN)
        internal_max_length = 1e9  # effectively disable further segmentation
    else:
        internal_max_length = maximal_length

    # 2) Build Δz field from ORIGINAL STL with the SAME params used in heightmap_gen.py
    xmin, ymin, dx, dy, dist_out, supported, s, blend = build_heightmap_field(
        stl_for_heightmap, grid_nx, grid_ny, z_tol, angle_deg, blend_mm, margin_mm
    )

    # 3) Build slowdown geometry index from FINAL geometry STL
    downward_triangles = triangle_data_from_mesh(surface_for_slowdown, max_angle_deg=downward_angle_deg)
    tri_grid, cell_size = build_triangle_spatial_index(downward_triangles, cell_size=2.0)

    # 4) Backtransform (rewrite Z) + slowdown
    data_bt = backtransform_data(
        data=data,
        zmin_clamp=z_desired + 0.2,   # same clamp convention as before
        maximal_length=internal_max_length,
        xmin=xmin, ymin=ymin, dx=dx, dy=dy,
        dist_out=dist_out, supported=supported, s=s, blend=blend,
        tri_grid=tri_grid, cell_size=cell_size,
        slow_feedrate=slow_feedrate
    )
    data_bt_string = ''.join(data_bt)

    # 4b) Smooth global feedrate transitions (fast <-> slow)
    lines = data_bt_string.splitlines(keepends=True)
    lines_smoothed = smooth_feedrate_transitions(
        lines,
        slow_feedrate=slow_feedrate,
        ramp_len=FEEDRATE_RAMP_LEN,  # global ramp length
    )
    data_bt_string = ''.join(lines_smoothed)

    # 5) Save final backtransformed code
    os.makedirs(out_dir, exist_ok=True)
    file_name = os.path.basename(in_file)
    output_path = os.path.join(out_dir, file_name)
    with open(output_path, 'w', newline="\n", encoding='utf-8') as f_gcode_bt:
        f_gcode_bt.write(data_bt_string)


    end = time.time()
    print('GCode generated in {:.1f}s, saved in {}'.format(end - start, output_path))
    return output_path


# ---------------------------------------------------------------------------
# Standalone test hook (optional)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example paths consistent with your structure
    in_file             = os.path.join('gcode_tf', 'Body2.gcode')        # sliced from DEFORMED STL
    stl_for_heightmap   = os.path.join('stl_parts', 'Body2.stl')         # ORIGINAL STL
    surface_for_slowdown= os.path.join('stl_parts', 'Body2.stl')         # FINAL geometry for slowdown
    out_dir             = 'gcode_parts'

    transformGCode(
        in_file=in_file,
        stl_for_heightmap=stl_for_heightmap,
        out_dir=out_dir,
        surface_for_slowdown=surface_for_slowdown,
        maximal_length=0.5,
        x_shift=0.0,
        y_shift=0.0,
        z_desired=0.1,
        downward_angle_deg=10.0,
        slow_feedrate=180.0,
        grid_nx=420, grid_ny=420, z_tol=0.05, angle_deg=20.0, blend_mm=0.35, margin_mm=0.0
    )
