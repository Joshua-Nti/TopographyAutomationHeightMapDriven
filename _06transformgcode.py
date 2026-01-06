# _06transformgcode.py
# Reverse-transform G-code sliced from a HEIGHTMAP-deformed STL back to printer space,
# now using the STORED heightmap (*.npz) produced by _04transformstl.py
# instead of recomputing the heightmap (no EDT rebuild here).
#
# Behavior:
#  - For most models: original stable behavior
#       * XY segmentation in backtransform_data (maximal_length)
#       * slowdown on downward-facing perimeters
#  - For selected files:
#       * Pre-segmentation of XY moves BEFORE reverse mapping
#       * Internal segmentation in backtransform_data effectively disabled
#
# Inputs:
#   - in_file: G-code produced by slicing the DEFORMED mesh
#   - heightmap_npz: saved heightmap file from transformSTL (e.g. test/heightmaps/<base>_heightmap.npz)
#   - surface_for_slowdown: FINAL geometry STL (for downward perimeter slowdown)
#
# Outputs:
#   - Writes backtransformed G-code to out_dir with the same basename
#
# Dependencies: numpy, numpy-stl (slowdown only), scipy not needed for heightmap anymore

import re
import os
import time
import math
import numpy as np
from stl import mesh  # only used for slowdown triangle extraction


# ---------------------------------------------------------------------------
#  CONFIG: which file gets pre-segmentation
# ---------------------------------------------------------------------------

PRESEG_MAX_LEN = 0.5
SEGMENT_TOGGLE = False

# Global ramp length for feedrate transitions (number of motion lines
# used to smooth fast->slow and slow->fast feedrate jumps).
FEEDRATE_RAMP_LEN = 7


def _should_presegment(in_file: str) -> bool:
    """
    Return True ONLY for these exact files:
        test/gcode_tf/test_2.gcode
        test/gcode_tf/test_4.gcode
        test/gcode_tf/test_5.gcode

    Any other G-code file → pre-segmentation DISABLED.
    """
    norm_path = os.path.normpath(os.path.abspath(in_file))

    TARGET_TAILS = [
        os.path.normpath(os.path.join("test", "gcode_tf", "test_2.gcode")),
        os.path.normpath(os.path.join("test", "gcode_tf", "test_4.gcode")),
        os.path.normpath(os.path.join("test", "gcode_tf", "test_5.gcode")),
    ]

    for tail in TARGET_TAILS:
        if norm_path.endswith(tail):
            print(f"[transformGCode] Pre-segmentation ENABLED for {tail}")
            return SEGMENT_TOGGLE

    print("[transformGCode] Pre-segmentation DISABLED for this file.")
    return False


# ---------------------------------------------------------------------------
# Heightmap utilities — LOAD STORED HEIGHTMAP (NO RECOMPUTE)
# ---------------------------------------------------------------------------

def _normalize_xy_to_meshgrid(X, Y, DZ):
    """
    Ensure X,Y are 2D meshgrids matching DZ.
    Supports:
      - X,Y already 2D arrays matching DZ
      - X,Y as 1D vectors (will meshgrid)
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    DZ = np.asarray(DZ)

    if X.ndim == 2 and Y.ndim == 2 and X.shape == DZ.shape and Y.shape == DZ.shape:
        return X, Y, DZ

    if X.ndim == 1 and Y.ndim == 1:
        XX, YY = np.meshgrid(X, Y)
        if XX.shape != DZ.shape:
            raise ValueError(
                f"Heightmap shape mismatch: DZ.shape={DZ.shape} but meshgrid(X,Y).shape={XX.shape}"
            )
        return XX, YY, DZ

    raise ValueError(
        f"Unsupported heightmap shapes: X.shape={X.shape}, Y.shape={Y.shape}, DZ.shape={DZ.shape}"
    )


def load_heightmap_npz(heightmap_npz: str):
    """
    Load stored heightmap from _04transformstl.py output NPZ:
      - X, Y (grid)
      - DZ (heightmap)
      - supported (mask)
      - xmin, ymin, dx, dy (grid metadata)

    Returns:
      xmin, ymin, dx, dy, DZ, supported, X, Y
    """
    if not os.path.isfile(heightmap_npz):
        raise FileNotFoundError(f"Heightmap NPZ not found: {heightmap_npz}")

    with np.load(heightmap_npz, allow_pickle=False) as data:
        keys = set(data.files)

        def pick(*cands):
            for k in cands:
                if k in keys:
                    return data[k]
            return None

        X = pick("X", "x", "XX")
        Y = pick("Y", "y", "YY")
        DZ = pick("DZ", "dz", "dZ", "deltaZ", "DeltaZ")
        supported = pick("supported", "mask", "footprint", "inside_mask", "valid", "inside")

        xmin = pick("xmin")
        ymin = pick("ymin")
        dx = pick("dx")
        dy = pick("dy")

        if X is None or Y is None or DZ is None:
            raise KeyError(
                f"Heightmap NPZ missing required keys. Expected at least X, Y, DZ. Found: {sorted(keys)}"
            )

        if supported is None:
            # If not present, derive supported from DZ == 0 (works for your pipeline)
            supported = (np.asarray(DZ) == 0.0)

        if xmin is None or ymin is None or dx is None or dy is None:
            # Derive from X,Y if metadata missing (still works)
            Xn, Yn, DZn = _normalize_xy_to_meshgrid(X, Y, DZ)
            xmin = float(np.min(Xn))
            ymin = float(np.min(Yn))
            # Assuming uniform grid
            dx = float(Xn[0, 1] - Xn[0, 0]) if Xn.shape[1] > 1 else 1.0
            dy = float(Yn[1, 0] - Yn[0, 0]) if Yn.shape[0] > 1 else 1.0
            X, Y, DZ = Xn, Yn, DZn
        else:
            xmin = float(xmin)
            ymin = float(ymin)
            dx = float(dx)
            dy = float(dy)
            X, Y, DZ = _normalize_xy_to_meshgrid(X, Y, DZ)

        supported = np.asarray(supported).astype(bool)
        if supported.shape != DZ.shape:
            # last-resort: derive from DZ==0
            supported = (np.asarray(DZ) == 0.0)

        return xmin, ymin, dx, dy, np.asarray(DZ, dtype=float), supported, X, Y


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


def dz_at_from_saved_heightmap(x, y, xmin, ymin, dx, dy, DZ, supported):
    """
    Compute Δz(x,y) directly from the STORED DZ field.
    Column-freeze behavior is preserved by returning 0 inside supported footprint.
    """
    u = int(round((x - xmin) / dx))
    v = int(round((y - ymin) / dy))
    ny, nx = supported.shape
    u = max(0, min(nx - 1, u))
    v = max(0, min(ny - 1, v))
    if supported[v, u]:
        return 0.0

    return float(bilinear_sample(DZ, x, y, xmin, ymin, dx, dy))


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
# G-code helpers (unchanged)
# ---------------------------------------------------------------------------

def insert_Z(row, z_value):
    pattern_Z = r'Z[-0-9.]+[.]?[0-9]*'
    m = re.search(pattern_Z, row)
    if m is not None:
        return re.sub(pattern_Z, ' Z' + str(round(z_value, 3)), row)
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
    if ramp_len is None:
        ramp_len = FEEDRATE_RAMP_LEN

    EPS = 1e-6
    out = list(lines)

    # PASS 1: FAST -> SLOW
    last_F = None
    for i in range(len(out)):
        line = out[i]
        F_here = extract_feedrate(line)
        if F_here is not None:
            if (
                last_F is not None
                and last_F > slow_feedrate + EPS
                and abs(F_here - slow_feedrate) < EPS
            ):
                ramp_indices = []
                j = i
                while j >= 0 and len(ramp_indices) < ramp_len:
                    l_j = out[j]
                    stripped_j = l_j.lstrip()
                    if stripped_j.startswith("G0") or stripped_j.startswith("G1"):
                        ramp_indices.append(j)
                    j -= 1

                ramp_indices.sort()
                total = len(ramp_indices)

                if total > 0:
                    F_start = last_F
                    F_target = slow_feedrate
                    for k, idx in enumerate(ramp_indices):
                        t = float(k + 1) / float(total)
                        F_k = F_start + (F_target - F_start) * t
                        out[idx] = clean_and_set_feedrate(out[idx], F_k)
                    last_F = slow_feedrate
                else:
                    last_F = F_here
            else:
                last_F = F_here

    # PASS 2: SLOW -> FAST
    result = out
    last_F = None
    i = 0
    n = len(result)

    while i < n:
        line = result[i]
        F_here = extract_feedrate(line)

        if F_here is not None:
            if (
                last_F is not None
                and abs(last_F - slow_feedrate) < EPS
                and F_here > slow_feedrate + EPS
            ):
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
                    result[i] = clean_and_set_feedrate(result[i], slow_feedrate)
                    last_F = slow_feedrate
                    i += 1
                    continue

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
                        t = float(k + 1) / float(total)
                        F_k = F_start + (F_target - F_start) * t
                        result[idx] = clean_and_set_feedrate(result[idx], F_k)

                    last_F = F_target
                    i = ramp_indices[-1] + 1
                    continue
                else:
                    result[i] = clean_and_set_feedrate(result[i], F_target)
                    last_F = F_target
                    i += 1
                    continue
            else:
                last_F = F_here

        i += 1

    return result


# ---------------------------------------------------------------------------
# Core: backtransform_data using STORED HEIGHTMAP DZ
# ---------------------------------------------------------------------------

def backtransform_data(
    data,
    zmin_clamp,
    maximal_length,
    # heightmap field components (loaded from NPZ):
    xmin, ymin, dx, dy, DZ, supported,
    # slowdown geometry:
    tri_grid,
    cell_size,
    slow_feedrate=180.0
):
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

        # Move line
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

        # Z := max(z_layer - DZ(x,y), zmin)
        z_vals = np.empty(num_segm + 1, dtype=float)
        for i, (xx, yy) in enumerate(zip(x_vals, y_vals)):
            delta_z = dz_at_from_saved_heightmap(xx, yy, xmin, ymin, dx, dy, DZ, supported)
            z_vals[i] = max(z_layer - delta_z, zmin_clamp)

        base_row = insert_Z(row, z_vals[0])
        base_row = replace_E(base_row, num_segm)

        replacement_rows = ""

        for j in range(num_segm):
            sub_x = x_vals[j + 1]
            sub_y = y_vals[j + 1]
            sub_z = z_vals[j + 1]

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

        x_old = x_new
        y_old = y_new
        new_data.append(replacement_rows)

    print("=== DEBUG backtransform_data ===")
    print("perimeter_true_count:", perimeter_true_count)
    print("total_subsegments:", total_subsegments)
    print("subsegments_near_downward:", subsegments_near_downward)
    print("subsegments_slow_candidates:", subsegments_slow_candidates)

    return new_data


# ---------------------------------------------------------------------------
# Pre-segmentation of G-code moves (for selected files)
# ---------------------------------------------------------------------------

def pre_subdivide_gcode_moves(data, max_seg_len=0.15):
    out = []

    pat_cmd = re.compile(r'^(G0|G1)\s')
    patX = re.compile(r'X([-0-9.]+)')
    patY = re.compile(r'Y([-0-9.]+)')
    patE = re.compile(r'E([-0-9.]+)')

    x_old = None
    y_old = None

    for row in data:
        stripped = row.strip()

        if not pat_cmd.match(stripped):
            out.append(row)
            continue

        mX = patX.search(row)
        mY = patY.search(row)
        mE = patE.search(row)

        if mX is None and mY is None:
            out.append(row)
            continue

        x_new = x_old if mX is None and x_old is not None else (float(mX.group(1)) if mX is not None else None)
        y_new = y_old if mY is None and y_old is not None else (float(mY.group(1)) if mY is not None else None)

        if x_old is None or y_old is None or x_new is None or y_new is None:
            out.append(row)
            x_old = x_new
            y_old = y_new
            continue

        dist = math.hypot(x_new - x_old, y_new - y_old)

        if dist <= max_seg_len or max_seg_len <= 0:
            out.append(row)
            x_old = x_new
            y_old = y_new
            continue

        N = int(math.ceil(dist / max_seg_len))
        xs = np.linspace(x_old, x_new, N + 1)
        ys = np.linspace(y_old, y_new, N + 1)

        e_total = float(mE.group(1)) if mE is not None else None
        e_per = (e_total / N) if e_total is not None else None

        for i in range(1, N + 1):
            xi = xs[i]
            yi = ys[i]
            new_line = row

            if mX is not None:
                new_line = patX.sub(f"X{xi:.3f}", new_line)
            else:
                new_line = new_line.rstrip() + f" X{xi:.3f}"

            if mY is not None:
                new_line = patY.sub(f"Y{yi:.3f}", new_line)
            else:
                new_line = new_line.rstrip() + f" Y{yi:.3f}"

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
# Public entry: transformGCode (HEIGHTMAP version, using stored NPZ)
# ---------------------------------------------------------------------------

def transformGCode(
    in_file: str,
    heightmap_npz: str,
    out_dir: str,
    surface_for_slowdown: str,
    maximal_length: float = 1.0,
    x_shift: float = 0.0,   # kept for interface parity; not applied here
    y_shift: float = 0.0,   # kept for interface parity; not applied here
    z_desired: float = 0.1,
    downward_angle_deg: float = 10.0,
    slow_feedrate: float = 180.0,
):
    """
    Reverse-transform G-code sliced from a HEIGHTMAP-deformed mesh back to printer space.

    in_file            : path to G-code sliced from the DEFORMED STL
    heightmap_npz      : stored NPZ from transformSTL (contains DZ + grid metadata)
    out_dir            : output directory for final G-code
    surface_for_slowdown : FINAL geometry STL for downward-facing slowdown detection
    """
    start = time.time()

    # 1) Read deformed-space G-code
    with open(in_file, 'r', encoding='utf-8', errors='ignore') as f_gcode:
        data = f_gcode.readlines()

    # 1b) Optional pre-segmentation
    use_preseg = _should_presegment(in_file)
    if use_preseg:
        data = pre_subdivide_gcode_moves(data, max_seg_len=PRESEG_MAX_LEN)
        internal_max_length = 1e9
    else:
        internal_max_length = maximal_length

    # 2) LOAD STORED HEIGHTMAP (no recompute)
    xmin, ymin, dx, dy, DZ, supported, _, _ = load_heightmap_npz(heightmap_npz)
    print(f"[transformGCode] Using stored heightmap: {heightmap_npz}")
    print(f"[transformGCode]   grid: {DZ.shape[1]}x{DZ.shape[0]}  dx={dx:.6f}  dy={dy:.6f}")

    # 3) Build slowdown geometry index from FINAL geometry STL
    downward_triangles = triangle_data_from_mesh(surface_for_slowdown, max_angle_deg=downward_angle_deg)
    tri_grid, cell_size = build_triangle_spatial_index(downward_triangles, cell_size=2.0)

    # 4) Backtransform Z + slowdown
    data_bt = backtransform_data(
        data=data,
        zmin_clamp=z_desired + 0.2,
        maximal_length=internal_max_length,
        xmin=xmin, ymin=ymin, dx=dx, dy=dy,
        DZ=DZ, supported=supported,
        tri_grid=tri_grid, cell_size=cell_size,
        slow_feedrate=slow_feedrate
    )
    data_bt_string = ''.join(data_bt)

    # 4b) Smooth global feedrate transitions
    lines = data_bt_string.splitlines(keepends=True)
    lines_smoothed = smooth_feedrate_transitions(
        lines,
        slow_feedrate=slow_feedrate,
        ramp_len=FEEDRATE_RAMP_LEN,
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
    in_file = os.path.join('test', 'gcode_tf', 'test_2.gcode')  # sliced from DEFORMED STL
    heightmap_npz = os.path.join('test', 'heightmaps', 'test_2_heightmap.npz')  # from transformSTL
    surface_for_slowdown = os.path.join('test', 'stl_tf', 'test_2.stl')  # FINAL geometry STL
    out_dir = os.path.join('test', 'gcode_parts')

    transformGCode(
        in_file=in_file,
        heightmap_npz=heightmap_npz,
        out_dir=out_dir,
        surface_for_slowdown=surface_for_slowdown,
        maximal_length=0.5,
        z_desired=0.1,
        downward_angle_deg=10.0,
        slow_feedrate=180.0,
    )
