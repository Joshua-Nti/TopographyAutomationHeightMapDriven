# _00all.py  — Clean, audited pipeline (heightmap nonplanar, planar base, 3-column cuts.txt)
import sys
import os
import time
import shutil

from _01analysestl import analyseSTL, segment_has_overhang
from _02cutstl import cutSTL
from _03refinemesh import refineMesh
from _04transformstl import transformSTL
from _05execslicer import execSlicer
from _06transformgcode import transformGCode
from _07combine import combineGCode
from _08movegcode import moveGCode

from _10config import (
    make_folder_dict,
    GEOMETRY_CONFIG,
    PIPELINE_CONFIG,
)

# Default STL if none given
DEFAULT_INPUT = "test.stl"


# --------------------------------------------------------------------------
#  UTILS
# --------------------------------------------------------------------------
def purge_heightmaps(heightmap_dir):
    """
    Delete the entire heightmaps folder before a new run.
    Ensures no stale files conflict when the input model changes.
    """
    if os.path.isdir(heightmap_dir):
        print(f"[PIPELINE] Purging old {heightmap_dir} ...")
        shutil.rmtree(heightmap_dir)

    os.makedirs(heightmap_dir, exist_ok=True)
    print(f"[PIPELINE] Fresh {heightmap_dir} ready.")


def _clear_folder(path):
    """
    Remove all files in a folder.
    Used to avoid mixing old and new runs inside the same base folder.
    """
    if not os.path.isdir(path):
        return
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if os.path.isfile(full):
            os.remove(full)


def createFoldersIfMissing(folder_dict: dict):
    """
    Ensure all working directories exist.
    """
    for path in folder_dict.values():
        os.makedirs(path, exist_ok=True)


def _load_transform_flags(cuts_txt_path):
    """
    Parse cuts.txt (3-column format) and extract per-segment transform flags.

    Expected NEW format:
        index  flag  z_value/TOP

    Returns
    -------
    list[int] or None
        flags[i] corresponds to segment i (0-based indexing).
        If no valid flags are found (e.g., legacy 1-column cuts.txt),
        returns None so the pipeline falls back to geometric overhang detection.
    """
    if not os.path.isfile(cuts_txt_path):
        return None

    flags = []
    has_flags = False

    with open(cuts_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if len(tokens) < 3:
                # Likely legacy format, ignore for flags
                continue
            # tokens: [index, flag, z_or_TOP]
            flag_str = tokens[1]
            try:
                flag = int(flag_str)
            except ValueError:
                continue
            flags.append(flag)
            has_flags = True

    return flags if has_flags else None


# --------------------------------------------------------------------------
#  HEIGHTMAP BACKTRANSFORM WRAPPER
# --------------------------------------------------------------------------

def _backtransform_and_slowdown(
    gcode_in: str,
    coarse_stl_original: str,
    out_dir: str,
) -> str:
    """
    Wrapper around transformGCode() so all numeric knobs live in one place.

    - gcode_in            : G-code sliced from the DEFORMED STL
    - coarse_stl_original : ORIGINAL segment geometry (unrefined) used to
                            rebuild the heightmap Δz(x,y) AND to detect
                            downward-facing surfaces for slowdown.
    - out_dir             : where final printer-space G-code is written.
    """

    max_seg_len = GEOMETRY_CONFIG["maximal_segment_length_mm"]
    down_angle  = GEOMETRY_CONFIG["downward_angle_deg"]
    slow_feed   = GEOMETRY_CONFIG["slow_feedrate_mm_per_min"]
    z_min       = GEOMETRY_CONFIG["z_desired_min_mm"]
    xy_x, xy_y  = GEOMETRY_CONFIG["xy_backtransform_shift_mm"]

    out_path = transformGCode(
        in_file=gcode_in,
        stl_for_heightmap=coarse_stl_original,      # ORIGINAL pre-deform STL
        out_dir=out_dir,
        surface_for_slowdown=coarse_stl_original,   # FINAL geometry for slowdown
        maximal_length=max_seg_len,
        x_shift=xy_x,
        y_shift=xy_y,
        z_desired=z_min,
        downward_angle_deg=down_angle,
        slow_feedrate=slow_feed,
        # MUST match transformSTL:
        grid_nx=420,
        grid_ny=420,
        z_tol=0.05,
        angle_deg=30.0,
        blend_mm=0.35,
        margin_mm=0.0,
    )
    return out_path


# --------------------------------------------------------------------------
#  SLICE + (OPTIONALLY) TRANSFORM ONE SEGMENT
# --------------------------------------------------------------------------

def sliceTransform(
    folder: dict,
    filename: str,
    bottom: bool = False,
    top: bool = False,
    transform_flag=None,
):
    """
    Process one segmented STL from stl_parts/:

      1. Copy a coarse original (unrefined) mesh to stl_coarse/
      2. Determine whether this segment is NONPLANAR based on:
           - transform_flag from cuts.txt (if provided), OR
           - fallback to geometric overhang detection.
      3. If NONPLANAR (and NOT forced planar bottom) →
           refine + heightmap deform + slice + backtransform
      4. Else → planar slice only (no deformation)
    """

    base, ext = os.path.splitext(filename)
    if ext.lower() != ".stl":
        print(f"  [sliceTransform] Skipping non-STL file: {filename}")
        return

    # Per-part paths
    stl_part    = os.path.join(folder["stl_parts"],  base + ".stl")
    stl_coarse  = os.path.join(folder["stl_parts"],  base + ".stl")
    stl_tf      = os.path.join(folder["stl_tf"],     base + ".stl")
    gcode_tf    = os.path.join(folder["gcode_tf"],   base + ".gcode")
    gcode_final = os.path.join(folder["gcode_parts"], base + ".gcode")

    # 1) Coarse copy (original geometry BEFORE refinement/deformation)
    if not os.path.exists(stl_part):
        print(f"  [sliceTransform] WARNING: missing {stl_part}, skipping.")
        return

    if not os.path.exists(stl_coarse):
        shutil.copyfile(stl_part, stl_coarse)
        print(f"  [sliceTransform] Saved coarse copy → {stl_coarse}")

    # 2) Decide NONPLANAR FLAG for this SEGMENT
    if transform_flag is None:
        # Fallback: compute from geometry (legacy behaviour)
        has_overhang = segment_has_overhang(stl_part)
        print(
            f"  [sliceTransform] segment_has_overhang({filename}) "
            f"(fallback) = {has_overhang}"
        )
    else:
        has_overhang = bool(transform_flag)
        print(
            f"  [sliceTransform] Using transform_flag from cuts.txt "
            f"for {filename} → {has_overhang} (flag={transform_flag})"
        )

    # 3) BOTTOM OVERRIDE — ONLY if this is a TRUE bottom (multi-part)
    #    If a model has only one segment (bottom=True & top=True),
    #    we DO NOT override; it may be deformed & reverse-transformed.
    if bottom and not top:
        if has_overhang:
            print(
                "  [sliceTransform] BOTTOM OVERRIDE: segment has overhangs "
                "but is forced planar to keep a stable base."
            )
        has_overhang = False

    # 4) NONPLANAR PATH  (overhanging AND not forced planar)
    if has_overhang:
        print(f"  [sliceTransform] Nonplanar segment → deform & backtransform: {filename}")

        # 4.1 Refine mesh IN-PLACE for smoother deformation
        refine_len = GEOMETRY_CONFIG["refine_edge_length_mm"]
        print(f"    Refining mesh (edge length ≤ {refine_len} mm)…")
        refineMesh(stl_part, refine_len)

        # 4.2 Apply heightmap (column-freeze) deformation
        print("    Applying column-freeze heightmap deformation (transformSTL)…")
        tf_stl_path = transformSTL(
            in_body=stl_part,
            in_transform=None,      # ignored in column-freeze mode
            out_dir=folder["stl_tf"],
            heightmap_dir=folder["heightmaps"],
            grid_nx=420,
            grid_ny=420,
            z_tol=0.05,
            angle_deg=30.0,
            blend_mm=0.35,
            margin_mm=0.0,
        )

        # 4.3 Slice the DEFORMED mesh
        print("    Slicing transformed STL (execSlicer, transformed=True)…")
        execSlicer(
            in_file=tf_stl_path,
            out_file=gcode_tf,
            bottom_stl=bottom,
            top_stl=top,
            transformed=True,
        )

        # 4.4 Reverse-transform G-code back to printer space + slowdown
        print("    Backtransforming & slowdown (transformGCode)…")
        _backtransform_and_slowdown(
            gcode_in=gcode_tf,
            coarse_stl_original=stl_coarse,
            out_dir=folder["gcode_parts"],
        )

    # 5) PLANAR PATH  (no overhang OR forced planar bottom)
    else:
        print(f"  [sliceTransform] Planar segment (no deformation): {filename}")

        # Slice the original planar segment directly
        execSlicer(
            in_file=stl_part,
            out_file=gcode_final,
            bottom_stl=bottom,
            top_stl=top,
            transformed=False,
        )

        # NOTE:
        # We DO NOT call transformGCode for planar parts, because the mesh
        # was never deformed. If in future you want "slowdown-only" on planar
        # parts, that should be a separate path using only the FINAL geometry.


# --------------------------------------------------------------------------
#  SLICE ALL SEGMENTS
# --------------------------------------------------------------------------

def sliceAll(folder: dict, segment_flags=None):
    """
    Walk through stl_parts/ in sorted order and process each segment.
    The first segment is 'bottom', the last is 'top'.

    If segment_flags is provided (list of 0/1), it is used to decide
    which segments are transformed. Otherwise, the pipeline falls back
    to geometric overhang detection.
    """

    parts = [
        f for f in os.listdir(folder["stl_parts"])
        if f.lower().endswith(".stl") and not f.startswith(".")
    ]
    parts.sort()

    if not parts:
        print("[sliceAll] No segmented STL parts found in stl_parts/.")
        return

    for idx, fname in enumerate(parts):
        bottom = (idx == 0)
        top    = (idx == len(parts) - 1)

        # Map flag list (0/1) to this segment index
        flag = None
        if segment_flags is not None and idx < len(segment_flags):
            flag = segment_flags[idx]

        print(f"\n=== Processing {fname}  | bottom={bottom} | top={top} | flag={flag} ===")
        sliceTransform(
            folder,
            fname,
            bottom=bottom,
            top=top,
            transform_flag=flag,
        )


# --------------------------------------------------------------------------
#  MAIN PIPELINE
# --------------------------------------------------------------------------

def main(input_stl: str):

    base = os.path.splitext(os.path.basename(input_stl))[0]
    folders = make_folder_dict(base)
    purge_heightmaps(folders["heightmaps"])

    print("\n=== Creating pipeline folders ===")
    createFoldersIfMissing(folders)

    # Clean working dirs to avoid mixing old/new runs
    for key in ["stl_parts", "stl_coarse", "stl_tf", "gcode_tf", "gcode_parts"]:
        if key in folders:
            print(f"[init] Clearing folder: {folders[key]}")
            _clear_folder(folders[key])

    cuts_txt    = os.path.join(folders["root"], "cuts.txt")
    working_stl = os.path.join(folders["root"], base + ".stl")

    cuts_already_exists = os.path.exists(cuts_txt)

    # 1. Analyse STL → only if cuts.txt does NOT already exist
    if cuts_already_exists:
        print("\n=== Using existing cuts.txt (skipping automatic analysis) ===")
        print("    cuts.txt:", cuts_txt)
    else:
        print("\n=== Analysing STL for cut heights & transform flags ===")
        analyseSTL(input_stl, cuts_txt)

    # 2. Copy STL into working directory (so we can safely mutate it in cutSTL)
    if PIPELINE_CONFIG.get("copy_input_to_work", True):
        print("Copying input STL into working root…")
        shutil.copyfile(input_stl, working_stl)
        stl_for_cut = working_stl
    else:
        stl_for_cut = input_stl

    # 3. Cut STL into vertical segments (uses safety-offset logic in _02cutstl)
    print("\n=== Cutting STL into parts ===")
    cutSTL(stl_for_cut, cuts_txt, folders["stl_parts"])

    # Load transform flags from cuts.txt (if present in 3-column form)
    segment_flags = _load_transform_flags(cuts_txt)
    if segment_flags is not None:
        print("[main] Loaded transform flags from cuts.txt:", segment_flags)
    else:
        print("[main] No valid transform flags found in cuts.txt → "
              "falling back to geometric overhang detection.")

    # 4. Slice + (if needed) deform segments
    print("\n=== Slicing all parts ===")
    sliceAll(folders, segment_flags=segment_flags)

    # 5. Combine per-part G-code
    combined_path = os.path.join(folders["root"], base + ".gcode")
    print("\n=== Combining G-code ===")
    combineGCode(folders["gcode_parts"], combined_path)

    # 6. Optional final XY shift on the whole merged toolpath
    shifted_path = os.path.join(base + "_moved.gcode")

    if PIPELINE_CONFIG.get("apply_final_shift", False):
        print("\n=== Applying final XY shift ===")
        x_off, y_off = PIPELINE_CONFIG["final_shift_xy_mm"]
        moveGCode(combined_path, shifted_path, x_off, y_off)
        print("Shifted file:", shifted_path)
    else:
        print("Skipping final XY shift (PIPELINE_CONFIG.apply_final_shift = False).")

    print("\n=== PIPELINE COMPLETE ===")
    print("Combined G-code (unshifted):", combined_path)


if __name__ == "__main__":
    stl_arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    main(stl_arg)