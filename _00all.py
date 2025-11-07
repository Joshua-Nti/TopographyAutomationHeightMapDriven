# _00all.py
import sys
import os
import time
import shutil

from _01analysestl import analyseSTL
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

# default entry if user doesn't pass an STL
in_file = "test.stl"


def sliceTransform(folder, filename, bottom=False, top=False):
    """
    Process a single STL part file:
    - create a coarse copy for slowdown detection
    - refine mesh
    - apply transform if tf_surface exists
    - slice with SuperSlicer
    - backtransform/slowdown G-code if nonplanar OR if config says so
    """

    base_no_ext, ext = os.path.splitext(filename)
    if ext.lower() == ".stl":
        part_name = base_no_ext
    elif ext == "":
        part_name = filename
    else:
        print(f"skip {filename}: not an STL")
        return

    stl_parts_path   = os.path.join(folder["stl_parts"],   part_name + ".stl")
    tf_surfaces_path = os.path.join(folder["tf_surfaces"], part_name + ".stl")
    stl_tf_path      = os.path.join(folder["stl_tf"],      part_name + ".stl")
    gcode_tf_path    = os.path.join(folder["gcode_tf"],    part_name + ".gcode")
    gcode_parts_path = os.path.join(folder["gcode_parts"], part_name + ".gcode")
    coarse_stl_path  = os.path.join(folder["stl_coarse"],  part_name + ".stl")

    # A part is considered "nonplanar" if there's a matching transform surface.
    is_nonplanar = os.path.exists(tf_surfaces_path)

    # Coarse copy (for slowdown geometry). We keep the original, unrefined STL here.
    if not os.path.exists(coarse_stl_path):
        shutil.copyfile(stl_parts_path, coarse_stl_path)

    # Step 1: refine mesh if we are going to transform
    # (We keep the same logic you had: refine only for nonplanar parts,
    #  because planar parts don't need dense tessellation.)
    if is_nonplanar:
        print("  refining mesh:", stl_parts_path)
        refineMesh(stl_parts_path, GEOMETRY_CONFIG["refine_edge_length_mm"])

        # Step 2: apply nonplanar transform
        print("  transforming STL using tf_surface:", tf_surfaces_path)
        transformSTL(stl_parts_path, tf_surfaces_path, folder["stl_tf"])

        # Step 3: slice transformed STL -> gcode_tf
        print("  slicing transformed part:", part_name)
        execSlicer(
            in_file=stl_tf_path,
            out_file=gcode_tf_path,
            bottom_stl=bottom,
            top_stl=top,
            transformed=True,
        )

        # Step 4: backtransform & slowdown -> gcode_parts
        print("  backtransform + slowdown:", part_name)
        _backtransform_and_slowdown(
            gcode_tf_path,
            tf_surfaces_path,
            folder["gcode_parts"],
            coarse_stl_path,
        )

    else:
        # Planar path
        print("  slicing planar part:", part_name)
        execSlicer(
            in_file=stl_parts_path,
            out_file=gcode_parts_path,
            bottom_stl=bottom,
            top_stl=top,
            transformed=False,
        )

        # Optionally also run backtransform/slowdown for planar parts if requested
        if PIPELINE_CONFIG["apply_backtransform_to_planar"]:
            print("  (config) also applying slowdown to planar part:", part_name)
            _backtransform_and_slowdown(
                gcode_parts_path,     # input is already planar G-code
                tf_surfaces_path,     # might not exist for planar -> you’d adapt if needed
                folder["gcode_parts"],
                coarse_stl_path,
            )


def _backtransform_and_slowdown(
    gcode_in,
    tf_surface_stl,
    out_dir,
    coarse_surface_for_slowdown,
):
    """
    Helper to call transformGCode() using GEOMETRY_CONFIG.
    This wraps all those numeric knobs in one place.
    """

    max_seg_len   = GEOMETRY_CONFIG["maximal_segment_length_mm"]
    down_angle    = GEOMETRY_CONFIG["downward_angle_deg"]
    slow_feed     = GEOMETRY_CONFIG["slow_feedrate_mm_per_min"]
    z_min         = GEOMETRY_CONFIG["z_desired_min_mm"]
    xy_shift_x, xy_shift_y = GEOMETRY_CONFIG["xy_backtransform_shift_mm"]

    # NOTE: our transformGCode signature from before was:
    # transformGCode(
    #   in_file,
    #   in_transform_for_interp,
    #   out_dir,
    #   surface_for_slowdown,
    #   maximal_length = ...,
    #   x_shift = ...,
    #   y_shift = ...,
    #   z_desired = ...,
    #   downward_angle_deg = ...,
    #   slow_feedrate = ...
    # )

    # ✅ Replace the old call with this:
    orig_stl_path = coarse_surface_for_slowdown

    transformGCode(
        in_file=gcode_in,
        stl_for_heightmap=orig_stl_path,          # ← MUST be the ORIGINAL (pre-deform) STL
        out_dir=out_dir,
        surface_for_slowdown=coarse_surface_for_slowdown,  # FINAL STL for slowdown checks
        maximal_length=max_seg_len,
        x_shift=xy_shift_x,
        y_shift=xy_shift_y,
        z_desired=z_min,
        downward_angle_deg=down_angle,
        slow_feedrate=slow_feed,
        # If you are NOT using the sidecar, also pass the exact heightmap params:
        grid_nx=420, grid_ny=420, z_tol=0.05,
        angle_deg=20, blend_mm=0.35, margin_mm=0.0
    )



def sliceAll(folder):
    """
    Go through all stl_parts/*.stl in sort order.
    Mark first as bottom=True, last as top=True,
    pass flags forward.
    """
    parts = [
        f for f in os.listdir(folder["stl_parts"])
        if f.lower().endswith(".stl") and not f.startswith(".")
    ]
    parts.sort()

    if not parts:
        print("No STL parts found.")
        return

    for idx, stlfile in enumerate(parts):
        bottom_flag = (idx == 0)
        top_flag    = (idx == len(parts) - 1)

        print("Processing:",
              stlfile,
              "| bottom =", bottom_flag,
              "| top =", top_flag)

        sliceTransform(
            folder,
            stlfile,
            bottom=bottom_flag,
            top=top_flag,
        )


def createFoldersIfMissing(folder_dict):
    """
    Ensure all working directories exist.
    """
    for key in folder_dict:
        p = folder_dict[key]
        if not os.path.exists(p):
            os.makedirs(p, exist_ok=True)


def main(input_stl):
    """
    Full pipeline:
    1. Build folder layout
    2. Analyse STL -> cuts.txt
    3. Copy STL into work dir, run cutSTL() to produce stl_parts
    4. sliceAll() -> slice each part (nonplanar or planar)
    5. combineGCode() merges all part G-codes
    6. moveGCode() shifts final merged toolpath on the bed (optional)
    """

    base_name = os.path.splitext(os.path.basename(input_stl))[0]
    folders = make_folder_dict(base_name)

    print("Create folders if missing …")
    t0 = time.time()
    createFoldersIfMissing(folders)

    cuts_txt_path = os.path.join(folders["root"], "cuts.txt")
    working_input_stl = os.path.join(folders["root"], base_name + ".stl")

    # Analyse STL (if we don't have cuts already)
    if not os.path.exists(cuts_txt_path):
        print("Analyse STL to determine cut heights …")
        analyseSTL(input_stl, cuts_txt_path)

    # Copy the input STL into working root (if config says so)
    if PIPELINE_CONFIG["copy_input_to_work"]:
        print("Copy input STL into working root …")
        shutil.copy(input_stl, working_input_stl)
        stl_for_cutting = working_input_stl
    else:
        stl_for_cutting = input_stl

    # Cut STL into individual vertical segments -> stl_parts/
    print("Cut STL into vertical parts …")
    cutSTL(
        in_stl=stl_for_cutting,
        cuts_file=cuts_txt_path,
        out_folder=folders["stl_parts"],
    )

    # Slice all parts (and transform/backtransform where needed)
    print("Slice all parts …")
    sliceAll(folders)

    # Combine final segments' G-code into one combined file
    combined_gcode_path = os.path.join(folders["root"], base_name + ".gcode")
    print("Combine G-code parts …")
    combineGCode(
        in_folder=folders["gcode_parts"],
        out_file=combined_gcode_path,
    )

    # Optionally apply final XY shift
    shifted_gcode_path = os.path.join(folders["root"], base_name + "_moved.gcode")

    if PIPELINE_CONFIG["apply_final_shift"]:
        print("Apply final XY shift to merged G-code …")
        x_off, y_off = PIPELINE_CONFIG["final_shift_xy_mm"]
        moveGCode(
            in_file=combined_gcode_path,
            out_file=shifted_gcode_path,
            x_off=x_off,
            y_off=y_off,
        )
        print("Shifted G-code:", shifted_gcode_path)
    else:
        print("Skipping final XY shift (config).")

    dt = time.time() - t0
    print("Pipeline finished in {:.1f}s".format(dt))
    print("Combined (unshifted):", combined_gcode_path)


if __name__ == "__main__":
    stl_arg = in_file
    if len(sys.argv) > 1:
        stl_arg = sys.argv[1]
    main(stl_arg)