# _08movegcode.py
import os
import re
from _10config import PIPELINE_CONFIG

def moveGCode(in_file, out_file, x_off=None, y_off=None):
    """
    Create a shifted copy of the merged (combined) G-code.

    Goal 🎯:
    - Keep purge / intro / warmup stuff at the original XY (usually near origin).
    - Move the actual printed object to a new XY offset (e.g. +100 / +100 mm).

    Trigger logic (same approach that already worked for you):
    1. We watch for the first occurrence of ';LAYER_CHANGE'. After we see it,
       we are now "armed": we know the real print is about to start soon.
       BUT we still do NOT shift yet.

    2. After we're armed, the first movement command (G0 or G1) that
       actually has an X or Y coordinate will activate shifting.
       From that point on, all following XY moves are shifted.

    Why this works ✅:
    - SuperSlicer typically inserts ';LAYER_CHANGE' just before it starts
      laying down the first layer of the actual model.
    - Purge lines before that comment are left unshifted, so the purge
      still happens at the origin.
    - After we hit the first "real" perimeter/infill move, we start shifting.

    Parameters
    ----------
    in_file : str
        Path to the merged, unshifted G-code (e.g. "<root>/<name>.gcode")
    out_file : str
        Path to write the shifted G-code (e.g. "<root>/<name>_moved.gcode")
    x_off, y_off : float or None
        Override XY shift in mm. If None, we read PIPELINE_CONFIG["final_shift_xy_mm"].

    Behavior is OS-independent. Paths are normalized using os.path.
    """

    # pull XY shift either from arguments or from config
    if x_off is None or y_off is None:
        x_shift, y_shift = PIPELINE_CONFIG["final_shift_xy_mm"]
    else:
        x_shift, y_shift = float(x_off), float(y_off)

    # regex to detect X and Y coordinates in motion lines
    pat_X = re.compile(r'X(-?\d+\.?\d*)')
    pat_Y = re.compile(r'Y(-?\d+\.?\d*)')

    # normalize absolute paths for clarity / portability
    in_abs  = os.path.abspath(in_file)
    out_abs = os.path.abspath(out_file)

    with open(in_abs, 'r') as f_in:
        lines = f_in.readlines()

    shifted_lines = []

    # state machine flags
    seen_layer_change = False   # set to True right after first ';LAYER_CHANGE'
    shift_active      = False   # once True, we start shifting all future XY moves

    for line in lines:
        stripped = line.strip()

        # 1. Arm the shift logic when we hit the first layer change marker.
        #    We do NOT immediately shift on this same line.
        if not seen_layer_change:
            if stripped.startswith(";LAYER_CHANGE"):
                seen_layer_change = True
                # write ';LAYER_CHANGE' line unmodified
                shifted_lines.append(line)
                continue

        # 2. If we've already seen ';LAYER_CHANGE' but haven't started shifting yet:
        #    The first subsequent motion line with X or Y will activate shifting.
        #
        #    We consider a "motion line" to be G0 or G1. We don't require extrusion E here,
        #    because in your working version you activated shift immediately
        #    on the first XY move after ';LAYER_CHANGE', even if it's travel.
        if (not shift_active) and seen_layer_change:
            is_move = stripped.startswith("G0") or stripped.startswith("G1")
            has_x = (" X" in line or stripped.startswith("X"))
            has_y = (" Y" in line or stripped.startswith("Y"))
            if is_move and (has_x or has_y):
                # activate shift from here on
                shift_active = True
                # fall through: this SAME line should be shifted

        if not shift_active:
            # Not yet shifting: write line as-is
            shifted_lines.append(line)
            continue

        # 3. Shift-active phase:
        #    From here on, all XY motion is shifted by x_shift/y_shift.
        def _shift_x(m):
            old_val = float(m.group(1))
            new_val = old_val + x_shift
            return f"X{new_val:.3f}"

        def _shift_y(m):
            old_val = float(m.group(1))
            new_val = old_val + y_shift
            return f"Y{new_val:.3f}"

        new_line = pat_X.sub(_shift_x, line)
        new_line = pat_Y.sub(_shift_y, new_line)

        shifted_lines.append(new_line)

    # ensure output directory exists (cross-platform safe)
    out_dir = os.path.dirname(out_abs)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(out_abs, 'w', newline="\n") as f_out:
        f_out.writelines(shifted_lines)

    print(f"[moveGCode] Armed on first ';LAYER_CHANGE'.")
    print(f"[moveGCode] Shift began on first XY move after that.")
    print(f"[moveGCode] Applied X+{x_shift}, Y+{y_shift} mm to model toolpath only.")
    print(f"[moveGCode] Output written to: {out_abs}")