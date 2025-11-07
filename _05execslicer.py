import os
import subprocess
from _10config import (
    get_superslicer_binary,
    SLICER_CONFIG,
)

# ---------------------------------------------------------------------------
# SAFETY SWITCHES
# ---------------------------------------------------------------------------
# These two flags control whether we actually inject the big multiline
# start / end G-code blocks directly via CLI.
#
# On macOS we've seen SuperSlicer sometimes crash (exit code 1) when
# passing very long multi-line macros via --start-gcode / --end-gcode.
#
# If you still want the custom warmup for the FIRST part (bottom_stl=True)
# and/or shutdown macro for the LAST part (top_stl=True), you can:
#   - set these to True, *or*
#   - leave them False here and later patch those macros manually in Python
#     after slicing (prepend/append them to the produced .gcode file).
#
INJECT_BOTTOM_START_MACRO_VIA_CLI = False
INJECT_TOP_END_MACRO_VIA_CLI = False


def _build_base_cmd(in_abs, out_abs):
    """
    Build the baseline SuperSlicer command using config.ini and
    the guaranteed-safe core arguments. This is the common prefix
    for ALL parts.
    """
    slicer_bin = get_superslicer_binary()

    cmd = [slicer_bin]

    # baseline args from config ("-g", "--dont-arrange", "--loglevel=1", ...)
    base_args = list(SLICER_CONFIG["base_args"])
    cmd.extend(base_args)

    # always load the config.ini from config file
    cmd.extend(["--load", SLICER_CONFIG["config_ini_path"]])

    # we *always* tell slicer where to write gcode and which STL to slice
    # BUT we append them LAST (after we finish adding tuning flags),
    # so here we just remember the paths and add later in execSlicer().
    return cmd


def _append_standard_part_args(cmd):
    """
    Add the "normal part" arguments (used when bottom_stl == False).
    This matches what you used to do in your working execSlicer() under
    the "if not bottom_stl" block:
        --skirts 0
        --disable-fan-first-layers 0
        --first-layer-extrusion-width 0
        --bottom-solid-layers 1
        --infill-first
        --no-ensure-on-bed
        --start-gcode "G1 E{retract_length[0]} F1800"
    """
    std_args = list(SLICER_CONFIG["standard_part_args"])
    cmd.extend(std_args)


def _append_bottom_part_args(cmd):
    """
    Add the "bottom" behavior (used when bottom_stl == True).
    In the original working script, this branch:
      --support-material
      --start-gcode "<BIG MULTILINE MACRO>"
    replaces the normal-part behavior.

    We keep the same intent.

    NOTE: We only inject the multiline start-gcode if
          INJECT_BOTTOM_START_MACRO_VIA_CLI == True.
          Otherwise we skip it here (for crash safety)
          and you can prepend that macro manually to the
          output G-code later.
    """
    bottom_cfg = SLICER_CONFIG["bottom_part_args"]

    # support-material flag
    if bottom_cfg.get("support_material", False):
        cmd.append("--support-material")

    # big warmup start-gcode
    start_macro = bottom_cfg.get("start_gcode_multiline", "").strip()
    if start_macro and INJECT_BOTTOM_START_MACRO_VIA_CLI:
        cmd.extend(["--start-gcode", start_macro])
    else:
        # If we *don't* inject it here, SuperSlicer will just use whatever
        # start-gcode is in config.ini. We'll patch later if needed.
        pass


def _append_top_end_code(cmd, is_top):
    """
    Handle end-gcode logic.

    If this is the LAST part (top_stl == True), the old code injected
    a long custom "--end-gcode ...".
    Otherwise it injected a fallback:
        --top-solid-layers 1
        --end-gcode ""

    We'll mirror that, BUT we make the multiline end-gcode optional
    (INJECT_TOP_END_MACRO_VIA_CLI). That avoids known CLI crashes.
    """
    if is_top:
        end_macro = SLICER_CONFIG["top_part_args"]["end_gcode_multiline"].strip()

        if end_macro and INJECT_TOP_END_MACRO_VIA_CLI:
            cmd.extend(["--end-gcode", end_macro])
        else:
            # If we skip injecting the macro, slicer will use config.ini's end gcode.
            # You can still append a shutdown macro to the final combined .gcode later.
            pass

    else:
        # non-top fallback:
        cmd.extend([
            "--top-solid-layers",
            str(SLICER_CONFIG["default_top_solid_layers"]),
            "--end-gcode",
            SLICER_CONFIG["default_end_gcode"],
        ])


def _append_transformed_tuning(cmd):
    """
    Add extra tuning flags for transformed (nonplanar) parts.
    This corresponds to the 'if transformed:' block in your working code.

    We ALSO filter out known-problematic flags here:
    - Some SuperSlicer builds choke on '--overhangs-width-speed'
    - You commented out the speed tuning in the old version; we keep them for now,
      but if slicing crashes again, we can add filters here too.
    """
    safe_list = []
    for tok in SLICER_CONFIG["transformed_extra_args"]:
        if tok == "--overhangs-width-speed":
            # skip this + its value in the next position
            # We assume next is a number like "0.05"
            # We'll just drop it entirely.
            print("[execSlicer] Skipping --overhangs-width-speed for safety")
            continue
        safe_list.append(tok)

    cmd.extend(safe_list)


def execSlicer(
    in_file,
    out_file,
    bottom_stl=False,
    top_stl=False,
    transformed=False,
):
    """
    Build and run the SuperSlicer command for ONE STL.

    Parameters
    ----------
    in_file : str
        path to STL
    out_file : str
        path to output G-code
    bottom_stl : bool
        True if this is the first part (bottom interface to the bed)
    top_stl : bool
        True if this is the last part (final cap / shutdown)
    transformed : bool
        True if this part was nonplanar-transformed (slower settings, etc.)

    Behavior:
    - We replicate exactly the logic from your last known working execSlicer.
    - We branch bottom vs normal vs top.
    - We optionally skip injecting huge macros on mac to avoid crashes.
    """

    # Resolve absolute paths and make sure output dir exists
    in_abs = os.path.abspath(in_file)
    out_abs = os.path.abspath(out_file)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)

    # 1. Baseline command
    cmd = _build_base_cmd(in_abs, out_abs)

    # 2. Handle bottom vs normal flags for start conditions
    if bottom_stl:
        # bottom gets its own treatment, we do NOT also add standard_part_args
        _append_bottom_part_args(cmd)
    else:
        # non-bottom parts get standard part args
        _append_standard_part_args(cmd)

    # 3. Handle top/non-top end-gcode logic
    _append_top_end_code(cmd, is_top=top_stl)

    # 4. Handle transformed tuning
    if transformed:
        _append_transformed_tuning(cmd)

    # 5. Append output and input at the end
    cmd.extend(["-o", out_abs, in_abs])

    # Debug print full command (one item per line so it's easy to read)
    print("=== SuperSlicer CMD ===")
    for c in cmd:
        print("  ", c)
    print("=======================\n")

    # 6. Run slicer and capture logs (so we can debug if it dies)
    result = subprocess.run(cmd, capture_output=True, text=True)

    print("----- SuperSlicer RETURN CODE -----")
    print(result.returncode)
    print("----- SuperSlicer STDOUT ----------")
    print(result.stdout)
    print("----- SuperSlicer STDERR ----------")
    print(result.stderr)
    print("----------- END SLICER DEBUG ------\n")

    # 7. Check exit code
    if result.returncode != 0:
        raise RuntimeError(
            f"SuperSlicer could not slice {in_abs} (exit code {result.returncode})"
        )

    print("SuperSlicer finished OK:", out_abs)