# _10config.py
import os
import platform

# ---------------------------------------------------------------------------
#  OS DETECTION
# ---------------------------------------------------------------------------

def get_system_name():
    """
    Returns human-readable platform identifier:
    - 'Darwin' for macOS
    - 'Windows' for Windows
    - 'Linux' for Linux
    """
    return platform.system()


# ---------------------------------------------------------------------------
#  EXTERNAL BINARIES (SuperSlicer / Slic3r)
# ---------------------------------------------------------------------------

def get_superslicer_binary():
    """
    Returns path to SuperSlicer (used for slicing STL -> GCODE).
    PLEASE ADJUST the Windows path to your install location.
    """
    system_name = get_system_name()

    if system_name == "Darwin":  # macOS
        return "/Applications/SuperSlicer.app/Contents/MacOS/SuperSlicer"

    if system_name == "Windows":
        # EXAMPLE. Change this to where SuperSlicer.exe actually lives:
        return r"C:\Program Files\SuperSlicer\superslicer.exe"

    # fallback for Linux or custom env in PATH
    return "superslicer"


def get_slic3r_binary():
    """
    Returns path to Slic3r / SuperSlicer console for --cut operations.
    This is used in _02cutstl.py.

    PLEASE ADJUST the Windows path to the console slicer you use.
    """
    system_name = get_system_name()

    if system_name == "Darwin":
        # Example for a local Slic3r bundle shipped with the project
        return "/Applications/Slic3r.app/Contents/MacOS/Slic3r"

    if system_name == "Windows":
        # EXAMPLE. Update this to your slic3r-console.exe or superslicer_console.exe:
        return r"C:\Program Files\Slic3r\Slic3r-console.exe"

    # fallback
    return "slic3r"


# ---------------------------------------------------------------------------
#  PROJECT FOLDER LAYOUT
# ---------------------------------------------------------------------------

def make_folder_dict(base_name):
    """
    Returns dictionary with all working/output subfolders for a given part root.
    Example base_name: 'test' for test.stl.
    We do NOT create them here, only describe them.
    """
    root_dir = base_name  # we keep folder name equal to base name

    return {
        "root":         os.path.join(root_dir),
        "stl_parts":    os.path.join(root_dir, "stl_parts"),
        "stl_coarse":   os.path.join(root_dir, "stl_coarse"),
        "stl_tf":       os.path.join(root_dir, "stl_tf"),
        "tf_surfaces":  os.path.join(root_dir, "tf_surfaces"),
        "gcode_tf":     os.path.join(root_dir, "gcode_tf"),
        "gcode_parts":  os.path.join(root_dir, "gcode_parts"),
    }


# ---------------------------------------------------------------------------
#  GEOMETRY / BACKTRANSFORM / QUALITY CONTROL SETTINGS
# ---------------------------------------------------------------------------
# Everything here is used by refineMesh, transformGCode, slowdown logic, etc.

GEOMETRY_CONFIG = {
    # How finely we refine STL triangles before applying the nonplanar transform.
    # This gets passed to refineMesh(stl, maxlength=...).
    "refine_edge_length_mm": 1.0,

    # Max XY length [mm] of each linear toolpath subsegment after subdivision
    # during backtransform. Smaller = more accurate following of surface,
    # but more G-code.
    "maximal_segment_length_mm": 2.0,

    # Angle tolerance [deg] for detecting downward-facing triangles.
    # Triangles whose normal is within this angle of -Z (downwards) are "critical".
    "downward_angle_deg": 10.0,

    # Feedrate [mm/min] for perimeters above downward-facing regions.
    # We inject F<slow_feedrate> once when entering the slow zone.
    "slow_feedrate_mm_per_min": 180.0,

    # Minimal allowed Z after backtransform. This clamps the toolpath upward
    # to avoid diving below a desired safety plane.
    "z_desired_min_mm": 6.0,

    # XY shift applied during backtransform in transformGCode().
    # (In practice you were using +90/+90.)
    "xy_backtransform_shift_mm": (90.0, 90.0),
}


# ---------------------------------------------------------------------------
#  CUTTING SETTINGS (used by _02cutstl.py)
# ---------------------------------------------------------------------------

CUT_CONFIG = {
    # We ignore cut planes at or below this height (e.g. 0.0)
    "ignore_cuts_at_or_below_mm": 0.0,

    # Whether we sort cuts from low to high.
    "sort_cuts_ascending": True,
}


# ---------------------------------------------------------------------------
#  SLICER SETTINGS / SUPER SLICER ARGUMENTS
# ---------------------------------------------------------------------------
# Everything that controls how execSlicer builds its command.

SLICER_CONFIG = {
    # Path to your slicer profile. You can also make this absolute if needed.
    "config_ini_path": "config.ini",

    # Arguments that always apply (before any special bottom/top/transformed tuning).
    "base_args": [
        "-g",
        "--dont-arrange",
        "--loglevel=1",
        # NOTE: we will insert ["--load", config_ini_path] dynamically in execSlicer
    ],

    # Normal (not-first-layer-special) parts default tuning.
    "standard_part_args": [
        "--skirts", "0",
        "--disable-fan-first-layers", "0",
        "--first-layer-extrusion-width", "0",
        "--bottom-solid-layers", "0",
        "--top-solid-layers", "2",
        "--infill-first",
        "--no-ensure-on-bed",
        "--start-gcode", "G1 E{retract_length[0]} F1800",
    ],

    # Bottom part settings (first printed segment).
    "bottom_part_args": {
        "support_material": False,
        "start_gcode_multiline": (
            "G90 ; use absolute coordinates\n"
            "M83 ; extruder relative mode\n"
            "M104 S[first_layer_temperature] ; set extruder temp\n"
            "M140 S[first_layer_bed_temperature] ; set bed temp\n"
            "M190 S[first_layer_bed_temperature] ; wait for bed temp\n"
            "M109 S[first_layer_temperature] ; wait for extruder temp\n"
            "G28 W ; home all without mesh bed level\n"
            "G80 ; mesh bed leveling\n"
            "{if filament_settings_id[initial_tool]=~/.*Prusament PA11.*/}\n"
            "G1 Z0.3 F720\n"
            "G1 Y-3 F1000 ; go outside print area\n"
            "G92 E0\n"
            "G1 X60 E9 F1000 ; intro line\n"
            "G1 X100 E9 F1000 ; intro line\n"
            "{else}\n"
            "G1 Z0.2 F720\n"
            "G1 Y-3 F1000 ; go outside print area\n"
            "G92 E0\n"
            "G1 X60 E9 F1000 ; intro line\n"
            "G1 X100 E12.5 F1000 ; intro line\n"
            "{endif}\n"
            "G92 E0\n"
            "M221 S{if layer_height<0.075}100{else}95{endif}\n"
            "; Do not change E values below. Excessive value can damage the printer.\n"
            "{if print_settings_id=~/.*(DETAIL @MK3|QUALITY @MK3).*/}"
            "M907 E430 ; set extruder motor current{endif}\n"
            "{if print_settings_id=~/.*(SPEED @MK3|DRAFT @MK3).*/}"
            "M907 E538 ; set extruder motor current{endif}"
        ),
    },

    # Top part settings (last printed segment).
    "top_part_args": {
        "end_gcode_multiline": (
            "{if max_layer_z < max_print_height}"
            "G1 Z{z_offset+min(max_layer_z+49, max_print_height)} F720 ; Move head up"
            "{endif}\n"
            "G4 ; wait\n"
            "M221 S100 ; reset flow\n"
            "M900 K0 ; reset LA\n"
            "{if print_settings_id=~/.*(DETAIL @MK3|QUALITY @MK3|@0.25 nozzle MK3).*/}"
            "M907 E538 ; reset extruder motor current"
            "{endif}\n"
            "M104 S0 ; turn off temperature\n"
            "M140 S0 ; turn off heatbed\n"
            "M107 ; fan off\n"
            "M84 ; disable motors\n"
            "; max_layer_z = [max_layer_z]"
        ),
    },

    # Tuning for transformed (nonplanar) parts.
    "transformed_extra_args": [

        "--bridge-flow-ratio", "0.5",

        "--extrusion-width", "0.3",
        "--perimeter-extrusion-width", "0.3",
        "--external-perimeter-extrusion-width", "0.3",
        "--infill-extrusion-width", "0.3",
        "--solid-infill-extrusion-width", "0.3",
        "--top-infill-extrusion-width", "0.3",

        "--perimeters", "4",
        "--extrusion-multiplier", "0.9",
        "--solid-infill-below-area", "0",

        "--bottom-solid-layers", "1",
        "--top-solid-min-thickness", "0",
        "--bottom-solid-min-thickness", "0",

        "--extra-perimeters-overhangs", 
    ],

    # Default behavior for parts that are NOT top parts:
    "default_top_solid_layers": 1,
    "default_end_gcode": "",
}


# ---------------------------------------------------------------------------
#  PIPELINE CONFIG (global behavior of _00all.py)
# ---------------------------------------------------------------------------

PIPELINE_CONFIG = {
    # Whether to run transformGCode() also for planar parts with no tf_surface.
    "apply_backtransform_to_planar": False,

    # Whether to apply final XY shift to merged G-code via moveGCode().
    "apply_final_shift": True,

    # Where to drop the final toolpath on the build plate.
    # This is used by _08movegcode.py at the very end.
    "final_shift_xy_mm": (100.0, 100.0),

    # Markers that tell moveGCode() when the "real part" starts,
    # so purge/intro lines are not shifted.
    "move_start_markers": [
        ";layer:",
        "; layer ",
        ";begin layer",
        "; printing",
    ],

    # Keep a copy of the STL in the working root and cut that copy.
    "copy_input_to_work": True,
}
