# _07combine.py  (segment-local safe seam travel)  — FIXED: natural numeric ordering
import os
import re

DEFAULT_TRAVEL_FEEDRATE = 6000.0   # fallback if nothing else is found


# --------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------

def _natural_key(filename: str):
    """
    Numeric-aware ordering so:
      test_1.gcode, test_2.gcode, ... test_10.gcode, test_11.gcode
    instead of lexicographic:
      test_1.gcode, test_10.gcode, test_11.gcode, test_2.gcode, ...
    """
    base = os.path.splitext(os.path.basename(filename))[0]

    # Prefer trailing digits (common pattern: *_<n>)
    m = re.search(r'(\d+)$', base)
    if m:
        prefix = base[:m.start(1)]
        return (prefix, int(m.group(1)))

    # Fallback: first digits anywhere
    m2 = re.search(r'(\d+)', base)
    if m2:
        prefix = base[:m2.start(1)]
        return (prefix, int(m2.group(1)))

    # No digits: keep stable by name after numbered ones
    return (base, float("inf"))


def _find_payload_start_index(lines):
    """Find index of first real printing ';TYPE:' line."""
    type_all = []
    type_non_custom = []

    for idx, row in enumerate(lines):
        if row.startswith(";TYPE:"):
            type_all.append(idx)
            if not row.startswith(";TYPE:Custom"):
                type_non_custom.append(idx)

    if type_non_custom:
        return type_non_custom[0]
    if type_all:
        return type_all[0]
    return 0


def _extract_print_payload(lines, is_first_file, context_lines_before_type=2):
    """For first file → keep all. For others → strip headers around first ';TYPE:'."""
    if is_first_file:
        return lines[:]

    t = _find_payload_start_index(lines)
    start_idx = max(t - context_lines_before_type, 0)
    return lines[start_idx:]


def _parse_F_from_line(line):
    parts = line.strip().split()
    for p in parts:
        if p.startswith("F"):
            try:
                return float(p[1:])
            except Exception:
                pass
    return None


def _is_feedrate_only_move(line):
    """Identify lines like: 'G1 F7200' or 'G0 F9000' — no XYZE."""
    stripped = line.strip()
    if not stripped.startswith(("G0", "G1")):
        return False
    parts = stripped.split()
    saw_F = False
    for p in parts[1:]:
        if p.startswith("F"):
            saw_F = True
        elif p.startswith(("X", "Y", "Z", "E")):
            return False   # has movement or extrusion → not feedrate-only
    return saw_F


def _find_first_internal_feedrate(lines):
    """
    Find the first feedrate *inside this segment itself* (ignoring comments).
    This is what we want to use for the seam travel of this segment.
    """
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(";"):
            continue
        f = _parse_F_from_line(line)
        if f is not None:
            return f
    return None


def _patch_first_movement(payload, travel_feedrate):
    """
    Patch one segment's payload:

      ✔ Removes all E-only lines before first XY
      ✔ Removes all feedrate-only lines before first XY
      ✔ Converts first XY to:
            G0 X.. Y.. Z.. F<travel_feedrate>
    """

    patched = []
    first_xy_fixed = False

    travel_F = travel_feedrate if travel_feedrate is not None else DEFAULT_TRAVEL_FEEDRATE

    for line in payload:
        stripped = line.strip()

        # keep comments as-is
        if stripped.startswith(";"):
            patched.append(line)
            continue

        # RULE 1: DELETE E-only commands BEFORE first XY
        if (not first_xy_fixed and
            ("E" in stripped) and
            ("X" not in stripped) and
            ("Y" not in stripped)):
            continue

        # RULE 2: DELETE feedrate-only moves BEFORE first XY
        if (not first_xy_fixed and _is_feedrate_only_move(line)):
            continue

        # RULE 3: FIRST XY MOVEMENT — rewrite to G0 travel with controlled F
        if (not first_xy_fixed and
            stripped.startswith(("G1", "G0")) and
            ("X" in stripped or "Y" in stripped)):

            parts = stripped.split()
            new_parts = ["G0"]   # enforce travel

            for p in parts[1:]:
                if p.startswith(("X", "Y", "Z")):
                    new_parts.append(p)
                # drop any E
                elif p.startswith("E"):
                    continue

            new_parts.append(f"F{travel_F:.0f}")
            patched.append(" ".join(new_parts) + "\n")
            first_xy_fixed = True
            continue

        patched.append(line)

    return patched


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------

def combineGCode(in_folder, out_file, context_lines_before_type=2):
    """
    Segment-local safe combiner:

      ✔ keep full header from first file
      ✔ strip headers from remaining files
      ✔ detect payload start correctly
      ✔ for each later segment:
          - use that segment's own first F (e.g. 1200 / 1800) as seam travel speed
          - if none exists, fall back to last global F, else DEFAULT_TRAVEL_FEEDRATE
      ✔ remove rogue F-only and E-only lines before first XY
      ✔ FIXED ORDER: numeric-aware sorting (1,2,3,...,10,11) not (1,10,11,2,...)
    """

    in_folder_abs = os.path.abspath(in_folder)
    out_file_abs  = os.path.abspath(out_file)

    print("[combineGCode] Combining from:", in_folder_abs)
    print("[combineGCode] Output:", out_file_abs)

    gcode_files = [
        f for f in os.listdir(in_folder_abs)
        if f.lower().endswith(".gcode") and not f.startswith(".")
    ]

    # --- FIX: numeric-aware order ---
    gcode_files.sort(key=_natural_key)

    print("[combineGCode] Order:", gcode_files)

    last_feedrate = None   # global modal feedrate across segments

    with open(out_file_abs, "w", newline="\n") as fout:
        for idx, fname in enumerate(gcode_files):
            full_path = os.path.join(in_folder_abs, fname)
            print(f"[combineGCode] Appending {full_path}")

            with open(full_path, "r") as fin:
                lines = fin.readlines()

            # Raw payload for this file (still unpatched)
            payload = _extract_print_payload(
                lines,
                is_first_file=(idx == 0),
                context_lines_before_type=context_lines_before_type,
            )

            # For non-first segments, patch the first XY move
            if idx > 0:
                # Segment-local preferred feedrate (e.g. the G1 F1200 inside test_2)
                local_feed = _find_first_internal_feedrate(payload)

                # If the segment has its own F, use that. Otherwise fall back.
                seam_travel_F = local_feed if local_feed is not None else last_feedrate

                payload = _patch_first_movement(payload, seam_travel_F)

            # Write combined output and keep tracking modal feedrate
            fout.write(f"; --- START OF SEGMENT {fname} ---\n")

            for line in payload:
                fout.write(line)

                fval = _parse_F_from_line(line)
                if fval is not None:
                    last_feedrate = fval

            fout.write(f"; --- END OF SEGMENT {fname} ---\n")

    print("[combineGCode] DONE:", out_file_abs)
