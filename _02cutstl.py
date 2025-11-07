import os
import subprocess
from _10config import get_slic3r_binary, CUT_CONFIG


def _read_cut_heights(cuts_file):
    """
    Read cut heights [mm] from cuts.txt.

    Behavior:
    - Parse each non-empty line as float (comma or dot decimal).
    - Drop duplicates.
    - Sort ascending.
    - Drop heights at or below CUT_CONFIG["ignore_cuts_at_or_below_mm"].
      (This handles the "0 mm should not force a cut" rule.)

    Returns
    -------
    cuts_sorted : list[float]
        e.g. [21.0, 25.7, 38.0, 46.3, 49.5]
    """
    ignore_min = CUT_CONFIG.get("ignore_cuts_at_or_below_mm", 0.0)

    raw_vals = []
    with open(cuts_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # accept "25,7" as "25.7"
            line = line.replace(",", ".")
            try:
                z_val = float(line)
            except ValueError:
                continue
            # apply ignore threshold
            if z_val > ignore_min:
                raw_vals.append(z_val)

    # unique + sort ascending
    raw_vals = sorted(set(raw_vals))

    return raw_vals


def cutSTL(in_stl, cuts_file, out_folder):
    """
    Cut an STL into stacked segments using Slic3r/SuperSlicer `--cut`,
    following the proven top-down strategy used in the working script.

    CORE IDEA (top-down peeling):
    --------------------------------
    1. We start with the *full* part in `in_stl`. This file WILL BE MODIFIED.
       (Important: upstream, you already copy the original STL into a
        working folder before calling cutSTL, so it's okay to mutate it.)

    2. We read the global cut heights from `cuts_file`, for example:
         [0, 21, 25.7, 38.0, 46.3, 49.5]
       We remove near-zero (0) because that "cut" is meaningless for the bottom.
       After cleanup we might have:
         [21, 25.7, 38.0, 46.3, 49.5]

    3. We then iterate these cut heights from TOP to BOTTOM:
         [49.5, 46.3, 38.0, 25.7, 21.0]
       For each cut height h:
         - We run: slic3r --dont-arrange --cut h -o out.stl <current_body.stl>
         - Slic3r creates:
             <current_body>.stl_upper.stl  (geometry ABOVE h)
             <current_body>.stl_lower.stl  (geometry BELOW h)
         - We IMMEDIATELY save the _upper.stl chunk aside as one final segment.
         - We REPLACE current_body.stl with _lower.stl.
           (So for the next, LOWER cut height we are cutting only the bottom remainder.)

       This exactly matches your "good" log:
       You kept cutting the SAME file, starting from the highest Z.

    4. After we've processed all cut heights,
       `in_stl` now holds the bottom-most chunk of the object.

    5. We then assemble the final stack in physical order bottom→top:
         [ bottom_chunk , ...peeled_upper_chunks reversed... ]

       Finally we rename them nicely into `out_folder`:
         <basename>_1.stl  (bottom segment)
         <basename>_2.stl
         ...
         <basename>_N.stl  (top segment)

    This ordering is important because downstream code assumes:
      *_1.stl is the first (lowest) thing to be printed = "bottom"
      *_N.stl is last (highest) = "top"
    """

    # Ensure output directory exists
    os.makedirs(out_folder, exist_ok=True)

    # Extract "test" from ".../test.stl"
    file_name = os.path.basename(in_stl)        # e.g. "test.stl"
    base_name, ext = os.path.splitext(file_name)
    if ext.lower() != ".stl":
        raise ValueError("cutSTL expects an .stl file as input")

    # --- 1. Read + sanitize cut heights
    cut_heights = _read_cut_heights(cuts_file)
    # Example after reading:
    #   [21.0, 25.7, 38.0, 46.3, 49.5]

    if len(cut_heights) == 0:
        # No valid cut → only one final part.
        final_dst = os.path.join(out_folder, f"{base_name}_1.stl")
        print("No valid cut heights → single segment:", final_dst)
        # We MOVE the file because downstream expects the part to live in stl_parts/
        os.replace(in_stl, final_dst)
        return

    # --- 2. Process cuts from highest Z to lowest Z
    # This is *critical* and matches your working behavior:
    # High cuts peel off thin top caps; after each cut the "_lower" chunk
    # becomes the new "body" for the next (lower) cut.
    cuts_desc = sorted(cut_heights, reverse=True)

    slic3r_bin = get_slic3r_binary()

    # We'll collect each peeled TOP chunk in the order we peel them.
    # So tmp_parts[0] will be the very topmost, tmp_parts[1] the next layer down, ...
    tmp_parts = []
    tmp_counter = 0

    for cut_z in cuts_desc:
        print(f"Cutting at Z = {cut_z} mm")

        # Build slic3r command
        cmd = [
            slic3r_bin,
            "--dont-arrange",
            "--cut", str(cut_z),
            "-o", "out.stl",
            in_stl
        ]
        print("Running:", " ".join(cmd))

        res = subprocess.run(cmd)
        if res.returncode != 0:
            raise RuntimeError(
                f"Slic3r --cut at Z={cut_z} failed (exit {res.returncode})"
            )

        upper_path = in_stl + "_upper.stl"
        lower_path = in_stl + "_lower.stl"

        # We expect both to exist if the cut plane truly lies inside the mesh.
        # In practice, on macOS Slic3r tends to generate both for valid cuts.
        # If one is missing, we throw because that's almost always geometry/logic mismatch.
        if not os.path.exists(upper_path):
            raise FileNotFoundError(
                f"Expected {upper_path} after cutting {in_stl} at Z={cut_z}, "
                "but it does not exist."
            )
        if not os.path.exists(lower_path):
            raise FileNotFoundError(
                f"Expected {lower_path} after cutting {in_stl} at Z={cut_z}, "
                "but it does not exist."
            )

        # Store the upper chunk immediately in a unique temp filename
        # inside out_folder, so it won't be overwritten by the next cut.
        tmp_name = os.path.join(out_folder, f"_tmp_part_{tmp_counter}.stl")
        os.replace(upper_path, tmp_name)
        tmp_parts.append(tmp_name)
        tmp_counter += 1

        # The lower part becomes the new "current body" that we continue cutting.
        # We overwrite in_stl with the lower geometry.
        os.replace(lower_path, in_stl)

    # After finishing all cuts, `in_stl` now contains the last/bottom-most chunk.
    bottom_chunk = in_stl

    # --- 3. Build ordered list from bottom to top
    # tmp_parts[0] is the very topmost piece (peeled first, from the largest cut_z),
    # tmp_parts[-1] is closer to the bottom.
    #
    # We want final physical stack: bottom -> ... -> top.
    #
    # So final order is:
    #   [bottom_chunk] + reversed(tmp_parts)
    ordered_parts = [bottom_chunk] + list(reversed(tmp_parts))

    print("Final bottom->top segment list:")
    for idx_debug, pth in enumerate(ordered_parts, start=1):
        print(f"  Segment {idx_debug}: {pth}")

    # --- 4. Rename into <base_name>_1.stl, _2.stl, ..., in out_folder
    for idx, src_path in enumerate(ordered_parts, start=1):
        dst_path = os.path.join(out_folder, f"{base_name}_{idx}.stl")
        print(f"Saving segment {idx} -> {dst_path}")
        os.replace(src_path, dst_path)

    print("Done cutting. Segments written to:", out_folder)


if __name__ == "__main__":
    import sys

    # Simple CLI fallback for manual testing:
    # python _02cutstl.py input.stl cuts.txt out_folder
    if len(sys.argv) != 4:
        print("Usage: python _02cutstl.py <input.stl> <cuts.txt> <out_folder>")
        sys.exit(1)

    in_stl_arg = sys.argv[1]
    cuts_arg   = sys.argv[2]
    out_dir    = sys.argv[3]

    cutSTL(in_stl_arg, cuts_arg, out_dir)